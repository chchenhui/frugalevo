# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations

_V0 = np.array([0.0, 0.0])
_V1 = np.array([1.0, 0.0])
_V2 = np.array([0.5, np.sqrt(3.0) / 2.0])
_TRI_AREA = 0.5 * np.sqrt(3.0) / 2.0

# Precompute all triplet indices once.
_TRIP = np.array(list(combinations(range(11), 3)), dtype=int)


def _min_area(pts):
    """Minimum (normalized) triangle area over all triplets of points."""
    a = pts[_TRIP[:, 0]]
    b = pts[_TRIP[:, 1]]
    c = pts[_TRIP[:, 2]]
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    return np.min(np.abs(cross)) / (2.0 * _TRI_AREA)


def _clip(pts):
    """Project points back into the equilateral triangle via barycentric clipping."""
    out = pts.copy()
    for i in range(out.shape[0]):
        p = out[i]
        # barycentric coordinates w.r.t. V0, V1, V2
        v2v = _V2 - _V0
        u = p - _V0
        b1 = (u[0] * v2v[1] - u[1] * v2v[0]) / (v2v[0] * 0.0 - v2v[0] * v2v[1] + v2v[1] * v2v[0])
        # solve properly: use explicit formulas
        y = p[1]
        x = p[0]
        h = np.sqrt(3.0) / 2.0
        l2 = y / h
        l1 = x - 0.5 * l2
        l0 = 1.0 - l1 - l2
        if l0 >= 0 and l1 >= 0 and l2 >= 0:
            continue
        # clamp lambdas and renormalize
        lam = np.array([l0, l1, l2])
        lam = np.maximum(lam, 0.0)
        lam /= lam.sum()
        out[i] = lam[0] * _V0 + lam[1] * _V1 + lam[2] * _V2
    return out


def _initial(seed):
    """Perturbed triangular lattice (order-3 grid + centroid), perturbed to break collinearity."""
    rng = np.random.default_rng(seed)
    pts = []
    k = 3
    for i in range(k + 1):
        for j in range(k + 1 - i):
            l = k - i - j
            pts.append((i * _V0 + j * _V1 + l * _V2) / k)
    pts.append(np.array([0.5, h := np.sqrt(3.0) / 6.0]))
    pts = np.array(pts)
    return _clip(pts + rng.normal(0.0, 0.02, pts.shape))


def _bottleneck_move(pts, rng, nb=8, step=0.02):
    """Move up to 3 points from the smallest-area triangles along their
    signed-area gradients (coordinated multi-point move)."""
    a = pts[_TRIP[:, 0]]
    b = pts[_TRIP[:, 1]]
    c = pts[_TRIP[:, 2]]
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    areas = 0.5 * np.abs(cross)
    order = np.argsort(areas)[:nb]
    s = np.where(cross >= 0, 1.0, -1.0)
    gi = 0.5 * s * np.stack([c[:, 1] - b[:, 1], b[:, 0] - c[:, 0]], axis=1)
    gj = 0.5 * s * np.stack([a[:, 1] - c[:, 1], c[:, 0] - a[:, 0]], axis=1)
    gk = 0.5 * s * np.stack([b[:, 1] - a[:, 1], a[:, 0] - b[:, 0]], axis=1)
    gmap = [gi, gj, gk]
    # membership weights of points in bottleneck triangles
    cnt = np.zeros(pts.shape[0])
    for t in order:
        cnt[_TRIP[t]] += 1.0
    # pick 1-3 points, membership-weighted
    k = 1 if rng.random() < 0.35 else (2 if rng.random() < 0.6 else 3)
    top = list(np.argsort(-cnt)[:k])
    dirs = np.zeros_like(pts)
    touched = set()
    for t in order:
        for local, pi in enumerate(_TRIP[t]):
            if pi in top and pi not in touched:
                g = gmap[local][t]
                nrm = np.hypot(g[0], g[1])
                if nrm > 1e-15:
                    dirs[pi] = g / nrm
                touched.add(pi)
    cand = pts + step * dirs
    return _clip(cand)


def _refine(pts, rng, steps=(0.04, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005)):
    best = _clip(pts)
    best_val = _min_area(best)
    n_dirs = 12
    ang = 2.0 * np.pi * np.arange(n_dirs) / n_dirs
    dirs = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    # Phase 1: coordinated bottleneck-gradient ascent with annealing reheats
    for _ in range(3):  # up to 3 passes (reheat cycles)
        b = best.copy()
        cur = best_val
        step = 0.02
        sigma = 0.006
        stall = 0
        for _ in range(1200):
            ok = False
            s = step
            for _ in range(5):
                cand = _bottleneck_move(b, rng, step=s)
                v = _min_area(cand)
                if v > cur + 1e-14:
                    b, cur = cand, v
                    ok = True
                    break
                s *= 0.5
            if ok:
                stall = 0
                step = min(step * 1.25, 0.02)
                if cur > best_val:
                    best_val, best = cur, b.copy()
            else:
                stall += 1
                step *= 0.5
                if step < 2e-5 or stall > 10:
                    # annealing kick from best, restart temperature
                    kick = _clip(best + rng.normal(0.0, sigma, best.shape))
                    kv = _min_area(kick)
                    if kv > best_val * 0.92 or kv > cur - 4.0 * sigma * sigma:
                        b, cur = kick, kv
                    sigma *= 0.75
                    if sigma < 5e-4:
                        break  # reheats exhausted
                    step = 0.02
                    stall = 0
    # Phase 2: single-point pattern-search polish
    for s in steps:
        improved = True
        while improved:
            improved = False
            for i in range(best.shape[0]):
                for d in dirs:
                    cand = best.copy()
                    cand[i] += s * d
                    cand = _clip(cand)
                    v = _min_area(cand)
                    if v > best_val + 1e-12:
                        best_val = v
                        best = cand
                        improved = True
    return best, best_val


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct an arrangement of 11 points on or inside the equilateral triangle with
    vertices (0,0), (1,0), (0.5, sqrt(3)/2), maximizing the minimum triangle area.

    Deterministic multi-start local search over a perturbed triangular lattice.
    Falls back to the initial configuration if refinement fails.
    """
    best_pts = None
    best_val = -1.0
    try:
        for seed in range(6):
            rng = np.random.default_rng(1234 + seed)
            init = _initial(1234 + seed)
            pts, val = _refine(init, rng)
            if val > best_val:
                best_val = val
                best_pts = pts
    except Exception:
        if best_pts is None:
            best_pts = _initial(1234)
    return np.asarray(best_pts, dtype=float)


# EVOLVE-BLOCK-END