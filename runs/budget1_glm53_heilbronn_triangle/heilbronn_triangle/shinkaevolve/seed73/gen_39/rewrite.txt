# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

H = np.sqrt(3.0) / 2.0
TRI_AREA = np.sqrt(3.0) / 4.0
TRIPLETS = np.array(list(combinations(range(11), 3)), dtype=int)


def _bary_from_xy(pts):
    x, y = pts[:, 0], pts[:, 1]
    w2 = 2.0 * y / np.sqrt(3.0)
    w1 = x - y / np.sqrt(3.0)
    w0 = 1.0 - w1 - w2
    return np.stack([w0, w1, w2], axis=1)


def _xy_from_bary(w):
    w = np.clip(w, 0.0, None)
    s = w.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    w = w / s
    return np.stack([w[:, 1] + 0.5 * w[:, 2], H * w[:, 2]], axis=1)


def _all_areas(pts):
    p = pts[TRIPLETS]
    a = p[:, 1] - p[:, 0]
    b = p[:, 2] - p[:, 0]
    return 0.5 * np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])


def _hard_min(pts):
    areas = _all_areas(pts)
    k = int(np.argmin(areas))
    return float(areas[k]), k


def _soft_obj(areas, k_sharp):
    # soft-min over the smallest 5% of triplet areas with sharpness k_sharp
    m = max(3, int(len(areas) * 0.05))
    smallest = np.partition(areas, m - 1)[:m]
    mx = smallest.max()
    return mx + np.log(np.mean(np.exp(-k_sharp * (smallest - mx)))) / k_sharp


def _phase1_annealed(pts, step0=0.02, iters=60):
    """Pattern search with soft-min sharpness annealed 30 -> 120."""
    pts = _xy_from_bary(_bary_from_xy(pts))
    step = step0
    best_soft = _soft_obj(_all_areas(pts), 30.0)
    for it in range(iters):
        k_sharp = 30.0 + 90.0 * it / max(1, iters - 1)
        improved = False
        for i in range(11):
            for d in range(2):
                for s in (+1, -1):
                    cand = pts.copy()
                    cand[i, d] += s * step
                    cand = _xy_from_bary(_bary_from_xy(cand))
                    v = _soft_obj(_all_areas(cand), k_sharp)
                    if v > best_soft + 1e-13:
                        pts, best_soft = cand, v
                        improved = True
        if not improved:
            step *= 0.5
            if step < 1e-6:
                break
    return pts


def _metropolis_phase(pts, rng, t_budget, step0=0.008):
    """Metropolis hill-climb on soft-min objective (annealed sharpness)."""
    pts = _xy_from_bary(_bary_from_xy(pts))
    cur_soft = _soft_obj(_all_areas(pts), 60.0)
    best = pts.copy()
    best_hard, _ = _hard_min(pts)
    step = step0
    deadline = time.time() + t_budget
    it = 0
    while time.time() < deadline:
        it += 1
        cand = pts.copy()
        i = int(rng.integers(11))
        cand[i] += rng.normal(0.0, step, size=2)
        cand = _xy_from_bary(_bary_from_xy(cand))
        v = _soft_obj(_all_areas(cand), 60.0)
        if v >= cur_soft or rng.random() < np.exp((v - cur_soft) * 40.0):
            pts, cur_soft = cand, v
            hv, _ = _hard_min(pts)
            if hv > best_hard:
                best, best_hard = pts.copy(), hv
        if it % 2000 == 0:
            step *= 0.7
            if step < 1e-5:
                break
    return best


def _greedy_polish(pts, step=0.004):
    """Deterministic coordinate polish on hard-min with step halving."""
    pts = _xy_from_bary(_bary_from_xy(pts))
    best, _ = _hard_min(pts)
    while step >= 1e-7:
        improved = False
        for i in range(11):
            for d in range(2):
                for s in (+1, -1):
                    cand = pts.copy()
                    cand[i, d] += s * step
                    cand = _xy_from_bary(_bary_from_xy(cand))
                    v, _ = _hard_min(cand)
                    if v > best + 1e-15:
                        pts, best = cand, v
                        improved = True
        if not improved:
            step *= 0.5
    return pts, best


def _lattice_seed():
    pts = []
    for r in range(5):
        y = H * (r + 0.5) / 5
        cnt = 5 - r
        for c in range(cnt):
            x = (c + 0.5) / cnt * (1.0 - y / np.sqrt(3.0)) + (y / np.sqrt(3.0)) * 0.5
            pts.append((x, y))
    pts = np.array(pts)
    if len(pts) > 11:
        cen = pts.mean(axis=0)
        d = np.linalg.norm(pts - cen, axis=1)
        keep = np.argsort(d)[::-1][:11]
        pts = pts[np.sort(keep)]
    return _xy_from_bary(_bary_from_xy(pts[:11]))


def _random_seed(rng):
    u, v = rng.random(11), rng.random(11)
    su = np.sqrt(u)
    w = np.stack([1 - su, su * (1 - v), su * v], axis=1)
    return _xy_from_bary(w)


def _optimize():
    rng = np.random.default_rng(12345)
    starts = [_lattice_seed()] + [_random_seed(rng) for _ in range(10)]
    results = []
    for s in starts:
        try:
            cand = _phase1_annealed(s)
            val, _ = _hard_min(cand)
            results.append((val, cand))
        except Exception:
            continue
    results.sort(key=lambda t: t[0], reverse=True)

    best_pts, best_val = _lattice_seed(), -1.0
    # Phase 2: Metropolis + vertex-kick restarts on the top candidates,
    # each finalized by deterministic greedy polish.
    deadline = time.time() + 2.5
    for val, cand in results[:3]:
        try:
            c = _metropolis_phase(cand, rng, t_budget=0.6)
            c, v = _greedy_polish(c)
            if v > best_val:
                best_pts, best_val = c, v
            # basin-hopping kicks on the minimal triangle's vertices
            base = c
            while time.time() < deadline:
                kick = base.copy()
                _, kmin = _hard_min(base)
                for j in TRIPLETS[kmin]:
                    kick[j] += rng.normal(0.0, 0.01, size=2)
                kick = _xy_from_bary(_bary_from_xy(kick))
                kick = _metropolis_phase(kick, rng, t_budget=0.3)
                kick, v = _greedy_polish(kick)
                if v > best_val + 1e-14:
                    best_pts, best_val = kick, v
                    base = kick
                else:
                    break
        except Exception:
            continue
    if not np.all(np.isfinite(best_pts)):
        best_pts = _lattice_seed()
    return _xy_from_bary(_bary_from_xy(best_pts))


try:
    _POINTS = _optimize()
except Exception:
    _POINTS = _lattice_seed()


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 11.

    Returns:
        points: np.ndarray of shape (11,2) with the x,y coordinates of the points.
    """
    return _POINTS.copy()

# EVOLVE-BLOCK-END