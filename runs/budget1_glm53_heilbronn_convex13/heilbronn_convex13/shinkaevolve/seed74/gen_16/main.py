# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations

_N = 13
_IDX = np.array(list(combinations(range(_N), 3)))  # (286, 3)


def _all_tri_areas(pts):
    """Vectorized areas of all C(13,3) triangles."""
    a = pts[_IDX[:, 0]]
    b = pts[_IDX[:, 1]]
    c = pts[_IDX[:, 2]]
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    return 0.5 * np.abs(cross)


def _hull_area(pts):
    p = sorted(map(tuple, pts), key=lambda t: (t[0], t[1]))
    p = list(dict.fromkeys(p))
    if len(p) < 3:
        return 0.0

    def cross(o, x, y):
        return (x[0]-o[0])*(y[1]-o[1]) - (x[1]-o[1])*(y[0]-o[0])

    lower = []
    for pt in p:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], pt) <= 0:
            lower.pop()
        lower.append(pt)
    upper = []
    for pt in reversed(p):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], pt) <= 0:
            upper.pop()
        upper.append(pt)
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return 0.0
    h = np.array(hull)
    s = 0.0
    for i in range(len(h)):
        x1, y1 = h[i]
        x2, y2 = h[(i + 1) % len(h)]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def _score(pts):
    ha = _hull_area(pts)
    if ha <= 1e-12:
        return 0.0
    return _all_tri_areas(pts).min() / ha


def _seeds():
    cands = []
    # 1) hexagonal rings: 6 outer + 6 inner + center (3-fold symmetric)
    t = np.linspace(0, 2 * np.pi, 7)[:-1]
    outer = np.stack([0.5 + 0.47 * np.cos(t), 0.5 + 0.47 * np.sin(t)], axis=1)
    inner = np.stack([0.5 + 0.26 * np.cos(t + np.pi / 6),
                      0.5 + 0.26 * np.sin(t + np.pi / 6)], axis=1)
    cands.append(np.vstack([outer, inner, [[0.5, 0.5]]]))

    # 2) two staggered hexagons, different radii
    inner2 = np.stack([0.5 + 0.33 * np.cos(t), 0.5 + 0.33 * np.sin(t)], axis=1)
    outer2 = np.stack([0.5 + 0.48 * np.cos(t + np.pi / 6),
                       0.5 + 0.48 * np.sin(t + np.pi / 6)], axis=1)
    cands.append(np.vstack([outer2, inner2, [[0.5, 0.5]]]))

    # 3) regular 13-gon on circle (boundary-only baseline)
    ang = np.linspace(0, 2 * np.pi, 14)[:-1]
    poly = np.stack([0.5 + 0.48 * np.cos(ang), 0.5 + 0.48 * np.sin(ang)], axis=1)
    cands.append(poly)

    # 4) jittered grid, fixed seed
    rng = np.random.default_rng(123)
    g = []
    for i in range(4):
        for j in range(4):
            g.append((0.125 + i * 0.25 + rng.random() * 0.05,
                      0.125 + j * 0.25 + rng.random() * 0.05))
    cands.append(np.array(g[:13]))

    # 5) triangular lattice, 13 points
    pts = []
    r = 0
    y = 0.08
    while len(pts) < 13:
        cnt = min(5 - abs(r - 2), 13 - len(pts))
        xs = np.linspace(0.1, 0.9, max(cnt, 1)) + 0.03 * (r % 2)
        for x in xs[:cnt]:
            pts.append((min(max(x, 0.02), 0.98), y))
        r += 1
        y += 0.19
    cands.append(np.array(pts))
    return cands


def _anneal(pts, iters=3000, t0=0.004, t1=1e-5, rng=None):
    """Worst-triangle targeted simulated annealing."""
    if rng is None:
        rng = np.random.default_rng(0)
    pts = pts.copy()
    best = _score(pts)
    best_pts = pts.copy()
    k_bot = 8  # target vertices from bottom-k triangles
    for it in range(iters):
        temp = t0 * (t1 / t0) ** (it / iters)
        step = 0.05 * (0.002 / 0.05) ** (it / iters) + 0.0005
        areas = _all_tri_areas(pts)
        order = np.argsort(areas)[:k_bot]
        verts = np.unique(_IDX[order].ravel())
        vi = verts[rng.integers(len(verts))]
        cand = pts.copy()
        cand[vi] += rng.normal(0, step, 2)
        cand[vi] = np.clip(cand[vi], 0.0, 1.0)
        # avoid duplicates
        if np.min(np.linalg.norm(np.delete(cand, vi, axis=0) - cand[vi], axis=1)) < 1e-6:
            continue
        s_new = _score(cand)
        if s_new >= best or rng.random() < np.exp((s_new - best) / max(temp, 1e-12)):
            pts = cand
            if s_new > best:
                best = s_new
                best_pts = cand.copy()
    return best_pts, best


def _polish(pts, iters=1500):
    """Greedy targeted refinement at small steps from annealing output."""
    pts = pts.copy()
    best = _score(pts)
    rng = np.random.default_rng(9)
    for it in range(iters):
        areas = _all_tri_areas(pts)
        order = np.argsort(areas)[:6]
        verts = np.unique(_IDX[order].ravel())
        improved = False
        for vi in verts:
            for d in range(2):
                for sgn in (1, -1):
                    for step in (0.004, 0.001):
                        cand = pts.copy()
                        cand[vi, d] = np.clip(cand[vi, d] + sgn * step, 0.0, 1.0)
                        s = _score(cand)
                        if s > best + 1e-12:
                            pts, best, improved = cand, s, True
        if not improved:
            break
    return pts, best


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points maximizing the smallest triangle
    area normalized by convex hull area, via worst-triangle targeted
    simulated annealing with structured multi-starts.
    """
    best_pts, best_val = None, -1.0
    seeds = _seeds()
    for si, cand in enumerate(seeds):
        rng = np.random.default_rng(100 + si)
        p, v = _anneal(cand, iters=2500, rng=rng)
        # short reheat then decay
        p2, v2 = _anneal(p, iters=800, t0=0.002, t1=1e-6, rng=np.random.default_rng(200 + si))
        if v2 > v:
            p, v = p2, v2
        p, v = _polish(p)
        if v > best_val:
            best_val, best_pts = v, p
    if best_pts is None:
        best_pts = np.random.default_rng(seed=42).random((13, 2))
    return best_pts


# EVOLVE-BLOCK-END
