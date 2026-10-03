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


def _init_ring(alt):
    """12 points on a ring (optionally alternating radii) plus the center."""
    ang = np.arange(12) * (2.0 * np.pi / 12.0)
    rad = np.where(np.arange(12) % 2 == 0, 1.0, 0.72) if alt else np.ones(12)
    pts = np.zeros((_N, 2))
    pts[:12, 0] = rad * np.cos(ang)
    pts[:12, 1] = rad * np.sin(ang)
    pts[12] = (0.0, 0.0)
    return pts


def _init_triangle():
    """3-fold symmetric configuration: corners + two inner rings + center."""
    pts = np.zeros((_N, 2))
    a0 = np.pi / 2.0
    for i in range(3):
        a = a0 + i * 2.0 * np.pi / 3.0
        pts[i] = (1.05 * np.cos(a), 1.05 * np.sin(a))
    for i in range(6):
        a = a0 + np.pi / 6.0 + i * np.pi / 3.0
        pts[3 + i] = (0.62 * np.cos(a), 0.62 * np.sin(a))
    for i in range(3):
        a = a0 + i * 2.0 * np.pi / 3.0
        pts[9 + i] = (0.30 * np.cos(a), 0.30 * np.sin(a))
    pts[12] = (0.0, 0.0)
    return pts


def _init_grid(rng):
    """Perturbed quasi-uniform grid in [-1,1]^2."""
    g = []
    for i in range(4):
        for j in range(4):
            g.append((i / 3.0, j / 3.0))
    g = np.array(g, dtype=float)
    keep = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14]
    pts = g[keep] * 2.0 - 1.0
    pts += rng.normal(0, 0.06, pts.shape)
    return pts


def _targeted_new(pts, t, i, step, rng):
    """Geometric move: push vertex i away from the line through the other
    two vertices of a small triangle (increases its area)."""
    old = pts[i]
    others = [int(x) for x in t if x != i]
    j, k2 = others[0], others[1]
    d = pts[k2] - pts[j]
    nrm = np.hypot(d[0], d[1]) + 1e-12
    nd = np.array([-d[1], d[0]]) / nrm
    side = d[0] * (old[1] - pts[j, 1]) - d[1] * (old[0] - pts[j, 0])
    if side < 0:
        nd = -nd
    return old + step * nd + rng.normal(0, step * 0.35, 2)


def _anneal(init, rng, iters, step0, temp0):
    """Targeted annealing on the smooth score (mean of K smallest areas),
    with axis-wise fallback moves on rejection for directional diversity."""
    K = 12
    pts = init.copy()
    a = _all_tri_areas(pts)
    smooth = float(np.sort(a)[:K].mean())
    best = pts.copy()
    best_min = float(a.min())
    step = step0
    T = temp0
    decay = temp0 ** (1.0 / iters)

    for it in range(iters):
        a = _all_tri_areas(pts)
        order = np.argsort(a)
        t = _IDX[order[rng.integers(K)]]
        i = int(t[rng.integers(3)])
        old = pts[i].copy()

        if rng.random() < 0.55:
            new = _targeted_new(pts, t, i, step, rng)
        else:
            new = old + rng.normal(0, step, 2)

        # reject degenerate moves (duplicate points)
        dists = np.hypot(*(pts - new).T)
        dists[i] = 10.0
        if dists.min() < 1e-6:
            new = None

        accepted = False
        if new is not None:
            pts[i] = new
            a2 = _all_tri_areas(pts)
            s2 = float(np.sort(a2)[:K].mean())
            if s2 >= smooth or rng.random() < np.exp((s2 - smooth) / max(T, 1e-12)):
                smooth = s2
                m2 = float(a2.min())
                if m2 > best_min:
                    best_min = m2
                    best = pts.copy()
                accepted = True
            else:
                pts[i] = old

        if not accepted:
            # axis-wise fallback: try +/- x, +/- y with the same step
            for d in range(2):
                for sgn in (1, -1):
                    cand = pts.copy()
                    cand[i, d] += sgn * step
                    pts[i] = cand[i]
                    a2 = _all_tri_areas(pts)
                    s2 = float(np.sort(a2)[:K].mean())
                    if s2 >= smooth or rng.random() < np.exp((s2 - smooth) / max(T, 1e-12)):
                        smooth = s2
                        m2 = float(a2.min())
                        if m2 > best_min:
                            best_min = m2
                            best = pts.copy()
                        accepted = True
                        break
                    else:
                        pts[i] = old
                if accepted:
                    break

        T *= decay
        if it % 600 == 599:
            step = max(step * 0.88, 0.0015)

    return best, best_min


def _polish(pts, rng, iters):
    """Strict local search: accept only moves that increase the true minimum."""
    pts = pts.copy()
    a = _all_tri_areas(pts)
    cur_min = float(a.min())
    step = 0.01
    for it in range(iters):
        order = np.argsort(a)
        t = _IDX[order[rng.integers(6)]]
        i = int(t[rng.integers(3)])
        old = pts[i].copy()
        new = _targeted_new(pts, t, i, step * 0.7, rng)
        dists = np.hypot(*(pts - new).T)
        dists[i] = 10.0
        if dists.min() < 1e-6:
            continue
        pts[i] = new
        a2 = _all_tri_areas(pts)
        m2 = float(a2.min())
        if m2 > cur_min:
            cur_min = m2
            a = a2
        else:
            pts[i] = old
        if it % 300 == 299:
            step = max(step * 0.9, 0.0005)
    return pts, cur_min


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points maximizing the smallest triangle
    area normalized by convex hull area, via worst-triangle targeted
    simulated annealing with structured multi-starts.
    """
    rng = np.random.default_rng(seed=42)
    inits = [_init_ring(False), _init_ring(True), _init_triangle(), _init_grid(rng)]

    global_best = None
    global_min = -1.0

    for init in inits:
        pts, m = _anneal(init, rng, iters=9000, step0=0.05, temp0=0.004)
        pts, m = _polish(pts, rng, iters=3000)
        if m > global_min:
            global_min = m
            global_best = pts

    # reheat cycles restarting from the global best
    for cyc in range(2):
        pts, m = _anneal(global_best.copy(), rng, iters=4000,
                         step0=0.02, temp0=0.002 * (0.5 ** cyc))
        pts, m = _polish(pts, rng, iters=2000)
        if m > global_min:
            global_min = m
            global_best = pts

    best_pts = global_best

    # rescale so the convex hull has unit area (scale-invariant problem)
    hull = _hull_area(best_pts)
    if hull > 1e-12:
        s = 1.0 / np.sqrt(hull)
        best_pts = (best_pts - best_pts.mean(axis=0)) * s + best_pts.mean(axis=0)

    return best_pts


# EVOLVE-BLOCK-END