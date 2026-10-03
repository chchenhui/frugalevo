# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations

N = 13
TRIPLES = np.array(list(combinations(range(N), 3)), dtype=int)
K_SMALL = 12  # number of smallest triangles tracked by the smooth score


def _areas(points: np.ndarray) -> np.ndarray:
    """Vectorized areas of all C(13,3) triangles."""
    p = points[TRIPLES[:, 0]]
    q = points[TRIPLES[:, 1]]
    r = points[TRIPLES[:, 2]]
    return 0.5 * np.abs(
        (q[:, 0] - p[:, 0]) * (r[:, 1] - p[:, 1])
        - (q[:, 1] - p[:, 1]) * (r[:, 0] - p[:, 0])
    )


def _hull_area(points: np.ndarray) -> float:
    """Area of the convex hull (monotone chain + shoelace)."""
    pts = points[np.lexsort((points[:, 1], points[:, 0]))]
    if len(pts) <= 2:
        return 0.0

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = np.array(lower[:-1] + upper[:-1])
    x, y = hull[:, 0], hull[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _init_ring(alt: bool) -> np.ndarray:
    """12 points on a ring (optionally alternating radii) plus the center."""
    ang = np.arange(12) * (2.0 * np.pi / 12.0)
    rad = np.where(np.arange(12) % 2 == 0, 1.0, 0.72) if alt else np.ones(12)
    pts = np.zeros((N, 2))
    pts[:12, 0] = rad * np.cos(ang)
    pts[:12, 1] = rad * np.sin(ang)
    pts[12] = (0.0, 0.0)
    return pts


def _init_triangle() -> np.ndarray:
    """3-fold symmetric configuration: corners + two inner rings + center."""
    pts = np.zeros((N, 2))
    a0 = np.pi / 2.0
    for i in range(3):  # outer corners
        a = a0 + i * 2.0 * np.pi / 3.0
        pts[i] = (1.05 * np.cos(a), 1.05 * np.sin(a))
    for i in range(6):  # mid ring, offset 30 degrees
        a = a0 + np.pi / 6.0 + i * np.pi / 3.0
        pts[3 + i] = (0.62 * np.cos(a), 0.62 * np.sin(a))
    for i in range(3):  # inner ring
        a = a0 + i * 2.0 * np.pi / 3.0
        pts[9 + i] = (0.30 * np.cos(a), 0.30 * np.sin(a))
    pts[12] = (0.0, 0.0)
    return pts


def _init_grid(rng) -> np.ndarray:
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
    """Targeted annealing on the smooth score (mean of K smallest areas)."""
    pts = init.copy()
    a = _areas(pts)
    smooth = float(np.sort(a)[:K_SMALL].mean())
    best = pts.copy()
    best_min = float(a.min())
    step = step0
    T = temp0
    decay = temp0 ** (1.0 / iters)

    for it in range(iters):
        a = _areas(pts)
        order = np.argsort(a)
        t = TRIPLES[order[rng.integers(K_SMALL)]]
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
            continue

        pts[i] = new
        a2 = _areas(pts)
        s2 = float(np.sort(a2)[:K_SMALL].mean())
        if s2 >= smooth or rng.random() < np.exp((s2 - smooth) / max(T, 1e-12)):
            smooth = s2
            m2 = float(a2.min())
            if m2 > best_min:
                best_min = m2
                best = pts.copy()
        else:
            pts[i] = old
        T *= decay
        if it % 600 == 599:
            step = max(step * 0.88, 0.0015)

    return best, best_min


def _polish(pts, rng, iters):
    """Strict local search: accept only moves that increase the true minimum."""
    pts = pts.copy()
    a = _areas(pts)
    cur_min = float(a.min())
    step = 0.01
    for it in range(iters):
        order = np.argsort(a)
        t = TRIPLES[order[rng.integers(6)]]
        i = int(t[rng.integers(3)])
        old = pts[i].copy()
        new = _targeted_new(pts, t, i, step * 0.7, rng)
        dists = np.hypot(*(pts - new).T)
        dists[i] = 10.0
        if dists.min() < 1e-6:
            continue
        pts[i] = new
        a2 = _areas(pts)
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
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
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
