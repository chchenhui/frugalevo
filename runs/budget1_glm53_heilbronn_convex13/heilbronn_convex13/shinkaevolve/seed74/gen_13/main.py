# EVOLVE-BLOCK-START
import numpy as np


def _min_triangle_area(pts):
    """Fast computation of the minimum area among all C(n,3) triangles."""
    n = len(pts)
    best = np.inf
    for i in range(n - 2):
        # vectorized over all (j,k) pairs with j>i, k>j
        v1 = pts[i + 1:, :] - pts[i]
        # cross products between all pairs
        m = len(v1)
        idx = np.triu_indices(m, k=1)
        # area = 0.5 * |cross(v_j - v_i, v_k - v_i)|
        # compute cross of all pairs of v1
        crosses = v1[:, 0][idx[0]] * v1[:, 1][idx[1]] - v1[:, 1][idx[0]] * v1[:, 0][idx[1]]
        a = 0.5 * np.min(np.abs(crosses)) if crosses.size else np.inf
        if a < best:
            best = a
    return best


def _hull_area(pts):
    """Convex hull area via monotone chain."""
    p = sorted(map(tuple, pts), key=lambda t: (t[0], t[1]))
    p = list(dict.fromkeys(p))
    if len(p) < 3:
        return 0.0

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

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


def _objective(pts):
    ha = _hull_area(pts)
    if ha <= 1e-12:
        return 0.0
    return _min_triangle_area(pts) / ha


def _refine(pts, steps=(0.06, 0.03, 0.015, 0.008, 0.004, 0.002, 0.001), iters=30):
    """Greedy coordinate-perturbation refinement of the min normalized area."""
    pts = pts.copy()
    best = _objective(pts)
    n = len(pts)
    rng = np.random.default_rng(7)
    for step in steps:
        for _ in range(iters):
            improved = False
            for i in range(n):
                for dim in range(2):
                    for sgn in (1, -1):
                        cand = pts.copy()
                        cand[i, dim] += sgn * step
                        v = _objective(cand)
                        if v > best + 1e-12:
                            pts, best, improved = cand, v, True
            if not improved:
                break
    return pts, best


def _candidates():
    """Deterministic structured initial configurations."""
    cands = []
    n = 13
    # 1) triangular lattice-like arrangement in unit square
    rows = [2, 3, 3, 3, 2]
    pts = []
    y = 0.1
    ys = np.linspace(0.08, 0.92, len(rows))
    xs_offsets = np.linspace(-0.04, 0.04, len(rows))
    for r, yy, off in zip(rows, ys, xs_offsets):
        xs = np.linspace(0.1, 0.9, r) + off
        for x in xs:
            pts.append((x, yy))
    cands.append(np.array(pts[:13]))

    # 2) hexagonal-ish ring: 6 outer on circle, 6 inner ring, 1 center
    t = np.linspace(0, 2 * np.pi, 7)[:-1]
    outer = np.stack([0.5 + 0.45 * np.cos(t), 0.5 + 0.45 * np.sin(t)], axis=1)
    inner = np.stack([0.5 + 0.24 * np.cos(t + np.pi / 6),
                      0.5 + 0.24 * np.sin(t + np.pi / 6)], axis=1)
    cand = np.vstack([outer, inner, [[0.5, 0.5]]])
    cands.append(cand)

    # 3) jittered grid, fixed seed
    rng = np.random.default_rng(42)
    g = np.array([(i / 3 + 0.08 + rng.random() * 0.02,
                   j / 3 + 0.08 + rng.random() * 0.02)
                  for i in range(4) for j in range(4)])[:13]
    cands.append(g)

    # 4) 3-fold symmetric configuration
    ang = 2 * np.pi / 3
    tri = []
    for k in range(3):
        a = k * ang
        tri.append((0.5 + 0.45 * np.cos(a), 0.5 + 0.45 * np.sin(a)))
        tri.append((0.5 + 0.28 * np.cos(a + 0.5), 0.5 + 0.28 * np.sin(a + 0.5)))
        tri.append((0.5 + 0.28 * np.cos(a - 0.5), 0.5 + 0.28 * np.sin(a - 0.5)))
        tri.append((0.5 + 0.12 * np.cos(a), 0.5 + 0.12 * np.sin(a)))
    cands.append(np.array(tri[:13]))
    return cands


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region in order
    to maximize the area of the smallest triangle formed by these points.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    best_pts = None
    best_val = -1.0
    for cand in _candidates():
        pts, val = _refine(cand)
        if val > best_val:
            best_val, best_pts = val, pts
    if best_pts is None:
        rng = np.random.default_rng(seed=42)
        best_pts = rng.random((13, 2))
    return best_pts


# EVOLVE-BLOCK-END
