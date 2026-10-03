# EVOLVE-BLOCK-START
import numpy as np


def _all_triple_indices(n):
    idx = [(i, j, k) for i in range(n) for j in range(i + 1, n) for k in range(j + 1, n)]
    return np.array(idx)


def _min_triangle_area(points, triples):
    p = points[triples]  # (T,3,2)
    a = p[:, 1] - p[:, 0]
    b = p[:, 2] - p[:, 0]
    areas = 0.5 * np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
    return areas.min(), areas


def _soft_objective(areas, frac=0.15):
    # soft-min over the smallest `frac` of triangle areas (maximize)
    k = max(3, int(len(areas) * frac))
    smallest = np.partition(areas, k - 1)[:k]
    # power-mean with negative exponent: smoothly emphasizes small areas
    return -np.log(np.mean(np.exp(-smallest * 200.0))) / 200.0


def _inside_triangle(pts, eps=1e-9):
    # barycentric containment in equilateral triangle (0,0),(1,0),(0.5,h)
    h = np.sqrt(3.0) / 2.0
    x, y = pts[:, 0], pts[:, 1]
    b = np.stack([1.0 - y / h - (x - 0.5 * y / h) * 0.0,  # placeholder replaced below
                  x - y / np.sqrt(3.0),
                  y - np.sqrt(3.0) * (1.0 - x)], axis=1)
    # proper barycentric coords: vertices v0=(0,0), v1=(1,0), v2=(0.5,h)
    v2y = h
    w1 = x - y / np.sqrt(3.0)
    w2 = y * 2.0 / (np.sqrt(3.0))
    w0 = 1.0 - w1 - w2
    return np.minimum(w0, np.minimum(w1, w2)) >= -eps


def _project_inside(pts):
    h = np.sqrt(3.0) / 2.0
    for i in range(len(pts)):
        x, y = pts[i]
        w1 = x - y / np.sqrt(3.0)
        w2 = 2.0 * y / np.sqrt(3.0)
        w0 = 1.0 - w1 - w2
        if min(w0, w1, w2) < 0:
            w = np.array([w0, w1, w2])
            w = np.clip(w, 0.0, None)
            w = w / w.sum()
            pts[i, 0] = w[1] + 0.5 * w[2]
            pts[i, 1] = h * w[2]
    return pts


def _pattern_search(pts, triples, steps, iters):
    pts = pts.copy()
    best_soft = _soft_objective(_all_areas(pts, triples))
    for it in range(iters):
        improved = False
        for i in range(len(pts)):
            for d in range(2):
                for s in (+1, -1):
                    cand = pts.copy()
                    cand[i, d] += s * steps
                    cand = _project_inside(cand)
                    val = _soft_objective(_all_areas(cand, triples))
                    if val > best_soft + 1e-12:
                        pts, best_soft = cand, val
                        improved = True
        if not improved:
            steps *= 0.5
            if steps < 1e-6:
                break
    return pts


def _all_areas(pts, triples):
    p = pts[triples]
    a = p[:, 1] - p[:, 0]
    b = p[:, 2] - p[:, 0]
    return 0.5 * np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])


def _hard_refine(pts, triples, steps, rounds):
    pts = pts.copy()
    best, _ = _min_triangle_area(pts, triples)
    for r in range(rounds):
        improved = False
        for i in range(len(pts)):
            for d in range(2):
                for s in (+1, -1):
                    cand = pts.copy()
                    cand[i, d] += s * steps
                    cand = _project_inside(cand)
                    val, _ = _min_triangle_area(cand, triples)
                    if val > best + 1e-12:
                        pts, best = cand, val
                        improved = True
        if not improved:
            steps *= 0.5
            if steps < 1e-6:
                break
    return pts


def _lattice_seed():
    h = np.sqrt(3.0) / 2.0
    pts = []
    rows = 5
    for r in range(rows):
        y = h * (r + 0.5) / rows
        cnt = rows - r
        for c in range(cnt):
            x = (c + 0.5) / cnt * (1.0 - y / np.sqrt(3.0)) + (y / np.sqrt(3.0)) * 0.5
            pts.append((x, y))
    pts = np.array(pts)
    if len(pts) > 11:
        # drop points closest to centroid to keep exactly 11
        cen = pts.mean(axis=0)
        d = np.linalg.norm(pts - cen, axis=1)
        keep = np.argsort(d)[::-1][:11]
        pts = pts[np.sort(keep)]
    return pts[:11]


def _optimize():
    n = 11
    triples = _all_triple_indices(n)
    rng = np.random.default_rng(12345)
    h = np.sqrt(3.0) / 2.0
    starts = [_lattice_seed()]
    for _ in range(12):
        u, v = rng.random(11), rng.random(11)
        # uniform sampling in triangle via barycentric coords
        su = np.sqrt(u)
        w0, w1, w2 = 1 - su, su * (1 - v), su * v
        p = np.stack([w1 + 0.5 * w2, h * w2], axis=1)
        starts.append(p)
    best_pts, best_val = None, -1.0
    for s in starts:
        cand = _pattern_search(s, triples, 0.02, 60)
        cand = _hard_refine(cand, triples, 0.01, 40)
        val, _ = _min_triangle_area(cand, triples)
        if val > best_val:
            best_pts, best_val = cand, val
    return _project_inside(best_pts)


try:
    _POINTS = _optimize()
except Exception:
    # graceful fallback: simple lattice arrangement
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