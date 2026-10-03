# EVOLVE-BLOCK-START
import numpy as np


def _min_area(points, idx_i, idx_j, idx_k):
    """Vectorized minimum absolute triangle area over all triplets (normalized by container area)."""
    pi = points[idx_i]
    pj = points[idx_j]
    pk = points[idx_k]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    return 0.5 * np.min(np.abs(cross)) / (np.sqrt(3) / 4.0)


def _clip_bary(p, eps=0.0):
    """Clip a point into the triangle with vertices (0,0),(1,0),(0.5,sqrt(3)/2) via barycentric coords."""
    s = np.sqrt(3.0)
    h = s / 2.0
    x, y = p
    # Barycentric weights: a (apex), b (origin), c ((1,0)).
    # Point = a*(0.5,h) + b*(0,0) + c*(1,0)  =>  y = a*h, x = c + 0.5*a
    a = y / h
    c = x - 0.5 * a
    b = 1.0 - a - c
    a = np.clip(a, eps, 1.0 - eps)
    b = np.clip(b, eps, 1.0 - eps)
    c = np.clip(c, eps, 1.0 - eps)
    t = a + b + c
    a, b, c = a / t, b / t, c / t
    x = c + 0.5 * a
    y = a * h
    return np.array([x, y])


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct an arrangement of 11 points on or inside an equilateral triangle maximizing the
    smallest triangle area (Heilbronn problem, n = 11).

    Returns:
        points: np.ndarray of shape (11,2).
    """
    n = 11
    try:
        rng = np.random.default_rng(20240517)
        s = np.sqrt(3.0)

        # ---- Warm start: symmetric (mirror about median x = 0.5) construction ----
        # 3 vertices + 3 edge midpoints + 5 interior points placed symmetrically.
        pts = [
            [0.0, 0.0],            # left base vertex
            [1.0, 0.0],            # right base vertex
            [0.5, s / 2.0],        # apex
            [0.5, 0.0],            # base midpoint
            [0.25, s / 4.0],       # left edge midpoint
            [0.75, s / 4.0],       # right edge midpoint
            [0.5, s / 12.0],       # centroid line, low
            [0.5, s / 4.0],        # centroid line, mid
            [0.25, s / 12.0],      # lower-left interior (mirror pair)
            [0.75, s / 12.0],      # lower-right interior
            [0.5, s / 3.0],        # upper interior on median
        ]
        points = np.array(pts, dtype=float)
        points = np.array([_clip_bary(p, eps=0.0) for p in points])

        # triplet indices
        ii, jj, kk = np.array(np.meshgrid(np.arange(n), np.arange(n), np.arange(n))).reshape(3, -1)
        mask = (ii < jj) & (jj < kk)
        idx_i, idx_j, idx_k = ii[mask], jj[mask], kk[mask]

        best = _min_area(points, idx_i, idx_j, idx_k)
        best_points = points.copy()

        # ---- Deterministic local search: one-point perturbations, shrink on stall ----
        step = 0.06
        budget = 3000
        it = 0
        while step > 1e-4 and it < budget:
            improved = False
            for i in range(n):
                for _ in range(4):
                    it += 1
                    if it >= budget:
                        break
                    cand = points.copy()
                    cand[i] = _clip_bary(points[i] + rng.normal(0.0, step, size=2))
                    if np.min(np.linalg.norm(np.delete(cand, i, axis=0) - cand[i], axis=1)) < 1e-7:
                        continue
                    a = _min_area(cand, idx_i, idx_j, idx_k)
                    if a > best + 1e-14:
                        points = cand
                        best = a
                        improved = True
            if not improved:
                step *= 0.6

        if best > _min_area(best_points, idx_i, idx_j, idx_k):
            best_points = points.copy()
        return best_points
    except Exception:
        # Graceful fallback: triangular lattice of 11 points
        s = np.sqrt(3.0)
        fallback = []
        m = 4
        for i in range(m + 1):
            for j in range(m + 1 - i):
                k = m - i - j
                fallback.append([(j + 0.5 * k) / m, (k * s / 2.0) / m])
        arr = np.zeros((11, 2))
        for idx in range(11):
            arr[idx] = fallback[idx % len(fallback)]
        return arr


# EVOLVE-BLOCK-END