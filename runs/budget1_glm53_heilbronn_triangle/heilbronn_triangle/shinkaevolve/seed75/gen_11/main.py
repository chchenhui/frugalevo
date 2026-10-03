# EVOLVE-BLOCK-START
import numpy as np


def _min_area(points: np.ndarray, idx_i, idx_j, idx_k) -> float:
    """Vectorized minimum absolute triangle area over all triplets (normalized by area of container)."""
    pi = points[idx_i]
    pj = points[idx_j]
    pk = points[idx_k]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    return 0.5 * np.min(np.abs(cross)) / (np.sqrt(3) / 4.0)


def _in_triangle(p):
    x, y = p
    return x >= -1e-12 and y >= -1e-12 and (x + y * np.sqrt(3)) <= 1 + 1e-12 and \
        y <= np.sqrt(3) * x + 1e-12 and y <= -np.sqrt(3) * (x - 1) + 1e-12


def _clip(p):
    # Simple clipping: reflect into the triangle by coordinate clamping in barycentric-like fashion
    x, y = p
    # interior of triangle: x in [0,1], y between the two slanted sides and below top
    s = np.sqrt(3.0)
    x = np.clip(x, 0.0, 1.0)
    ylo = max(0.0, s * x - s, -s * x)  # placeholder, use exact bounds below
    ylow = np.clip(s * (x - 1.0), 0.0, None)
    yhigh_left = s * x
    yhigh = min(s, yhigh_left + 0.0)
    # Upper boundary: line from (0,0)-(0.5,s): y = s*x for x<=0.5; from (0.5,s)-(1,0): y = s*(1-x)
    if x <= 0.5:
        yhi = s * x
    else:
        yhi = s * (1.0 - x)
    y = np.clip(y, 0.0, yhi)
    return np.array([x, y])


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct an arrangement of 11 points on or inside an equilateral triangle with
    vertices (0,0), (1,0), (0.5, sqrt(3)/2), maximizing the smallest triangle area.

    Returns:
        points: np.ndarray of shape (11,2).
    """
    n = 11
    try:
        rng = np.random.default_rng(12345)
        s = np.sqrt(3.0)

        # ---- Warm start: symmetric arrangement using vertices + edge points + interior ring ----
        h = s / 2.0
        pts = [
            [0.0, 0.0],        # left base vertex
            [1.0, 0.0],        # right base vertex
            [0.5, h],          # apex
            [0.5, 0.0],        # base midpoint
            [0.25, h / 2.0],   # left edge midpoint
            [0.75, h / 2.0],   # right edge midpoint
            [0.25, h / 6.0],   # lower-left interior
            [0.75, h / 6.0],   # lower-right interior
            [0.5, h / 3.0],    # center-low
            [0.5, 2.0 * h / 3.0],  # center-high
            [0.25, 5.0 * h / 6.0], # upper-left interior
        ]
        points = np.array(pts, dtype=float)

        # Precompute triplet indices
        ii, jj, kk = np.array(np.meshgrid(np.arange(n), np.arange(n), np.arange(n))).reshape(3, -1)
        mask = (ii < jj) & (jj < kk)
        idx_i, idx_j, idx_k = ii[mask], jj[mask], kk[mask]

        best = _min_area(points, idx_i, idx_j, idx_k)
        best_points = points.copy()

        # ---- Two-phase deterministic search with global-best tracking ----
        total_iters = 0
        max_iters = 6000

        def try_move(cur, cur_area, step, two_point=False):
            """Attempt a random (one- or two-point) move; return (points, area, improved)."""
            nonlocal total_iters
            total_iters += 1
            cand = cur.copy()
            if two_point:
                i1, i2 = rng.choice(n, size=2, replace=False)
                cand[i1] = _clip(cur[i1] + rng.normal(0.0, step, size=2))
                cand[i2] = _clip(cur[i2] + rng.normal(0.0, step, size=2))
                moved = (i1, i2)
            else:
                i1 = int(rng.integers(n))
                cand[i1] = _clip(cur[i1] + rng.normal(0.0, step, size=2))
                moved = (i1,)
            ok = True
            for i in moved:
                if np.min(np.linalg.norm(np.delete(cand, i, axis=0) - cand[i], axis=1)) < 1e-6:
                    ok = False
                    break
            if not ok:
                return cur, cur_area, False
            a = _min_area(cand, idx_i, idx_j, idx_k)
            if a > cur_area + 1e-13:
                return cand, a, True
            return cur, cur_area, False

        # Phase 1: coarse exploration, occasionally move two points to escape
        # single-point local optima (common in Heilbronn arrangements).
        step = 0.08
        while step > 5e-3 and total_iters < max_iters:
            improved = False
            for _ in range(60):
                if total_iters >= max_iters:
                    break
                two_pt = rng.random() < 0.25
                points, best, imp = try_move(points, best, step, two_point)
                if imp:
                    improved = True
                    if best > _min_area(best_points, idx_i, idx_j, idx_k):
                        best_points = points.copy()
            if not improved:
                step *= 0.6

        # Phase 2: fine refinement, single-point moves only.
        points = best_points.copy()
        best = _min_area(points, idx_i, idx_j, idx_k)
        step = 0.01
        while step > 1e-5 and total_iters < max_iters:
            improved = False
            for _ in range(80):
                if total_iters >= max_iters:
                    break
                points, best, imp = try_move(points, best, step, False)
                if imp:
                    improved = True
                    if best > _min_area(best_points, idx_i, idx_j, idx_k):
                        best_points = points.copy()
            if not improved:
                step *= 0.6

        # Phase 3: bounded restarts from perturbed best to escape local optima.
        restarts = 0
        while total_iters < max_iters and restarts < 5:
            restarts += 1
            points = best_points.copy()
            # perturb all points slightly
            for i in range(n):
                points[i] = _clip(points[i] + rng.normal(0.0, 0.02, size=2))
            cur = _min_area(points, idx_i, idx_j, idx_k)
            step = 0.03
            while step > 1e-5 and total_iters < max_iters:
                improved = False
                for _ in range(60):
                    if total_iters >= max_iters:
                        break
                    points, cur, imp = try_move(points, cur, step, False)
                    if imp:
                        improved = True
                        if cur > _min_area(best_points, idx_i, idx_j, idx_k):
                            best_points = points.copy()
                if not improved:
                    step *= 0.6

        return best_points
    except Exception:
        # Graceful fallback: simple spread lattice
        s = np.sqrt(3.0)
        fallback = []
        m = 4
        for i in range(m + 1):
            for j in range(m + 1 - i):
                k = m - i - j
                fallback.append([(j + 0.5 * k) / m, (k * s / 2.0) / m])
        arr = np.zeros((n, 2))
        for idx in range(n):
            arr[idx] = fallback[idx % len(fallback)]
        return arr


# EVOLVE-BLOCK-END