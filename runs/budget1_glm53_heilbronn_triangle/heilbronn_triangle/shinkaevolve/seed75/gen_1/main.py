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

        # ---- Warm start: triangular lattice points inside the triangle ----
        pts = []
        rows = 5  # rows of lattice: 1,2,3,4,... gives triangular numbers; pick pattern summing to 11
        # lattice on triangle grid: barycentric (i,j,k)/m
        m = 4
        for i in range(m + 1):
            for j in range(m + 1 - i):
                k = m - i - j
                # barycentric -> cartesian (avoid duplicated vertices collapse)
                x = (j + 0.5 * k) / m
                y = (k * s / 2.0) / m
                pts.append([x, y])
        pts = pts[:n]
        while len(pts) < n:
            pts.append([rng.random(), rng.random() * 0.5])
        points = np.array(pts, dtype=float)

        # Precompute triplet indices
        ii, jj, kk = np.array(np.meshgrid(np.arange(n), np.arange(n), np.arange(n))).reshape(3, -1)
        mask = (ii < jj) & (jj < kk)
        idx_i, idx_j, idx_k = ii[mask], jj[mask], kk[mask]

        best = _min_area(points, idx_i, idx_j, idx_k)
        best_points = points.copy()

        # ---- Deterministic annealing: single-point perturbations ----
        step = 0.08
        total_iters = 0
        max_iters = 4000
        while step > 1e-4 and total_iters < max_iters:
            improved = False
            for i in range(n):
                for t in range(6):  # multiple trials per point per temperature
                    total_iters += 1
                    if total_iters >= max_iters:
                        break
                    delta = rng.normal(0.0, step, size=2)
                    cand = points.copy()
                    cand[i] = _clip(points[i] + delta)
                    # avoid coincident points
                    if np.min(np.linalg.norm(np.delete(cand, i, axis=0) - cand[i], axis=1)) < 1e-6:
                        continue
                    a = _min_area(cand, idx_i, idx_j, idx_k)
                    if a > best + 1e-12:
                        points = cand
                        best = a
                        improved = True
            if not improved:
                step *= 0.6

        if best > _min_area(best_points, idx_i, idx_j, idx_k):
            best_points = points
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
