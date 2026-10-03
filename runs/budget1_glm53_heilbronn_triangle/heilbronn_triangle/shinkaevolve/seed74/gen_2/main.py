# EVOLVE-BLOCK-START
import numpy as np

TRI = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]])
AREA_TOTAL = np.sqrt(3.0) / 4.0


def _project(pts):
    """Project points back into the equilateral triangle using barycentric coords."""
    A, B, C = TRI
    v0, v1, v2 = B - A, C - A, TRI[2] - TRI[1]
    out = pts.copy()
    for i, p in enumerate(out):
        # barycentric coordinates
        d00 = v0 @ v0
        d01 = v0 @ v1
        d11 = v1 @ v1
        dp0 = (p - A) @ v0
        dp1 = (p - A) @ v1
        den = d00 * d11 - d01 * d01
        u = (d11 * dp0 - d01 * dp1) / den
        v = (d00 * dp1 - d01 * dp0) / den
        w = 1.0 - u - v
        if u < 0:  # outside edge AB
            t = np.clip((p - A) @ v0 / d00, 0.0, 1.0)
            out[i] = A + t * v0
        elif v < 0:  # outside edge AC
            t = np.clip((p - A) @ v1 / d11, 0.0, 1.0)
            out[i] = A + t * v1
        elif w < 0:  # outside edge BC
            t = np.clip((p - TRI[1]) @ v2 / (v2 @ v2), 0.0, 1.0)
            out[i] = TRI[1] + t * v2
    return out


def _min_area(pts):
    """Minimum normalized triangle area over all triplets (vectorized)."""
    n = len(pts)
    if n < 3:
        return 0.0
    idx = np.array([(i, j, k) for i in range(n) for j in range(i + 1, n)
                    for k in range(j + 1, n)])
    a = pts[idx[:, 0]]
    b = pts[idx[:, 1]]
    c = pts[idx[:, 2]]
    areas = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) -
                         (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    return float(areas.min() / AREA_TOTAL)


def _pattern_search(pts, steps=(0.08, 0.04, 0.02, 0.01, 0.005, 0.002, 0.001)):
    pts = _project(pts)
    best = _min_area(pts)
    for step in steps:
        improved = True
        while improved:
            improved = False
            for i in range(len(pts)):
                for dx, dy in ((step, 0), (-step, 0), (0, step), (0, -step),
                               (step, step), (step, -step), (-step, step), (-step, -step)):
                    cand = pts.copy()
                    cand[i] += (dx, dy)
                    cand = _project(cand)
                    a = _min_area(cand)
                    if a > best + 1e-12:
                        pts, best = cand, a
                        improved = True
    return pts, best


def heilbronn_triangle11() -> np.ndarray:
    n = 11
    rng = np.random.default_rng(12345)
    starts = []
    # Seeded random starts
    for _ in range(4):
        starts.append(rng.random((n, 2)) * 0.8 + 0.1)
    # Structured start: points along a perturbed hexagonal-ish lattice
    t = np.linspace(0.05, 0.95, 6)
    starts.append(np.column_stack([t, 0.05 + 0.02 * np.sin(t * 10.0)]))
    starts.append(np.column_stack([t * 0.7 + 0.15, 0.4 + 0.02 * np.cos(t * 10.0)]))

    best_pts, best_val = None, -1.0
    for s in starts:
        try:
            p, v = _pattern_search(s)
            if v > best_val:
                best_pts, best_val = p, v
        except Exception:
            continue

    if best_pts is None or best_val <= 0:
        # Safe fallback: interior points (still valid output)
        best_pts = np.array([[0.1 + 0.08 * i, 0.1 + 0.05 * (i % 3)]
                             for i in range(n)])
        best_pts = _project(best_pts)

    return np.asarray(best_pts, dtype=float)


# EVOLVE-BLOCK-END
