# EVOLVE-BLOCK-START
import numpy as np


_TRIP_IDX = None


def _min_triangle_area(pts: np.ndarray) -> float:
    """Vectorized minimum area over all C(n,3) triplets (indices cached)."""
    global _TRIP_IDX
    n = pts.shape[0]
    if _TRIP_IDX is None or _TRIP_IDX.shape[0] != n:
        _TRIP_IDX = np.array(
            [(a, b, c) for a in range(n) for b in range(a + 1, n)
             for c in range(b + 1, n)]
        )
    idx = _TRIP_IDX
    p0 = pts[idx[:, 0]]
    p1 = pts[idx[:, 1]]
    p2 = pts[idx[:, 2]]
    areas = 0.5 * np.abs(
        (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
        - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])
    )
    return float(areas.min())


def _bary_fold(pts, A, B, C):
    """Project points back into triangle ABC via barycentric clamping."""
    v0, v1 = B - A, C - A
    v2 = pts - A
    d00, d01, d11 = v0 @ v0, v0 @ v1, v1 @ v1
    d20, d21 = v2 @ v0, v2 @ v1
    den = d00 * d11 - d01 * d01
    b1 = (d11 * d20 - d01 * d21) / den
    b2 = (d00 * d21 - d01 * d20) / den
    b0 = 1.0 - b1 - b2
    b = np.stack([b0, b1, b2], axis=1)
    b = np.maximum(b, 0.0)
    b /= b.sum(axis=1, keepdims=True)
    return b @ np.stack([A, B, C])


def _inside(pts: np.ndarray) -> np.ndarray:
    """Boolean mask: points inside/on the equilateral triangle (0,0),(1,0),(0.5,√3/2)."""
    h = np.sqrt(3.0) / 2.0
    x, y = pts[:, 0], pts[:, 1]
    return (
        (y >= -1e-12)
        & (y <= np.sqrt(3.0) * x + 1e-12)
        & (y <= -np.sqrt(3.0) * (x - 1.0) + 1e-12)
    )


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct an arrangement of 11 points inside/on the equilateral triangle
    (0,0),(1,0),(0.5,sqrt(3)/2) maximizing the minimum triangle area.

    Strategy: symmetric lattice-like seed (vertices, edge midpoints, coarse
    interior grid) refined by a deterministic seeded hill climb.

    Returns:
        points: np.ndarray of shape (11,2).
    """
    n = 11
    h = np.sqrt(3.0) / 2.0
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, h])
    verts = np.stack([A, B, C])
    rng = np.random.default_rng(20240501)

    def to_xy(bary):
        return bary @ verts

    # --- Deterministic starting configurations ---
    seeds = []

    # Seed 0: hand-designed symmetric layout
    seeds.append(np.array(
        [
            [0.0, 0.0], [1.0, 0.0], [0.5, h], [0.5, 0.0],
            [0.25, h / 2.0], [0.75, h / 2.0], [0.5, h / 3.0],
            [0.5, 2.0 * h / 3.0], [0.25, h / 6.0],
            [0.75, h / 6.0], [0.5, h / 6.0],
        ]
    ))

    # Seed 1: triangular-lattice-like (rows of 1,2,3,4,1 pts on horizontal bands)
    rows = [1.0, 2.0 / 3.0, 1.0 / 3.0, 0.0]
    seed1 = []
    counts = [1, 2, 3, 4]
    for yy, cnt in zip(rows, counts):
        for t in np.linspace(0.0, 1.0, cnt):
            # interpolate across the horizontal chord at height yy*h
            xl = 0.5 * (1.0 - yy)
            xr = 0.5 * (1.0 + yy)
            seed1.append([xl + t * (xr - xl), yy * h])
    # 11th point: centroid-ish
    seed1.append([0.5, h / 3.0])
    seeds.append(np.array(seed1))

    # Seeds 2-4: random Dirichlet barycentric samples
    for _ in range(3):
        bary = rng.dirichlet(alpha=np.ones(3) * 4.0, size=n)
        seeds.append(to_xy(bary))

    def anneal(pts, iters, s0, s1, rng):
        """Annealed single-point hill-climb with barycentric folding."""
        pts = _bary_fold(pts.copy(), A, B, C)
        val = _min_triangle_area(pts)
        for it in range(iters):
            step = s0 * (s1 / s0) ** (it / max(iters - 1, 1))
            i = it % n
            cand = pts.copy()
            cand[i] = cand[i] + rng.normal(0.0, step, size=2)
            cand = _bary_fold(cand, A, B, C)
            v2 = _min_triangle_area(cand)
            if v2 > val + 1e-14:
                pts, val = cand, v2
        return pts, val

    # --- Multi-start annealed schedule ---
    best_pts, best_val = None, -1.0
    for s_idx, seed in enumerate(seeds):
        rng_s = np.random.default_rng(1000 + s_idx)
        pts, val = anneal(seed, 5000, 0.02, 1e-4, rng_s)
        if val > best_val:
            best_pts, best_val = pts, val

    # --- Fine polish on the best result ---
    rng_p = np.random.default_rng(99999)
    for it in range(10000):
        step = 1e-3 * (1e-5 / 1e-3) ** (it / 9999)
        i = it % n
        cand = best_pts.copy()
        cand[i] = cand[i] + rng_p.normal(0.0, step, size=2)
        cand = _bary_fold(cand, A, B, C)
        v2 = _min_triangle_area(cand)
        if v2 > best_val + 1e-15:
            best_pts, best_val = cand, v2

    points = best_pts

    # Safety: fold ensures feasibility, but verify shape/finite
    if (not np.all(np.isfinite(points))
            or points.shape != (n, 2)):
        points = _bary_fold(seeds[0], A, B, C)

    return np.asarray(points, dtype=float)


# EVOLVE-BLOCK-END