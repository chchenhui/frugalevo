# EVOLVE-BLOCK-START
import numpy as np


def _hull_area(points):
    """Area of the convex hull of the point set (monotonic chain)."""
    pts = np.unique(np.round(points, 12), axis=0)
    if len(pts) < 3:
        return 1e-12
    idx = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[idx]
    if len(pts) == 3:
        return 0.5 * abs(np.cross(pts[1] - pts[0], pts[2] - pts[0]))

    def half(seq):
        h = []
        for p in seq:
            while len(h) >= 2 and np.cross(h[-1] - h[-2], p - h[-2]) <= 1e-12:
                h.pop()
            h.append(p)
        return h

    lower = half(pts)
    upper = half(pts[::-1])
    hull = np.array(lower[:-1] + upper[:-1])
    if len(hull) < 3:
        # degenerate (collinear) fallback
        return 1e-12
    x, y = hull[:, 0], hull[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _min_triangle_area(points):
    """Smallest area among all C(13,3) triangles, vectorized."""
    p = np.asarray(points, dtype=float)
    n = len(p)
    i, j, k = np.triu_indices(n, 3)[:3] if False else np.where(np.triu(np.ones((n, n), bool), 1))
    # build all triples explicitly
    tri = np.array([(a, b, c) for a in range(n) for b in range(a + 1, n)
                    for c in range(b + 1, n)], dtype=int)
    A, B, C = p[tri[:, 0]], p[tri[:, 1]], p[tri[:, 2]]
    area = 0.5 * np.abs(np.cross(B - A, C - A))
    return area.min()


def _score(points):
    ha = _hull_area(points)
    if ha <= 1e-12:
        return 0.0, 0.0
    mta = _min_triangle_area(points)
    return mta / ha, mta


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Strategy: start from a symmetric, strong baseline (regular dodecagon on the
    unit circle plus its center), then run a seeded stochastic hill-climbing
    refinement that perturbs one point at a time and accepts only improvements
    of the normalized minimum-triangle-area objective.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    rng = np.random.default_rng(seed=42)

    # --- Initialization: regular dodecagon + center (scaled for unit hull area) ---
    angles = np.arange(12) * (2.0 * np.pi / 12.0)
    ring = np.column_stack([np.cos(angles), np.sin(angles)])
    points = np.vstack([ring, np.zeros((1, 2))])  # center point last

    best_score, _ = _score(points)

    # --- Deterministic local search (coordinate-wise perturbations) ---
    n_iters = 3000
    step = 0.08
    for it in range(n_iters):
        i = int(rng.integers(0, n))
        old = points[i].copy()
        trial = points.copy()
        trial[i] = old + rng.normal(0.0, step, size=2)
        s, _ = _score(trial)
        if s > best_score + 1e-15:
            points = trial
            best_score = s
        # anneal the step size on a fixed schedule (reproducible)
        if (it + 1) % 500 == 0:
            step *= 0.7

    # --- Normalize so the convex hull has unit area ---
    ha = _hull_area(points)
    if ha > 1e-12:
        points = points / np.sqrt(ha)

    return points


# EVOLVE-BLOCK-END