# EVOLVE-BLOCK-START
import numpy as np


def _hull_area(pts: np.ndarray) -> float:
    """Area of the convex hull via the shoelace formula on hull vertices."""
    try:
        from scipy.spatial import ConvexHull
        ch = ConvexHull(pts)
        v = pts[ch.vertices]
    except Exception:
        # Fallback: sort by angle around centroid (works for near-convex sets)
        c = pts.mean(axis=0)
        d = pts - c
        order = np.argsort(np.arctan2(d[:, 1], d[:, 0]))
        v = pts[order]
    x, y = v[:, 0], v[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _min_tri_area(pts: np.ndarray, tri: np.ndarray) -> float:
    """Minimum absolute area over all C(13,3) triangles (vectorized cross products)."""
    a = pts[tri[:, 0]]
    b = pts[tri[:, 1]]
    c = pts[tri[:, 2]]
    areas = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                         - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    return areas.min()


def heilbronn_convex13() -> np.ndarray:
    """
    Construct 13 points maximizing the smallest normalized triangle area.

    Approach: deterministic hill-climbing local search.
    - Start from a regular 13-gon (strong symmetric baseline, ~0.0176 normalized).
    - Objective: min triangle area / convex hull area (scale-invariant).
    - Iteratively perturb points with an adaptive step size, accepting only
      improvements; multiple seeds of the RNG are fixed for reproducibility.
    - All 286 triangle areas computed via vectorized cross products.
    """
    n = 13
    rng = np.random.default_rng(seed=42)

    # Precompute all triangle index triples
    tri = np.array([(i, j, k)
                    for i in range(n) for j in range(i + 1, n)
                    for k in range(j + 1, n)], dtype=int)

    # Initialization: regular 13-gon on unit circle
    angles = 2 * np.pi * np.arange(n) / n
    best = np.column_stack([np.cos(angles), np.sin(angles)])

    def score(p):
        return _min_tri_area(p, tri) / _hull_area(p)

    best_score = score(best)
    step = 0.05
    iters = 30000
    since_improve = 0

    for it in range(iters):
        cand = best + rng.normal(0, step, size=(n, 2))
        s = score(cand)
        if s > best_score:
            best, best_score = cand, s
            since_improve = 0
        else:
            since_improve += 1
            if since_improve > 400:  # shrink step when stuck
                step *= 0.7
                since_improve = 0
                if step < 1e-4:
                    break

    # Normalize to a unit-area convex region (affine scaling of both axes
    # changes areas uniformly, so the normalized score is preserved).
    a = _hull_area(best)
    best = best / np.sqrt(a)

    return best


# EVOLVE-BLOCK-END
