# EVOLVE-BLOCK-START
import numpy as np


def _barycentric_fold(pts, A, B, C):
    """Map possibly-outside points back into triangle ABC via barycentric clamping."""
    v0 = B - A
    v1 = C - A
    v2 = pts - A
    d00 = v0 @ v0
    d01 = v0 @ v1
    d11 = v1 @ v1
    d20 = v2 @ v0
    d21 = v2 @ v1
    den = d00 * d11 - d01 * d01
    b1 = (d11 * d20 - d01 * d21) / den
    b2 = (d00 * d21 - d01 * d20) / den
    b0 = 1.0 - b1 - b2
    b = np.stack([b0, b1, b2], axis=1)
    b = np.maximum(b, 0.0)
    b /= b.sum(axis=1, keepdims=True)
    verts = np.stack([A, B, C])
    return b @ verts


def _min_area(pts):
    """Vectorized minimum (non-degenerate) triangle area over all triplets."""
    n = len(pts)
    i, j, k = np.triu_indices(n, 3)[:3] if False else (None, None, None)
    idx = np.array([(a, b, c) for a in range(n) for b in range(a + 1, n)
                    for c in range(b + 1, n)])
    p0 = pts[idx[:, 0]]
    p1 = pts[idx[:, 1]]
    p2 = pts[idx[:, 2]]
    areas = 0.5 * np.abs((p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
                          - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0]))
    return areas.min()


def heilbronn_triangle11() -> np.ndarray:
    n = 11
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, np.sqrt(3) / 2])

    rng = np.random.default_rng(12345)

    # Deterministic spread-out starting configuration (interior, no lattice collinearity)
    bary = rng.dirichlet(alpha=np.ones(3) * 2.0, size=n)
    # nudge points away from the extreme center to spread them
    bary = 0.5 + 0.5 * (bary - bary.mean(axis=0))
    bary = np.maximum(bary, 0.02)
    bary /= bary.sum(axis=1, keepdims=True)
    verts = np.stack([A, B, C])
    pts = bary @ verts

    best_val = _min_area(pts)

    # Annealed hill-climbing with barycentric folding for feasibility
    step = 0.08
    total_iters = 6000
    for it in range(total_iters):
        i = it % n
        cand = pts.copy()
        delta = rng.normal(0.0, step, size=2)
        cand[i] += delta
        cand = _barycentric_fold(cand, A, B, C)
        val = _min_area(cand)
        if val > best_val + 1e-14:
            pts, best_val = cand, val
        if it % (n * 40) == n * 40 - 1:
            step *= 0.82  # geometric annealing
            if step < 1e-4:
                break

    # Final fine polish: tiny local perturbations per point
    step = 1e-3
    for it in range(2000):
        i = it % n
        cand = pts.copy()
        cand[i] += rng.normal(0.0, step, size=2)
        cand = _barycentric_fold(cand, A, B, C)
        val = _min_area(cand)
        if val > best_val + 1e-15:
            pts, best_val = cand, val

    points = np.asarray(pts, dtype=float)
    assert points.shape == (n, 2)
    return points


# EVOLVE-BLOCK-END
