# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations


def _all_triples(n):
    return np.array(list(combinations(range(n), 3)), dtype=int)


def _min_area(points, triples):
    p = points[triples]  # (T,3,2)
    a = p[:, 1] - p[:, 0]
    b = p[:, 2] - p[:, 0]
    areas = 0.5 * np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
    idx = int(np.argmin(areas))
    return areas[idx], idx


def _project(pts):
    """Project points back into the unit equilateral triangle via barycentric clipping.

    Accepts any shape (..., 2); flattens internally and restores the shape.
    """
    orig_shape = pts.shape
    flat = pts.reshape(-1, 2)
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, np.sqrt(3.0) / 2.0])
    v1 = B - A
    v2 = C - A
    det = v1[0] * v2[1] - v1[1] * v2[0]
    rel = flat - A
    b1 = (rel[:, 0] * v2[1] - rel[:, 1] * v2[0]) / det
    b2 = (v1[0] * rel[:, 1] - v1[1] * rel[:, 0]) / det
    b3 = 1.0 - b1 - b2
    b = np.stack([b3, b1, b2], axis=1)  # weights for A, B, C
    b = np.clip(b, 0.0, 1.0)
    b /= b.sum(axis=1, keepdims=True)
    verts = np.stack([A, B, C], axis=0)
    out = b @ verts
    return out.reshape(orig_shape)


def _min_area_batch(batch, triples):
    """Compute the minimum triangle area for each configuration in a batch.

    batch: np.ndarray of shape (K, n, 2). Returns array of K minimum areas.
    """
    p = batch[:, triples]  # (K, T, 3, 2)
    a = p[:, :, 1] - p[:, :, 0]
    b = p[:, :, 2] - p[:, :, 0]
    areas = 0.5 * np.abs(a[:, :, 0] * b[:, :, 1] - a[:, :, 1] * b[:, :, 0])
    return areas.min(axis=1)


def _areas_all(points, triples):
    """Vectorized area of every triangle triplet."""
    p = points[triples]
    a = p[:, 1] - p[:, 0]
    b = p[:, 2] - p[:, 0]
    return 0.5 * np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])


def _init_points(rng, n):
    """Random point set inside the triangle via uniform barycentric coordinates."""
    u = rng.random((n, 2))
    su = np.sqrt(u[:, 0])
    b = np.stack([1 - su, su * (1 - u[:, 1]), su * u[:, 1]], axis=1)
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, np.sqrt(3.0) / 2.0])
    verts = np.stack([A, B, C], axis=0)
    return b @ verts


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct an arrangement of 11 points inside the unit equilateral triangle to
    maximize the smallest triangle area over all point triplets.

    Deterministic random-restart local search: at each step the worst triplet is
    found and its points are perturbed (with shrinking step size) to improve the
    global minimum area. Points are kept inside the triangle by barycentric
    projection.

    Returns:
        points: np.ndarray of shape (11,2) with the x,y coordinates of the points.
    """
    n = 11
    triples = _all_triples(n)
    rng = np.random.default_rng(12345)

    best_pts = None
    best_val = -1.0

    K = 24          # candidates generated per point perturbation (best-of-K move)
    KWORST = 6      # number of worst triplets perturbed jointly per iteration

    for restart in range(30):
        pts = _init_points(rng, n)
        step = 0.05
        cur_val, _ = _min_area(pts, triples)
        # Early bail-out on hopeless restarts to spend budget on good basins.
        if best_val > 0 and cur_val < 0.5 * best_val:
            continue
        while step > 1e-7:
            improved = False
            # Collect the KWORST smallest-area triplets and perturb their points.
            areas = _areas_all(pts, triples)
            worst_order = np.argsort(areas)[:KWORST]
            mover_set = sorted(set(triples[worst_order].ravel().tolist()))
            for pi in mover_set:
                cands = np.repeat(pts[None], K, axis=0)  # (K, n, 2)
                cands[:, pi, :] += rng.normal(0.0, step, size=(K, 2))
                cands = _project(cands)
                vals = _min_area_batch(cands, triples)
                j = int(np.argmax(vals))
                if vals[j] > cur_val + 1e-12:
                    pts = cands[j]
                    cur_val = float(vals[j])
                    improved = True
            if not improved:
                step *= 0.6
        if cur_val > best_val:
            best_val = cur_val
            best_pts = pts.copy()

    # Fallback safety: should never trigger, but guarantees a valid output.
    if best_pts is None:
        best_pts = _project(_init_points(np.random.default_rng(0), n))

    return np.asarray(best_pts, dtype=float)


# EVOLVE-BLOCK-END