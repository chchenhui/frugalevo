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
    import time
    n = 11
    triples = _all_triples(n)
    rng = np.random.default_rng(12345)

    s3 = np.sqrt(3.0)
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, s3 / 2.0])
    verts = np.stack([A, B, C], axis=0)

    def _lattice_start(m, jitter):
        lat = []
        for i in range(m + 1):
            for j in range(m + 1 - i):
                k = m - i - j
                lat.append([(j + 0.5 * k) / m, (k * s3 / 2.0) / m])
        lat = np.array(lat)
        if len(lat) >= n:
            pts = lat[:n]
        else:
            extra = _init_points(rng, n - len(lat))
            pts = np.vstack([lat, extra])
        pts = _project(pts + rng.normal(0.0, jitter, size=pts.shape))
        return pts

    # Structured warm starts: optima for Heilbronn sit mostly on the boundary,
    # so seed with vertices + edge/interior lattices (jittered to avoid
    # collinear degenerate starts that trap local search at zero area).
    starts = []
    starts.append(np.vstack([verts, _init_points(rng, n - 3)]) if n >= 3 else _init_points(rng, n))
    for m in (3, 4, 5):
        starts.append(_lattice_start(m, 0.01))
    # Edge-heavy start: 3 vertices + 6 edge points + 2 interior
    edge_pts = []
    for (P, Q, cnt) in [(A, B, 3), (B, C, 2), (C, A, 2)]:
        for t in range(1, cnt + 1):
            edge_pts.append(P + (Q - P) * (t / (cnt + 1)))
    edge_pts = np.array(edge_pts)
    interior = _init_points(rng, n - len(edge_pts) - 3)
    starts.append(_project(np.vstack([verts, edge_pts, interior])
                           + rng.normal(0.0, 0.005, size=(n, 2))))

    best_pts = None
    best_val = -1.0

    K = 32  # candidates generated per point perturbation (best-of-K move)
    t0 = time.time()
    time_budget = 10.0

    def _perp_dir(pa, pb, pc):
        """Unit vector from c perpendicular to line ab, on c's side (increases area)."""
        ex, ey = pb[0] - pa[0], pb[1] - pa[1]
        L = np.hypot(ex, ey) + 1e-12
        ux, uy = ex / L, ey / L
        dx, dy = pc[0] - pa[0], pc[1] - pa[1]
        if -uy * dx + ux * dy >= 0:
            return np.array([-uy, ux])
        return np.array([uy, -ux])

    def _argmin_polish(pts, triples, iters=600):
        """Deterministic polish: expand the argmin (worst) triangle geometrically."""
        step = 0.01
        cur_val, widx = _min_area(pts, triples)
        for _ in range(iters):
            if time.time() - t0 > time_budget * 0.95:
                break
            _, widx = _min_area(pts, triples)
            i, j, k = triples[widx]
            dirs = {
                i: _perp_dir(pts[j], pts[k], pts[i]),
                j: _perp_dir(pts[i], pts[k], pts[j]),
                k: _perp_dir(pts[i], pts[j], pts[k]),
            }
            improved = False
            for trial in [(i, j, k), (i, j), (i, k), (j, k), (i,), (j,), (k,)]:
                cand = pts.copy()
                for m in trial:
                    cand[m] = pts[m] + step * dirs[m]
                cand = _project(cand[None])[0]
                v, _ = _min_area(cand, triples)
                if v > cur_val + 1e-14:
                    pts = cand
                    cur_val = float(v)
                    improved = True
                    break
            if not improved:
                step *= 0.7
                if step < 1e-7:
                    break
        return pts, cur_val
    restart = 0
    while time.time() - t0 < time_budget:
        if restart < len(starts):
            pts = starts[restart].copy()
        else:
            pts = _init_points(rng, n)
        restart += 1
        step = 0.05
        cur_val, _ = _min_area(pts, triples)
        while step > 1e-7:
            improved = False
            _, widx = _min_area(pts, triples)
            worst = triples[widx]
            for pi in worst:
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
        # deterministic geometric polish on the worst triplet
        pts, cur_val = _argmin_polish(pts, triples)
        if cur_val > best_val:
            best_val = cur_val
            best_pts = pts.copy()

    # Fallback safety: should never trigger, but guarantees a valid output.
    if best_pts is None:
        best_pts = _project(_init_points(np.random.default_rng(0), n))

    return np.asarray(best_pts, dtype=float)


# EVOLVE-BLOCK-END