# EVOLVE-BLOCK-START
import numpy as np

try:
    from scipy.optimize import minimize as _scipy_minimize
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside an equilateral triangle with
    vertices (0,0), (1,0), (0.5, sqrt(3)/2) in order to maximize the area of the
    smallest triangle formed by these points. Here n = 11.

    Uses a fast deterministic multi-start local search in direct (x, y)
    coordinates. At each iteration the worst triangle is identified and its
    vertices are moved along gradient-informed directions (perpendicular to the
    opposite side), projected back onto the triangle for feasibility. A shrinking
    step size provides annealing; randomized escape moves on the worst triplet
    help escape local stalls. A hardcoded fallback configuration is returned if
    anything fails.

    Returns:
        points: np.ndarray of shape (11,2) with the x,y coordinates of the points.
    """
    n = 11
    sqrt3 = np.sqrt(3.0)
    V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, sqrt3 / 2.0]])

    from itertools import combinations
    triples = np.array(list(combinations(range(n), 3)))   # (C,3)

    def all_areas(pts):
        p = pts[triples]                                   # (C,3,2)
        cross = ((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
                 - (p[:, 1, 1] - p[:, 0, 1]) * (p[:, 2, 0] - p[:, 0, 0]))
        return 0.5 * np.abs(cross)

    def min_area(pts):
        return float(np.min(all_areas(pts)))

    def project(pts):
        """Project points onto the triangle via barycentric clipping."""
        x, y = pts[:, 0], pts[:, 1]
        wC = 2.0 * y / sqrt3
        wB = x - 0.5 * wC
        wA = 1.0 - wB - wC
        W = np.stack([wA, wB, wC], axis=1)
        W = np.clip(W, 0.0, None)
        W /= W.sum(axis=1, keepdims=True)
        return W @ V

    def worst_grad_move(pts, step):
        """Move the three vertices of the worst triangle away from their
        opposite sides (exact gradient directions of that triangle's area)."""
        areas = all_areas(pts)
        k = int(np.argmin(areas))
        i, j, l = triples[k]
        cand = pts.copy()

        def perp(P, Q, R):
            d = Q - P
            L = np.hypot(d[0], d[1])
            if L < 1e-12:
                return np.zeros(2)
            nrm = np.array([-d[1], d[0]]) / L
            cross = d[0] * (R - P)[1] - d[1] * (R - P)[0]
            return nrm if cross >= 0 else -nrm

        cand[i] = pts[i] + step * perp(pts[j], pts[l], pts[i])
        cand[j] = pts[j] + step * perp(pts[i], pts[l], pts[j])
        cand[l] = pts[l] + step * perp(pts[i], pts[j], pts[l])
        return project(cand)

    def local_search(start, iters=400):
        pts = project(np.asarray(start, dtype=float))
        cur = min_area(pts)
        best, best_v = pts.copy(), cur
        step = 0.06
        for _ in range(iters):
            cand = worst_grad_move(pts, step)
            v = min_area(cand)
            if v >= cur:
                pts, cur = cand, v
                if v > best_v:
                    best, best_v = cand.copy(), v
            else:
                # randomized escape move on the worst triplet's points
                cand2 = pts.copy()
                k = int(np.argmin(all_areas(pts)))
                for idx in triples[k]:
                    cand2[idx] = cand2[idx] + step * _rng.normal(size=2)
                cand2 = project(cand2)
                v2 = min_area(cand2)
                if v2 >= cur:
                    pts, cur = cand2, v2
                    if v2 > best_v:
                        best, best_v = cand2.copy(), v2
                step *= 0.98
            if step < 1e-5:
                break
        return best, best_v

    _rng = np.random.default_rng(12345)

    # Deterministic starting configurations
    base = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.5, sqrt3 / 2.0],
        [0.5, 0.0], [0.25, sqrt3 / 4.0], [0.75, sqrt3 / 4.0],
        [0.25, sqrt3 / 12.0], [0.75, sqrt3 / 12.0],
        [0.5, sqrt3 / 6.0], [0.125, sqrt3 / 8.0], [0.875, sqrt3 / 8.0],
    ])
    starts = [base]
    for _ in range(6):
        starts.append(project(base + 0.05 * _rng.normal(size=base.shape)))
    for _ in range(6):
        b = _rng.random((n, 3)) + 0.2
        b /= b.sum(axis=1, keepdims=True)
        starts.append(b @ V)

    best_pts = base
    best_val = min_area(base)
    for s in starts:
        try:
            pts, val = local_search(s)
            if np.all(np.isfinite(pts)) and val > best_val:
                best_val = val
                best_pts = pts
        except Exception:
            continue

    # Final validity check (numerical safety)
    best_pts = np.asarray(best_pts, dtype=float)
    if best_pts.shape != (n, 2) or not np.all(np.isfinite(best_pts)):
        best_pts = base
    return best_pts


# EVOLVE-BLOCK-END