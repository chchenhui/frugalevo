# EVOLVE-BLOCK-START
import numpy as np

try:
    from scipy.optimize import minimize as _scipy_minimize
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

VERTS = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2.0]])


def _in_triangle(pts, tol=1e-12):
    (x1, y1), (x2, y2), (x3, y3) = VERTS
    s = (x2 - x1) * (pts[:, 1] - y1) - (y2 - y1) * (pts[:, 0] - x1)
    t = (x3 - x2) * (pts[:, 1] - y2) - (y3 - y2) * (pts[:, 0] - x2)
    u = (x1 - x3) * (pts[:, 1] - y3) - (y1 - y3) * (pts[:, 0] - x3)
    return ((s >= -tol) & (t >= -tol) & (u >= -tol))


def _bary_repair(pts):
    A = np.array([[1.0, VERTS[0, 0], VERTS[0, 1]],
                  [1.0, VERTS[1, 0], VERTS[1, 1]],
                  [1.0, VERTS[2, 0], VERTS[2, 1]]])
    b = np.column_stack([np.ones(len(pts)), pts])
    lam = np.linalg.solve(A.T, b.T).T  # barycentric coords
    lam = np.clip(lam, 0.0, None)
    lam /= lam.sum(axis=1, keepdims=True)
    return lam @ VERTS


def _min_area(pts):
    n = len(pts)
    best = np.inf
    worst = None
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                a = 0.5 * abs((pts[j, 0] - pts[i, 0]) * (pts[k, 1] - pts[i, 1]) -
                              (pts[k, 0] - pts[i, 0]) * (pts[j, 1] - pts[i, 1]))
                if a < best:
                    best = a
                    worst = (i, j, k)
    return best, worst


def _initial_layouts():
    """Several deterministic, well-spread seed layouts (no near-degenerate triplets)."""
    s3 = np.sqrt(3.0)
    layouts = []
    # Layout A: hand-built known-good configuration (vertices + edge + interior)
    layouts.append(np.array([
        [0.0, 0.0], [1.0, 0.0], [0.5, s3 / 2.0],
        [0.5, 0.0], [0.25, s3 / 4.0], [0.75, s3 / 4.0],
        [0.25, s3 / 12.0], [0.75, s3 / 12.0],
        [0.5, s3 / 6.0], [0.125, s3 / 8.0], [0.875, s3 / 8.0],
    ]))
    # Layout B: uniform barycentric lattice-ish interior points
    rng = np.random.default_rng(777)
    layouts.append(np.array([
        [0.0, 0.0], [1.0, 0.0], [0.5, s3 / 2.0],
        [0.5, 0.0], [0.25, s3 / 4.0], [0.75, s3 / 4.0],
        [1.0 / 3.0, s3 / 9.0], [2.0 / 3.0, s3 / 9.0],
        [0.5, s3 / 3.0], [0.5, s3 / 18.0], [0.25, s3 / 36.0],
    ]))
    # Layout C: jittered version of A for extra diversity
    jitter = layouts[0] + rng.normal(scale=0.02, size=layouts[0].shape)
    layouts.append(jitter)
    return layouts


def _polish(pts, iters=1500, step=0.015):
    """Gradient-informed polish: move worst-triplet points perpendicular
    away from the line through the other two points (the exact direction
    that increases that triplet's signed area)."""
    for _ in range(iters):
        cur, worst = _min_area(pts)
        if worst is None:
            break
        i, j, k = worst
        p = pts
        # perpendicular "escape" directions for each of the 3 points
        dirs = {}
        dirs[i] = _perp(p[j], p[k], p[i])
        dirs[j] = _perp(p[i], p[k], p[j])
        dirs[k] = _perp(p[i], p[j], p[k])
        improved = False
        # try moving all three, then pairs, then single points
        trials = [(i, j, k), (i, j), (j, k), (i, k), (i,), (j,), (k,)]
        for trial in trials:
            cand = pts.copy()
            for m_ in trial:
                cand[m_] = pts[m_] + step * dirs[m_]
            cand = _bary_repair(cand)
            new, _ = _min_area(cand)
            if new > cur + 1e-15:
                pts = cand
                improved = True
                break
        if not improved:
            step *= 0.85
            if step < 1e-7:
                break
    return pts


def _perp(a, b, c):
    """Unit direction from point c away from line through a,b (increases |area|)."""
    ex, ey = b[0] - a[0], b[1] - a[1]
    L = np.hypot(ex, ey) + 1e-12
    ux, uy = ex / L, ey / L          # unit vector along edge
    dx, dy = c[0] - a[0], c[1] - a[1]
    # signed perpendicular distance component
    px, py = -uy * dx + ux * dy, ux * dx + uy * dy
    # perpendicular pointing away from edge, on the side where c lies
    if px >= 0:
        return np.array([-uy, ux])
    return np.array([uy, -ux])


def _stochastic_refine(pts, rng, budget=2500, step0=0.03):
    """Shrinking-step stochastic hill climb: random one-point perturbations,
    accepted only if they raise the minimum triangle area."""
    pts = pts.copy()
    best, _ = _min_area(pts)
    step = step0
    it = 0
    n = len(pts)
    while step > 1e-5 and it < budget:
        improved = False
        for i in range(n):
            for _t in range(3):
                it += 1
                if it >= budget:
                    break
                cand = pts.copy()
                cand[i] = pts[i] + rng.normal(0.0, step, size=2)
                cand = _bary_repair(cand)
                if np.min(np.linalg.norm(np.delete(cand, i, axis=0) - cand[i], axis=1)) < 1e-9:
                    continue
                val, _ = _min_area(cand)
                if val > best + 1e-15:
                    pts, best = cand, val
                    improved = True
        if not improved:
            step *= 0.6
    return pts


def _areas(pts, triples_arr):
    """Vectorized areas of all point triples; triples_arr has shape (C,3)."""
    p = pts[triples_arr]
    cross = ((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
             - (p[:, 1, 1] - p[:, 0, 1]) * (p[:, 2, 0] - p[:, 0, 0]))
    return 0.5 * np.abs(cross)


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct 11 points inside the unit equilateral triangle maximizing the
    minimum triangle area (Heilbronn problem, n=11).

    Strategy: scipy Powell over softmax-barycentric logits with a top-k
    temperature-annealed softmin objective (only the k smallest triplet areas
    enter the log-sum-exp, focusing the search on binding constraints).
    Falls back to the best known hard-coded layout on any failure.

    Returns:
        points: np.ndarray of shape (11,2) of x,y coordinates.
    """
    n = 11
    s3 = np.sqrt(3.0)
    V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, s3 / 2.0]])
    fallback = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.5, s3 / 2.0],
        [0.5, 0.0], [0.25, s3 / 4.0], [0.75, s3 / 4.0],
        [0.25, s3 / 12.0], [0.75, s3 / 12.0],
        [0.5, s3 / 6.0], [0.125, s3 / 8.0], [0.875, s3 / 8.0],
    ])
    try:
        from itertools import combinations
        triples_arr = np.array(list(combinations(range(n), 3)))

        def unpack(theta):
            W = theta.reshape(n, 3)
            W = W - W.max(axis=1, keepdims=True)
            E = np.exp(W)
            W = E / E.sum(axis=1, keepdims=True)
            return W @ V

        K = 20  # focus softmin on the k binding (smallest-area) triples

        def softmin_obj(theta, temp):
            a = _areas(unpack(theta), triples_arr)
            amin = a.min()
            ak = np.partition(a, K)[:K]          # k smallest areas
            return temp * np.log(np.sum(np.exp(-(ak - amin) / temp))) + amin

        def argmin_polish(pts, iters=200):
            """Exact-argmin polish: perturb only the worst triplet's points
            via softmax logits (kept only if the exact min area improves)."""
            pts = pts.copy()
            for _ in range(iters // 2):
                val, worst = _min_area(pts)
                if worst is None:
                    break
                i, j, k = worst

                def obj3(theta):
                    W = theta.reshape(3, 3)
                    W = W - W.max(axis=1, keepdims=True)
                    E = np.exp(W)
                    W = E / E.sum(axis=1, keepdims=True)
                    cand = pts.copy()
                    cand[[i, j, k]] = W @ V
                    a = _areas(cand, triples_arr)
                    amin = a.min()
                    ak = np.partition(a, K)[:K]
                    return temp_fine * np.log(np.sum(np.exp(-(ak - amin) / temp_fine))) + amin

                # initial logits from current barycentric-ish coords
                cur = pts[[i, j, k]]
                lam0 = np.linalg.solve(
                    np.array([[1.0, V[0, 0], V[0, 1]],
                              [1.0, V[1, 0], V[1, 1]],
                              [1.0, V[2, 0], V[2, 1]]]).T,
                    np.column_stack([np.ones(3), cur]).T).T
                lam0 = np.clip(lam0, 1e-3, None)
                lam0 /= lam0.sum(axis=1, keepdims=True)
                theta0 = np.log(lam0).ravel()
                try:
                    res = _scipy_minimize(obj3, theta0, method="Powell",
                                          options={"maxiter": 100, "xtol": 1e-6, "ftol": 1e-9})
                    if np.all(np.isfinite(res.x)):
                        W = res.x.reshape(3, 3)
                        W = W - W.max(axis=1, keepdims=True)
                        E = np.exp(W)
                        W = E / E.sum(axis=1, keepdims=True)
                        cand = pts.copy()
                        cand[[i, j, k]] = W @ V
                        new_val, _ = _min_area(cand)
                        if new_val > val + 1e-14:
                            pts = cand
                        else:
                            break
                    else:
                        break
                except Exception:
                    break
            return pts

        # Deterministic starts: hand-crafted barycentric seeds + fixed-seed noise
        bary = np.array([
            [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5],
            [0.75, 0.25, 0.0], [0.25, 0.75, 0.0], [0.25, 0.25, 0.5],
            [0.625, 0.125, 0.25], [0.125, 0.625, 0.25],
        ])
        rng = np.random.default_rng(12345)
        starts = [np.log(np.clip(bary, 1e-6, 1.0)).ravel()]
        for _ in range(3):
            starts.append(rng.normal(scale=1.5, size=(n, 3)).ravel())

        best_pts = fallback
        best_val, _ = _min_area(fallback)
        temp_fine = 1e-6  # fine-temperature used inside argmin_polish

        if _HAVE_SCIPY:
            for s in starts:
                theta = np.array(s, dtype=float)
                try:
                    for temp in (5e-3, 1e-3, 1e-4, 1e-5, 1e-6):
                        res = _scipy_minimize(
                            softmin_obj, theta, args=(temp,),
                            method="Powell",
                            options={"maxiter": 350, "xtol": 1e-7, "ftol": 1e-10},
                        )
                        if np.all(np.isfinite(res.x)):
                            theta = res.x
                    pts = unpack(theta)
                    # final feasibility-aware touch-up (kept only if it helps)
                    pts_r = _bary_repair(pts)
                    pts_r = _polish(pts_r, iters=200, step=0.006)
                    val_r, _ = _min_area(pts_r)
                    val, _ = _min_area(pts)
                    if val_r > val:
                        pts, val = pts_r, val_r
                    # exact-argmin polish on the truly worst triplet
                    pts_p = argmin_polish(pts, iters=60)
                    val_p, _ = _min_area(pts_p)
                    if val_p > val:
                        pts, val = pts_p, val_p
                    if val > best_val:
                        best_val, best_pts = val, pts
                except Exception:
                    continue

        best_pts = np.asarray(best_pts, dtype=float)
        if best_pts.shape != (n, 2) or not np.all(np.isfinite(best_pts)):
            best_pts = fallback
        if not np.all(_in_triangle(best_pts, tol=1e-9)):
            best_pts = _bary_repair(best_pts)
        return best_pts
    except Exception:
        return np.asarray(fallback, dtype=float)


# EVOLVE-BLOCK-END