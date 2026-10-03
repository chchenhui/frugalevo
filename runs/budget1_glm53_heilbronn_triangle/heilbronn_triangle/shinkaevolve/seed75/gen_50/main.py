# EVOLVE-BLOCK-START
import numpy as np

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


try:
    from scipy.optimize import minimize as _scipy_minimize
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


def _softmin_refine(pts, n_temps=(2e-2, 5e-3, 1e-3), k_worst=20):
    """Top-k softmin smooth refinement (softmax barycentric parametrization).
    Applies log-sum-exp over only the k smallest triplet areas so the smooth
    objective concentrates its signal on the binding constraints instead of
    diluting it across all 165 triples."""
    if not _HAVE_SCIPY:
        return pts
    n = len(pts)
    V = VERTS
    A = np.array([[1.0, V[0, 0], V[0, 1]],
                  [1.0, V[1, 0], V[1, 1]],
                  [1.0, V[2, 0], V[2, 1]]])
    b = np.column_stack([np.ones(n), pts])
    lam0 = np.linalg.solve(A.T, b.T).T
    lam0 = np.clip(lam0, 1e-3, None)
    lam0 /= lam0.sum(axis=1, keepdims=True)
    theta = np.log(lam0).ravel()

    ii = []
    jj = []
    kk = []
    for a in range(n - 2):
        for c in range(a + 1, n - 1):
            for d in range(c + 1, n):
                ii.append(a)
                jj.append(c)
                kk.append(d)
    ii = np.array(ii)
    jj = np.array(jj)
    kk = np.array(kk)

    def unpack(th):
        W = th.reshape(n, 3)
        W = W - W.max(axis=1, keepdims=True)
        E = np.exp(W)
        W = E / E.sum(axis=1, keepdims=True)
        return W @ V

    def obj(th, temp):
        p = unpack(th)
        pi, pj, pk = p[ii], p[jj], p[kk]
        cross = ((pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1])
                 - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0]))
        a = 0.5 * np.abs(cross)
        if len(a) > k_worst:
            part = np.partition(a, k_worst - 1)[:k_worst]
        else:
            part = a
        m = part.min()
        return -(m - temp * np.log(np.sum(np.exp(-(part - m) / temp))))

    def _worst_triplet_idx(p):
        pi, pj, pk = p[ii], p[jj], p[kk]
        cross = ((pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1])
                 - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0]))
        a = 0.5 * np.abs(cross)
        return int(np.argmin(a))

    def _refine_worst_triplet(th):
        """Powell on only the current argmin triplet's 9 logits."""
        if not _HAVE_SCIPY:
            return th
        p = unpack(th)
        t = _worst_triplet_idx(p)
        i0, j0, k0 = ii[t], jj[t], kk[t]

        def obj_trip(th9, temp=2e-4):
            th = th.copy()
            th.reshape(n, 3)[[i0, j0, k0]] = th9.reshape(3, 3)
            p = unpack(th)
            pi, pj, pk = p[ii], p[jj], p[kk]
            cross = ((pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1])
                     - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0]))
            a = 0.5 * np.abs(cross)
            part = np.partition(a, 10)[:11]
            m = part.min()
            return -(m - temp * np.log(np.sum(np.exp(-(part - m) / temp))))

        cur = _min_area(unpack(th))[0]
        for _round in range(2):
            p = unpack(th)
            t = _worst_triplet_idx(p)
            i0, j0, k0 = ii[t], jj[t], kk[t]
            th9 = th.reshape(n, 3)[[i0, j0, k0]].copy().ravel()
            try:
                res = _scipy_minimize(obj_trip, th9, method="Powell",
                                      options={"maxiter": 60, "xtol": 1e-6, "ftol": 1e-10})
                if np.all(np.isfinite(res.x)):
                    cand_th = th.copy()
                    cand_th.reshape(n, 3)[[i0, j0, k0]] = res.x.reshape(3, 3)
                    if _min_area(unpack(cand_th))[0] > _min_area(unpack(th))[0] + 1e-14:
                        th = cand_th
            except Exception:
                break
        del cur
        return th

    cur_val, _ = _min_area(pts)
    for temp in n_temps:
        try:
            res = _scipy_minimize(obj, theta, args=(temp,), method="Powell",
                                  options={"maxiter": 300, "xtol": 1e-6, "ftol": 1e-9})
            if np.all(np.isfinite(res.x)):
                theta = res.x
            theta = _refine_worst_triplet(theta)
        except Exception:
            break
    cand = unpack(theta)
    if not np.all(np.isfinite(cand)):
        return pts
    val, _ = _min_area(cand)
    if val > cur_val:
        return cand
    return pts


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct 11 points inside the unit equilateral triangle maximizing the
    minimum triangle area (Heilbronn problem, n=11).

    Returns:
        points: np.ndarray of shape (11,2) of x,y coordinates.
    """
    s3 = np.sqrt(3.0)
    fallback = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.5, s3 / 2.0],
        [0.5, 0.0], [0.25, s3 / 4.0], [0.75, s3 / 4.0],
        [0.25, s3 / 12.0], [0.75, s3 / 12.0],
        [0.5, s3 / 6.0], [0.125, s3 / 8.0], [0.875, s3 / 8.0],
    ])
    try:
        rng = np.random.default_rng(20240517)
        best_pts, best_val = None, -np.inf
        bases = _initial_layouts()
        # Multi-restart: each base layout plus deterministic jittered variants.
        seeds = []
        for layout in bases:
            seeds.append(np.asarray(layout, dtype=float))
            for _r in range(6):
                seeds.append(np.asarray(layout, dtype=float) + rng.normal(scale=0.05, size=(11, 2)))
        for seed in seeds:
            cand = _bary_repair(seed)
            cand = _polish(cand, iters=400, step=0.015)
            cand = _stochastic_refine(cand, rng)
            cand = _polish(cand, iters=300, step=0.008)
            val, _ = _min_area(cand)
            if val > best_val:
                best_val, best_pts = val, cand
        # Final smooth top-k softmin refinement on the best candidate
        cand = _softmin_refine(best_pts)
        val, _ = _min_area(cand)
        if val > best_val:
            best_val, best_pts = val, cand
        if best_pts is None or best_pts.shape != (11, 2) or not np.all(np.isfinite(best_pts)):
            raise RuntimeError("bad result")
        if not np.all(_in_triangle(best_pts, tol=1e-9)):
            best_pts = _bary_repair(best_pts)
        pts = best_pts
    except Exception:
        pts = fallback
    return np.asarray(pts, dtype=float)


# EVOLVE-BLOCK-END