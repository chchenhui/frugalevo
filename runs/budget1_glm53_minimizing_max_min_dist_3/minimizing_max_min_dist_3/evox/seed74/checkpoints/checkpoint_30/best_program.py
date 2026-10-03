# EVOLVE-BLOCK-START
import numpy as np


def _ratio_sq(pts):
    """True objective: (dmin/dmax)^2."""
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    iu = np.triu_indices(len(pts), 1)
    dm = d[iu]
    return (dm.min() ** 2) / (dm.max() ** 2)


def _optimize(pts, iters=1500, lr=0.03):
    """Adam gradient ascent on softmin(d^2)/max(d^2) with decaying temperature."""
    n = pts.shape[0]
    m = np.zeros_like(pts)
    v = np.zeros_like(pts)
    b1, b2, eps = 0.9, 0.999, 1e-8
    pts = pts - pts.mean(axis=0)
    scale = np.linalg.norm(pts, axis=1).max()
    if scale > 0:
        pts = pts / scale
    for t_i in range(iters):
        diff = pts[:, None, :] - pts[None, :, :]
        d2 = (diff ** 2).sum(-1)
        iu = np.triu_indices(n, 1)
        du = d2[iu]
        t = max(0.02, 0.3 * du.min()) * (1.0 - t_i / (2.0 * iters))
        e = np.exp(-(du - du.min()) / t)
        Z = e.sum()
        softmin = -t * np.log(Z) + du.min()
        w = e / Z
        mx = du.max()
        k = du.argmax()
        grad_pair = (w * mx - softmin * (np.arange(len(du)) == k)) / (mx ** 2)
        g = np.zeros((n, n))
        g[iu] = grad_pair
        g = g + g.T
        grad = 2.0 * (g[:, :, None] * diff).sum(axis=1)
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * grad ** 2
        mh = m / (1 - b1 ** (t_i + 1))
        vh = v / (1 - b2 ** (t_i + 1))
        pts = pts + lr * mh / (np.sqrt(vh) + eps)
        pts = pts - pts.mean(axis=0)
    return pts


def _slsqp_polish(pts):
    """Epigraph SLSQP polish: maximize t s.t. d2_ij >= t and d2_ij <= 1.

    Linear objective with smooth quadratic constraints (equivalent to
    maximizing dmin with dmax fixed at 1, since the ratio is scale
    invariant). Much better behaved for SLSQP than the nonsmooth
    sqrt-of-min formulation.
    """
    try:
        from scipy.optimize import minimize
    except Exception:
        return pts
    n = pts.shape[0]
    pts = pts - pts.mean(axis=0)
    s = np.linalg.norm(pts, axis=1).max()
    if s > 0:
        pts = pts / s
    iu = np.triu_indices(n, 1)
    d2_0 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)[iu]
    t0 = min(d2_0.min(), 0.999 * d2_0.max())
    x0 = np.concatenate([pts.ravel(), [t0]])
    k = 3 * n

    def neg_t(x):
        return -x[-1]

    ip, jp = iu[0], iu[1]
    m = len(ip)

    def cons_lo(x):
        p = x[:k].reshape(n, 3)
        d2 = ((p[:, None, :] - p[None, :, :]) ** 2).sum(-1)[iu]
        return d2 - x[-1]

    def cons_lo_jac(x):
        p = x[:k].reshape(n, 3)
        diff = p[ip] - p[jp]
        J = np.zeros((m, k + 1))
        for a in range(3):
            J[np.arange(m), 3 * ip + a] = 2 * diff[:, a]
            J[np.arange(m), 3 * jp + a] = -2 * diff[:, a]
        J[:, -1] = -1.0
        return J

    def cons_hi(x):
        p = x[:k].reshape(n, 3)
        d2 = ((p[:, None, :] - p[None, :, :]) ** 2).sum(-1)[iu]
        return 1.0 - d2

    def cons_hi_jac(x):
        p = x[:k].reshape(n, 3)
        diff = p[ip] - p[jp]
        J = np.zeros((m, k + 1))
        for a in range(3):
            J[np.arange(m), 3 * ip + a] = -2 * diff[:, a]
            J[np.arange(m), 3 * jp + a] = 2 * diff[:, a]
        return J

    res = minimize(neg_t, x0, jac=lambda x: np.concatenate(
                       [np.zeros(k), [-1.0]]), method="SLSQP",
                   constraints=[{"type": "ineq", "fun": cons_lo,
                                 "jac": cons_lo_jac},
                                {"type": "ineq", "fun": cons_hi,
                                 "jac": cons_hi_jac}],
                   options={"maxiter": 500, "ftol": 1e-14})
    out = res.x[:k].reshape(n, 3)
    if np.all(np.isfinite(out)) and _ratio_sq(out) > _ratio_sq(pts):
        return out
    return pts


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Construct 14 points in 3D maximizing (dmin/dmax)^2.

    Multi-restart Adam gradient ascent on a smooth surrogate
    softmin(d^2)/max(d^2) (log-sum-exp with decaying temperature),
    which tracks the true nonsmooth objective dmin^2/dmax^2.
    The best configuration across restarts (judged by the true squared
    ratio) is returned, followed by a low-lr polish, then an exact
    SLSQP polish on the true nonsmooth objective.
    """
    n, d = 14, 3
    rng = np.random.RandomState(12345)

    best_pts = None
    best_val = -1.0

    inits = []
    # structured spherical starts
    for s in range(10):
        p = rng.randn(n, d)
        p /= np.linalg.norm(p, axis=1, keepdims=True)
        inits.append(p * (1.0 + 0.05 * rng.randn(n, 1)))
    # random starts
    for s in range(20):
        inits.append(rng.randn(n, d))
    # structured seeds: icosahedron (12 verts) + 2 jittered extras,
    # and two-shell designs (6+8, 8+6) — known-good basins for n=14
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ], dtype=float)
    for _ in range(4):
        extra = rng.randn(2, 3)
        extra /= np.linalg.norm(extra, axis=1, keepdims=True)
        p = np.vstack([ico, extra])
        p *= (1.0 + 0.05 * rng.randn(n, 1))
        inits.append(p)
    for split in (6, 8):
        for _ in range(3):
            a = rng.randn(split, 3)
            a /= np.linalg.norm(a, axis=1, keepdims=True)
            b = rng.randn(n - split, 3)
            b /= np.linalg.norm(b, axis=1, keepdims=True)
            b *= 0.55
            inits.append(np.vstack([a, b]))
    # two poles + staggered equatorial rings: the best-known 14-point
    # packings are pole+ring / antiprism-like; random starts miss them.
    for m_ring, r_ring, z_ring in ((6, 1.0, 0.45), (12, 1.0, 0.0), (6, 0.9, 0.55)):
        for _ in range(3):
            ang = 2 * np.pi * np.arange(m_ring) / m_ring + 0.3 * rng.randn()
            ring = np.stack([r_ring * np.cos(ang),
                             r_ring * np.sin(ang),
                             z_ring * (1 + 0.1 * rng.randn(m_ring))], axis=1)
            poles = np.array([[0, 0, 1.6], [0, 0, -1.6]]) * (1 + 0.05 * rng.randn(2, 1))
            p = np.vstack([poles, ring])
            if len(p) < n:
                extra = rng.randn(n - len(p), 3)
                extra /= np.linalg.norm(extra, axis=1, keepdims=True)
                extra *= 0.7
                p = np.vstack([p, extra])
            inits.append(p[:n])
    # antiprism seeds: two staggered n/2-gons on parallel planes
    for _ in range(3):
        ang1 = 2 * np.pi * np.arange(7) / 7 + 0.2 * rng.randn()
        ang2 = ang1 + np.pi / 7 + 0.2 * rng.randn()
        h = 0.55 * (1 + 0.1 * rng.randn())
        p = np.vstack([
            np.stack([np.cos(ang1), np.sin(ang1), np.full(7, h)], axis=1),
            np.stack([np.cos(ang2), np.sin(ang2), np.full(7, -h)], axis=1),
        ])
        p *= (1.0 + 0.05 * rng.randn(n, 1))
        inits.append(p)

    # optimize every start, keep the top-3 candidates by true ratio
    cands = []
    for init in inits:
        pts = _optimize(init.copy())
        cands.append((_ratio_sq(pts), pts))
    cands.sort(key=lambda c: -c[0])
    cands = cands[:5]

    # exact epigraph SLSQP polish on each top candidate
    for val, pts in cands:
        polished = _slsqp_polish(pts.copy())
        v2 = _ratio_sq(polished)
        if v2 > best_val:
            best_val = v2
            best_pts = polished
        if val > best_val:
            best_val = val
            best_pts = pts

    # alternating fine Adam + SLSQP polish from the best configuration
    for lr, it in ((0.01, 800), (0.005, 600)):
        polished = _optimize(best_pts.copy(), iters=it, lr=lr)
        v = _ratio_sq(polished)
        if v > best_val:
            best_pts = polished
            best_val = v
        polished = _slsqp_polish(best_pts.copy())
        v = _ratio_sq(polished)
        if v > best_val:
            best_pts = polished
            best_val = v

    # jittered local restarts around the incumbent best:
    # small perturbations + short Adam + SLSQP escape local optima
    for trial in range(24):
        sigma = 0.03 * (0.5 ** (trial // 8))
        jit = best_pts + sigma * rng.randn(*best_pts.shape)
        cand = _optimize(jit, iters=400, lr=0.015)
        cand = _slsqp_polish(cand)
        v = _ratio_sq(cand)
        if v > best_val:
            best_pts = cand
            best_val = v

    # micro-refinement: very small decaying jitters (down to ~1e-4)
    # followed by a tight SLSQP pass. At this scale the epigraph
    # formulation with analytic Jacobians can resolve improvements
    # in the 6th decimal of the squared ratio that coarse jitter
    # and Adam (whose temperature floor limits precision) cannot.
    for trial in range(30):
        sigma = 3e-3 * (0.7 ** (trial // 6))
        jit = best_pts + sigma * rng.randn(*best_pts.shape)
        cand = _slsqp_polish(jit)
        v = _ratio_sq(cand)
        if v > best_val:
            best_pts = cand
            best_val = v
        # occasionally re-run a very-low-lr Adam from the incumbent
        # to let the smooth surrogate re-balance near-degenerate
        # contact pairs before the next SLSQP pass
        if trial % 5 == 4:
            cand = _slsqp_polish(_optimize(best_pts.copy(),
                                           iters=300, lr=0.002))
            v = _ratio_sq(cand)
            if v > best_val:
                best_pts = cand
                best_val = v

    return np.asarray(best_pts, dtype=float)


# EVOLVE-BLOCK-END
