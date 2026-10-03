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
    """Exact nonsmooth polish: maximize dmin s.t. dmax <= 1 via SLSQP.

    The smooth surrogate gets close; the exact constrained formulation
    lets SLSQP settle onto the true optimum where dmin is maximized
    while all pairwise distances stay within the diameter bound.
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

    def neg_dmin(x):
        d2 = ((x.reshape(n, 3)[:, None, :] - x.reshape(n, 3)[None, :, :]) ** 2).sum(-1)
        return -np.sqrt(d2[iu].min())

    def cons_f(x):
        d2 = ((x.reshape(n, 3)[:, None, :] - x.reshape(n, 3)[None, :, :]) ** 2).sum(-1)
        return 1.0 - d2[iu]

    res = minimize(neg_dmin, pts.ravel(), method="SLSQP",
                   constraints=[{"type": "ineq", "fun": cons_f}],
                   options={"maxiter": 400, "ftol": 1e-12})
    out = res.x.reshape(n, 3)
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
    ratio) is returned, followed by a low-lr polish from the best point.
    """
    n, d = 14, 3
    rng = np.random.RandomState(12345)

    best_pts = None
    best_val = -1.0

    inits = []
    # structured spherical starts
    for s in range(6):
        p = rng.randn(n, d)
        p /= np.linalg.norm(p, axis=1, keepdims=True)
        inits.append(p * (1.0 + 0.05 * rng.randn(n, 1)))
    # random starts
    for s in range(10):
        inits.append(rng.randn(n, d))

    for init in inits:
        pts = _optimize(init.copy())
        val = _ratio_sq(pts)
        if val > best_val:
            best_val = val
            best_pts = pts

    # fine polish from the best configuration
    polished = _optimize(best_pts.copy(), iters=800, lr=0.01)
    if _ratio_sq(polished) > best_val:
        best_pts = polished
        best_val = _ratio_sq(polished)

    # exact constrained polish on the true nonsmooth objective:
    # maximize dmin subject to all pairwise distances <= 1 (SLSQP)
    best_pts = _slsqp_polish(best_pts)
    best_val = _ratio_sq(best_pts)

    # second fine polish from the SLSQP solution
    polished = _optimize(best_pts.copy(), iters=600, lr=0.005)
    if _ratio_sq(polished) > best_val:
        best_pts = polished
        best_val = _ratio_sq(polished)

    return np.asarray(best_pts, dtype=float)


# EVOLVE-BLOCK-END
