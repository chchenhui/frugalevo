# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y) coordinates of the 14 points.

    """

    n = 14
    d = 3

    from scipy.optimize import minimize
    from scipy.special import logsumexp

    def sq_dists(x):
        P = x.reshape(n, d)
        diff = P[:, None, :] - P[None, :, :]
        D2 = np.sum(diff * diff, axis=2)
        iu = np.triu_indices(n, k=1)
        return D2[iu]

    def objective(x, p):
        d2 = sq_dists(x)
        # smooth min and max of squared distances (log-sum-exp)
        soft_min = -logsumexp(-p * d2) / p
        soft_max = logsumexp(p * d2) / p
        # maximize soft_min / soft_max  ->  minimize its negative
        return -(soft_min / soft_max)

    def exact_grad(x):
        """Value and analytic gradient of -(dmin^2/dmax^2)."""
        d2 = sq_dists(x)
        P = x.reshape(n, d)
        im = int(np.argmin(d2))
        iM = int(np.argmax(d2))
        iu = np.triu_indices(n, k=1)
        a, b = iu[0][im], iu[1][im]
        c, e = iu[0][iM], iu[1][iM]
        dmin2 = d2[im]
        dmax2 = max(d2[iM], 1e-18)
        r2 = dmin2 / dmax2
        grad = np.zeros((n, d))
        # f = r^2 ; df/da = 2*(Pa-Pb)/dmax2 for the min pair
        umin = (P[a] - P[b]) / max(dmin2, 1e-18)
        grad[a] += 2.0 * umin / dmax2
        grad[b] -= 2.0 * umin / dmax2
        # contribution from dmax pair: -2*dmin2*(Pc-Pe)/dmax2^2
        umax = (P[c] - P[e]) / dmax2
        grad[c] -= 2.0 * dmin2 * umax / dmax2
        grad[e] += 2.0 * dmin2 * umax / dmax2
        return -r2, grad.ravel()

    rng = np.random.default_rng(42)
    best_x = None
    best_true = -np.inf
    # structured starts: icosahedron + 2 extra points along each axis
    t = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ], dtype=float)
    structured = []
    for axis in range(3):
        for rad in (2.0, 3.0):
            extra = np.zeros((2, 3))
            extra[0, axis] = rad
            extra[1, axis] = -rad
            x0 = np.vstack([ico, extra])
            assert x0.shape == (n, d), f"seed shape {x0.shape}"
            structured.append(x0.ravel())
    for trial in range(10):
        if trial < len(structured):
            x0 = structured[trial].copy()
        else:
            x0 = rng.standard_normal(n * d) * 2.0
        # progressive sharpening of the soft min/max
        for p in (4.0, 16.0, 48.0):
            res = minimize(objective, x0, args=(p,), method="L-BFGS-B",
                           options={"maxiter": 500})
            x0 = res.x
        # final polish on the exact (dmin/dmax)^2 objective
        try:
            res = minimize(exact_grad, x0, jac=True, method="L-BFGS-B",
                           options={"maxiter": 300, "ftol": 1e-16,
                                    "gtol": 1e-12})
            cand = res.x
            if np.all(np.isfinite(cand)) and cand.size == n * d:
                if sq_dists(cand).min() / sq_dists(cand).max() > \
                        sq_dists(x0).min() / sq_dists(x0).max():
                    x0 = cand
        except Exception:
            pass
        d2 = sq_dists(x0)
        true = d2.min() / d2.max()
        if true > best_true:
            best_true = true
            best_x = x0.copy()

    points = best_x.reshape(n, d)
    # center and normalize (scale-invariant objective)
    points = points - points.mean(axis=0)
    scale = np.max(np.linalg.norm(points, axis=1))
    if scale > 0:
        points = points / scale

    return points


# EVOLVE-BLOCK-END