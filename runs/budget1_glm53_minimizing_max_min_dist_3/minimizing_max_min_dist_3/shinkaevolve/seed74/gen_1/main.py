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

    rng = np.random.default_rng(42)
    best_x = None
    best_true = -np.inf
    for trial in range(10):
        if trial == 0:
            # structured start: 12 points on icosahedron ring + 2 poles
            t = (1 + np.sqrt(5)) / 2
            ico = np.array([
                [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
                [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
                [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
            ])
            poles = np.array([[0, 0, 3.0], [0, 0, -3.0]])
            x0 = np.vstack([ico, poles]).ravel()
        else:
            x0 = rng.standard_normal(n * d) * 2.0
        # progressive sharpening of the soft min/max
        for p in (4.0, 16.0, 48.0):
            res = minimize(objective, x0, args=(p,), method="L-BFGS-B",
                           options={"maxiter": 500})
            x0 = res.x
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