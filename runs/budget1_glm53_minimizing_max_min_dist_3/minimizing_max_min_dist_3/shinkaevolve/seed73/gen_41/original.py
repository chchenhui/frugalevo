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

    iu = np.triu_indices(n, 1)

    def score(P):
        D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
        dmin = D[iu].min()
        dmax = D[iu].max()
        if dmax <= 0:
            return -1.0
        return (dmin / dmax) ** 2

    def objective(flat):
        P = flat.reshape(n, d)
        diff = P[:, None, :] - P[None, :, :]
        dist = np.sqrt((diff ** 2).sum(-1) + 1e-12)
        dm = dist[iu]
        beta = 20.0
        # smooth min and max of pairwise distances
        wmin = np.exp(-beta * dm)
        dmin_s = (dm * wmin).sum() / wmin.sum()
        wmax = np.exp(beta * dm)
        dmax_s = (dm * wmax).sum() / wmax.sum()
        # objective: maximize dmin/dmax  ->  minimize -ratio (smooth)
        return -(dmin_s / dmax_s)

    best_s, best_P = -1.0, None
    for seed in range(60):
        rng = np.random.RandomState(seed)
        P0 = rng.randn(n, d)
        if seed == 0:
            # hint: cube vertices + antipodal-ish start
            cube = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)
            P0 = np.vstack([cube[:7], -cube[:7]])
        elif seed == 1:
            # hint: start from random points in a ball (not on a sphere)
            P0 = rng.randn(n, d) * rng.rand(n, 1)
        res = minimize(objective, P0.ravel(), method="L-BFGS-B",
                       options={"maxiter": 500, "maxfun": 2000})
        P = res.x.reshape(n, d)
        s = score(P)
        if s > best_s:
            best_s, best_P = s, P

    # center and normalize for a clean output
    best_P = best_P - best_P.mean(axis=0)
    best_P = best_P / np.abs(best_P).max()
    return best_P


# EVOLVE-BLOCK-END