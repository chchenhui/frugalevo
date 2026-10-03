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

    def ratio_of(pts):
        dm = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        iu = np.triu_indices(n, 1)
        dists = dm[iu]
        return dists.min() / dists.max()

    best_pts = None
    best_ratio = -1.0

    for seed in range(6):
        rng = np.random.RandomState(seed)
        # start on the unit sphere (scaling/translation invariant anyway)
        x = rng.randn(n, d)
        x /= np.linalg.norm(x, axis=1, keepdims=True)

        def obj(z):
            t = z[-1]
            return -t  # maximize t

        def obj_grad(z):
            g = np.zeros_like(z)
            g[-1] = -1.0
            return g

        def make_constraints():
            cons = []
            for i in range(n):
                for j in range(i + 1, n):
                    cons.append({
                        'type': 'ineq',
                        'fun': (lambda z, i=i, j=j:
                                np.sum((z[3*i:3*i+3] - z[3*j:3*j+3])**2) - z[-1]),
                        'jac': (lambda z, i=i, j=j: _pair_grad(z, i, j)),
                    })
            return cons

        def _pair_grad(z, i, j):
            g = np.zeros(len(z))
            di = 2.0 * (z[3*i:3*i+3] - z[3*j:3*j+3])
            g[3*i:3*i+3] = di
            g[3*j:3*j+3] = -di
            g[-1] = -1.0
            return g

        z0 = np.concatenate([x.ravel(), [0.1]])
        res = minimize(obj, z0, jac=obj_grad,
                       constraints=make_constraints(),
                       method='SLSQP',
                       options={'maxiter': 300, 'ftol': 1e-10})
        pts = res.x[:-1].reshape(n, d)
        r = ratio_of(pts)
        if r > best_ratio:
            best_ratio = r
            best_pts = pts

    # normalize: scale so max pairwise distance equals 1 (objective invariant)
    dm = np.linalg.norm(best_pts[:, None, :] - best_pts[None, :, :], axis=-1)
    best_pts = best_pts / dm.max()

    return best_pts


# EVOLVE-BLOCK-END