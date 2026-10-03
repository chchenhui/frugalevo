# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y) coordinates of the 14 points.

    """

    from scipy.optimize import minimize
    from scipy.special import logsumexp

    n = 14
    d = 3
    iu = np.triu_indices(n, 1)

    def unit(x):
        p = x.reshape(n, d)
        return p / np.linalg.norm(p, axis=1, keepdims=True)

    def dists(p):
        diff = p[:, None, :] - p[None, :, :]
        D2 = (diff * diff).sum(-1)
        return np.sqrt(np.maximum(D2[iu], 1e-12))

    def ratio(p):
        D = dists(p)
        return D.min() / D.max()

    def tammes14_seed():
        # Two staggered hexagonal rings at height +-h plus two poles.
        # For 14 points the optimal ring height satisfies
        # (2/3)*r^2 = (1-h)^2  with r^2 = 1 - h^2, giving
        # h = 1/sqrt(5) and r = 2/sqrt(5).
        h = 1.0 / np.sqrt(5.0)
        r = 2.0 / np.sqrt(5.0)
        ang = np.pi / 6.0
        pts = []
        pts.append([0.0, 0.0, 1.0])
        pts.append([0.0, 0.0, -1.0])
        for k in range(6):
            a = k * (np.pi / 3.0)
            pts.append([r * np.cos(a), r * np.sin(a), h])
            pts.append([r * np.cos(a + ang), r * np.sin(a + ang), -h])
        return np.array(pts)

    rng = np.random.default_rng(0)
    best_pts = None
    best_r = -1.0

    seeds = [tammes14_seed()]
    for _ in range(9):
        p = rng.normal(size=(n, d))
        seeds.append(p / np.linalg.norm(p, axis=1, keepdims=True))

    for pts in seeds:

        # Annealed soft-min maximization of minimum distance on the unit sphere.
        for T in (0.08, 0.03, 0.012, 0.005, 0.002):
            def loss(x, T=T):
                p = unit(x)
                D = dists(p)
                return T * logsumexp(-D / T)

            res = minimize(loss, pts.ravel(), method="L-BFGS-B",
                           options={"maxiter": 500, "maxfun": 2000})
            pts = unit(res.x)

        # Final refinement on the true objective dmin/dmax (normalized to
        # unit radius of gyration keeps scale fixed while allowing the
        # diameter to shrink, exactly what the metric rewards).
        def loss2(x):
            p = x.reshape(n, d)
            p = p - p.mean(0)
            D = dists(p)
            lo = T * logsumexp(-D / T)
            hi = 0.005 * logsumexp(D / 0.005)
            return lo + hi

        res = minimize(loss2, pts.ravel(), method="L-BFGS-B",
                       options={"maxiter": 800, "maxfun": 3000})
        p2 = res.x.reshape(n, d)
        p2 -= p2.mean(0)
        r = ratio(p2)
        if r > best_r:
            best_r = r
            best_pts = p2.copy()

        r = ratio(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()

    return best_pts


# EVOLVE-BLOCK-END