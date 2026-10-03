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
    rng = np.random.default_rng(42)

    # Structured initial guess: 2 poles + two staggered hexagonal rings
    angles1 = np.arange(6) * (np.pi / 3)
    angles2 = angles1 + np.pi / 6
    z_ring = 0.42
    pts = []
    pts.append([0.0, 0.0, 1.0])
    pts.append([0.0, 0.0, -1.0])
    for a in angles1:
        r = np.sqrt(1 - z_ring**2)
        pts.append([r * np.cos(a), r * np.sin(a), z_ring])
    for a in angles2:
        r = np.sqrt(1 - z_ring**2)
        pts.append([r * np.cos(a), r * np.sin(a), -z_ring])
    pts = np.array(pts)

    from scipy.optimize import minimize

    def unpack(x):
        # work on the unit sphere via spherical coordinates (n, 2): theta, phi
        th = x[0::2]
        ph = x[1::2]
        p = np.stack([np.cos(th) * np.cos(ph), np.sin(th) * np.cos(ph),
                      np.sin(ph)], axis=1)
        return p

    def objective(x):
        p = unpack(x)
        dd = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
        iu = np.triu_indices(n, 1)
        dists = dd[iu]
        dmax = dists.max()
        # smooth surrogate for dmin to keep gradients useful
        dmin = dists.min()
        soft = -np.log(np.sum(np.exp(-(dists / (dmax + 1e-12)) * 50.0))) / 50.0
        return -(dmin / dmax) - 1e-4 * soft

    # initial parameterization
    th = np.arctan2(pts[:, 1], pts[:, 0])
    ph = np.arcsin(np.clip(pts[:, 2], -1, 1))
    x0 = np.empty(2 * n)
    x0[0::2] = th
    x0[1::2] = ph

    best_pts = pts
    best_val = -objective(x0)
    for trial in range(3):
        if trial > 0:
            x = x0 + 0.15 * rng.standard_normal(x0.shape)
        else:
            x = x0
        res = minimize(objective, x, method="SLSQP",
                       options={"maxiter": 400, "ftol": 1e-12})
        val = -res.fun
        if val > best_val:
            best_val = val
            best_pts = unpack(res.x)

    # normalize so the configuration lives in [-1, 1]^3 (convenience only)
    c = best_pts.mean(axis=0)
    best_pts = best_pts - c
    scale = np.abs(best_pts).max()
    best_pts = best_pts / scale

    return np.asarray(best_pts, dtype=float)


# EVOLVE-BLOCK-END