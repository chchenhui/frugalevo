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

    # icosahedron vertices (edge length 2/sin(2pi/5) normalized to circumradius 1)
    phi = (1 + 5 ** 0.5) / 2
    ico = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]], dtype=float)
    ico /= np.linalg.norm(ico[0])

    def rot(P, axis, ang):
        axis = np.asarray(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return P @ (np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K))

    seeds = []
    # icosahedron + antipodal poles at various radii and orientations
    for h in (0.2, 0.35, 0.5, 0.7, 1.0, 1.3, 1.6):
        poles = np.array([[0, 0, h], [0, 0, -h]], dtype=float)
        seeds.append(np.vstack([ico, poles]))
    for ang in (0.3, 0.8, 1.5):
        seeds.append(np.vstack([rot(ico, [1, 0.3, 0.2], ang), [[0, 0, 1.0], [0, 0, -1.0]]]))
    # cube vertices + face centers hints
    cube = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)
    seeds.append(np.vstack([cube[:7], -cube[:7]]))
    # a few random starts with fixed seeds
    for seed in range(6):
        rng = np.random.RandomState(seed)
        seeds.append(rng.randn(n, d))

    best_s, best_P = -1.0, None
    for P0 in seeds:
        res = minimize(objective, P0.ravel(), method="L-BFGS-B",
                       options={"maxiter": 800, "maxfun": 4000})
        P = res.x.reshape(n, d)
        s = score(P)
        if s > best_s:
            best_s, best_P = s, P
        if best_s >= 0.2395:
            break

    # center and normalize for a clean output
    best_P = best_P - best_P.mean(axis=0)
    best_P = best_P / np.abs(best_P).max()
    return best_P


# EVOLVE-BLOCK-END