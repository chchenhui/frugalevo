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

    def score(P):
        D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
        iu = np.triu_indices(n, 1)
        dmin = D[iu].min()
        dmax = D[iu].max()
        if dmax <= 0:
            return -1.0, P
        return (dmin / dmax) ** 2, P

    def relax(P, steps=4000, step0=0.05):
        P = P / np.linalg.norm(P, axis=1, keepdims=True)
        for t in range(steps):
            step = step0 * (1.0 - t / steps) + 1e-4
            diff = P[:, None, :] - P[None, :, :]
            dist = np.linalg.norm(diff, axis=-1)
            np.fill_diagonal(dist, 1.0)
            # repulsive force ~ 1/dist
            F = -diff / dist[:, :, None] ** 3
            forces = F.sum(axis=1)
            # extra force on the closest pair to push the minimum up
            iu = np.triu_indices(n, 1)
            dm = dist[iu]
            k = np.argmin(dm)
            i, j = iu[0][k], iu[1][k]
            forces[i] += (P[i] - P[j]) / dm[k] ** 3
            forces[j] += (P[j] - P[i]) / dm[k] ** 3
            P = P + step * forces
            P = P / np.linalg.norm(P, axis=1, keepdims=True)
        return P

    best_s, best_P = -1.0, None
    for seed in range(12):
        rng = np.random.RandomState(seed)
        P0 = rng.randn(n, d)
        # start from a reasonable configuration: cube + antipodal pairs hint
        if seed == 0:
            cube = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)
            P0 = np.vstack([cube[:7], -cube[:7]])
        P = relax(P0)
        s, P = score(P)
        if s > best_s:
            best_s, best_P = s, P

    # center and normalize for a clean output
    best_P = best_P - best_P.mean(axis=0)
    best_P = best_P / np.abs(best_P).max()
    return best_P


# EVOLVE-BLOCK-END