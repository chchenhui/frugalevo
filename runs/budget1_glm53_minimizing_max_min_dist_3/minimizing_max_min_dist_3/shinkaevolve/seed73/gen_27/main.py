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

    def fibonacci_sphere(k):
        i = np.arange(k) + 0.5
        phi = np.pi * (1.0 + 5.0 ** 0.5) * i
        z = 1.0 - 2.0 * i / k
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)

    # structured seed: icosahedron (12 verts) + 2 poles
    t = (1.0 + 5.0 ** 0.5) / 2.0
    ico = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], dtype=float)
    ico /= np.linalg.norm(ico, axis=1, keepdims=True)

    cube = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)

    seeds = []
    # structured seeds first (most reliable)
    for k in range(8):
        rng = np.random.RandomState(100 + k)
        Q, _ = np.linalg.qr(rng.randn(d, d))
        base = np.vstack([ico, [[0, 0, 1.0], [0, 0, -1.0]]])
        P0 = base @ Q.T + 0.05 * rng.randn(n, d)
        seeds.append(P0)
    seeds.append(fibonacci_sphere(n))
    seeds.append(np.vstack([cube[:7], -cube[:7]]))
    seeds.append(cube[:7].tolist() and np.vstack([cube[:7], -cube[:7]]) + 0.01 * np.random.RandomState(7).randn(n, d))
    # random seeds (ball and gaussian)
    for seed in range(16):
        rng = np.random.RandomState(seed)
        if seed % 2 == 0:
            P0 = rng.randn(n, d)
        else:
            P0 = rng.randn(n, d) * rng.rand(n, 1)
        seeds.append(P0)

    cands = []
    for P0 in seeds:
        res = minimize(objective, P0.ravel(), method="L-BFGS-B",
                       options={"maxiter": 500, "maxfun": 2000})
        P = res.x.reshape(n, d)
        cands.append((score(P), P))

    cands.sort(key=lambda c: -c[0])

    # polish top-7 candidates with restarts from their optimized positions
    best_s, best_P = cands[0]
    for s0, P0 in cands[:7]:
        rng = np.random.RandomState(200)
        for trial in range(4):
            start = P0 if trial == 0 else P0 + 0.02 * rng.randn(n, d)
            res = minimize(objective, start.ravel(), method="L-BFGS-B",
                           options={"maxiter": 800, "maxfun": 3000})
            P = res.x.reshape(n, d)
            s = score(P)
            if s > best_s:
                best_s, best_P = s, P

    # center and normalize for a clean output
    best_P = best_P - best_P.mean(axis=0)
    best_P = best_P / np.abs(best_P).max()
    return best_P


# EVOLVE-BLOCK-END