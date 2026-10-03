# EVOLVE-BLOCK-START
import numpy as np

N = 14
D = 3
IU = np.triu_indices(N, 1)


def pdist(P):
    diff = P[:, None, :] - P[None, :, :]
    return np.sqrt((diff * diff).sum(-1))


def score(P):
    d = pdist(P)[IU]
    return d.min() / d.max()


def normalize(P):
    P = P - P.mean(axis=0)
    dm = pdist(P)[IU].max()
    return P / dm


def relax(P, iters=600, seed_rng=None):
    """Repulsion relaxation under unit-diameter normalization."""
    P = normalize(P)
    best = P.copy()
    best_s = score(P)
    for it in range(iters):
        frac = it / iters
        alpha = 0.4 * (1.0 - frac) ** 1.5 + 0.01
        diff = P[:, None, :] - P[None, :, :]
        d = np.sqrt((diff * diff).sum(-1))
        dd = d[IU]
        dmin = dd.min()
        # adaptive target: push just above current minimum, capped
        t = min(dmin * 1.06 + 0.005, 0.75)
        mask = dd < t
        if mask.any():
            I = IU[0][mask]
            J = IU[1][mask]
            dm_ = dd[mask][:, None]
            dirs = diff[I, J] / np.maximum(dm_, 1e-12)
            w = (alpha * (t - dd[mask]))[:, None]
            step = w * dirs
            np.add.at(P, I, step)
            np.add.at(P, J, -step)
        P = normalize(P)
        s = score(P)
        if s > best_s:
            best_s = s
            best = P.copy()
    return best, best_s


def perturb(P, rng, scale):
    return normalize(P + rng.normal(scale=scale, size=P.shape))


def min_max_dist_dim3_14() -> np.ndarray:
    rng = np.random.default_rng(20240517)

    seeds = []
    for _ in range(24):
        seeds.append(rng.normal(size=(N, D)))
    # structured seeds: icosahedron + 2 poles, cubic lattice picks
    p = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [-1, p, 0], [1, p, 0], [-1, -p, 0], [1, -p, 0],
        [0, -1, p], [0, 1, p], [0, -1, -p], [0, 1, -p],
        [p, 0, -1], [p, 0, 1], [-p, 0, -1], [-p, 0, 1],
    ], dtype=float)
    for s in (0.3, 0.6, 0.9):
        seeds.append(np.vstack([ico * s, [[0, 0, 1.0], [0, 0, -1.0]]]))
    g = np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])).reshape(3, -1).T
    for _ in range(4):
        seeds.append(g[rng.choice(len(g), size=N, replace=False)])
    # two-cluster seeds encourage non-spherical (small-diameter) optima
    for _ in range(6):
        a = rng.normal(size=(7, D)) + np.array([1.5, 0, 0])
        b = rng.normal(size=(7, D)) - np.array([1.5, 0, 0])
        seeds.append(np.vstack([a, b]))

    best, best_s = None, -1.0
    for s0 in seeds:
        P, s = relax(normalize(s0), iters=500)
        if s > best_s:
            best_s, best = s, P
        # basin-hopping around current best occasionally
        if s > best_s - 0.02:
            for k in range(3):
                Q, sq_ = relax(perturb(P, rng, 0.05 * (k + 1)), iters=300)
                if sq_ > best_s:
                    best_s, best = sq_, Q

    # final polish: tight relaxation restarts around best
    for k in range(8):
        Q, s = relax(perturb(best, rng, 0.02), iters=400)
        if s > best_s:
            best_s, best = s, Q

    best = normalize(best)
    assert np.isfinite(best).all() and best.shape == (N, D)
    return best
# EVOLVE-BLOCK-END