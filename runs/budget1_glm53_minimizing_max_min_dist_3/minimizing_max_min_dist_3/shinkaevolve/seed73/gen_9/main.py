# EVOLVE-BLOCK-START
import numpy as np

N = 14
D = 3


def pair_data(P):
    diff = P[:, None, :] - P[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))
    np.fill_diagonal(dist, np.inf)
    iu = np.triu_indices(N, 1)
    return diff, dist, iu


def evaluate(P):
    _, dist, iu = pair_data(P)
    dm = dist[iu]
    dmin, dmax = dm.min(), dm.max()
    if dmax <= 0:
        return -1.0
    return (dmin / dmax) ** 2


def project_sphere(P):
    return P / np.linalg.norm(P, axis=1, keepdims=True)


def relax(P, steps=1500, step0=0.03, kick_every=300, kick_scale=0.02, seed=0):
    rng = np.random.RandomState(seed)
    P = project_sphere(P)
    for t in range(steps):
        frac = 1.0 - t / steps
        step = step0 * frac + 1e-4
        diff, dist, iu = pair_data(P)
        dm = dist[iu]
        dmin = dm.min()
        # inverse-square repulsion from everyone
        F = -diff / dist[:, :, None] ** 3
        forces = F.sum(axis=1)
        # strong extra force on all pairs within 15% of current min
        close = np.where(dm < dmin * 1.15)[0]
        for k in close:
            i, j = iu[0][k], iu[1][k]
            f = (P[i] - P[j]) / max(dm[k], 1e-9) ** 3
            forces[i] += f
            forces[j] -= f
        # damp radial component, move, reproject
        r = np.linalg.norm(P, axis=1, keepdims=True)
        forces = forces - (forces * P).sum(1, keepdims=True) * P / r ** 2
        P = P + step * forces
        P = project_sphere(P)
        if kick_every and t % kick_every == 0 and t > 0:
            P = project_sphere(P + kick_scale * frac * rng.randn(N, D))
    return P


def initializers():
    seeds = []
    cube = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)
    seeds.append(project_sphere(np.vstack([cube[:7], -cube[:7]])))
    # icosahedron vertices + centers
    phi = (1 + 5 ** 0.5) / 2
    ico = []
    for a in (-1, 1):
        for b in (-phi, phi):
            ico += [[0, a, b], [a, b, 0], [b, 0, a]]
    ico = np.array(ico, dtype=float)
    c = ico.mean(0)
    ico = project_sphere(ico - c)
    seeds.append(np.vstack([ico, -ico[:2]]))
    for s in range(10):
        rng = np.random.RandomState(100 + s)
        seeds.append(project_sphere(rng.randn(N, D)))
    return seeds


def min_max_dist_dim3_14() -> np.ndarray:
    best_s, best_P = -1.0, None
    for idx, P0 in enumerate(initializers()):
        P = relax(P0, seed=idx)
        # fine polish: small-step re-relax
        P = relax(P, steps=400, step0=0.005, kick_every=0, seed=idx)
        s = evaluate(P)
        if s > best_s:
            best_s, best_P = s, P.copy()
    best_P = best_P - best_P.mean(0)
    best_P = best_P / np.abs(best_P).max()
    return best_P


# EVOLVE-BLOCK-END