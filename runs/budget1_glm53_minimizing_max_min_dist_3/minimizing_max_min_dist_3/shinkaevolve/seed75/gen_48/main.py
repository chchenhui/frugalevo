# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

N = 14
D = 3
IU = np.triu_indices(N, 1)
_IJ = np.array([(i, j) for i in range(N) for j in range(i + 1, N)])
_IJ0 = _IJ[:, 0]
_IJ1 = _IJ[:, 1]


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


def d6_family(R, h, z0):
    """Two staggered hexagonal rings (30 deg offset) at z=+-h plus two poles."""
    th = np.arange(6) * (np.pi / 3.0)
    r1 = np.stack([R * np.cos(th), R * np.sin(th), np.full(6, h)], axis=1)
    r2 = np.stack([R * np.cos(th + np.pi / 6), R * np.sin(th + np.pi / 6),
                   np.full(6, -h)], axis=1)
    poles = np.array([[0.0, 0.0, z0], [0.0, 0.0, -z0]])
    return np.vstack([r1, r2, poles])


def polish(P0, taus=(0.02, 0.008, 0.003, 0.001, 0.0004, 0.00015)):
    """Annealed soft-min/soft-max L-BFGS polish on squared distances.

    The surrogate tau*(logsumexp(-d2/tau) + logsumexp(d2/tau)) is smooth,
    scale-aware, and converges to -(dmin^2 - dmax^2) as tau -> 0, so
    maximizing it drives the true ratio toward a local optimum with
    coordinated moves of all points.
    """
    P = normalize(P0)
    x = P.ravel()

    def obj(z, tau):
        X = z.reshape(N, D)
        d2 = np.sum((X[_IJ0] - X[_IJ1]) ** 2, axis=1)
        return tau * (logsumexp(-d2 / tau) + logsumexp(d2 / tau))

    for tau in taus:
        try:
            res = minimize(lambda z, t=tau: obj(z, t), x, method="L-BFGS-B",
                           options={"maxiter": 400, "ftol": 1e-15,
                                    "gtol": 1e-13})
            if np.isfinite(res.x).all():
                x = res.x
        except Exception:
            break
    X = normalize(x.reshape(N, D))
    return X


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
    # parametric D6 family: staggered hex rings + poles (known good basin)
    d6_best_r, d6_best = -1.0, None
    for h in np.linspace(0.15, 1.4, 26):
        for z0 in np.linspace(max(h, 0.2) + 0.05, 2.6, 48):
            P = d6_family(1.0, h, z0)
            r = score(P)
            if r > d6_best_r:
                d6_best_r, d6_best = r, P
    seeds.append(d6_best)
    # coarse local refinement of the D6 parameters
    for _ in range(8):
        hh, z0 = None, None
        for h in np.linspace(0.3, 1.0, 8):
            for z0c in np.linspace(max(h, 0.2) + 0.05, 2.0, 12):
                P = d6_family(1.0, h, z0c)
                r = score(P)
                if r > d6_best_r:
                    d6_best_r, d6_best = r, P
    seeds.append(d6_best)

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

    # gradient-based soft-ratio polish of the current best
    Q = polish(best)
    q = score(Q)
    if q > best_s:
        best_s, best = q, Q
    # jittered polish restarts (coordinated moves can escape relax plateaus)
    for k in range(6):
        Q = polish(best + 0.03 * (0.7 ** k) * rng.normal(size=best.shape))
        q = score(Q)
        if q > best_s + 1e-12:
            best_s, best = q, Q
        # alternate: jitter then relax then polish
        Qr, _ = relax(perturb(best, rng, 0.02), iters=200)
        Qp = polish(Qr)
        q = score(Qp)
        if q > best_s + 1e-12:
            best_s, best = q, Qp

    # exact symmetric D6 candidate as a guarded final comparison
    if d6_best is not None:
        Pd6 = normalize(d6_best)
        if score(Pd6) > best_s:
            best_s, best = score(Pd6), Pd6

    best = normalize(best)
    assert np.isfinite(best).all() and best.shape == (N, D)
    return best
# EVOLVE-BLOCK-END