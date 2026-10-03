# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

N = 14
D = 3
_IU = np.triu_indices(N, 1)


def pair_dists(P):
    Dm = np.linalg.norm(P[:, None] - P[None, :], axis=-1)
    return Dm[_IU]


def ratio(P):
    d = pair_dists(P)
    return d.min() / d.max()


def d6_family(R, h, z0):
    """Two staggered hexagonal rings (30 deg offset) at z=+-h plus two poles."""
    th = np.arange(6) * (np.pi / 3.0)
    r1 = np.stack([R * np.cos(th), R * np.sin(th), np.full(6, h)], axis=1)
    r2 = np.stack([R * np.cos(th + np.pi / 6), R * np.sin(th + np.pi / 6),
                   np.full(6, -h)], axis=1)
    poles = np.array([[0.0, 0.0, z0], [0.0, 0.0, -z0]])
    return np.vstack([r1, r2, poles])


def soft_ratio_obj(z, tau):
    R, h, z0 = z
    if R <= 0 or h <= 0 or z0 <= h:
        return 1e6
    P = d6_family(R, h, z0)
    d = pair_dists(P)
    # maximize soft-min - soft-max (negative for minimize)
    smin = -tau * logsumexp(-d / tau)
    smax = tau * logsumexp(d / tau)
    return -(smin - smax)


def polish(P0, taus=(0.05, 0.02, 0.008, 0.003, 0.001, 0.0004)):
    """Annealed soft-min/soft-max L-BFGS-B polish, scale-invariant objective."""
    P = (P0 - P0.mean(axis=0)).copy()
    P = P / pair_dists(P).max()
    x = P.ravel()
    IJ = np.array([(i, j) for i in range(N) for j in range(i + 1, N)])

    def obj(z, tau=0.01):
        X = z.reshape(N, D)
        diff = X[IJ[:, 0]] - X[IJ[:, 1]]
        d2 = np.sum(diff * diff, axis=1)
        return tau * (logsumexp(-d2 / tau) + logsumexp(d2 / tau))

    for tau in taus:
        res = minimize(lambda z, t=tau: obj(z, t), x, method="L-BFGS-B",
                       options={"maxiter": 300, "ftol": 1e-14, "gtol": 1e-12})
        if np.isfinite(res.x).all():
            x = res.x
    X = x.reshape(N, D)
    X = X - X.mean(axis=0)
    X = X / pair_dists(X).max()
    return X


def min_max_dist_dim3_14() -> np.ndarray:
    rng = np.random.default_rng(7)

    # --- Stage 1: grid search over the 2-DOF D6 family (fix scale by R) ---
    best_param, best_r = None, -1.0
    for h in np.linspace(0.15, 1.4, 26):
        for z0 in np.linspace(max(h, 0.2) + 0.05, 2.6, 48):
            P = d6_family(1.0, h, z0)
            r = ratio(P)
            if r > best_r:
                best_r, best_param = r, (1.0, h, z0)

    # --- Stage 2: Nelder-Mead refine the 3 params (R adds scale redundancy, fine) ---
    res = minimize(lambda z: soft_ratio_obj(z, 0.002), np.array(best_param),
                   method="Nelder-Mead",
                   options={"xatol": 1e-10, "fatol": 1e-14, "maxiter": 4000})
    if np.isfinite(res.x).all():
        R, h, z0 = res.x
        if R > 0 and h > 0 and z0 > h:
            r = ratio(d6_family(R, h, z0))
            if r > best_r:
                best_r, best_param = r, (R, h, z0)

    # --- Stage 3: full polish from exact symmetric optimum ---
    P0 = d6_family(*best_param)
    P0 = P0 - P0.mean(axis=0)
    P0 = P0 / pair_dists(P0).max()
    best = polish(P0)
    best_r = ratio(best)

    # --- Stage 4: jittered restarts to escape any numerical plateau ---
    for k in range(8):
        mag = 0.03 * (0.7 ** k)
        Q = best + mag * rng.normal(size=best.shape)
        Q = polish(Q)
        r = ratio(Q)
        if r > best_r + 1e-12:
            best_r, best = r, Q.copy()

    # final exact symmetric candidate comparison (parametric optimum may beat polish)
    Psym = d6_family(*best_param)
    Psym = (Psym - Psym.mean(axis=0)) / pair_dists(Psym).max()
    if ratio(Psym) > best_r:
        best, best_r = Psym, ratio(Psym)

    # --- Stage 5: tiny high-precision polish of the winner ---
    Q = polish(best, taus=(0.001, 0.0003, 0.0001))
    if ratio(Q) > best_r:
        best, best_r = Q, ratio(Q)

    # --- Stage 6: targeted pair/triple hill-climb on min- and max-distance pairs ---
    def renorm(P):
        P = P - P.mean(axis=0)
        return P / pair_dists(P).max()

    def hillclimb(P0, iters=400, step0=0.02):
        P = renorm(P0.copy())
        r_best = ratio(P)
        step = step0
        iu = np.array([(i, j) for i in range(N) for j in range(i + 1, N)])
        for it in range(iters):
            d = pair_dists(P)
            ia, ib = iu[d.argmin()]
            ja, jb = iu[d.argmax()]
            improved = False
            u_min = P[ia] - P[ib]
            n_min = np.linalg.norm(u_min)
            if n_min > 1e-12:
                u_min /= n_min
            u_max = P[ja] - P[jb]
            n_max = np.linalg.norm(u_max)
            if n_max > 1e-12:
                u_max /= n_max
            for mag in (step, step / 2, step / 4):
                # move 1: push min pair apart
                Q = P.copy()
                Q[ia] += mag * u_min
                Q[ib] -= mag * u_min
                Q = renorm(Q)
                r = ratio(Q)
                if r > r_best + 1e-14:
                    P, r_best, improved = Q, r, True
                    break
                # move 2: pull max pair inward
                Q = P.copy()
                Q[ja] -= mag * u_max
                Q[jb] += mag * u_max
                Q = renorm(Q)
                r = ratio(Q)
                if r > r_best + 1e-14:
                    P, r_best, improved = Q, r, True
                    break
                # move 3 (triple): do both simultaneously
                if (ia, ib) != (ja, jb):
                    Q = P.copy()
                    Q[ia] += mag * u_min
                    Q[ib] -= mag * u_min
                    Q[ja] -= mag * u_max
                    Q[jb] += mag * u_max
                    Q = renorm(Q)
                    r = ratio(Q)
                    if r > r_best + 1e-14:
                        P, r_best, improved = Q, r, True
                        break
            if not improved:
                step *= 0.6
                if step < 1e-7:
                    break
        return P, r_best

    for k in range(6):
        Q, r = hillclimb(best, iters=300, step0=0.02 * (0.7 ** k))
        if r > best_r + 1e-14:
            best, best_r = Q, r
        else:
            break

    points = np.asarray(best, dtype=float)
    assert points.shape == (N, D) and np.isfinite(points).all()
    assert pair_dists(points).max() > 0
    return points
# EVOLVE-BLOCK-END