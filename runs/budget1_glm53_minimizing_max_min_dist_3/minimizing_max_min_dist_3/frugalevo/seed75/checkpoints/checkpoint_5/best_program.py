# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import pdist


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Point-group orbit family search: 2 antipodal poles + two staggered
    hexagonal rings with independent radii/heights/twist (6 parameters).

    Approach:
    - Family P(h, z1, r1, z2, r2, a): poles at (0,0,+/-h); ring 1 at
      (z1, r1) with angles 2*pi*k/6; ring 2 at (z2, r2) with angles
      2*pi*k/6 + a. Off-sphere radii r1 != r2 realize configurations a
      unit-sphere-constrained search cannot represent.
    - Maximize the exact dmin/dmax ratio over this family with
      differential_evolution (popsize 20, maxiter 200, seed 7).
    - Polish the best family optimum in the full 42-dim point space with
      a trust-constr epigraph NLP to escape the family boundary.
    - Keep a safe fallback; return the best valid incumbent.
    """
    n = 14
    k = 6
    ang = 2.0 * np.pi * np.arange(k) / k
    c6, s6 = np.cos(ang), np.sin(ang)

    def build(theta):
        h, z1, r1, z2, r2, a = theta
        P = np.empty((n, 3))
        P[0] = (0.0, 0.0, h)
        P[1] = (0.0, 0.0, -h)
        ca, sa = np.cos(a), np.sin(a)
        P[2:8, 0] = r1 * c6
        P[2:8, 1] = r1 * s6
        P[2:8, 2] = z1
        P[8:14, 0] = r2 * (c6 * ca - s6 * sa)
        P[8:14, 1] = r2 * (c6 * sa + s6 * ca)
        P[8:14, 2] = z2
        return P

    def exact_ratio(P):
        d = pdist(P)
        return d.min() / d.max()

    def family_obj(theta):
        return -exact_ratio(build(theta))

    # --- Stage 1: differential evolution on the 6-parameter family ---
    bounds = [(0.1, 2.0), (-1.5, 1.5), (0.1, 2.0),
              (-1.5, 1.5), (0.1, 2.0), (0.0, np.pi / 6)]
    de = differential_evolution(family_obj, bounds, popsize=30, maxiter=500,
                                tol=1e-12, polish=True, seed=7)
    best_theta = de.x.copy()

    # Collect distinct near-optimal family candidates for polishing.
    fam = [(de.x.copy(), -de.fun)]
    for th in de.population:
        th = np.asarray(th, dtype=np.float64)
        r = exact_ratio(build(th))
        if r > -de.fun - 1e-3:
            fam.append((th.copy(), r))
    fam.sort(key=lambda t: -t[1])
    distinct = []
    for th, r in fam:
        if all(np.linalg.norm(th - u) > 1e-2 for u, _ in distinct):
            distinct.append((th, r))
    distinct = distinct[:5]

    # --- Stage 2: smooth-epigraph polish from top distinct family optima ---
    # Maximize dmin/dmax via a smooth log-sum-exp surrogate:
    #   f_beta = LSE(beta*d)/beta - (-LSE(-beta*d)/beta)  -> (dmax - dmin).
    # Softmax weights give correct analytic gradients over ALL active pairs,
    # letting the polish escape the 6-parameter family subspace. Sharpness
    # is annealed so the smooth landscape guides, then the exact ratio decides.
    def unpack(x):
        return x.reshape(n, 3)

    iu = np.triu_indices(n, 1)

    def smooth_obj_grad(x, beta):
        Q = unpack(x)
        diff = Q[:, None, :] - Q[None, :, :]
        Dsq = (diff ** 2).sum(-1)
        D = np.sqrt(np.maximum(Dsq, 1e-300))
        d = D[iu]
        amax, amin = d.max(), d.min()
        emax = np.exp(beta * (d - amax))
        emin = np.exp(-beta * (d - amin))
        obj = np.log(emax.sum()) / beta + np.log(emin.sum()) / beta
        wmax = emax / emax.sum()
        wmin = emin / emin.sum()
        w = wmax - wmin                      # gradient weight per pair
        G = diff * np.where(D > 1e-12, 1.0 / np.where(D > 1e-12, D, 1.0), 0.0)[:, :, None]
        ii, jj = iu
        grad = np.zeros_like(Q)
        contrib = w[:, None] * G[ii, jj]
        np.add.at(grad, ii, contrib)
        np.add.at(grad, jj, -contrib)
        return obj, grad.ravel()

    rng = np.random.default_rng(11)
    starts = [th for th, _ in distinct]
    for s in range(len(starts)):
        pert = starts[s].copy()
        pert[:5] += rng.normal(scale=0.03, size=5)
        pert[5] = (pert[5] + rng.normal(scale=0.01)) % (np.pi / 6)
        starts.append(pert)

    best_P = build(best_theta)
    best_r = exact_ratio(best_P)
    for th0 in starts:
        x = build(th0).ravel()
        try:
            for beta in (50.0, 200.0, 1000.0):
                res = minimize(smooth_obj_grad, x, args=(beta,),
                               method="L-BFGS-B", jac=True,
                               options={"maxiter": 200})
                x = res.x
            P = unpack(x)
            r = exact_ratio(P)
            if r > best_r:
                best_r, best_P = r, P.copy()
        except Exception:
            continue

    # --- Fallback / validation ---
    best_P = np.asarray(best_P, dtype=np.float64)
    if (not np.all(np.isfinite(best_P)) or best_P.shape != (n, 3)
            or pdist(best_P).max() <= 0.0):
        rng = np.random.default_rng(42)
        best_P = rng.normal(size=(n, 3))
        best_P /= np.linalg.norm(best_P, axis=1, keepdims=True)
    return best_P


# EVOLVE-BLOCK-END
