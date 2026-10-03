# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Approach: use the D6d two-staggered-hexagonal-ring family plus two polar
    cap points (the topology of the best-known 14-point 3D packings), which
    the cube+octahedron family cannot reach. Five parameters (ring radius r,
    ring height h, cap height c, twist phi, global scale handled by
    normalization) are globally optimized with seeded differential_evolution
    on the exact squared ratio (dmin/dmax)^2. The best seed is expanded to 42
    free coordinates and polished with L-BFGS-B on a smooth soft-min/soft-max
    surrogate. A cube+octahedron fallback preserves validity if optimization
    fails; a finite (14,3) array is always returned.
    """
    n, d = 14, 3
    rng = np.random.RandomState(42)
    iu = np.triu_indices(n, 1)

    def ratio(P):
        D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
        dd = D[iu]
        return dd.min() / dd.max()

    def sq_ratio(P):
        D2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
        dd = D2[iu]
        return dd.min() / dd.max()

    def two_rings(r, h1, h2, c, phi):
        # Extended D6d family: independent ring heights break the
        # up-down symmetry that saturated the 5-parameter version.
        P = np.zeros((n, d))
        ang1 = np.arange(6) * np.pi / 3.0
        ang2 = ang1 + np.pi / 6.0 + phi
        P[:6, 0] = r * np.cos(ang1)
        P[:6, 1] = r * np.sin(ang1)
        P[:6, 2] = h1
        P[6:12, 0] = r * np.cos(ang2)
        P[6:12, 1] = r * np.sin(ang2)
        P[6:12, 2] = -h2
        P[12, 2] = c
        P[13, 2] = -c
        return P

    def cube_octa(s, a):
        P = np.zeros((n, d))
        k = 0
        for x in (-1, 1):
            for y in (-1, 1):
                for z in (-1, 1):
                    P[k] = (x * s, y * s, z * s)
                    k += 1
        for j in range(3):
            P[8 + 2 * j, j] = a
            P[9 + 2 * j, j] = -a
        return P

    # Global search over the two-ring + caps parameter family
    best_P = cube_octa(1.0, 1.0)
    best_r = ratio(best_P)
    try:
        from scipy.optimize import differential_evolution

        def neg_sr(p):
            P = two_rings(p[0], p[1], p[2], p[3], p[4])
            return -sq_ratio(P)

        de = differential_evolution(
            neg_sr,
            bounds=[(0.2, 0.9), (0.05, 0.6), (0.05, 0.6), (0.4, 1.3),
                    (0.0, np.pi / 3.0)],
            seed=0, popsize=15, maxiter=80, tol=1e-12,
        )
        if np.all(np.isfinite(de.x)):
            # Nelder-Mead polish of the DE winner on the exact squared
            # ratio before expanding to 42 free coordinates.
            nm = minimize(neg_sr, de.x, method="Nelder-Mead",
                          options={"maxiter": 2000, "xatol": 1e-10,
                                   "fatol": 1e-12})
            for p in (nm.x, de.x):
                if np.all(np.isfinite(p)):
                    P = two_rings(*p)
                    r = ratio(P)
                    if r > best_r:
                        best_r, best_P = r, P
    except Exception:
        pass

    # Smooth surrogate: minimize (softmax(d^2) - softmin(d^2)) / softmax(d^2)
    def soft_obj(x, p):
        P = x.reshape(n, d)
        D2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
        d2 = D2[iu]
        dmin_soft = -np.log(np.exp(-p * d2).sum()) / p
        dmax_soft = np.log(np.exp(p * d2).sum()) / p
        return (dmax_soft - dmin_soft) / max(dmax_soft, 1e-12)

    starts = [best_P.ravel()]
    for _ in range(10):
        starts.append((best_P + 0.03 * rng.randn(n, d)).ravel())

    for x0 in starts:
        res = minimize(soft_obj, x0, args=(40.0,), method="L-BFGS-B",
                       options={"maxiter": 2000})
        if not np.all(np.isfinite(res.x)):
            continue
        P = res.x.reshape(n, d)
        r = ratio(P)
        if r > best_r:
            best_r, best_P = r, P

    # Final polish from the incumbent with a sharper surrogate
    res = minimize(soft_obj, best_P.ravel(), args=(80.0,), method="L-BFGS-B",
                   options={"maxiter": 3000})
    if np.all(np.isfinite(res.x)):
        P = res.x.reshape(n, d)
        if ratio(P) > best_r:
            best_P = P

    # Exact max-min refinement: SLSQP on the true problem (maximize t s.t.
    # d_ij^2 >= t, d_ij^2 <= 1), with analytic Jacobians. This removes the
    # surrogate bias of the soft-min/soft-max polish at the active set.
    def con_vals(z):
        P = z[:42].reshape(n, d)
        D2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
        return D2[iu]

    def con_jac(z):
        P = z[:42].reshape(n, d)
        J = np.zeros((91, 43))
        for k, (i, j) in enumerate(zip(*iu)):
            diff = 2.0 * (P[i] - P[j])
            J[k, 3 * i:3 * i + 3] = diff
            J[k, 3 * j:3 * j + 3] = -diff
        J[:, 42] = -1.0
        return J

    # Bounded multistart SLSQP: incumbent plus 4 deterministic perturbations,
    # each hard-limited to 800 iterations; exact-ratio incumbent selection.
    starts = [np.array(best_P, dtype=float, copy=True)]
    for _ in range(4):
        starts.append(best_P + 0.02 * rng.randn(n, d))

    for P0 in starts:
        try:
            Q0 = np.array(P0, dtype=float, copy=True)
            D2 = ((Q0[:, None, :] - Q0[None, :, :]) ** 2).sum(-1)
            d2 = D2[iu]
            Q0 = Q0 / np.sqrt(d2.max())
            t0 = max(d2.min() / d2.max(), 1e-6)
            z0 = np.concatenate([Q0.ravel(), [t0]])
            res = minimize(
                lambda z: -z[42], z0,
                jac=lambda z: np.concatenate([np.zeros(42), [-1.0]]),
                method="SLSQP",
                constraints=[
                    {"type": "ineq", "fun": lambda z: con_vals(z) - z[42],
                     "jac": con_jac},
                    {"type": "ineq", "fun": lambda z: 1.0 - con_vals(z),
                     "jac": lambda z: -con_jac(z)},
                ],
                options={"ftol": 1e-12, "maxiter": 800},
            )
            if np.all(np.isfinite(res.x)):
                P = res.x[:42].reshape(n, d)
                if ratio(P) > best_r:
                    best_r, best_P = ratio(P), P
        except Exception:
            continue

    return best_P


# EVOLVE-BLOCK-END
