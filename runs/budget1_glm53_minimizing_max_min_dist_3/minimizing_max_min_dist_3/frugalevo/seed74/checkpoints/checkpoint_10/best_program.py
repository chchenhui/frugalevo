# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Approach: build a deterministic ladder of structurally distinct
    polyhedral seeds — 12-vertex icosahedron, cuboctahedron, and a
    dodecahedron-vertex subset, each combined with 2 antipodal points on a
    symmetry axis (3 axis directions x 2 heights per family, plus 6
    perturbed copies of the best seed) — and refine every seed with the
    exact max-min SLSQP reformulation (maximize t s.t. d_ij^2 >= t,
    d_ij^2 <= 1) with analytic Jacobians, capped at 600 iterations per
    start. The best refined configuration is further polished with the
    soft-min/soft-max surrogate and additional exact SLSQP restarts. A
    cube+octahedron fallback preserves validity if optimization fails; a
    finite (14,3) array is always returned.
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

    best_P = cube_octa(1.0, 1.0)
    best_r = ratio(best_P)

    # ---- Polyhedral seed ladder -------------------------------------
    # 12-vertex polyhedra (different contact topologies than the D6d
    # hexagonal-ring family) + 2 antipodal points on a symmetry axis.
    phi_g = (1.0 + np.sqrt(5.0)) / 2.0
    icosa = np.array(
        [(0., sg1, sg2 * phi_g) for sg1 in (-1, 1) for sg2 in (-1, 1)]
        + [(sg1, sg2 * phi_g, 0.) for sg1 in (-1, 1) for sg2 in (-1, 1)]
        + [(sg2 * phi_g, 0., sg1) for sg1 in (-1, 1) for sg2 in (-1, 1)])
    cubocta = np.array(
        [(sg1, sg2, 0.) for sg1 in (-1, 1) for sg2 in (-1, 1)]
        + [(sg1, 0., sg2) for sg1 in (-1, 1) for sg2 in (-1, 1)]
        + [(0., sg1, sg2) for sg1 in (-1, 1) for sg2 in (-1, 1)])
    dodeca12 = np.array(
        [(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        + [(0., sg1 / phi_g, sg2 * phi_g)
           for sg1 in (-1, 1) for sg2 in (-1, 1)])
    families = [icosa, cubocta, dodeca12]
    axes = [np.array([1., 0., 0.]),
            np.array([0., 1., phi_g]),
            np.array([1., 1., 1.])]
    axes = [a / np.linalg.norm(a) for a in axes]

    def con_vals_loc(P):
        D2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
        return D2[iu]

    def con_jac_loc(P):
        J = np.zeros((91, 42))
        for k, (i, j) in enumerate(zip(*iu)):
            diff = 2.0 * (P[i] - P[j])
            J[k, 3 * i:3 * i + 3] = diff
            J[k, 3 * j:3 * j + 3] = -diff
        return J

    def slsqp_refine(P0, maxiter=600):
        """Exact max-min SLSQP: maximize t s.t. d_ij^2 >= t, d_ij^2 <= 1."""
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
                {"type": "ineq",
                 "fun": lambda z: con_vals_loc(z[:42].reshape(n, d)) - z[42],
                 "jac": lambda z: np.hstack(
                     [con_jac_loc(z[:42].reshape(n, d)),
                      -np.ones((91, 1))])},
                {"type": "ineq",
                 "fun": lambda z: 1.0 - con_vals_loc(z[:42].reshape(n, d)),
                 "jac": lambda z: np.hstack(
                     [-con_jac_loc(z[:42].reshape(n, d)),
                      np.zeros((91, 1))])},
            ],
            options={"ftol": 1e-12, "maxiter": maxiter},
        )
        if not np.all(np.isfinite(res.x)):
            return None
        return res.x[:42].reshape(n, d)

    seeds = []
    for V in families:
        Vn = V / np.sqrt((V ** 2).sum(1)).max()
        for ax in axes:
            for h in (0.35, 0.6):
                seeds.append(np.vstack([Vn, h * ax, -h * ax]))

    for P0 in seeds:
        try:
            P = slsqp_refine(P0)
            if P is not None:
                r = ratio(P)
                if r > best_r:
                    best_r, best_P = r, P
        except Exception:
            continue

    # Perturbed copies of the best polyhedral seed explore nearby basins
    base = np.array(best_P, dtype=float, copy=True)
    for _ in range(6):
        try:
            P = slsqp_refine(base + 0.05 * rng.randn(n, d))
            if P is not None:
                r = ratio(P)
                if r > best_r:
                    best_r, best_P = r, P
        except Exception:
            continue

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

    # Bounded multistart SLSQP: incumbent plus 12 deterministic perturbations
    # with a larger step (0.05) so the exact KKT solve can escape the basin
    # of the soft-surrogate optimum, each hard-limited to 800 iterations;
    # exact-ratio incumbent selection.
    starts = [np.array(best_P, dtype=float, copy=True)]
    for _ in range(12):
        starts.append(best_P + 0.05 * rng.randn(n, d))

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

    # Interleaved re-polish: one sharper soft-surrogate pass from the SLSQP
    # incumbent, then two more exact SLSQP restarts from perturbations of it.
    res = minimize(soft_obj, best_P.ravel(), args=(120.0,), method="L-BFGS-B",
                   options={"maxiter": 3000})
    if np.all(np.isfinite(res.x)):
        P = res.x.reshape(n, d)
        r = ratio(P)
        if r > best_r:
            best_r, best_P = r, P
    for _ in range(2):
        P0 = best_P + 0.03 * rng.randn(n, d)
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
                r = ratio(P)
                if r > best_r:
                    best_r, best_P = r, P
        except Exception:
            continue

    return best_P


# EVOLVE-BLOCK-END
