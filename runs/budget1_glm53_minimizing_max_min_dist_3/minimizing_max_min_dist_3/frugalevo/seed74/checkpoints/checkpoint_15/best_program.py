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
    # (0, 1/phi, phi) is the dodecahedron 5-fold axis (through opposite
    # dodecahedron vertices), matching the dodeca12 family far better
    # than the generic 3-fold (1,1,1) direction.
    axes = [np.array([1., 0., 0.]),
            np.array([0., 1., phi_g]),
            np.array([0., 1.0 / phi_g, phi_g])]
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
            # Small deterministic perturbation breaks the exact polyhedral
            # symmetry so SLSQP is not pinned to the symmetric KKT point
            # with a degenerate active set; same start count, new basins.
            P = slsqp_refine(P0 + 0.02 * rng.randn(n, d))
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

    # Annealed repulsion dynamics: minimize E_p = sum_{i<j} d_ij^{-p} with
    # exact renormalization dmax <- 1 between annealing stages. Unlike the
    # fixed-beta soft-min/soft-max surrogate (whose optimum lies strictly
    # interior to the true max-min optimum), this objective is unbiased:
    # dmax is pinned exactly at 1 and only dmin is effectively maximized.
    # Annealing p from 2 to 20 is a smooth continuation that can migrate
    # points across near-degenerate arrangements. Each p-stage is solved
    # properly with L-BFGS-B and an analytic gradient (not a fixed-step
    # Euler loop), so the dynamics actually converge at each temperature.
    def repulse(P0, ps=(2, 3, 4, 6, 8, 12, 16, 20, 28, 40), steps=300):
        P = np.array(P0, dtype=float, copy=True)

        def energy(x, p):
            Pp = x.reshape(n, d)
            diff = Pp[:, None, :] - Pp[None, :, :]
            d2 = (diff ** 2).sum(-1)
            d2v = d2[iu]
            d2v = np.maximum(d2v, 1e-24)
            return (d2v ** (-p / 2.0)).sum()

        def egrad(x, p):
            Pp = x.reshape(n, d)
            diff = Pp[:, None, :] - Pp[None, :, :]
            d2 = (diff ** 2).sum(-1)
            np.fill_diagonal(d2, np.inf)
            w = np.where(np.isfinite(d2), d2 ** (-(p / 2.0 + 1.0)), 0.0)
            g = -p * (w[:, :, None] * diff).sum(1)
            return g.ravel()

        mid_done = False
        for p in ps:
            x0 = P.ravel()
            res = minimize(energy, x0, args=(p,), jac=egrad,
                           method="L-BFGS-B",
                           options={"maxiter": steps, "ftol": 1e-16,
                                    "gtol": 1e-14})
            if np.all(np.isfinite(res.x)):
                P = res.x.reshape(n, d)
            # Exact renormalization: pin dmax = 1 after each stage.
            D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
            m = D[iu].max()
            if np.isfinite(m) and m > 0:
                P = P / m
            # Adaptive mid-schedule re-centering: once the annealing passes
            # the sharpness where the active set is essentially fixed (p=8),
            # snap back to an exact max-min KKT point so the remaining
            # high-p stages polish the true objective instead of drifting
            # along the repulsion field. Done once per trajectory to keep
            # the compute contract.
            if (not mid_done) and p >= 8:
                mid_done = True
                try:
                    Pm = slsqp_finish(P, maxiter=400)
                    if Pm is not None and np.all(np.isfinite(Pm)):
                        P = Pm
                except Exception:
                    pass
        return P

    # Annealed-repulsion multistart: incumbent plus 5 deterministic
    # perturbations (6 starts total). Each repulsion trajectory is followed
    # by an immediate exact max-min SLSQP finish so the unbiased repulsion
    # basin is always converted to a true max-min point before incumbent
    # selection (repulsion output alone is not KKT for the exact problem).
    def slsqp_finish(P0, maxiter=800):
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
            options={"ftol": 1e-12, "maxiter": maxiter},
        )
        if np.all(np.isfinite(res.x)):
            return res.x[:42].reshape(n, d)
        return None

    starts = [np.array(best_P, dtype=float, copy=True)]
    for _ in range(5):
        starts.append(best_P + 0.03 * rng.randn(n, d))

    for P0 in starts:
        try:
            P = repulse(P0)
            if np.all(np.isfinite(P)):
                Pf = slsqp_finish(P)
                if Pf is not None:
                    P = Pf
                r = ratio(P)
                if r > best_r:
                    best_r, best_P = r, P
        except Exception:
            continue

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
    try:
        P = repulse(np.array(best_P, dtype=float, copy=True))
        if np.all(np.isfinite(P)):
            Pf = slsqp_finish(P)
            if Pf is not None:
                P = Pf
            r = ratio(P)
            if r > best_r:
                best_r, best_P = r, P
    except Exception:
        pass
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

    # Final alternating polish: sharper soft surrogate then exact SLSQP,
    # repeated until no improvement (max 4 rounds). At a near-optimal point
    # the active set is nearly fixed, so alternating removes the surrogate
    # bias and then re-sharpens the min/max estimate, closing the last
    # fraction of a percent of the ratio.
    for _ in range(4):
        improved = False
        try:
            P = repulse(np.array(best_P, dtype=float, copy=True))
            if np.all(np.isfinite(P)):
                Pf = slsqp_finish(P)
                if Pf is not None:
                    P = Pf
                r = ratio(P)
                if r > best_r:
                    best_r, best_P = r, P
                    improved = True
        except Exception:
            pass
        try:
            Q0 = np.array(best_P, dtype=float, copy=True)
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
                options={"ftol": 1e-14, "maxiter": 800},
            )
            if np.all(np.isfinite(res.x)):
                P = res.x[:42].reshape(n, d)
                r = ratio(P)
                if r > best_r:
                    best_r, best_P = r, P
                    improved = True
        except Exception:
            pass
        if not improved:
            break

    return best_P


# EVOLVE-BLOCK-END
