# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution, NonlinearConstraint
from scipy.spatial.distance import pdist


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Trilayer orbit family search (D3h decomposition): 2 antipodal poles +
    two independent staggered triangles + one hexagon (10 parameters).

    Approach:
    - Family P(h, z_t1, r_t1, a_t1, z_t2, r_t2, a_t2, z_m, r_m, a_m):
      poles at (0,0,+/-h); triangle 1 at height z_t1, radius r_t1, angles
      2*pi*k/3 + a_t1; triangle 2 at height z_t2, radius r_t2, angles
      2*pi*k/3 + a_t2; hexagon at z_m, r_m, angles 2*pi*k/6 + a_m.
      14 = 2 + 3 + 3 + 6; split triangle heights/radii/twists give a
      contact topology unreachable by any two-layer D6h arrangement.
    - Maximize the exact dmin/dmax ratio over this family with
      differential_evolution (popsize 30, maxiter 600, seed 7).
    - Polish the best family candidates in the full 42-dim point space
      with a trust-constr epigraph NLP to escape the family boundary.
    - Keep a safe fallback; return the best valid incumbent.
    """
    n = 14
    ang3 = 2.0 * np.pi * np.arange(3) / 3
    ang6 = 2.0 * np.pi * np.arange(6) / 6
    c3, s3 = np.cos(ang3), np.sin(ang3)
    c6, s6 = np.cos(ang6), np.sin(ang6)

    def build(theta):
        h, z_t1, r_t1, a_t1, z_t2, r_t2, a_t2, z_m, r_m, a_m = theta
        P = np.empty((n, 3))
        P[0] = (0.0, 0.0, h)
        P[1] = (0.0, 0.0, -h)
        c1, s1 = np.cos(a_t1), np.sin(a_t1)
        c2, s2 = np.cos(a_t2), np.sin(a_t2)
        cm, sm = np.cos(a_m), np.sin(a_m)
        P[2:5, 0] = r_t1 * (c3 * c1 - s3 * s1)
        P[2:5, 1] = r_t1 * (c3 * s1 + s3 * c1)
        P[2:5, 2] = z_t1
        P[5:8, 0] = r_t2 * (c3 * c2 - s3 * s2)
        P[5:8, 1] = r_t2 * (c3 * s2 + s3 * c2)
        P[5:8, 2] = z_t2
        P[8:14, 0] = r_m * (c6 * cm - s6 * sm)
        P[8:14, 1] = r_m * (c6 * sm + s6 * cm)
        P[8:14, 2] = z_m
        return P

    def exact_ratio(P):
        d = pdist(P)
        return d.min() / d.max()

    def family_obj(theta):
        return -exact_ratio(build(theta))

    # --- Stage 1: differential evolution on the 10-parameter trilayer family ---
    bounds = [(0.1, 2.0),
              (-1.5, 1.5), (0.05, 2.0), (0.0, 2.0 * np.pi / 3),
              (-1.5, 1.5), (0.05, 2.0), (0.0, 2.0 * np.pi / 3),
              (-1.5, 1.5), (0.05, 2.0), (0.0, np.pi / 6)]
    de = differential_evolution(family_obj, bounds, popsize=30, maxiter=600,
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

    # --- Stage 2: exact epigraph NLP polish (mechanism: epigraph-constrained) ---
    # Variables z = (P.ravel(), t), 43 dims. Maximize t s.t.
    #   d^2_k >= t for all 91 pairs, d^2_k <= 1 for all pairs (fixes scale).
    # This optimizes the exact ratio (no finite-beta surrogate bias), with
    # analytic constraint Jacobians via trust-constr. Starts: family optimum,
    # near-optimal family points, and seeded random sphere configurations.
    iu = np.triu_indices(n, 1)

    def pair_sq_and_jac(Q):
        """Squared distances (91,) and Jacobian wrt flattened Q (91, 42)."""
        ii, jj = iu
        diff = Q[ii] - Q[jj]                     # (91, 3)
        dsq = (diff ** 2).sum(-1)
        J = np.zeros((91, 42))
        rows = np.repeat(np.arange(91), 3)
        cols_i = (ii[:, None] * 3 + np.arange(3)).ravel()
        cols_j = (jj[:, None] * 3 + np.arange(3)).ravel()
        vals = 2.0 * diff.ravel()
        J[rows, cols_i] = vals
        J[rows, cols_j] = -vals
        return dsq, J

    def eg_obj(z):
        return -z[-1]

    def eg_grad(z):
        g = np.zeros_like(z)
        g[-1] = -1.0
        return g

    def rescale(P):
        d = pdist(P)
        return P / d.max()

    def solve_epigraph(x0):
        P0 = rescale(x0.reshape(n, 3))
        dsq0, _ = pair_sq_and_jac(P0)
        t0 = dsq0.min() - 1e-4
        z = np.concatenate([P0.ravel(), [t0]])
        lower = NonlinearConstraint(
            lambda z: pair_sq_and_jac(z[:42].reshape(n, 3))[0] - z[42],
            0.0, np.inf,
            jac=lambda z: np.hstack([pair_sq_and_jac(
                z[:42].reshape(n, 3))[1],
                -np.ones((91, 1))]))
        upper = NonlinearConstraint(
            lambda z: 1.0 - pair_sq_and_jac(z[:42].reshape(n, 3))[0],
            0.0, np.inf,
            jac=lambda z: np.hstack([-pair_sq_and_jac(
                z[:42].reshape(n, 3))[1],
                np.zeros((91, 1))]))
        res = minimize(eg_obj, z, jac=eg_grad, method="trust-constr",
                       constraints=[lower, upper],
                       options={"maxiter": 800, "gtol": 1e-13,
                                "xtol": 1e-13})
        return res.x[:42].reshape(n, 3)

    rng = np.random.default_rng(0)
    theta_starts = [best_theta] + [th for th, _ in distinct[:4]]
    starts = [build(th) for th in theta_starts]
    for _ in range(15):
        Q = rng.normal(size=(n, 3))
        Q /= np.linalg.norm(Q, axis=1, keepdims=True)
        starts.append(Q)

    best_P = build(best_theta)
    best_r = exact_ratio(best_P)
    cands = []
    for P0 in starts:
        try:
            P = solve_epigraph(P0)
            if not np.all(np.isfinite(P)):
                continue
            r = exact_ratio(P)
            cands.append(P.copy())
            if r > best_r:
                best_r, best_P = r, P.copy()
        except Exception:
            continue

    # --- Stage 2b: feasibility-bisection polish (mechanism: feasibility-
    # bisection). For a fixed target t, solve the pure feasibility problem
    # "d^2_ij >= t for all 91 pairs, d^2_ij <= 1 for all pairs" with SLSQP
    # over the 42 point coordinates only (no epigraph variable), then
    # bisect t within [t_inc - 0.01, t_inc + 0.02] around the incumbent's
    # squared dmin (after rescaling dmax^2 = 1). Applied to the top 3
    # trust-constr incumbents; <= 6 bisection rounds per candidate, each
    # SLSQP solve capped at 300 iterations. Every accepted iterate is
    # checked with the exact pdist ratio, so a failed bisection is a no-op.
    def feasibility_bisect(P0, max_rounds=6):
        """Bisect the min-distance target t with SLSQP feasibility solves."""
        P0 = rescale(np.asarray(P0, dtype=np.float64).copy())
        r_best = exact_ratio(P0)
        P_best = P0.copy()
        dsq0, _ = pair_sq_and_jac(P0)
        lo = max(dsq0.min() - 0.01, 0.0)
        hi = dsq0.min() + 0.02
        t_hold = [lo]

        def cons_fun(z):
            dsq, _ = pair_sq_and_jac(z.reshape(n, 3))
            return np.concatenate([dsq - t_hold[0], 1.0 - dsq])

        def cons_jac(z):
            _, J = pair_sq_and_jac(z.reshape(n, 3))
            return np.vstack([J, -J])

        for _ in range(max_rounds):
            t_hold[0] = 0.5 * (lo + hi)
            z0 = P_best.ravel().copy()
            res = minimize(lambda z: 0.0, z0,
                           jac=lambda z: np.zeros(42, dtype=np.float64),
                           method="SLSQP",
                           constraints=[{"type": "ineq",
                                         "fun": cons_fun,
                                         "jac": cons_jac}],
                           options={"maxiter": 300, "ftol": 1e-12})
            Q = np.asarray(res.x, dtype=np.float64).reshape(n, 3)
            if not np.all(np.isfinite(Q)):
                hi = t_hold[0]
                continue
            if cons_fun(res.x).min() >= -1e-9:
                lo = t_hold[0]
                r = exact_ratio(Q)
                if r > r_best:
                    r_best, P_best = r, Q.copy()
            else:
                hi = t_hold[0]
        return P_best

    try:
        cands = sorted(cands, key=lambda p: -exact_ratio(p))[:3]
        for Pc in cands:
            Q = feasibility_bisect(Pc)
            if np.all(np.isfinite(Q)):
                r = exact_ratio(Q)
                if r > best_r:
                    best_r, best_P = r, Q.copy()
    except Exception:
        pass

    # Active-set SLSQP refinement: solve the epigraph problem restricted to
    # the incumbent's active contact graph (pairs at dmin and dmax, plus a
    # slack band), where SQP converges quadratically to ~1e-12. Grow the
    # active set if an excluded pair becomes violated (<= 5 rounds, <= 20
    # SLSQP solves, each capped at 400 iterations). Accept only if the
    # exact pdist ratio improves.
    def active_set_refine(P0, max_rounds=5, max_solves=20):
        P = np.asarray(P0, dtype=np.float64).copy()
        best_local = P.copy()
        best_r_loc = exact_ratio(P)
        solves = 0
        for _ in range(max_rounds):
            if solves >= max_solves:
                break
            d = pdist(P)
            dmin, dmax = d.min(), d.max()
            Amin = np.where(d <= dmin + 1e-6)[0]
            Amax = np.where(d >= dmax - 1e-6)[0]
            slack = np.where((d > dmin + 1e-6) & (d < dmin + 0.02))[0]
            A = np.unique(np.concatenate([Amin, slack]))

            def obj(z):
                return -z[-1]

            def grad(z):
                g = np.zeros_like(z)
                g[-1] = -1.0
                return g

            def cons_min(z):
                dsq, _ = pair_sq_and_jac(z[:42].reshape(n, 3))
                return np.concatenate([dsq[A] - z[42],
                                       1.0 - dsq[Amax]])

            def cons_jac(z):
                _, J = pair_sq_and_jac(z[:42].reshape(n, 3))
                Jm = np.hstack([J[A], -np.ones((A.size, 1))])
                Jx = np.hstack([-J[Amax], np.zeros((Amax.size, 1))])
                return np.vstack([Jm, Jx])

            P0r = P / pdist(P).max()
            dsq0, _ = pair_sq_and_jac(P0r)
            z0 = np.concatenate([P0r.ravel(),
                                 [dsq0.min() - 1e-5]])
            res = minimize(obj, z0, jac=obj, method="SLSQP",
                           constraints=[{"type": "ineq",
                                         "fun": cons_min,
                                         "jac": cons_jac}],
                           options={"maxiter": 400, "ftol": 1e-14})
            solves += 1
            if not np.all(np.isfinite(res.x)):
                break
            Q = res.x[:42].reshape(n, 3)
            if not np.all(np.isfinite(Q)):
                break
            r = exact_ratio(Q)
            if r > best_r_loc:
                best_r_loc, best_local = r, Q.copy()
            # Grow active set with any pair now below the target.
            dsq_new, _ = pair_sq_and_jac(Q)
            viol = np.where(dsq_new < res.x[42] - 1e-9)[0]
            new = [k for k in viol if k not in set(A.tolist())]
            if not new:
                break
            A = np.unique(np.concatenate([A, np.asarray(new, dtype=int)]))
            P = Q
        return best_local

    try:
        Q = active_set_refine(best_P)
        if np.all(np.isfinite(Q)):
            r = exact_ratio(Q)
            if r > best_r:
                best_r, best_P = r, Q.copy()
    except Exception:
        pass

    # --- Fallback / validation ---
    best_P = np.asarray(best_P, dtype=np.float64)
    if (not np.all(np.isfinite(best_P)) or best_P.shape != (n, 3)
            or pdist(best_P).max() <= 0.0):
        rng = np.random.default_rng(42)
        best_P = rng.normal(size=(n, 3))
        best_P /= np.linalg.norm(best_P, axis=1, keepdims=True)
    return best_P


# EVOLVE-BLOCK-END
