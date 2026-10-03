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
    for P0 in starts:
        try:
            P = solve_epigraph(P0)
            if not np.all(np.isfinite(P)):
                continue
            r = exact_ratio(P)
            if r > best_r:
                best_r, best_P = r, P.copy()
        except Exception:
            continue

    # Jitter re-polish: the trilayer family optimum often sits on a KKT
    # boundary of the epigraph NLP; small seeded perturbations of the
    # incumbent, each re-solved exactly, escape the stalled basin without
    # changing the mechanism. Three scales x four restarts, hard-bounded.
    rng2 = np.random.default_rng(11)
    for scale in (1e-3, 3e-4, 1e-4):
        for _ in range(4):
            if best_r < 1e-12:
                break
            Qp = best_P.copy()
            Qp += scale * rng2.normal(size=Qp.shape)
            try:
                P = solve_epigraph(Qp.ravel())
                if np.all(np.isfinite(P)):
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
