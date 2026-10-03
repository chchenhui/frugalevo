# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """Build many spherical packing seeds, then maximin-refine them in 3D.

    The final SLSQP stage maximizes the squared minimum distance while all
    pairwise squared distances are constrained to be at most one.
    """
    from scipy.optimize import minimize

    c = np.sqrt(3.0) / 2.0
    h = np.sqrt((2.0 * c - 1.0) / (3.0 + 2.0 * c))
    a = np.arange(6, dtype=float) * (np.pi / 3.0)
    theta0 = np.r_[np.full(6, np.arccos(h)), np.full(6, np.arccos(-h))]
    phi0 = np.r_[a, a + np.pi / 6.0]

    def points(x):
        th, ph = x[:12], x[12:24]
        q = np.column_stack((np.sin(th) * np.cos(ph),
                             np.sin(th) * np.sin(ph),
                             np.cos(th)))
        return np.vstack(([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]], q))

    def distances2(p):
        g = p @ p.T
        return 2.0 - 2.0 * g[np.triu_indices(14, 1)]

    def initial_vector(th, ph):
        p = points(np.r_[th, ph])
        return np.r_[th, ph, distances2(p).min() * 0.995]

    def constraints(x):
        return distances2(points(x)) - x[24]

    rng = np.random.default_rng(714029)
    starts = [initial_vector(theta0, phi0)]
    for scale in (0.18, 0.35, 0.60):
        for _ in range(6):
            th = np.clip(theta0 + rng.normal(0.0, scale, 12),
                         1.0e-4, np.pi - 1.0e-4)
            ph = phi0 + rng.normal(0.0, scale, 12)
            starts.append(initial_vector(th, ph))

    # Include a few genuinely different spherical topologies.  Even when
    # their spherical value is inferior, they can enter a better nonspherical
    # basin after the diameter-constrained refinement below.
    for _ in range(4):
        th = np.arccos(rng.uniform(-0.96, 0.96, 12))
        ph = rng.uniform(-np.pi, np.pi, 12)
        starts.append(initial_vector(th, ph))

    best = points(starts[0])
    best_value = distances2(best).min()
    sphere_seeds = [best]
    bounds = [(1.0e-5, np.pi - 1.0e-5)] * 12 + [(-10.0, 10.0)] * 12 + [(0.0, 4.0)]

    for start in starts:
        result = minimize(
            lambda x: -x[24],
            start,
            method="SLSQP",
            bounds=bounds,
            constraints={"type": "ineq", "fun": constraints},
            options={"maxiter": 1800, "ftol": 1.0e-11, "disp": False},
        )
        candidate = points(result.x)
        value = distances2(candidate).min()
        sphere_seeds.append(candidate)
        if value > best_value:
            best, best_value = candidate, value

    # The spherical restriction is convenient for finding a good initial
    # packing, but it is not part of the actual objective.  Refine in the
    # full diameter-constrained configuration space.  Point zero is fixed at
    # the origin only to remove translation invariance.
    ii, jj = np.triu_indices(14, 1)

    def free_distances2(p):
        q = p[ii] - p[jj]
        return np.einsum("ij,ij->i", q, q)

    def unpack(x):
        return np.vstack((np.zeros(3), x[:39].reshape(13, 3)))

    def pack(p, noise=None):
        p = p.copy()
        p -= p[0]
        if noise is not None:
            p[1:] += noise
        d = free_distances2(p)
        p /= np.sqrt(d.max())
        d = free_distances2(p)
        return np.r_[p[1:].ravel(), d.min()]

    def free_constraints(x):
        d = free_distances2(unpack(x))
        return np.r_[d - x[39], 1.0 - d]

    def free_constraint_jacobian(x):
        p = unpack(x)
        jac = np.zeros((182, 40))
        for k, (i, j) in enumerate(zip(ii, jj)):
            v = 2.0 * (p[i] - p[j])
            if i:
                jac[k, 3 * (i - 1):3 * i] = v
                jac[91 + k, 3 * (i - 1):3 * i] = -v
            if j:
                jac[k, 3 * (j - 1):3 * j] = -v
                jac[91 + k, 3 * (j - 1):3 * j] = v
            jac[k, 39] = -1.0
        return jac

    # All starts are normalized to diameter one, hence are feasible before
    # optimization.  Small perturbations let SLSQP leave the spherical local
    # optimum while the analytic constraint Jacobian keeps this stage cheap.
    free_rng = np.random.default_rng(290317)

    # Do not throw away near-best spherical optima: their contact graphs can
    # lead to distinct local optima once radial motion is permitted.
    seed_pool = sorted(
        sphere_seeds, key=lambda p: distances2(p).min(), reverse=True
    )[:12]
    free_starts = [pack(p) for p in seed_pool]
    for seed in seed_pool[:6]:
        for scale in (0.012, 0.035, 0.080, 0.150):
            free_starts.append(
                pack(seed, free_rng.normal(0.0, scale, (13, 3)))
            )

    best_ratio = best_value / 4.0
    free_bounds = [(-1.05, 1.05)] * 39 + [(0.0, 1.0)]
    free_cons = {"type": "ineq", "fun": free_constraints,
                 "jac": free_constraint_jacobian}

    def refine_free(start):
        return minimize(
            lambda x: -x[39],
            start,
            jac=lambda x: np.r_[np.zeros(39), -1.0],
            method="SLSQP",
            bounds=free_bounds,
            constraints=free_cons,
            options={"maxiter": 2500, "ftol": 2.0e-12, "disp": False},
        )

    for start in free_starts:
        result = refine_free(start)
        candidate = unpack(result.x)
        d = free_distances2(candidate)
        ratio = d.min() / d.max()
        if ratio > best_ratio:
            best, best_ratio = candidate, ratio

    # A sequential shake-and-polish pass is important: the best free-space
    # configuration is generally not in the same basin as its spherical seed.
    for scale in (0.008, 0.020, 0.050, 0.100):
        for _ in range(3):
            result = refine_free(
                pack(best, free_rng.normal(0.0, scale, (13, 3)))
            )
            candidate = unpack(result.x)
            d = free_distances2(candidate)
            ratio = d.min() / d.max()
            if ratio > best_ratio:
                best, best_ratio = candidate, ratio

    return best


# EVOLVE-BLOCK-END
