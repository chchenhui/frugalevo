# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def min_max_dist_dim3_14() -> np.ndarray:
    """Generate a spherical-repulsion seed and maximize the minimum distance
    under diameter-one pairwise constraints with deterministic basin search."""
    n = 14
    rng = np.random.default_rng(20260912)
    ii, jj = np.triu_indices(n, 1)

    def ratio(p):
        delta = p[ii] - p[jj]
        distances = np.sqrt(np.sum(delta * delta, axis=1))
        return np.min(distances) / np.max(distances)

    # First obtain a well-separated spherical code.  The sphere is only used
    # to create high quality starting points; the final optimization is free
    # to move points anywhere in R^3.
    best = None
    best_ratio = -1.0
    for _ in range(8):
        p = rng.normal(size=(n, 3))
        p /= np.linalg.norm(p, axis=1)[:, None]
        p += 0.015 * rng.normal(size=(n, 3))
        p /= np.linalg.norm(p, axis=1)[:, None]

        for power in (4, 8, 16, 28):
            for it in range(1500):
                delta = p[:, None, :] - p[None, :, :]
                d2 = np.sum(delta * delta, axis=2)
                np.fill_diagonal(d2, 1.0)

                weights = d2 ** (-(power + 2.0) / 2.0)
                np.fill_diagonal(weights, 0.0)
                force = np.sum(weights[:, :, None] * delta, axis=1)

                # Keep the repulsion update tangent to the unit sphere.
                force -= np.sum(force * p, axis=1)[:, None] * p
                maximum_force = np.max(np.linalg.norm(force, axis=1))
                if maximum_force > 0.0:
                    step = 0.055 * (1.0 - 0.82 * it / 1499.0)
                    p += step * force / maximum_force
                    p /= np.linalg.norm(p, axis=1)[:, None]

        value = ratio(p)
        if value > best_ratio:
            best = p.copy()
            best_ratio = value

    # Scale diameter to one.  Translation is removed by fixing point zero.
    distances = np.sqrt(np.sum((best[ii] - best[jj]) ** 2, axis=1))
    p0 = (best - best[0]) / np.max(distances)

    # The evaluator scores the squared ratio.  With diameter fixed to one,
    # maximize u = (minimum distance)^2 directly rather than introducing
    # t and the less well-conditioned nonlinear term t*t.
    z0 = np.r_[p0[1:].ravel(), ratio(p0) ** 2 * 0.999999]

    def unpack(z):
        p = np.zeros((n, 3))
        p[1:] = z[:-1].reshape(n - 1, 3)
        return p, z[-1]

    # Every squared distance is at least u and at most one.  Thus u is
    # exactly the evaluator's squared min/max ratio after normalization.
    def constraints(z):
        p, u = unpack(z)
        delta = p[ii] - p[jj]
        d2 = np.sum(delta * delta, axis=1)
        return np.r_[d2 - u, 1.0 - d2]

    def constraint_jacobian(z):
        """Return the analytic Jacobian of all lower and upper constraints."""
        p, t = unpack(z)
        delta = p[ii] - p[jj]
        m = len(ii)
        jac = np.zeros((2 * m, 3 * (n - 1) + 1))
        rows = np.arange(m)
        xyz = np.arange(3)

        # Point zero is fixed.  Fill all endpoint coordinate derivatives at
        # once, avoiding a Python loop in this frequently called routine.
        mask_i = ii != 0
        ri = rows[mask_i]
        ci = 3 * (ii[mask_i, None] - 1) + xyz
        jac[ri[:, None], ci] = 2.0 * delta[mask_i]

        mask_j = jj != 0
        rj = rows[mask_j]
        cj = 3 * (jj[mask_j, None] - 1) + xyz
        jac[rj[:, None], cj] = -2.0 * delta[mask_j]

        jac[m:, :-1] = -jac[:m, :-1]
        # d^2 - u has constant derivative -1 with respect to u.
        jac[:m, -1] = -1.0
        return jac

    objective_jacobian = lambda z: np.r_[np.zeros(3 * (n - 1)), -1.0]

    def polish(start, iterations):
        return minimize(
            lambda z: -z[-1],
            start,
            jac=objective_jacobian,
            constraints={
                "type": "ineq",
                "fun": constraints,
                "jac": constraint_jacobian,
            },
            method="SLSQP",
            options={"maxiter": iterations, "ftol": 1e-12, "disp": False},
        )

    result = polish(z0, 2500)
    incumbent = result.x.copy()
    incumbent_points, _ = unpack(incumbent)
    incumbent_ratio = ratio(incumbent_points)

    # Explore a range of contact-graph transitions.  Since the Jacobian is
    # vectorized, these deterministic restarts remain inexpensive.  Each
    # trial is rescaled before SLSQP so every upper-distance constraint starts
    # feasible, while its lower-distance variable is safely feasible too.
    for scale in (0.003, 0.006, 0.012, 0.02, 0.03, 0.045, 0.065, 0.09, 0.12):
        for _ in range(12):
            trial, _ = unpack(incumbent)
            trial[1:] += scale * rng.normal(size=(n - 1, 3))
            trial /= np.max(np.sqrt(np.sum(
                (trial[ii] - trial[jj]) ** 2, axis=1
            )))
            trial_z = np.r_[trial[1:].ravel(), ratio(trial) ** 2 * 0.99999]

            local = polish(trial_z, 1600)
            candidate, _ = unpack(local.x)
            candidate_ratio = ratio(candidate)
            if np.all(np.isfinite(candidate)) and candidate_ratio > incumbent_ratio:
                incumbent = local.x.copy()
                incumbent_points = candidate
                incumbent_ratio = candidate_ratio

    if np.all(np.isfinite(incumbent_points)) and incumbent_ratio > best_ratio:
        return incumbent_points
    return best


# EVOLVE-BLOCK-END
