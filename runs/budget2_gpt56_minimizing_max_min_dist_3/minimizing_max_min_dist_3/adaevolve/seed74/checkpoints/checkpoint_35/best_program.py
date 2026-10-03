# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, root


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

    def kkt_contact_polish(points):
        """Refine the detected min/max contact graph by solving its KKT system."""
        # Put the incumbent in the exact diameter-one gauge before identifying
        # contacts.  This prevents small SLSQP feasibility residuals from
        # changing which pairs are classified as diameter contacts.
        q = points - points[0]
        d2 = np.sum((q[ii] - q[jj]) ** 2, axis=1)
        diameter2 = np.max(d2)
        if not np.isfinite(diameter2) or diameter2 <= 0.0:
            return None
        q /= np.sqrt(diameter2)
        d2 = np.sum((q[ii] - q[jj]) ** 2, axis=1)
        u = np.min(d2)

        # A contact tolerance is deliberately much tighter than the spacing
        # between the two distance levels.  The final root solve subsequently
        # imposes these selected equalities exactly.
        tolerance = 3e-5
        lower = np.flatnonzero(d2 <= u + tolerance)
        upper = np.flatnonzero(d2 >= 1.0 - tolerance)
        active_pairs = np.r_[lower, upper]
        signs = np.r_[np.ones(len(lower)), -np.ones(len(upper))]
        if len(lower) == 0 or len(upper) == 0:
            return None

        # Unknowns are the 39 non-translation coordinates, u, and one KKT
        # multiplier per active contact.  The equations are the contact
        # equalities plus coordinate and u stationarity.  Rotation remains a
        # harmless null gauge, for which Levenberg--Marquardt is robust.
        contact_count = len(active_pairs)

        def kkt_equations(x):
            r = np.zeros((n, 3))
            r[1:] = x[:3 * (n - 1)].reshape(n - 1, 3)
            v = x[3 * (n - 1)]
            multipliers = x[3 * (n - 1) + 1:]
            contact_residual = np.empty(contact_count)
            stationarity = np.zeros(3 * (n - 1))
            lower_sum = 0.0

            for k, pair in enumerate(active_pairs):
                a, b = ii[pair], jj[pair]
                delta = r[a] - r[b]
                squared = np.dot(delta, delta)
                if k < len(lower):
                    contact_residual[k] = squared - v
                    lower_sum += multipliers[k]
                    factor = 2.0 * multipliers[k]
                else:
                    contact_residual[k] = 1.0 - squared
                    factor = -2.0 * multipliers[k]

                if a:
                    stationarity[3 * (a - 1):3 * a] += factor * delta
                if b:
                    stationarity[3 * (b - 1):3 * b] -= factor * delta

            # For L = u + sum(lambda_l*(d^2-u)) +
            # sum(lambda_u*(1-d^2)), stationarity in u is 1-sum(lambda_l).
            return np.r_[contact_residual, stationarity, 1.0 - lower_sum]

        # Estimate multipliers from the incumbent stationarity equations.  A
        # good multiplier initialization substantially improves root's local
        # convergence compared with starting all multipliers at zero.
        g = np.zeros((3 * (n - 1) + 1, contact_count))
        for k, pair in enumerate(active_pairs):
            a, b = ii[pair], jj[pair]
            delta = q[a] - q[b]
            factor = 2.0 * signs[k]
            if a:
                g[3 * (a - 1):3 * a, k] += factor * delta
            if b:
                g[3 * (b - 1):3 * b, k] -= factor * delta
            if k < len(lower):
                g[-1, k] = 1.0
        multipliers = np.linalg.lstsq(
            g, np.r_[np.zeros(3 * (n - 1)), 1.0], rcond=None
        )[0]
        start = np.r_[q[1:].ravel(), u, multipliers]

        solved = root(
            kkt_equations, start, method="lm",
            options={"ftol": 1e-13, "xtol": 1e-13, "gtol": 1e-13,
                     "maxiter": 8000},
        )
        if not np.all(np.isfinite(solved.x)):
            return None

        candidate = np.zeros((n, 3))
        candidate[1:] = solved.x[:3 * (n -1)].reshape(n - 1, 3)
        candidate_d2 = np.sum(
            (candidate[ii] - candidate[jj]) ** 2, axis=1
        )
        candidate_diameter2 = np.max(candidate_d2)
        if candidate_diameter2 <= 0.0 or not np.isfinite(candidate_diameter2):
            return None
        candidate /= np.sqrt(candidate_diameter2)
        candidate_d2 = np.sum(
            (candidate[ii] - candidate[jj]) ** 2, axis=1
        )
        candidate_u = solved.x[3 * (n - 1)] / candidate_diameter2

        # Verify that the graph used by root is still the graph represented by
        # the returned geometry, rather than accepting a different branch of
        # the algebraic equations.
        if (np.max(np.abs(candidate_d2[lower] - candidate_u)) > 2e-6 or
                np.max(np.abs(candidate_d2[upper] - 1.0)) > 2e-6 or
                np.min(candidate_d2) <= 0.0):
            return None
        return candidate

    # SLSQP supplies the basin and active graph; root removes residual error
    # from the resulting fixed-contact KKT equations.
    refined = kkt_contact_polish(incumbent_points)
    if refined is not None:
        refined_ratio = ratio(refined)
        if np.isfinite(refined_ratio) and refined_ratio > incumbent_ratio:
            incumbent_points = refined
            incumbent_ratio = refined_ratio

    if np.all(np.isfinite(incumbent_points)) and incumbent_ratio > best_ratio:
        return incumbent_points
    return best


# EVOLVE-BLOCK-END
