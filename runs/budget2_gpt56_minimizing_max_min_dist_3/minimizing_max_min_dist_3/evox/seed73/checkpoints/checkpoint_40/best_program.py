# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """Construct 14 points by deterministic multistart annealing, then solve
    a centered unit-minimum-separation diameter minimization problem with
    sequential quadratic programming."""
    n = 14
    rng = np.random.default_rng(918273)

    def normalize_rows(x):
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    def distances(x):
        z = x[:, None, :] - x[None, :, :]
        return np.sum(z * z, axis=2)

    def value_after_move(d, i, newd):
        keep = np.arange(n) != i
        fixed = d[np.ix_(keep, keep)]
        offdiag = ~np.eye(n - 1, dtype=bool)
        return min(np.min(fixed[offdiag]), np.min(newd)) / max(
            np.max(fixed), np.max(newd)
        )

    best_points = None
    best_value = -np.inf

    # This deterministic budget was empirically stronger than spreading the
    # same seed sequence across many shallow basins.
    for restart in range(14):
        points = normalize_rows(rng.normal(size=(n, 3)))

        # A sphere is useful for generating well-separated starts, but is
        # not imposed during the actual Euclidean diameter optimization.
        for it in range(700):
            delta = points[:, None, :] - points[None, :, :]
            q = np.sum(delta * delta, axis=2) + np.eye(n)
            force = np.sum(delta * q[:, :, None] ** -3.0, axis=1)
            force -= np.sum(force * points, axis=1, keepdims=True) * points
            points = normalize_rows(
                points + 0.011 * force / (1.0 + 0.0025 * it)
            )

        d = distances(points)
        current = value_after_move(d, 0, d[0, 1:])

        # Radial freedom permits nonspherical diameter packings.  Scoring
        # only recomputes the thirteen distances incident to one point.
        # A slower cooling schedule gives each independently generated
        # packing enough opportunity to leave its initial spherical basin.
        total = 76000
        for it in range(total):
            i = rng.integers(n)
            frac = it / total
            scale = 0.105 * (1.0 - frac) ** 1.55 + 0.0010
            candidate_point = points[i] + scale * rng.normal(size=3)
            keep = np.arange(n) != i
            newd = np.sum((points[keep] - candidate_point) ** 2, axis=1)
            candidate = value_after_move(d, i, newd)
            temperature = 0.0032 * (1.0 - frac) ** 2 + 0.000008

            if candidate >= current or rng.random() < np.exp(
                (candidate - current) / temperature
            ):
                points[i] = candidate_point
                d[i, keep] = newd
                d[keep, i] = newd
                current = candidate

            # Translation is irrelevant, but recentering prevents random
            # walk of the coordinate origin during unrestricted moves.
            if it % 4000 == 3999:
                points -= np.mean(points, axis=0)

        # Small greedy moves resolve the nearly active contact graph.
        for it in range(18000):
            i = rng.integers(n)
            scale = 0.008 * (1.0 - it / 18000.0) + 0.00003
            candidate_point = points[i] + scale * rng.normal(size=3)
            keep = np.arange(n) != i
            newd = np.sum((points[keep] - candidate_point) ** 2, axis=1)
            candidate = value_after_move(d, i, newd)

            if candidate > current:
                points[i] = candidate_point
                d[i, keep] = newd
                d[keep, i] = newd
                current = candidate

        if current > best_value:
            best_value = current
            best_points = points.copy()

    # Scale so the smallest squared separation is one, then directly
    # minimize the largest squared separation.  This is equivalent to the
    # evaluator's squared min-distance/diameter objective, but avoids its
    # nonsmooth quotient during the final polish.
    best_points -= np.mean(best_points, axis=0)
    upper = np.triu_indices(n, 1)
    final_d = distances(best_points)
    best_points /= np.sqrt(np.min(final_d[upper]))

    try:
        from scipy.optimize import minimize

        initial_d = distances(best_points)[upper]
        x0 = np.concatenate(
            (best_points.ravel(), [float(np.max(initial_d))])
        )

        def objective(x):
            """Return the diameter-squared auxiliary variable."""
            return x[-1]

        def objective_jacobian(x):
            """Return the constant objective gradient."""
            g = np.zeros_like(x)
            g[-1] = 1.0
            return g

        def constraints(x):
            """Keep every squared pair distance between one and t."""
            p = x[:-1].reshape(n, 3)
            delta = p[upper[0]] - p[upper[1]]
            d2 = np.einsum("ij,ij->i", delta, delta)
            return np.concatenate((d2 - 1.0, x[-1] - d2, [x[-1]]))

        def constraint_jacobian(x):
            """Return analytic derivatives of all distance constraints."""
            p = x[:-1].reshape(n, 3)
            delta = p[upper[0]] - p[upper[1]]
            m = len(upper[0])
            jac = np.zeros((2 * m + 1, 3 * n + 1))
            rows = np.arange(m)
            for axis in range(3):
                jac[rows, 3 * upper[0] + axis] = 2.0 * delta[:, axis]
                jac[rows, 3 * upper[1] + axis] = -2.0 * delta[:, axis]
            jac[m:2 * m, :-1] = -jac[:m, :-1]
            jac[m:2 * m, -1] = 1.0
            jac[-1, -1] = 1.0
            return jac

        def centered(x):
            """Fix the translational null modes by keeping the centroid zero."""
            return np.mean(x[:-1].reshape(n, 3), axis=0)

        def centered_jacobian(x):
            """Return the constant Jacobian of the centroid equations."""
            jac = np.zeros((3, 3 * n + 1))
            for axis in range(3):
                jac[axis, axis:3 * n:3] = 1.0 / n
            return jac

        solver_constraints = (
            {
                "type": "ineq",
                "fun": constraints,
                "jac": constraint_jacobian,
            },
            {
                "type": "eq",
                "fun": centered,
                "jac": centered_jacobian,
            },
        )

        polished = minimize(
            objective,
            x0,
            jac=objective_jacobian,
            constraints=solver_constraints,
            method="SLSQP",
            options={"maxiter": 2200, "ftol": 2e-14, "disp": False},
        )

        # Restart from the first active-set solution.  The first solve usually
        # identifies the contact graph, while the second can reduce small
        # residual differences among simultaneous diameter contacts.
        if np.all(np.isfinite(polished.x)):
            polished = minimize(
                objective,
                polished.x,
                jac=objective_jacobian,
                constraints=solver_constraints,
                method="SLSQP",
                options={"maxiter": 1200, "ftol": 1e-14, "disp": False},
            )

        candidate = polished.x[:-1].reshape(n, 3)
        candidate -= np.mean(candidate, axis=0)
        candidate_d = distances(candidate)
        candidate_ratio = (
            np.min(candidate_d[upper]) / np.max(candidate_d[upper])
        )
        original_d = distances(best_points)
        original_ratio = (
            np.min(original_d[upper]) / np.max(original_d[upper])
        )
        if np.isfinite(candidate_ratio) and candidate_ratio > original_ratio:
            best_points = candidate
    except Exception:
        # The deterministic annealed packing remains a valid fallback.
        pass

    best_points -= np.mean(best_points, axis=0)
    return np.asarray(best_points, dtype=float)


# EVOLVE-BLOCK-END
