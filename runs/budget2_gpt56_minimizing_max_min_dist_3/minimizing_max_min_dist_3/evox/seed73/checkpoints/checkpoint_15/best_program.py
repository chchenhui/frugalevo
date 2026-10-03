# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """Generate many repulsive spherical starts, then use unrestricted
    incremental annealing and long greedy coordinate polishing to maximize
    the squared minimum-distance to diameter ratio for exactly 14 points."""
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

    # The objective has many nearly degenerate local optima.  Independent
    # basins are especially valuable because only the best realization is
    # returned and evaluator time does not affect combined_score.
    # A smaller set of independent basins leaves time for a deterministic
    # constrained polish of the strongest stochastic realization.
    # More independent basins is valuable here: the final constrained
    # solver can substantially improve a good basin, but cannot reliably
    # change its contact-graph topology.
    for restart in range(32):
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

        # The final stage is deliberately long: improvements near the
        # optimum require resolving several nearly equal active distances.
        for it in range(40000):
            i = rng.integers(n)
            scale = 0.008 * (1.0 - it / 40000.0) + 0.000015
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

    # Convert the final nonsmooth ratio problem into a smooth constrained
    # problem: scale so d_min^2 = 1, then minimize t subject to
    # 1 <= d_ij^2 <= t for every pair.  This is exactly equivalent to
    # maximizing the evaluator's squared min-distance/diameter ratio.
    best_points -= np.mean(best_points, axis=0)
    upper = np.triu_indices(n, 1)
    final_d = distances(best_points)
    min_d2 = np.min(final_d[upper])
    best_points /= np.sqrt(min_d2)

    try:
        from scipy.optimize import minimize

        initial_d = distances(best_points)[upper]
        x0 = np.concatenate(
            (best_points.ravel(), [float(np.max(initial_d))])
        )

        def objective(x):
            return x[-1]

        def objective_jacobian(x):
            g = np.zeros_like(x)
            g[-1] = 1.0
            return g

        def constraints(x):
            p = x[:-1].reshape(n, 3)
            delta = p[upper[0]] - p[upper[1]]
            d2 = np.einsum("ij,ij->i", delta, delta)
            return np.concatenate((d2 - 1.0, x[-1] - d2, [x[-1]]))

        def constraint_jacobian(x):
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

        polished = minimize(
            objective,
            x0,
            jac=objective_jacobian,
            constraints={
                "type": "ineq",
                "fun": constraints,
                "jac": constraint_jacobian,
            },
            method="SLSQP",
            # This problem has many nearly active distance constraints.
            # Permit additional SQP iterations to equalize those contacts;
            # evaluation time is not part of the combined score.
            options={"maxiter": 4000, "ftol": 1e-13, "disp": False},
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
        # Preserve the reliable stochastic construction if SciPy is absent
        # or a platform-specific optimizer failure occurs.
        pass

    return np.asarray(best_points, dtype=float)


# EVOLVE-BLOCK-END
