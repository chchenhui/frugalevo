# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """Use deterministic multistart diameter-constrained SLSQP with adaptive basin escapes."""
    n = 14
    pairs_i, pairs_j = np.triu_indices(n, 1)
    m = len(pairs_i)

    def normalize(points: np.ndarray) -> np.ndarray:
        points = points - points.mean(axis=0, keepdims=True)
        differences = points[pairs_i] - points[pairs_j]
        diameter = np.sqrt(np.max(np.sum(differences * differences, axis=1)))
        return points / diameter

    def quality(points: np.ndarray) -> float:
        differences = points[pairs_i] - points[pairs_j]
        squared = np.sum(differences * differences, axis=1)
        return float(np.min(squared) / np.max(squared))

    # A staggered two-ring antiprism is a much better feasible start than
    # independent Gaussian points.
    angles = 2.0 * np.pi * np.arange(7) / 7.0
    r = 1.0
    h = np.sqrt(np.sin(np.pi / 7.0) ** 2 - np.sin(np.pi / 14.0) ** 2)
    antiprism = np.vstack((
        np.column_stack((r * np.cos(angles), r * np.sin(angles), np.full(7, h))),
        np.column_stack((r * np.cos(angles + np.pi / 7.0),
                         r * np.sin(angles + np.pi / 7.0), np.full(7, -h))),
    ))
    best = normalize(antiprism)
    best_value = quality(best)

    try:
        from scipy.optimize import minimize
    except ImportError:
        return best

    # The constraint vector is:
    #   |p_i-p_j|^2 >= t  for every pair,
    #   |p_i-p_j|^2 <= 1  for every pair.
    # Analytic derivatives make SLSQP substantially more reliable than
    # finite-difference optimization for the 182 pair constraints.
    def constraints(z: np.ndarray) -> np.ndarray:
        points = z[:-1].reshape(n, 3)
        delta = points[pairs_i] - points[pairs_j]
        dist2 = np.sum(delta * delta, axis=1)
        return np.concatenate((dist2 - z[-1], 1.0 - dist2))

    def constraints_jacobian(z: np.ndarray) -> np.ndarray:
        points = z[:-1].reshape(n, 3)
        delta = points[pairs_i] - points[pairs_j]
        jac = np.zeros((2 * m, 3 * n + 1))
        rows = np.arange(m)
        for coordinate in range(3):
            first = 3 * pairs_i + coordinate
            second = 3 * pairs_j + coordinate
            derivative = 2.0 * delta[:, coordinate]
            jac[rows, first] = derivative
            jac[rows, second] = -derivative
        jac[rows, -1] = -1.0
        jac[m:, :-1] = -jac[:m, :-1]
        return jac

    centroid_jacobian = np.zeros((3, 3 * n + 1))
    for coordinate in range(3):
        centroid_jacobian[coordinate, coordinate:3 * n:3] = 1.0 / n

    rng = np.random.default_rng(20260912)
    starts = [best]

    # Add contact-rich 14-point polyhedral families.  These are deliberately
    # perturbed slightly: exact symmetry can otherwise keep SLSQP in a
    # symmetric, but suboptimal, stationary contact graph.
    cuboctahedron = np.array(
        [(a, b, 0.0) for a in (-1.0, 1.0) for b in (-1.0, 1.0)] +
        [(a, 0.0, b) for a in (-1.0, 1.0) for b in (-1.0, 1.0)] +
        [(0.0, a, b) for a in (-1.0, 1.0) for b in (-1.0, 1.0)]
    )
    for pole_height in (1.05, 1.35, 1.7, 2.1, 2.6):
        structured = np.vstack((
            cuboctahedron,
            ((0.0, 0.0, pole_height), (0.0, 0.0, -pole_height)),
        ))
        starts.append(normalize(structured))
        starts.append(normalize(
            structured + 0.018 * rng.standard_normal((n, 3))
        ))

    cube_axes = np.array(
        [(a, b, c) for a in (-1.0, 1.0)
                   for b in (-1.0, 1.0)
                   for c in (-1.0, 1.0)] +
        [(a, 0.0, 0.0) for a in (-2.0, 2.0)] +
        [(0.0, a, 0.0) for a in (-2.0, 2.0)] +
        [(0.0, 0.0, a) for a in (-2.0, 2.0)]
    )
    for noise in (0.0, 0.012, 0.04):
        starts.append(normalize(
            cube_axes + noise * rng.standard_normal((n, 3))
        ))

    # Fine perturbations retain the useful antiprism contact graph, while
    # larger perturbations permit transitions to quite different packings.
    for scale in (0.004, 0.008, 0.015, 0.03, 0.06, 0.11, 0.18, 0.28, 0.42):
        for _ in range(5):
            starts.append(normalize(
                best + scale * rng.standard_normal((n, 3))
            ))

    # Boundary-heavy starts complement the antiprism: optimum
    # diameter-constrained packings frequently have many near-boundary points.
    for _ in range(32):
        spherical = rng.standard_normal((n, 3))
        spherical /= np.linalg.norm(spherical, axis=1, keepdims=True)
        starts.append(normalize(spherical))
    for _ in range(28):
        starts.append(normalize(rng.standard_normal((n, 3))))

    # Re-starting from local optima can cross contact-graph basin boundaries
    # which independent random starts do not reliably reach.  The objective
    # has narrow high-quality basins, so spend additional deterministic work
    # exploring local-optimum perturbations; runtime is not score-weighted.
    initial_count = len(starts)
    max_starts = initial_count + 900
    start_index = 0
    while start_index < len(starts) and start_index < max_starts:
        start = starts[start_index]
        start_index += 1
        start_dist2 = np.sum((start[pairs_i] - start[pairs_j]) ** 2, axis=1)
        z0 = np.concatenate((start.ravel(), [0.98 * np.min(start_dist2)]))
        result = minimize(
            fun=lambda z: -z[-1],
            x0=z0,
            jac=lambda z: np.r_[np.zeros(3 * n), -1.0],
            method="SLSQP",
            constraints=[
                {"type": "ineq", "fun": constraints, "jac": constraints_jacobian},
                {
                    "type": "eq",
                    "fun": lambda z: z[:-1].reshape(n, 3).mean(axis=0),
                    "jac": lambda z: centroid_jacobian,
                },
            ],
            options={"maxiter": 1600, "ftol": 8e-14, "disp": False},
        )
        if np.all(np.isfinite(result.x)):
            candidate = normalize(result.x[:-1].reshape(n, 3))
            candidate_value = quality(candidate)
            if candidate_value > best_value:
                best = candidate
                best_value = candidate_value

            if len(starts) < max_starts:
                # Very small displacements preserve a promising contact graph
                # while allowing inactive contacts to exchange roles.  The
                # larger displacements are needed to leave a whole basin when
                # the present contact graph is only locally optimal.
                for scale in (0.0025, 0.007, 0.018, 0.055, 0.14):
                    if len(starts) >= max_starts:
                        break
                    starts.append(normalize(
                        candidate + scale * rng.standard_normal((n, 3))
                    ))

    return best


# EVOLVE-BLOCK-END
