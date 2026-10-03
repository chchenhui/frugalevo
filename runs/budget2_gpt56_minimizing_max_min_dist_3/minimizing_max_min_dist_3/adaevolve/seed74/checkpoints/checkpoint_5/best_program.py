# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def min_max_dist_dim3_14() -> np.ndarray:
    """Construct 14 points by spherical repulsion followed by diameter-one
    constrained optimization of the minimum pairwise Euclidean distance."""
    n = 14
    rng = np.random.default_rng(20260912)
    ii, jj = np.triu_indices(n, 1)

    def actual_ratio(points):
        differences = points[ii] - points[jj]
        distances = np.sqrt(np.sum(differences * differences, axis=1))
        return np.min(distances) / np.max(distances)

    # Spherical repulsion provides consistently good, non-degenerate initial
    # configurations.  Several deterministic restarts reduce local-minimum
    # sensitivity before the direct maximin polishing step below.
    best = None
    best_ratio = -1.0
    # The repulsion landscape has several distinct local spherical codes.
    # Extra fixed-seed restarts are cheap relative to the final SLSQP solve
    # and make it much more likely that polishing starts in the best basin.
    for _ in range(64):
        points = rng.normal(size=(n, 3))
        points /= np.linalg.norm(points, axis=1)[:, None]
        points += 0.015 * rng.normal(size=(n, 3))
        points /= np.linalg.norm(points, axis=1)[:, None]

        for power in (4, 8, 16, 28):
            for iteration in range(1500):
                delta = points[:, None, :] - points[None, :, :]
                distance_squared = np.sum(delta * delta, axis=2)
                np.fill_diagonal(distance_squared, 1.0)

                weights = distance_squared ** (-(power + 2.0) / 2.0)
                np.fill_diagonal(weights, 0.0)
                force = np.sum(weights[:, :, None] * delta, axis=1)

                # Project the repulsive force onto the sphere tangent planes.
                force -= np.sum(force * points, axis=1)[:, None] * points
                norm = np.max(np.linalg.norm(force, axis=1))
                if norm > 0.0:
                    step = 0.055 * (1.0 - 0.82 * iteration / 1499.0)
                    points += step * force / norm
                    points /= np.linalg.norm(points, axis=1)[:, None]

        value = actual_ratio(points)
        if value > best_ratio:
            best = points.copy()
            best_ratio = value

    # Uniform scale is immaterial, so constrain every distance to be at most
    # one and directly maximize the common lower-distance variable t.
    distances = np.sqrt(np.sum((best[ii] - best[jj]) ** 2, axis=1))
    initial_points = (best - best[0]) / np.max(distances)
    initial_t = actual_ratio(initial_points) * 0.999999
    initial = np.r_[initial_points[1:].ravel(), initial_t]

    def unpack(vector):
        points = np.zeros((n, 3))
        points[1:] = vector[:-1].reshape(n - 1, 3)
        return points, vector[-1]

    def constraints(vector):
        points, t = unpack(vector)
        delta = points[ii] - points[jj]
        distance_squared = np.sum(delta * delta, axis=1)
        return np.r_[distance_squared - t * t, 1.0 - distance_squared]

    def constraint_jacobian(vector):
        points, t = unpack(vector)
        delta = points[ii] - points[jj]
        pair_count = len(ii)
        jacobian = np.zeros((2 * pair_count, 3 * (n - 1) + 1))

        for k, (a, b) in enumerate(zip(ii, jj)):
            gradient = 2.0 * delta[k]
            if a:
                jacobian[k, 3 * (a - 1):3 * a] = gradient
                jacobian[pair_count + k, 3 * (a - 1):3 * a] = -gradient
            if b:
                jacobian[k, 3 * (b - 1):3 * b] = -gradient
                jacobian[pair_count + k, 3 * (b - 1):3 * b] = gradient

        jacobian[:pair_count, -1] = -2.0 * t
        return jacobian

    result = minimize(
        lambda vector: -vector[-1],
        initial,
        jac=lambda vector: np.r_[np.zeros(3 * (n - 1)), -1.0],
        constraints={
            "type": "ineq",
            "fun": constraints,
            "jac": constraint_jacobian,
        },
        method="SLSQP",
        options={"maxiter": 2500, "ftol": 1e-12, "disp": False},
    )

    polished, _ = unpack(result.x)
    if np.all(np.isfinite(polished)) and actual_ratio(polished) > best_ratio:
        return polished
    return best


# EVOLVE-BLOCK-END
