# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """Optimize spherical point sets with tangent-plane annealing and polishing.

    The objective is the evaluator's squared minimum/maximum distance ratio.
    Tangent proposals preserve the sphere exactly, while several independent
    restarts and a final greedy cooling phase improve exploration and local
    nonsmooth max-min optimization.
    """
    n = 14
    rng = np.random.default_rng(20260912)
    pairs = np.triu_indices(n, 1)

    index = np.arange(n, dtype=float)
    z = 1.0 - 2.0 * (index + 0.5) / n
    golden = np.pi * (3.0 - np.sqrt(5.0))
    theta = golden * index
    seed = np.column_stack((
        np.sqrt(1.0 - z * z) * np.cos(theta),
        np.sqrt(1.0 - z * z) * np.sin(theta),
        z,
    ))

    def objective(x):
        delta = x[:, None, :] - x[None, :, :]
        distance2 = np.sum(delta * delta, axis=2)[pairs]
        return float(np.min(distance2) / np.max(distance2))

    best_value = -np.inf
    best_points = seed.copy()

    # Use more independent basins, with slightly shorter annealing phases.
    # This increases the chance of finding a better spherical-code topology
    # without materially increasing the total number of objective evaluations.
    for restart in range(8):
        if restart == 0:
            points = seed.copy()
        elif restart == 1:
            # A second deterministic low-discrepancy phase gives a useful
            # alternative to both the Fibonacci seed and random starts.
            phase = 0.5 * golden
            theta2 = golden * index + phase
            points = np.column_stack((
                np.sqrt(1.0 - z * z) * np.cos(theta2),
                np.sqrt(1.0 - z * z) * np.sin(theta2),
                z,
            ))
        else:
            points = rng.normal(size=(n, 3))
            points /= np.linalg.norm(points, axis=1, keepdims=True)

        value = objective(points)
        steps = 60000
        for step in range(steps):
            t = step / float(steps - 1)
            # Tangent moves are more effective than perturbing and
            # renormalizing in Cartesian coordinates.
            sigma = 0.24 * (1.0 - t) ** 0.82 + 0.00035
            k = int(rng.integers(n))
            direction = rng.normal(size=3)
            direction -= np.dot(direction, points[k]) * points[k]
            length = np.linalg.norm(direction)
            if length < 1.0e-14:
                continue
            direction /= length

            candidate = points.copy()
            candidate[k] = points[k] + sigma * direction
            candidate[k] /= np.linalg.norm(candidate[k])
            candidate_value = objective(candidate)
            delta = candidate_value - value
            temperature = 0.0038 * (1.0 - t) ** 1.25 + 2.0e-8

            if delta >= 0.0 or rng.random() < np.exp(delta / temperature):
                points = candidate
                value = candidate_value
                if value > best_value:
                    best_value = value
                    best_points = points.copy()

        # Deterministic fine polishing removes late-temperature wandering.
        for step in range(14000):
            sigma = 0.0025 * (1.0 - step / 14000.0) + 1.0e-6
            k = int(rng.integers(n))
            direction = rng.normal(size=3)
            direction -= np.dot(direction, points[k]) * points[k]
            length = np.linalg.norm(direction)
            if length < 1.0e-14:
                continue
            candidate = points.copy()
            candidate[k] = points[k] + sigma * direction / length
            candidate[k] /= np.linalg.norm(candidate[k])
            candidate_value = objective(candidate)
            if candidate_value >= value:
                points, value = candidate, candidate_value
                if value > best_value:
                    best_value = value
                    best_points = points.copy()

    return np.asarray(best_points, dtype=float)


# EVOLVE-BLOCK-END
