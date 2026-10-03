# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Construct a reproducible 14-point diameter packing by optimizing the
    exact minimum-squared-distance to maximum-squared-distance ratio.
    """
    count = 14
    upper = np.triu_indices(count, 1)

    def normalize(points: np.ndarray) -> np.ndarray:
        return points / np.maximum(
            np.linalg.norm(points, axis=1, keepdims=True), 1e-12
        )

    def distances(points: np.ndarray) -> np.ndarray:
        delta = points[:, None, :] - points[None, :, :]
        return np.einsum("ijk,ijk->ij", delta, delta)

    def score(points: np.ndarray) -> float:
        values = distances(points)[upper]
        return float(values.min() / values.max())

    def gradient(points: np.ndarray, beta: float) -> np.ndarray:
        squared = distances(points)
        values = np.maximum(squared[upper], 1e-12)
        logs = np.log(values)

        low = -beta * logs
        low -= low.max()
        low = np.exp(low)
        low /= low.sum()

        high = beta * logs
        high -= high.max()
        high = np.exp(high)
        high /= high.sum()

        weights = np.zeros_like(squared)
        weights[upper] = low - high
        weights[(upper[1], upper[0])] = low - high

        delta = points[:, None, :] - points[None, :, :]
        inverse = np.zeros_like(squared)
        mask = squared > 1e-12
        inverse[mask] = 1.0 / squared[mask]
        return 2.0 * np.sum(weights[:, :, None] * delta * inverse[:, :, None], axis=1)

    def refine(seed: np.ndarray) -> tuple[np.ndarray, float]:
        points = seed.copy()
        velocity = np.zeros_like(points)
        best_points = points.copy()
        best_score = score(points)

        for beta, iterations, initial_step in (
            (7.0, 360, 0.040),
            (16.0, 420, 0.026),
            (38.0, 460, 0.016),
            (90.0, 520, 0.009),
        ):
            for iteration in range(iterations):
                direction = gradient(points, beta)
                direction -= np.sum(direction * points, axis=1, keepdims=True) * points

                velocity = 0.68 * velocity + direction
                velocity -= np.sum(velocity * points, axis=1, keepdims=True) * points
                velocity *= np.minimum(
                    1.0,
                    2.5 / np.maximum(
                        np.linalg.norm(velocity, axis=1, keepdims=True), 1e-12
                    ),
                )

                fraction = iteration / max(iterations - 1, 1)
                step = initial_step * (1.0 - 0.70 * fraction)
                points = normalize(points + step * velocity)

                if iteration % 12 == 0 or iteration == iterations - 1:
                    candidate_score = score(points)
                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_points = points.copy()

        return best_points, best_score

    cube = np.array(
        [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)],
        dtype=float,
    )
    cube_axis_seed = normalize(np.vstack((cube, np.eye(3), -np.eye(3))))

    indices = np.arange(count, dtype=float)
    golden = np.pi * (3.0 - np.sqrt(5.0))
    z = 1.0 - 2.0 * (indices + 0.5) / count
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    fibonacci_seed = np.column_stack(
        (radius * np.cos(golden * indices), radius * np.sin(golden * indices), z)
    )

    rng = np.random.default_rng(917341)
    seeds = [cube_axis_seed, fibonacci_seed]
    seeds.extend(normalize(rng.normal(size=(count, 3))) for _ in range(14))

    best_points = cube_axis_seed
    best_score = score(best_points)
    for seed in seeds:
        candidate, candidate_score = refine(seed)
        if candidate_score > best_score:
            best_points, best_score = candidate, candidate_score

    return np.asarray(best_points, dtype=float)


# EVOLVE-BLOCK-END