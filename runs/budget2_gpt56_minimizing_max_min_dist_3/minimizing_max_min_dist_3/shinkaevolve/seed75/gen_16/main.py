# EVOLVE-BLOCK-START
import numpy as np


def _normalize_rows(points: np.ndarray) -> np.ndarray:
    """Project every point onto the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / np.maximum(norms, 1e-12)


def _pairwise_squared_distances(points: np.ndarray) -> np.ndarray:
    """Return the complete squared Euclidean distance matrix."""
    delta = points[:, None, :] - points[None, :, :]
    return np.einsum("ijk,ijk->ij", delta, delta)


def _true_ratio_score(points: np.ndarray, upper: np.ndarray) -> float:
    """Score using the evaluator's squared-distance ratio convention."""
    distances = _pairwise_squared_distances(points)
    pair_values = distances[upper]
    return float(np.min(pair_values) / np.max(pair_values))


def _cube_octahedron_seed() -> np.ndarray:
    """A strong symmetric 14-point seed: cube vertices plus axis vertices."""
    cube = np.array(
        [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)],
        dtype=float,
    )
    axes = np.vstack((np.eye(3), -np.eye(3)))
    return _normalize_rows(np.vstack((cube, axes)))


def _fibonacci_seed(count: int) -> np.ndarray:
    """Deterministic near-uniform spherical initialization."""
    indices = np.arange(count, dtype=float)
    golden = np.pi * (3.0 - np.sqrt(5.0))
    z = 1.0 - 2.0 * (indices + 0.5) / count
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    angle = golden * indices
    return np.column_stack((radius * np.cos(angle), radius * np.sin(angle), z))


def _smooth_ratio_gradient(
    points: np.ndarray,
    upper: np.ndarray,
    beta: float,
) -> np.ndarray:
    """
    Gradient of a differentiable approximation to
    log(min squared distance) - log(max squared distance).
    """
    distances = _pairwise_squared_distances(points)
    values = np.maximum(distances[upper], 1e-12)
    log_values = np.log(values)

    min_logits = -beta * log_values
    min_logits -= np.max(min_logits)
    min_weights = np.exp(min_logits)
    min_weights /= np.sum(min_weights)

    max_logits = beta * log_values
    max_logits -= np.max(max_logits)
    max_weights = np.exp(max_logits)
    max_weights /= np.sum(max_weights)

    coefficients = min_weights - max_weights

    weight_matrix = np.zeros_like(distances)
    weight_matrix[upper] = coefficients
    weight_matrix[(upper[1], upper[0])] = coefficients

    differences = points[:, None, :] - points[None, :, :]
    inverse_distances = np.zeros_like(distances)
    nonzero = distances > 1e-12
    inverse_distances[nonzero] = 1.0 / distances[nonzero]

    return 2.0 * np.sum(
        weight_matrix[:, :, None] * differences * inverse_distances[:, :, None],
        axis=1,
    )


def _optimize_seed(
    initial_points: np.ndarray,
    upper: np.ndarray,
    stages: tuple[tuple[float, int, float], ...],
) -> tuple[np.ndarray, float]:
    """Run selected projected-gradient continuation stages from one seed."""
    points = initial_points.copy()
    velocity = np.zeros_like(points)

    best_points = points.copy()
    best_score = _true_ratio_score(points, upper)

    for beta, iterations, initial_step in stages:
        for iteration in range(iterations):
            gradient = _smooth_ratio_gradient(points, upper, beta)

            # Keep updates tangent to the sphere before reprojection.
            gradient -= np.sum(gradient * points, axis=1, keepdims=True) * points

            velocity = 0.68 * velocity + gradient
            velocity -= np.sum(velocity * points, axis=1, keepdims=True) * points

            velocity_norm = np.linalg.norm(velocity, axis=1, keepdims=True)
            velocity *= np.minimum(1.0, 2.5 / np.maximum(velocity_norm, 1e-12))

            progress = iteration / max(iterations - 1, 1)
            step = initial_step * (1.0 - 0.70 * progress)
            points = _normalize_rows(points + step * velocity)

            if iteration % 12 == 0 or iteration == iterations - 1:
                score = _true_ratio_score(points, upper)
                if score > best_score:
                    best_score = score
                    best_points = points.copy()

    return best_points, best_score


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Construct 14 reproducible 3D points with a high minimum/maximum
    pairwise-distance ratio.

    Returns
    -------
    np.ndarray
        Finite array of shape (14, 3).
    """
    count = 14
    upper = np.triu_indices(count, k=1)
    rng = np.random.default_rng(917341)

    seeds = [_cube_octahedron_seed(), _fibonacci_seed(count)]

    # Multi-start seeds explore different basins while remaining reproducible.
    for _ in range(14):
        random_points = rng.normal(size=(count, 3))
        seeds.append(_normalize_rows(random_points))

    # The broad first pass identifies favorable contact graphs before the
    # expensive high-beta stages.  Ranking is always by the exact objective,
    # rather than by its smooth surrogate.
    screening_stages = (
        (6.0, 200, 0.042),
        (16.0, 280, 0.026),
    )
    finalists: list[tuple[float, np.ndarray]] = []
    for seed in seeds:
        candidate, score = _optimize_seed(seed, upper, screening_stages)
        finalists.append((score, candidate))

    finalists.sort(key=lambda item: item[0], reverse=True)

    # A few distinct survivors retain basin diversity.  Their longer,
    # sharper continuation resolves the small set of active close and far
    # pairs that controls the true diameter-packing ratio.
    refinement_stages = (
        (34.0, 520, 0.017),
        (85.0, 720, 0.010),
        (180.0, 760, 0.0055),
    )
    best_score, best_points = finalists[0]
    for _, seed in finalists[:4]:
        candidate, score = _optimize_seed(seed, upper, refinement_stages)
        if score > best_score:
            best_points = candidate
            best_score = score

    return np.asarray(best_points, dtype=float)


# EVOLVE-BLOCK-END