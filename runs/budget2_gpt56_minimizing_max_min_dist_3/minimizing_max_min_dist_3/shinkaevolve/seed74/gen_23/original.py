# EVOLVE-BLOCK-START
import numpy as np


def _pairwise_geometry(points: np.ndarray):
    """Return pairwise displacement vectors, squared distances, and mask."""
    delta = points[:, None, :] - points[None, :, :]
    squared = np.einsum("ijk,ijk->ij", delta, delta)
    np.fill_diagonal(squared, 1.0)
    mask = ~np.eye(points.shape[0], dtype=bool)
    return delta, squared, mask


def _normalize(points: np.ndarray) -> np.ndarray:
    """Remove irrelevant translation and uniform scale."""
    points = points - points.mean(axis=0, keepdims=True)
    rms = np.sqrt(np.mean(np.sum(points * points, axis=1)))
    return points / rms


def _exact_ratio(points: np.ndarray) -> float:
    """Compute the actual unsmoothed minimum/maximum distance ratio."""
    _, squared, mask = _pairwise_geometry(points)
    values = squared[mask]
    return float(np.sqrt(values.min() / values.max()))


def _soft_extrema_gradient(points: np.ndarray, sharpness: float) -> np.ndarray:
    """
    Gradient of a differentiable approximation of
    log(min pair distance) - log(max pair distance).
    """
    delta, squared, mask = _pairwise_geometry(points)
    log_distance = 0.5 * np.log(squared)

    valid_logs = log_distance[mask]
    low_shift = valid_logs.min()
    high_shift = valid_logs.max()

    near_weights = np.zeros_like(squared)
    far_weights = np.zeros_like(squared)

    near_values = np.exp(-sharpness * (valid_logs - low_shift))
    far_values = np.exp(sharpness * (valid_logs - high_shift))

    near_weights[mask] = near_values / near_values.sum()
    far_weights[mask] = far_values / far_values.sum()

    # Pair (i,j) appears twice in the symmetric matrix.  The factor is
    # harmless under normalized ascent and improves the update magnitude.
    coefficients = near_weights - far_weights
    return 2.0 * np.sum(
        coefficients[:, :, None] * delta / squared[:, :, None],
        axis=1,
    )


def _refine(
    seed_points: np.ndarray, schedule: int = 0, iterations: int = 1500
) -> np.ndarray:
    """Refine one layout with a selected soft-extrema continuation path."""
    points = _normalize(seed_points.astype(float, copy=True))
    best_points = points.copy()
    best_ratio = _exact_ratio(points)
    velocity = np.zeros_like(points)

    for step in range(iterations):
        progress = step / max(iterations - 1, 1)

        # Different continuations favor different contact graphs.  In
        # particular, delaying the very sharp phase permits coordinated
        # motion before individual bottleneck pairs dominate the gradient.
        if schedule == 0:
            sharpness = 6.0 + 94.0 * progress * progress
        elif schedule == 1:
            sharpness = 6.0 + 194.0 * progress ** 1.5
        else:
            if progress < 0.68:
                sharpness = 8.0 + 47.0 * (progress / 0.68) ** 1.4
            else:
                sharpness = 55.0 + 245.0 * ((progress - 0.68) / 0.32) ** 2.0

        gradient = _soft_extrema_gradient(points, sharpness)

        gradient -= gradient.mean(axis=0, keepdims=True)
        gradient_norm = np.sqrt(np.mean(np.sum(gradient * gradient, axis=1)))
        if gradient_norm > 0.0:
            gradient /= gradient_norm

        velocity = 0.82 * velocity + 0.18 * gradient
        step_size = 0.045 * (1.0 - 0.65 * progress)
        points = _normalize(points + step_size * velocity)

        # The smooth objective is only a surrogate, so retain the best
        # actual configuration rather than assuming the endpoint is best.
        if step % 24 == 23 or step == iterations - 1:
            ratio = _exact_ratio(points)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = points.copy()

    return best_points


def _structured_initializers() -> list:
    """Build complementary symmetric and seeded-random starting layouts."""
    initializers = []

    # Two staggered hexagonal layers plus two poles.
    angles = np.arange(6, dtype=float) * (np.pi / 3.0)
    lower = np.column_stack((np.cos(angles), np.sin(angles), -0.48 * np.ones(6)))
    upper = np.column_stack(
        (
            np.cos(angles + np.pi / 6.0),
            np.sin(angles + np.pi / 6.0),
            0.48 * np.ones(6),
        )
    )
    initializers.append(
        np.vstack((lower, upper, [[0.0, 0.0, -1.35], [0.0, 0.0, 1.35]]))
    )

    # Cube corners and axial points provide a distinct high-symmetry basin.
    cube = np.array(
        [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)],
        dtype=float,
    )
    axial = 1.65 * np.eye(3)
    initializers.append(np.vstack((cube, axial, -axial)))

    # Fixed seeds retain reproducibility while exploring less symmetric basins.
    for seed in (104729, 271828, 314159, 161803):
        rng = np.random.default_rng(seed)
        initializers.append(rng.normal(size=(14, 3)))

    return initializers


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Construct fourteen 3D points by optimizing their minimum-to-maximum
    Euclidean pairwise-distance ratio.
    """
    best_points = None
    best_ratio = -np.inf

    for initial in _structured_initializers():
        # Portfolio continuation is inexpensive for this fourteen-point
        # problem and substantially reduces dependence on a single
        # surrogate-temperature path.
        for schedule in range(3):
            candidate = _refine(initial, schedule=schedule)
            ratio = _exact_ratio(candidate)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = candidate

    return np.asarray(best_points, dtype=float)


# EVOLVE-BLOCK-END