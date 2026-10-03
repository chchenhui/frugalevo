# EVOLVE-BLOCK-START
import numpy as np


def _normalize_rows(points: np.ndarray) -> np.ndarray:
    """Remove translation and fix RMS radius without imposing cosphericity."""
    centered = points - np.mean(points, axis=0, keepdims=True)
    scale = np.sqrt(np.mean(np.sum(centered * centered, axis=1)))
    return centered / max(float(scale), 1e-12)


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
    axes = np.vstack((np.eye(3), -np.eye(3))) * np.sqrt(3.0)
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
    active_mix: float,
) -> np.ndarray:
    """
    Gradient of the smooth log-ratio surrogate, optionally blended with
    equal-weight forces on the currently active close and far contacts.
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

    # The exact objective is controlled by sets of contacts, not necessarily
    # one pair.  Equal weighting prevents late softmax updates from chasing
    # whichever nearly tied pair happens to be infinitesimally most extreme.
    if active_mix > 0.0:
        minimum = float(np.min(values))
        maximum = float(np.max(values))
        close = values <= minimum * 1.014
        far = values >= maximum * 0.986
        active = np.zeros_like(values)
        active[close] = 1.0 / max(int(np.sum(close)), 1)
        active[far] -= 1.0 / max(int(np.sum(far)), 1)
        coefficients = (1.0 - active_mix) * coefficients + active_mix * active

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
) -> tuple[np.ndarray, float]:
    """Run a staged projected-gradient refinement from one initial seed."""
    points = initial_points.copy()
    velocity = np.zeros_like(points)

    best_points = points.copy()
    best_score = _true_ratio_score(points, upper)

    stages = (
        (7.0, 360, 0.040, 0.00),
        (16.0, 420, 0.026, 0.00),
        (38.0, 480, 0.016, 0.10),
        (90.0, 560, 0.009, 0.32),
        (180.0, 360, 0.005, 0.52),
    )

    for beta, iterations, initial_step, active_mix in stages:
        for iteration in range(iterations):
            gradient = _smooth_ratio_gradient(points, upper, beta, active_mix)

            # Only translation and overall scale are immaterial.  Unlike a
            # spherical projection, retain independent radial motion.
            gradient -= np.mean(gradient, axis=0, keepdims=True)
            radial = np.sum(gradient * points) / max(
                float(np.sum(points * points)), 1e-12
            )
            gradient -= radial * points

            velocity = 0.72 * velocity + gradient
            velocity -= np.mean(velocity, axis=0, keepdims=True)
            radial = np.sum(velocity * points) / max(
                float(np.sum(points * points)), 1e-12
            )
            velocity -= radial * points

            velocity_norm = np.sqrt(np.mean(np.sum(velocity * velocity, axis=1)))
            if velocity_norm > 3.0:
                velocity *= 3.0 / velocity_norm

            progress = iteration / max(iterations - 1, 1)
            step = initial_step * (1.0 - 0.70 * progress)
            points = _normalize_rows(points + step * velocity)

            if iteration % 12 == 0 or iteration == iterations - 1:
                score = _true_ratio_score(points, upper)
                if score > best_score:
                    best_score = score
                    best_points = points.copy()

    # The continuation objective still gives small weight to inactive pairs.
    # Polish its best saved state against the actual min/max contact sets,
    # accepting a step only when the evaluator's exact squared ratio improves.
    points = best_points.copy()
    velocity = np.zeros_like(points)

    for tolerance, base_step, iterations in (
        (0.030, 0.018, 90),
        (0.014, 0.010, 120),
        (0.006, 0.005, 150),
    ):
        for _ in range(iterations):
            squared = _pairwise_squared_distances(points)
            values = squared[upper]
            minimum = float(np.min(values))
            maximum = float(np.max(values))

            close = values <= minimum * (1.0 + tolerance)
            far = values >= maximum * (1.0 - tolerance)
            coefficients = np.zeros_like(values)
            coefficients[close] = 1.0 / max(int(np.sum(close)), 1)
            coefficients[far] -= 1.0 / max(int(np.sum(far)), 1)

            delta = points[upper[0]] - points[upper[1]]
            pair_force = (
                2.0 * coefficients[:, None] * delta
                / np.maximum(values[:, None], 1e-12)
            )
            force = np.zeros_like(points)
            np.add.at(force, upper[0], pair_force)
            np.add.at(force, upper[1], -pair_force)

            force -= np.mean(force, axis=0, keepdims=True)
            radial = np.sum(force * points) / max(
                float(np.sum(points * points)), 1e-12
            )
            force -= radial * points

            velocity = 0.45 * velocity + force
            velocity -= np.mean(velocity, axis=0, keepdims=True)
            radial = np.sum(velocity * points) / max(
                float(np.sum(points * points)), 1e-12
            )
            velocity -= radial * points

            norm = np.sqrt(np.mean(np.sum(velocity * velocity, axis=1)))
            if norm < 1e-14:
                break

            accepted = False
            for multiplier in (1.0, 0.5, 0.25, 0.125, 0.0625):
                trial = _normalize_rows(
                    points + base_step * multiplier * velocity / norm
                )
                trial_score = _true_ratio_score(trial, upper)
                if trial_score > best_score + 1e-13:
                    points = trial
                    best_points = trial.copy()
                    best_score = trial_score
                    accepted = True
                    break

            if not accepted:
                velocity *= 0.25
                if np.sqrt(np.mean(np.sum(velocity * velocity, axis=1))) < 1e-10:
                    break

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

    cube_axis = _cube_octahedron_seed()
    seeds = [cube_axis, _fibonacci_seed(count)]

    # The cube--axis arrangement is a valuable contact-graph prior, but the
    # best unconstrained packing need not preserve its antipodal symmetry.
    # Small deterministic perturbations retain that basin while freeing both
    # radial coordinates and contact multiplicities.
    for magnitude in (0.018, 0.045, 0.085, 0.140):
        seeds.append(
            _normalize_rows(cube_axis + magnitude * rng.normal(size=(count, 3)))
        )

    # Multi-start seeds explore different basins while remaining reproducible.
    for _ in range(12):
        random_points = rng.normal(size=(count, 3))
        seeds.append(_normalize_rows(random_points))

    best_points = seeds[0]
    best_score = _true_ratio_score(best_points, upper)

    for seed in seeds:
        candidate, score = _optimize_seed(seed, upper)
        if score > best_score:
            best_points = candidate
            best_score = score

    return np.asarray(best_points, dtype=float)


# EVOLVE-BLOCK-END