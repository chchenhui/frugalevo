# EVOLVE-BLOCK-START
import numpy as np


_N_POINTS = 14
_PAIR_I, _PAIR_J = np.triu_indices(_N_POINTS, 1)


def _normalize(points: np.ndarray) -> np.ndarray:
    """Remove translation and arbitrary uniform scale."""
    centered = points - points.mean(axis=0, keepdims=True)
    rms_radius = np.sqrt(np.mean(np.sum(centered * centered, axis=1)))
    return centered / max(rms_radius, 1.0e-14)


def _squared_distances(points: np.ndarray) -> np.ndarray:
    delta = points[_PAIR_I] - points[_PAIR_J]
    return np.einsum("ij,ij->i", delta, delta)


def _exact_score(points: np.ndarray) -> float:
    """Evaluator-equivalent squared minimum-distance / diameter-squared."""
    distances = _squared_distances(points)
    return float(distances.min() / distances.max())


def _soft_contact_direction(points: np.ndarray, sharpness: float) -> np.ndarray:
    """
    Gradient ascent direction for a log-soft-min minus log-soft-max
    pair-distance objective.  Pair storage is triangular, avoiding
    unnecessary NxN temporary arrays.
    """
    delta = points[_PAIR_I] - points[_PAIR_J]
    squared = np.einsum("ij,ij->i", delta, delta)
    log_squared = np.log(np.maximum(squared, 1.0e-15))

    low_origin = log_squared.min()
    high_origin = log_squared.max()

    low_raw = np.exp(-sharpness * (log_squared - low_origin))
    high_raw = np.exp(sharpness * (log_squared - high_origin))
    weights = low_raw / low_raw.sum() - high_raw / high_raw.sum()

    # d log(||xi-xj||^2) / d xi = 2 (xi-xj) / ||xi-xj||^2.
    pair_force = 2.0 * weights[:, None] * delta / squared[:, None]
    direction = np.zeros_like(points)
    np.add.at(direction, _PAIR_I, pair_force)
    np.add.at(direction, _PAIR_J, -pair_force)
    return direction


def _schedule_sharpness(progress: float, schedule: int) -> float:
    """Several continuation paths target different eventual contact graphs."""
    if schedule == 0:
        return 6.0 + 105.0 * progress * progress
    if schedule == 1:
        return 7.0 + 210.0 * progress ** 1.45
    if progress < 0.66:
        return 8.0 + 52.0 * (progress / 0.66) ** 1.35
    return 60.0 + 250.0 * ((progress - 0.66) / 0.34) ** 2.0


def _refine(seed: np.ndarray, schedule: int, iterations: int = 1450) -> np.ndarray:
    """
    Follow a smooth contact objective while periodically selecting by the
    nonsmoothed evaluator objective.  Revert sustained surrogate-induced
    regressions to the best exact contact configuration.
    """
    points = _normalize(seed)
    best = points.copy()
    best_score = _exact_score(points)
    velocity = np.zeros_like(points)
    regressions = 0
    step_factor = 1.0
    damped_steps = 0

    for step in range(iterations):
        progress = step / float(iterations - 1)
        direction = _soft_contact_direction(
            points, _schedule_sharpness(progress, schedule)
        )
        direction -= direction.mean(axis=0, keepdims=True)

        norm = np.sqrt(np.mean(np.sum(direction * direction, axis=1)))
        if norm > 1.0e-14:
            direction /= norm

        persistence = 0.64 if damped_steps > 0 else 0.82
        velocity = persistence * velocity + (1.0 - persistence) * direction
        step_size = step_factor * 0.046 * (1.0 - 0.66 * progress)
        points = _normalize(points + step_size * velocity)
        damped_steps = max(0, damped_steps - 1)

        if step % 10 == 9 or step == iterations - 1:
            score = _exact_score(points)
            if score > best_score:
                best_score = score
                best = points.copy()
                regressions = 0
            elif score < best_score * (1.0 - 3.0e-4):
                regressions += 1
            else:
                regressions = max(0, regressions - 1)

            # A few bad exact checkpoints indicate that the increasingly
            # sharp surrogate has left the useful contact basin.  Restart
            # from the authoritative exact-best state with gentler dynamics.
            if regressions >= 3:
                points = best.copy()
                velocity.fill(0.0)
                step_factor *= 0.72
                damped_steps = 90
                regressions = 0

    return best


def _layer_seed(radius: float, height: float, twist: float, pole: float) -> np.ndarray:
    """Two staggered six-rings with two axial points."""
    angles = np.arange(6, dtype=float) * (np.pi / 3.0)
    lower = np.column_stack(
        (radius * np.cos(angles), radius * np.sin(angles), -height * np.ones(6))
    )
    upper = np.column_stack(
        (
            radius * np.cos(angles + twist),
            radius * np.sin(angles + twist),
            height * np.ones(6),
        )
    )
    return np.vstack((lower, upper, [[0.0, 0.0, -pole], [0.0, 0.0, pole]]))


def _initializers() -> list:
    """
    Produce geometrically distinct high-quality basins.  Small fixed
    perturbations deliberately break exact symmetries, which allows the
    contact graph to rearrange during refinement.
    """
    seeds = [
        _layer_seed(0.946, 0.405, np.pi / 6.0, 1.000),
        _layer_seed(0.920, 0.385, np.pi / 6.0, 0.980),
        _layer_seed(0.975, 0.430, np.pi / 6.0, 1.030),
        _layer_seed(0.950, 0.415, 0.455, 1.000),
        _layer_seed(0.955, 0.400, 0.585, 1.000),
    ]

    cube = np.array(
        [[x, y, z] for x in (-1.0, 1.0)
         for y in (-1.0, 1.0)
         for z in (-1.0, 1.0)],
        dtype=float,
    )
    axes = np.array(
        [
            [1.55, 0.0, 0.0], [-1.55, 0.0, 0.0],
            [0.0, 1.67, 0.0], [0.0, -1.67, 0.0],
            [0.0, 0.0, 1.60], [0.0, 0.0, -1.60],
        ],
        dtype=float,
    )
    seeds.append(np.vstack((cube, axes)))

    rng = np.random.default_rng(482917)
    perturbed = []
    for seed in seeds:
        perturbed.append(seed + 0.018 * rng.standard_normal(seed.shape))

    # Two reproducible nonsymmetric starts complement the geometric seeds.
    perturbed.append(rng.standard_normal((_N_POINTS, 3)))
    perturbed.append(rng.standard_normal((_N_POINTS, 3)))
    return perturbed


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Return exactly fourteen finite 3D points with a large minimum pairwise
    distance relative to their diameter.
    """
    best_points = None
    best_score = -np.inf

    for seed in _initializers():
        for schedule in range(3):
            candidate = _refine(seed, schedule)
            score = _exact_score(candidate)
            if score > best_score:
                best_score = score
                best_points = candidate

    return np.asarray(best_points, dtype=float)


# EVOLVE-BLOCK-END