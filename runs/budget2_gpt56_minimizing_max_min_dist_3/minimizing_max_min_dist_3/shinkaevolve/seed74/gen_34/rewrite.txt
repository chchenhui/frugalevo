# EVOLVE-BLOCK-START
import numpy as np


_N = 14
_II, _JJ = np.triu_indices(_N, 1)


def _normalize(points: np.ndarray) -> np.ndarray:
    points = points - points.mean(axis=0, keepdims=True)
    rms = np.sqrt(np.mean(np.sum(points * points, axis=1)))
    return points / max(float(rms), 1.0e-14)


def _normalize_batch(points: np.ndarray) -> np.ndarray:
    points = points - points.mean(axis=1, keepdims=True)
    rms = np.sqrt(np.mean(np.sum(points * points, axis=2), axis=1))
    return points / np.maximum(rms, 1.0e-14)[:, None, None]


def _score(points: np.ndarray) -> float:
    delta = points[_II] - points[_JJ]
    squared = np.einsum("ij,ij->i", delta, delta)
    return float(np.min(squared) / np.max(squared))


def _batch_score(points: np.ndarray) -> np.ndarray:
    delta = points[:, _II] - points[:, _JJ]
    squared = np.sum(delta * delta, axis=2)
    return np.min(squared, axis=1) / np.max(squared, axis=1)


def _contact_gradient(points: np.ndarray, sharpness: float) -> np.ndarray:
    """
    Gradient of log(soft-min pair squared distance) minus
    log(soft-max pair squared distance).  Pairwise logarithms make the
    continuation insensitive to the otherwise arbitrary point scale.
    """
    delta = points[_II] - points[_JJ]
    squared = np.einsum("ij,ij->i", delta, delta)
    logs = np.log(np.maximum(squared, 1.0e-15))

    low_origin = float(np.min(logs))
    high_origin = float(np.max(logs))
    low = np.exp(-sharpness * (logs - low_origin))
    high = np.exp(sharpness * (logs - high_origin))
    weights = low / np.sum(low) - high / np.sum(high)

    force = 2.0 * weights[:, None] * delta / squared[:, None]
    grad = np.zeros_like(points)
    np.add.at(grad, _II, force)
    np.add.at(grad, _JJ, -force)

    grad -= grad.mean(axis=0, keepdims=True)
    radial = np.sum(grad * points) / max(np.sum(points * points), 1.0e-15)
    grad -= radial * points

    norm = np.sqrt(np.mean(np.sum(grad * grad, axis=1)))
    if norm > 1.0e-14:
        grad /= norm
    return grad


def _sharpness(t: float, schedule: int) -> float:
    if schedule == 0:
        return 5.5 + 140.0 * t ** 1.65
    if schedule == 1:
        return 7.0 + 250.0 * t ** 1.28
    if schedule == 2:
        if t < 0.58:
            return 8.0 + 58.0 * (t / 0.58) ** 1.40
        return 66.0 + 280.0 * ((t - 0.58) / 0.42) ** 2.0
    if t < 0.40:
        return 10.0
    return 10.0 + 360.0 * ((t - 0.40) / 0.60) ** 2.35


def _run_continuation(
    seed: np.ndarray,
    schedule: int,
    iterations: int,
    step0: float,
) -> np.ndarray:
    """
    Smooth basin search with periodic exact-objective recovery.  Returning
    to the best exact checkpoint prevents a late sharp surrogate from
    discarding a high-quality contact graph.
    """
    points = _normalize(seed)
    best = points.copy()
    best_value = _score(points)
    velocity = np.zeros_like(points)
    step_factor = 1.0
    regressions = 0
    damped = 0

    for k in range(iterations):
        t = k / max(iterations - 1, 1)
        direction = _contact_gradient(points, _sharpness(t, schedule))

        momentum = 0.80 if damped == 0 else 0.58
        velocity = momentum * velocity + (1.0 - momentum) * direction
        step = step_factor * step0 * (1.0 - 0.69 * t)
        points = _normalize(points + step * velocity)
        damped = max(0, damped - 1)

        if k % 12 == 11 or k == iterations - 1:
            value = _score(points)
            if value > best_value:
                best_value = value
                best = points.copy()
                regressions = 0
            elif value < best_value * (1.0 - 2.0e-4):
                regressions += 1
            else:
                regressions = max(0, regressions - 1)

            if regressions >= 3:
                points = best.copy()
                velocity.fill(0.0)
                step_factor *= 0.72
                damped = 70
                regressions = 0

    return best


def _layer_seed(radius: float, height: float, twist: float, pole: float) -> np.ndarray:
    angle = np.arange(6, dtype=float) * (np.pi / 3.0)
    lower = np.column_stack((
        radius * np.cos(angle),
        radius * np.sin(angle),
        -height * np.ones(6),
    ))
    upper = np.column_stack((
        radius * np.cos(angle + twist),
        radius * np.sin(angle + twist),
        height * np.ones(6),
    ))
    return np.vstack((lower, upper, [[0.0, 0.0, -pole], [0.0, 0.0, pole]]))


def _icosahedral_seed() -> np.ndarray:
    phi = 0.5 * (1.0 + np.sqrt(5.0))
    ico = np.array(
        [
            [-1.0, phi, 0.0], [1.0, phi, 0.0],
            [-1.0, -phi, 0.0], [1.0, -phi, 0.0],
            [0.0, -1.0, phi], [0.0, 1.0, phi],
            [0.0, -1.0, -phi], [0.0, 1.0, -phi],
            [phi, 0.0, -1.0], [phi, 0.0, 1.0],
            [-phi, 0.0, -1.0], [-phi, 0.0, 1.0],
        ],
        dtype=float,
    )
    extra = np.array([[0.0, 0.0, 1.75], [0.0, 0.0, -1.75]])
    return np.vstack((ico, extra))


def _initial_portfolio(rng: np.random.Generator) -> list:
    seeds = [
        _layer_seed(0.946, 0.405, np.pi / 6.0, 1.000),
        _layer_seed(0.920, 0.385, np.pi / 6.0, 0.980),
        _layer_seed(0.975, 0.430, np.pi / 6.0, 1.030),
        _layer_seed(0.950, 0.415, 0.455, 1.000),
        _layer_seed(0.955, 0.400, 0.585, 1.000),
        _icosahedral_seed(),
    ]

    cube = np.array(
        [[a, b, c] for a in (-1.0, 1.0)
         for b in (-1.0, 1.0)
         for c in (-1.0, 1.0)],
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

    starts = []
    for seed in seeds:
        starts.append(_normalize(seed + 0.025 * rng.standard_normal(seed.shape)))
        starts.append(_normalize(seed + 0.045 * rng.standard_normal(seed.shape)))

    for _ in range(8):
        starts.append(_normalize(rng.standard_normal((_N, 3))))

    return starts


def _polish(seed: np.ndarray) -> np.ndarray:
    """
    Exact-monotone local portfolio.  All candidate moves are scored with the
    evaluator-equivalent criterion, so this phase cannot reduce the result.
    """
    rng = np.random.default_rng(616271)
    best = _normalize(seed)
    best_value = _score(best)

    branches = (
        (180.0, 600.0, 0.0120, 0.00030, 330),
        (350.0, 1100.0, 0.0080, 0.00014, 390),
        (650.0, 1800.0, 0.0045, 0.00007, 400),
    )

    for begin, end, first_step, last_step, count in branches:
        points = best.copy()
        value = best_value

        for k in range(count):
            t = k / max(count - 1, 1)
            beta = begin * (end / begin) ** t
            step = first_step * (last_step / first_step) ** t
            direction = _contact_gradient(points, beta)

            scales = np.array([1.0, 0.55, 0.28, 0.12], dtype=float)
            trials = points[None, :, :] + step * scales[:, None, None] * direction

            if k % 19 == 0:
                noise = rng.standard_normal((2, _N, 3))
                noise -= noise.mean(axis=1, keepdims=True)
                norm = np.sqrt(np.mean(np.sum(noise * noise, axis=2), axis=1))
                noise /= norm[:, None, None]
                trials = np.concatenate((
                    trials,
                    points[None, :, :] + 0.16 * step * noise,
                ))

            trials = _normalize_batch(trials)
            values = _batch_score(trials)
            choice = int(np.argmax(values))
            if values[choice] > value + 1.0e-15:
                points = trials[choice]
                value = float(values[choice])

        if value > best_value:
            best = points
            best_value = value

    return best


def min_max_dist_dim3_14() -> np.ndarray:
    """Return exactly fourteen finite three-dimensional points."""
    rng = np.random.default_rng(917263)
    screened = []

    # Cheap broad basin screening over structured and unrestricted starts.
    for index, seed in enumerate(_initial_portfolio(rng)):
        candidate = _run_continuation(
            seed,
            index % 4,
            iterations=500,
            step0=0.047,
        )
        screened.append((_score(candidate), candidate))

    screened.sort(key=lambda item: item[0], reverse=True)

    # Spend the larger continuation budget only on promising contact graphs.
    finalists = []
    for rank, (_, seed) in enumerate(screened[:7]):
        for offset in range(2):
            candidate = _run_continuation(
                seed,
                (rank + offset) % 4,
                iterations=1050,
                step0=0.037 if offset == 0 else 0.031,
            )
            finalists.append((_score(candidate), candidate))

    finalists.sort(key=lambda item: item[0], reverse=True)
    best_points = finalists[0][1]

    # Two independent sharp local branches provide protection against a
    # single nearly-degenerate active-contact assignment.
    polished_a = _polish(best_points)
    polished_b = _polish(finalists[min(2, len(finalists) - 1)][1])

    if _score(polished_b) > _score(polished_a):
        polished_a = polished_b

    return np.asarray(polished_a, dtype=np.float64)


# EVOLVE-BLOCK-END