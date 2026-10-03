# EVOLVE-BLOCK-START
import numpy as np


# The three fixed vertices define the convex reference region.  Its area is 1/2,
# so |cross product| is exactly the corresponding normalized triangle area.
_HULL_VERTICES = np.array(
    [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ],
    dtype=np.float64,
)

_TRIANGLES = np.asarray(
    [
        (i, j, k)
        for i in range(13)
        for j in range(i + 1, 13)
        for k in range(j + 1, 13)
    ],
    dtype=np.intp,
)

_CACHED_POINTS = None


def _decode_parameters(parameters: np.ndarray) -> np.ndarray:
    """
    Convert unconstrained optimization parameters into points strictly inside
    the fixed triangular convex region using barycentric softmax coordinates.
    """
    values = np.asarray(parameters, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]

    logits = values.reshape(values.shape[0], 10, 2)
    logits = np.concatenate(
        (logits, np.zeros((logits.shape[0], 10, 1), dtype=np.float64)),
        axis=2,
    )
    logits -= np.max(logits, axis=2, keepdims=True)
    weights = np.exp(logits)
    weights /= np.sum(weights, axis=2, keepdims=True)

    # Barycentric weights correspond to vertices:
    # weight 0 -> (0,0), weight 1 -> (1,0), weight 2 -> (0,1).
    interior = np.stack((weights[:, :, 1], weights[:, :, 2]), axis=2)
    hull = np.broadcast_to(_HULL_VERTICES, (values.shape[0], 3, 2))
    return np.concatenate((hull, interior), axis=1)


def _population_scores(parameters: np.ndarray) -> np.ndarray:
    """
    Return the normalized area of the smallest triangle for every candidate.
    Evaluation is fully vectorized over the candidate population.
    """
    points = _decode_parameters(parameters)
    triples = points[:, _TRIANGLES, :]
    first = triples[:, :, 1, :] - triples[:, :, 0, :]
    second = triples[:, :, 2, :] - triples[:, :, 0, :]
    doubled_areas = np.abs(first[:, :, 0] * second[:, :, 1] - first[:, :, 1] * second[:, :, 0])
    return np.min(doubled_areas, axis=1)


def _initial_population(rng: np.random.Generator, size: int) -> np.ndarray:
    """
    Create a mixed low-discrepancy / random initial population in logit space.
    The structured component avoids spending the entire search near clustered
    random configurations.
    """
    dimension = 20
    population = rng.normal(0.0, 1.45, size=(size, dimension))

    # Deterministic stratified perturbations produce broadly distributed
    # barycentric coordinates while retaining diverse local arrangements.
    for row in range(size // 3):
        phase = (row + 0.5) / max(1, size // 3)
        pattern = np.arange(10, dtype=np.float64)
        a = (pattern * 0.6180339887498949 + phase) % 1.0
        b = (pattern * 0.4142135623730950 + 0.37 * phase) % 1.0
        population[row, 0::2] = 3.0 * (a - 0.5) + rng.normal(0.0, 0.35, 10)
        population[row, 1::2] = 3.0 * (b - 0.5) + rng.normal(0.0, 0.35, 10)

    return np.clip(population, -6.0, 6.0)


def _differential_evolution(rng: np.random.Generator) -> np.ndarray:
    """
    Global deterministic search using one-to-one DE selection.  All expensive
    geometry calls operate on batches rather than individual candidates.
    """
    population_size = 72
    generations = 420
    dimension = 20

    population = _initial_population(rng, population_size)
    scores = _population_scores(population)

    for generation in range(generations):
        donors = np.empty_like(population)
        for index in range(population_size):
            choices = np.arange(population_size)
            choices = choices[choices != index]
            a, b, c = rng.choice(choices, size=3, replace=False)
            scale = 0.72 if generation < generations * 0.65 else 0.48
            donors[index] = population[a] + scale * (population[b] - population[c])

        donors = np.clip(donors, -7.0, 7.0)
        mask = rng.random((population_size, dimension)) < 0.84
        forced = rng.integers(0, dimension, size=population_size)
        mask[np.arange(population_size), forced] = True
        trials = np.where(mask, donors, population)

        trial_scores = _population_scores(trials)
        accepted = trial_scores > scores
        population[accepted] = trials[accepted]
        scores[accepted] = trial_scores[accepted]

    return population[np.argmax(scores)].copy()


def _local_refinement(seed: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Batched stochastic hill climbing around the best global candidate.
    Strict improvement only keeps the result deterministic and monotonic.
    """
    best = seed.copy()
    best_score = float(_population_scores(best)[0])

    for stage, sigma in enumerate((0.32, 0.18, 0.095, 0.045, 0.020, 0.009)):
        iterations = 45 if stage < 3 else 65
        for _ in range(iterations):
            candidates = best[None, :] + rng.normal(0.0, sigma, size=(40, best.size))
            candidates = np.clip(candidates, -7.0, 7.0)
            scores = _population_scores(candidates)
            winner = int(np.argmax(scores))
            if scores[winner] > best_score:
                best = candidates[winner]
                best_score = float(scores[winner])

    return best


def _valid_configuration(points: np.ndarray) -> bool:
    """Lightweight integrity verification before caching the result."""
    if points.shape != (13, 2) or not np.all(np.isfinite(points)):
        return False
    if np.min(np.linalg.norm(points[:, None] - points[None, :], axis=2) + np.eye(13)) < 1e-10:
        return False
    return float(_population_scores(_encode_for_validation(points))[0]) > 0.0


def _encode_for_validation(points: np.ndarray) -> np.ndarray:
    """
    Convert already valid interior coordinates to logits solely for the final
    validation path.  Hull coordinates are omitted because they are fixed.
    """
    interior = np.clip(points[3:], 1e-12, 1.0)
    w1 = interior[:, 0]
    w2 = interior[:, 1]
    w0 = np.clip(1.0 - w1 - w2, 1e-12, 1.0)
    return np.column_stack((np.log(w1 / w0), np.log(w2 / w0))).reshape(1, -1)


def heilbronn_convex13() -> np.ndarray:
    """
    Construct a deterministic 13-point configuration inside a convex triangle.

    Returns
    -------
    np.ndarray
        Array of shape (13, 2).  The first three entries are convex-hull
        vertices and all remaining points lie strictly inside that hull.
    """
    global _CACHED_POINTS

    if _CACHED_POINTS is not None:
        return _CACHED_POINTS.copy()

    rng = np.random.default_rng(20240513)
    global_best = _differential_evolution(rng)
    refined = _local_refinement(global_best, rng)
    points = _decode_parameters(refined)[0]

    if not _valid_configuration(points):
        # Graceful deterministic fallback: still a valid nondegenerate
        # triangular-region arrangement if an unexpected numerical failure occurs.
        fallback = np.array(
            [
                [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
                [0.16, 0.12], [0.38, 0.10], [0.66, 0.11],
                [0.11, 0.35], [0.34, 0.31], [0.58, 0.29],
                [0.12, 0.64], [0.31, 0.53], [0.50, 0.43],
                [0.22, 0.74],
            ],
            dtype=np.float64,
        )
        points = fallback

    _CACHED_POINTS = np.asarray(points, dtype=np.float64)
    return _CACHED_POINTS.copy()


# EVOLVE-BLOCK-END