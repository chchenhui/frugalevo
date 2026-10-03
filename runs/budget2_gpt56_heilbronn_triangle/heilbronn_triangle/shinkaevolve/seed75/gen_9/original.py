# EVOLVE-BLOCK-START
import numpy as np


# The equilateral target triangle is represented by affine coordinates:
# (u, v) -> (u + 0.5*v, sqrt(3)/2 * v), with u >= 0, v >= 0, u + v <= 1.
_SQRT3_OVER_2 = np.sqrt(3.0) / 2.0
_N_POINTS = 11
_FIXED_CORNERS = np.array(
    [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ],
    dtype=float,
)
_TRIPLES = np.array(
    [
        (i, j, k)
        for i in range(_N_POINTS - 2)
        for j in range(i + 1, _N_POINTS - 1)
        for k in range(j + 1, _N_POINTS)
    ],
    dtype=np.intp,
)


def _project_to_simplex(points: np.ndarray) -> np.ndarray:
    """Map affine coordinates safely into the unit triangular simplex."""
    projected = np.abs(np.asarray(points, dtype=float)).copy()
    totals = projected[..., 0] + projected[..., 1]
    mask = totals > 1.0

    if np.any(mask):
        projected[mask] /= totals[mask, None]

    return np.clip(projected, 0.0, 1.0)


def _normalized_triangle_areas(population: np.ndarray) -> np.ndarray:
    """
    Return normalized triangle areas for each population member.

    In (u, v) affine coordinates, the determinant equals physical area
    divided by the area of the enclosing equilateral triangle.
    """
    all_points = np.concatenate(
        (
            np.broadcast_to(
                _FIXED_CORNERS,
                (population.shape[0], _FIXED_CORNERS.shape[0], 2),
            ),
            population,
        ),
        axis=1,
    )

    a = all_points[:, _TRIPLES[:, 0]]
    b = all_points[:, _TRIPLES[:, 1]]
    c = all_points[:, _TRIPLES[:, 2]]

    return np.abs(
        (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1])
        - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])
    )


def _fitness(population: np.ndarray) -> np.ndarray:
    """
    Score layouts primarily by their minimum area.

    A small contribution from the next-smallest areas breaks otherwise
    frequent ties and gives the evolutionary process a smoother direction.
    """
    areas = _normalized_triangle_areas(population)
    smallest = np.partition(areas, 5, axis=1)[:, :6]
    return smallest[:, 0] + 0.025 * np.mean(smallest[:, 1:], axis=1)


def _seed_population(rng: np.random.Generator, count: int) -> np.ndarray:
    """Create diversified initial layouts using jittered triangular samples."""
    population = np.empty((count, 8, 2), dtype=float)

    base = np.array(
        [
            [0.16, 0.08],
            [0.42, 0.06],
            [0.73, 0.08],
            [0.08, 0.34],
            [0.34, 0.28],
            [0.58, 0.25],
            [0.15, 0.62],
            [0.40, 0.48],
        ],
        dtype=float,
    )

    for index in range(count):
        jitter = rng.normal(0.0, 0.135, size=(8, 2))
        population[index] = _project_to_simplex(base + jitter)

    return population


def _evolve_layout() -> np.ndarray:
    """
    Deterministically search for a high-quality eleven point arrangement.

    The three enclosing-triangle corners are retained explicitly. Eight
    additional points are evolved in affine simplex coordinates.
    """
    rng = np.random.default_rng(11031987)

    population_size = 192
    elite_count = 28
    generations = 850

    population = _seed_population(rng, population_size)
    scores = _fitness(population)

    best_index = int(np.argmax(scores))
    best_layout = population[best_index].copy()
    best_score = float(scores[best_index])

    for generation in range(generations):
        ranking = np.argsort(scores)[::-1]
        elites = population[ranking[:elite_count]]

        progress = generation / max(generations - 1, 1)
        step = 0.105 * (1.0 - progress) + 0.006

        children = np.empty_like(population)
        children[:elite_count] = elites

        for child_index in range(elite_count, population_size):
            parent = elites[rng.integers(elite_count)].copy()

            if rng.random() < 0.32:
                ia, ib, ic = rng.integers(elite_count, size=3)
                differential = elites[ia] + 0.55 * (elites[ib] - elites[ic])
                parent = 0.55 * parent + 0.45 * differential

            changed = rng.random(8) < 0.48
            if not np.any(changed):
                changed[rng.integers(8)] = True

            parent[changed] += rng.normal(
                0.0,
                step,
                size=(int(np.sum(changed)), 2),
            )
            children[child_index] = _project_to_simplex(parent)

        population = children
        scores = _fitness(population)

        current_index = int(np.argmax(scores))
        current_score = float(scores[current_index])
        if current_score > best_score:
            best_score = current_score
            best_layout = population[current_index].copy()

    return best_layout


def _affine_to_cartesian(affine_points: np.ndarray) -> np.ndarray:
    """Convert unit-simplex affine coordinates to Cartesian coordinates."""
    result = np.empty_like(affine_points)
    result[:, 0] = affine_points[:, 0] + 0.5 * affine_points[:, 1]
    result[:, 1] = _SQRT3_OVER_2 * affine_points[:, 1]
    return result


def _fallback_layout() -> np.ndarray:
    """Provide a valid deterministic arrangement if optimization ever fails."""
    affine = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.16, 0.08],
            [0.45, 0.05],
            [0.76, 0.08],
            [0.09, 0.35],
            [0.35, 0.28],
            [0.62, 0.24],
            [0.15, 0.63],
            [0.41, 0.47],
        ],
        dtype=float,
    )
    return _affine_to_cartesian(affine)


def _build_cached_layout() -> np.ndarray:
    """Build and validate the cached Cartesian output once."""
    try:
        interior = _evolve_layout()
        affine = np.vstack((_FIXED_CORNERS, interior))
        result = _affine_to_cartesian(affine)

        if result.shape != (11, 2) or not np.all(np.isfinite(result)):
            return _fallback_layout()

        return result
    except Exception:
        return _fallback_layout()


_CACHED_POINTS = _build_cached_layout()


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct a deterministic arrangement of 11 points in or on the
    equilateral triangle with vertices (0,0), (1,0), and (0.5,sqrt(3)/2).

    Returns:
        np.ndarray: Cartesian point coordinates with shape (11, 2).
    """
    return _CACHED_POINTS.copy()


# EVOLVE-BLOCK-END
