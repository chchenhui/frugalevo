# EVOLVE-BLOCK-START
import numpy as np


_N = 13
_FIXED = 3
_MOVABLE = 10
_EPS = 1.0e-10

# Unit-side equilateral reference triangle.  Its area is sqrt(3)/4.
_HULL = np.array(
    (
        (0.0, 0.0),
        (1.0, 0.0),
        (0.5, 0.86602540378443864676),
    ),
    dtype=np.float64,
)
_CENTER = _HULL.mean(axis=0)
_HULL_DOUBLE_AREA = abs(
    (_HULL[1, 0] - _HULL[0, 0]) * (_HULL[2, 1] - _HULL[0, 1])
    - (_HULL[1, 1] - _HULL[0, 1]) * (_HULL[2, 0] - _HULL[0, 0])
)

_TRIPLES = np.asarray(
    [
        (i, j, k)
        for i in range(_N)
        for j in range(i + 1, _N)
        for k in range(j + 1, _N)
    ],
    dtype=np.intp,
)

_INCIDENT = tuple(
    np.flatnonzero(np.any(_TRIPLES == i, axis=1))
    for i in range(_N)
)

_CACHE = None


def _areas(points: np.ndarray) -> np.ndarray:
    """Normalized triangle areas for one configuration."""
    tri = points[_TRIPLES]
    u = tri[:, 1] - tri[:, 0]
    v = tri[:, 2] - tri[:, 0]
    return np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]) / _HULL_DOUBLE_AREA


def _batch_areas(points: np.ndarray) -> np.ndarray:
    """Normalized triangle areas for a configuration batch."""
    tri = points[:, _TRIPLES]
    u = tri[:, :, 1] - tri[:, :, 0]
    v = tri[:, :, 2] - tri[:, :, 0]
    return np.abs(u[:, :, 0] * v[:, :, 1] - u[:, :, 1] * v[:, :, 0]) / _HULL_DOUBLE_AREA


def _tail_key(values: np.ndarray, count: int = 18) -> np.ndarray:
    """Lower-tail objective used only as a tie-breaker after the bottleneck."""
    low = np.partition(values, count - 1, axis=-1)[..., :count]
    low.sort(axis=-1)
    weights = np.linspace(1.0, 0.12, count, dtype=np.float64)
    return low @ weights


def _project_triangle(points: np.ndarray) -> np.ndarray:
    """
    Project Cartesian points into the equilateral hull through barycentric
    coordinates.  This is stable for both individual points and batches.
    """
    result = np.asarray(points, dtype=np.float64).copy()
    q = result - _HULL[0]
    # For q = b*(1,0) + c*(.5,sqrt(3)/2).
    c = q[..., 1] / _HULL[2, 1]
    b = q[..., 0] - 0.5 * c
    a = 1.0 - b - c

    bary = np.stack((a, b, c), axis=-1)
    bary = np.maximum(bary, 2.0e-5)
    bary /= bary.sum(axis=-1, keepdims=True)
    return bary[..., 1:2] * _HULL[1] + bary[..., 2:3] * _HULL[2]


def _orbit_points(params: np.ndarray) -> np.ndarray:
    """
    Decode compact orbit parameters.

    The first three values are radii and the next three are phases.  A small
    free displacement of the central point deliberately breaks the exact
    symmetry after the global orbit search has found a well-separated layout.
    """
    p = np.asarray(params, dtype=np.float64)
    batch = p.reshape(-1, 8)
    result = np.empty((batch.shape[0], _N, 2), dtype=np.float64)
    result[:, :3] = _HULL

    radii = np.clip(batch[:, :3], 0.07, 0.91)
    phases = batch[:, 3:6]
    center_shift = np.clip(batch[:, 6:8], -0.075, 0.075)

    # Three interior orbits, each containing three points.
    for ring in range(3):
        angles = phases[:, ring, None] + (2.0 * np.pi / 3.0) * np.arange(3)
        result[:, 3 + 3 * ring:6 + 3 * ring, 0] = (
            _CENTER[0] + radii[:, ring, None] * np.cos(angles)
        )
        result[:, 3 + 3 * ring:6 + 3 * ring, 1] = (
            _CENTER[1] + radii[:, ring, None] * np.sin(angles)
        )

    result[:, 12] = _CENTER + center_shift
    result[:, 3:] = _project_triangle(result[:, 3:])
    return result


def _orbit_score(params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = _orbit_points(params)
    values = _batch_areas(points)
    return values.min(axis=1), _tail_key(values)


def _global_orbit_search(rng: np.random.Generator) -> np.ndarray:
    """
    Elite-distribution search in an eight-dimensional orbit parameter space.

    Unlike coordinate differential evolution, generations are sampled from a
    covariance adapted to the actual elite geometry, which is useful because
    ring radius and angular phase changes are strongly coupled.
    """
    population = 96
    generations = 135

    mean = np.array(
        (0.25, 0.45, 0.68, 0.16, 0.69, 1.37, 0.0, 0.0),
        dtype=np.float64,
    )
    scale = np.array(
        (0.16, 0.16, 0.15, 0.55, 0.55, 0.55, 0.035, 0.035),
        dtype=np.float64,
    )
    covariance = np.diag(scale * scale)

    best = mean.copy()
    best_min = -np.inf
    best_tail = -np.inf

    for generation in range(generations):
        noise = rng.multivariate_normal(np.zeros(8), covariance, size=population)
        candidates = mean + noise

        # Every few generations retain broad phase diversity.
        if generation % 9 == 0:
            candidates[:18, :3] = rng.uniform(0.10, 0.86, size=(18, 3))
            candidates[:18, 3:6] = rng.uniform(0.0, 2.0 * np.pi / 3.0, size=(18, 3))
            candidates[:18, 6:] = rng.normal(0.0, 0.025, size=(18, 2))

        candidates[:, :3] = np.clip(candidates[:, :3], 0.06, 0.92)
        candidates[:, 3:6] %= (2.0 * np.pi / 3.0)
        candidates[:, 6:] = np.clip(candidates[:, 6:], -0.07, 0.07)

        minima, tails = _orbit_score(candidates)
        order = np.lexsort((tails, minima))
        winner = order[-1]

        if (
            minima[winner] > best_min + 1.0e-14
            or (
                abs(minima[winner] - best_min) <= 1.0e-14
                and tails[winner] > best_tail
            )
        ):
            best = candidates[winner].copy()
            best_min = float(minima[winner])
            best_tail = float(tails[winner])

        elite = candidates[order[-18:]]
        elite_mean = elite.mean(axis=0)

        # Circular phases are represented near their current mean before
        # covariance estimation, preventing a 0 / 2pi-period boundary jump.
        adjusted = elite.copy()
        for column in range(3, 6):
            delta = (adjusted[:, column] - elite_mean[column] + np.pi / 3.0) % (
                2.0 * np.pi / 3.0
            ) - np.pi / 3.0
            adjusted[:, column] = elite_mean[column] + delta

        centered = adjusted - adjusted.mean(axis=0)
        empirical = centered.T @ centered / max(1, len(elite) - 1)
        floor = np.diag(
            np.array(
                (0.010, 0.010, 0.010, 0.020, 0.020, 0.020, 0.0015, 0.0015)
            ) ** 2
        )
        covariance = 0.72 * covariance + 0.28 * empirical + floor
        mean = 0.64 * mean + 0.36 * elite_mean
        mean[3:6] %= (2.0 * np.pi / 3.0)

    return _orbit_points(best[None, :])[0]


def _incident_trials(
    points: np.ndarray,
    values: np.ndarray,
    index: int,
    locations: np.ndarray,
) -> np.ndarray:
    """Update only triangle constraints affected by moving one point."""
    count = locations.shape[0]
    result = np.broadcast_to(values, (count, values.size)).copy()
    ids = _INCIDENT[index]
    triples = _TRIPLES[ids]

    coords = np.broadcast_to(points[triples], (count, len(ids), 3, 2)).copy()
    where = np.where(triples[None, :, :, None] == index)
    coords[where[0], where[1], where[2]] = locations[where[0]]

    u = coords[:, :, 1] - coords[:, :, 0]
    v = coords[:, :, 2] - coords[:, :, 0]
    result[:, ids] = (
        np.abs(u[:, :, 0] * v[:, :, 1] - u[:, :, 1] * v[:, :, 0])
        / _HULL_DOUBLE_AREA
    )
    return result


def _local_bundle_polish(points: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Constraint-bundle coordinate polishing.

    Moves are selected from vertices in active small-area triangles; batches
    contain both isotropic perturbations and an altitude-increasing drift away
    from the opposite edge of a sampled active constraint.
    """
    points = points.copy()
    values = _areas(points)
    current_min = float(values.min())
    current_tail = float(_tail_key(values[None, :])[0])

    for sigma, rounds, batch in (
        (0.040, 150, 56),
        (0.018, 210, 64),
        (0.0070, 240, 72),
        (0.0024, 230, 80),
        (0.0008, 160, 84),
    ):
        for _ in range(rounds):
            active = np.argpartition(values, 17)[:18]
            tri_id = int(active[rng.integers(active.size)])
            tri = _TRIPLES[tri_id]
            movable = tri[tri >= _FIXED]
            index = int(movable[rng.integers(movable.size)])

            # Normal to the opposite edge, oriented to raise this active area.
            others = tri[tri != index]
            edge = points[others[1]] - points[others[0]]
            normal = np.array((-edge[1], edge[0]))
            signed = np.cross(edge, points[index] - points[others[0]])
            if signed < 0.0:
                normal = -normal
            norm = np.linalg.norm(normal)
            if norm > 1.0e-14:
                normal /= norm

            proposals = points[index] + sigma * rng.normal(size=(batch, 2))
            proposals += (
                normal[None, :]
                * sigma
                * rng.uniform(0.0, 1.6, size=(batch, 1))
            )
            proposals[: batch // 5] = points[index] + sigma * rng.normal(
                0.0, 1.8, size=(batch // 5, 2)
            )
            proposals = _project_triangle(proposals)

            trial_values = _incident_trials(points, values, index, proposals)
            minima = trial_values.min(axis=1)
            tails = _tail_key(trial_values)
            order = np.lexsort((tails, minima))
            winner = int(order[-1])

            if (
                minima[winner] > current_min + 1.0e-14
                or (
                    abs(minima[winner] - current_min) <= 1.0e-14
                    and tails[winner] > current_tail + 1.0e-14
                )
            ):
                points[index] = proposals[winner]
                values = trial_values[winner]
                current_min = float(minima[winner])
                current_tail = float(tails[winner])

    return points


def _fallback() -> np.ndarray:
    return np.array(
        (
            (0.0, 0.0),
            (1.0, 0.0),
            (0.5, 0.8660254037844386),
            (0.220, 0.110),
            (0.525, 0.120),
            (0.770, 0.105),
            (0.140, 0.340),
            (0.460, 0.300),
            (0.725, 0.355),
            (0.240, 0.590),
            (0.535, 0.570),
            (0.405, 0.730),
            (0.490, 0.435),
        ),
        dtype=np.float64,
    )


def _valid(points: np.ndarray) -> bool:
    if points.shape != (_N, 2) or not np.all(np.isfinite(points)):
        return False
    if _areas(points).min() <= 1.0e-9:
        return False
    q = points[3:] - _HULL[0]
    c = q[:, 1] / _HULL[2, 1]
    b = q[:, 0] - 0.5 * c
    a = 1.0 - b - c
    return bool(np.min(np.column_stack((a, b, c))) >= -1.0e-9)


def heilbronn_convex13() -> np.ndarray:
    """
    Return thirteen deterministic points inside an equilateral convex hull.

    The hull is fixed by the first three points; all remaining points are
    strictly projected into that hull.  Triangle areas are optimized relative
    to this hull's area, making the construction scale invariant.
    """
    global _CACHE

    if _CACHE is not None:
        return _CACHE.copy()

    rng = np.random.default_rng(13031957)
    try:
        points = _global_orbit_search(rng)
        points = _local_bundle_polish(points, rng)
        if not _valid(points):
            points = _fallback()
    except (FloatingPointError, ValueError, RuntimeError, np.linalg.LinAlgError):
        points = _fallback()

    _CACHE = np.asarray(points, dtype=np.float64)
    return _CACHE.copy()


# EVOLVE-BLOCK-END
