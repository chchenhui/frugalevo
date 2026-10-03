# EVOLVE-BLOCK-START
import numpy as np


_N = 13
_FIXED = 4
_MOVABLE = 9
_EPS = 1.0e-8

_CORNERS = np.array(
    (
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
    ),
    dtype=np.float64,
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
    np.flatnonzero(np.any(_TRIPLES == point, axis=1))
    for point in range(_N)
)

_CACHE = None


def _double_areas(points: np.ndarray) -> np.ndarray:
    """Return absolute triangle determinants for one point configuration."""
    triangles = points[_TRIPLES]
    u = triangles[:, 1] - triangles[:, 0]
    v = triangles[:, 2] - triangles[:, 0]
    return np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])


def _batch_incident_areas(
    points: np.ndarray,
    old_areas: np.ndarray,
    point_index: int,
    locations: np.ndarray,
) -> np.ndarray:
    """
    Evaluate a batch of moves using only triangles incident to the moved point.

    All unaffected triangle determinants are copied from old_areas, avoiding
    repeated full recomputation during the local search.
    """
    count = locations.shape[0]
    result = np.broadcast_to(old_areas, (count, old_areas.size)).copy()

    triangle_ids = _INCIDENT[point_index]
    triples = _TRIPLES[triangle_ids]

    coords = points[triples][None, :, :, :].repeat(count, axis=0)
    positions = np.where(triples[None, :, :, None] == point_index)
    coords[positions[0], positions[1], positions[2]] = locations[positions[0]]

    u = coords[:, :, 1] - coords[:, :, 0]
    v = coords[:, :, 2] - coords[:, :, 0]
    result[:, triangle_ids] = np.abs(u[:, :, 0] * v[:, :, 1] - u[:, :, 1] * v[:, :, 0])
    return result


def _tail_key(areas: np.ndarray, count: int = 16) -> np.ndarray:
    """
    A floor-oriented scalar objective.

    The true minimum is dominant, while the remaining lower tail prevents
    moves which merely replace one limiting degenerate triangle by another.
    """
    low = np.partition(areas, count - 1, axis=1)[:, :count]
    low.sort(axis=1)
    weights = np.linspace(1.0, 0.10, count, dtype=np.float64)
    return low @ weights


def _soft_floor(areas: np.ndarray, width: float) -> np.ndarray:
    """Stable soft-minimum used during exploratory search."""
    minimum = np.min(areas, axis=1)
    return minimum - width * np.log(
        np.exp(-(areas - minimum[:, None]) / width).sum(axis=1)
    )


def _gradient_for_point(
    points: np.ndarray,
    areas: np.ndarray,
    point_index: int,
    active_count: int = 18,
) -> np.ndarray:
    """
    Compute a geometry-aware ascent direction for active triangle areas.

    For every small incident triangle, the determinant gradient supplies the
    normal to the opposite edge.  Their weighted combination directly raises
    the altitude of the selected point relative to limiting edges.
    """
    active = np.argpartition(areas, active_count - 1)[:active_count]
    direction = np.zeros(2, dtype=np.float64)
    threshold = float(np.partition(areas, active_count - 1)[active_count - 1])

    for triangle_id in active:
        tri = _TRIPLES[triangle_id]
        where = np.flatnonzero(tri == point_index)
        if where.size == 0:
            continue

        a, b, c = points[tri]
        signed = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        sign = 1.0 if signed >= 0.0 else -1.0

        role = int(where[0])
        if role == 0:
            gradient = np.array((b[1] - c[1], c[0] - b[0]))
        elif role == 1:
            gradient = np.array((c[1] - a[1], a[0] - c[0]))
        else:
            gradient = np.array((a[1] - b[1], b[0] - a[0]))

        weight = 1.0 / max(areas[triangle_id], 0.25 * threshold, 1.0e-4)
        direction += weight * sign * gradient

    norm = float(np.linalg.norm(direction))
    return direction / norm if norm > 1.0e-14 else direction


def _initial_configuration(rng: np.random.Generator, restart: int) -> np.ndarray:
    """Build a stratified but deliberately non-collinear starting arrangement."""
    points = np.empty((_N, 2), dtype=np.float64)
    points[:_FIXED] = _CORNERS

    if restart % 3 == 0:
        base = np.array(
            (
                (0.16, 0.15), (0.47, 0.12), (0.79, 0.19),
                (0.13, 0.47), (0.52, 0.42), (0.84, 0.53),
                (0.23, 0.79), (0.54, 0.83), (0.76, 0.72),
            ),
            dtype=np.float64,
        )
        noise = rng.normal(0.0, 0.075, size=base.shape)
        points[_FIXED:] = base + noise
    elif restart % 3 == 1:
        order = rng.permutation(_MOVABLE)
        cells = np.column_stack((order % 3, order // 3)).astype(np.float64)
        points[_FIXED:] = (cells + rng.uniform(0.14, 0.86, size=(9, 2))) / 3.0
    else:
        points[_FIXED:] = rng.uniform(0.075, 0.925, size=(_MOVABLE, 2))

    points[_FIXED:] = np.clip(points[_FIXED:], 0.035, 0.965)
    return points


def _choose_point(rng: np.random.Generator, areas: np.ndarray) -> int:
    """Select a movable point biased toward active constraints."""
    if rng.random() < 0.84:
        active = np.argpartition(areas, 13)[:14]
        tri = _TRIPLES[active[rng.integers(active.size)]]
        movable = tri[tri >= _FIXED]
        if movable.size:
            return int(movable[rng.integers(movable.size)])
    return int(rng.integers(_FIXED, _N))


def _run_restart(rng: np.random.Generator, restart: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Two-stage active constraint search.

    The first stage accepts soft-floor improvements and occasional annealed
    moves.  The second stage uses a strict lexicographic maximin objective.
    """
    points = _initial_configuration(rng, restart)
    areas = _double_areas(points)
    best_points = points.copy()
    best_areas = areas.copy()

    exploratory_steps = 600
    batch_size = 40

    for iteration in range(exploratory_steps):
        progress = iteration / max(1, exploratory_steps - 1)
        point_index = _choose_point(rng, areas)
        center = points[point_index]
        gradient = _gradient_for_point(points, areas, point_index)

        step = 0.090 * (1.0 - progress) + 0.008
        noise = rng.normal(0.0, 1.0, size=(batch_size, 2))
        drift = gradient[None, :] * rng.uniform(0.25, 1.45, size=(batch_size, 1))

        proposals = center + step * (0.72 * noise + 0.90 * drift)
        proposals[: batch_size // 5] = center + step * rng.normal(
            0.0, 1.65, size=(batch_size // 5, 2)
        )
        proposals = np.clip(proposals, 0.026, 0.974)

        trial_areas = _batch_incident_areas(points, areas, point_index, proposals)
        width = 0.020 * (1.0 - progress) + 0.0030
        scores = _soft_floor(trial_areas, width)
        winner = int(np.argmax(scores))

        current = float(_soft_floor(areas[None, :], width)[0])
        candidate = float(scores[winner])
        temperature = 0.0028 * (1.0 - progress) ** 2 + 1.0e-5

        if candidate >= current or rng.random() < np.exp((candidate - current) / temperature):
            points[point_index] = proposals[winner]
            areas = trial_areas[winner]

        if float(areas.min()) > float(best_areas.min()):
            best_points = points.copy()
            best_areas = areas.copy()

    points = best_points
    areas = best_areas

    for scale, rounds, polish_batch in (
        (0.020, 260, 56),
        (0.008, 320, 64),
        (0.0028, 360, 72),
        (0.0009, 260, 72),
    ):
        current_key = _tail_key(areas[None, :])[0]

        for _ in range(rounds):
            point_index = _choose_point(rng, areas)
            center = points[point_index]
            gradient = _gradient_for_point(points, areas, point_index, active_count=24)

            proposals = center + scale * (
                0.64 * rng.normal(size=(polish_batch, 2))
                + 1.25 * gradient[None, :] * rng.uniform(
                    -0.10, 1.15, size=(polish_batch, 1)
                )
            )
            proposals = np.clip(proposals, 0.022, 0.978)

            trial_areas = _batch_incident_areas(points, areas, point_index, proposals)
            minima = trial_areas.min(axis=1)
            tail = _tail_key(trial_areas)
            order = np.lexsort((tail, minima))
            winner = int(order[-1])

            candidate_key = float(tail[winner])
            if (
                minima[winner] > areas.min() + 1.0e-14
                or (
                    abs(minima[winner] - areas.min()) <= 1.0e-14
                    and candidate_key >= current_key
                )
            ):
                points[point_index] = proposals[winner]
                areas = trial_areas[winner]
                current_key = candidate_key

    return points, areas


def _fallback() -> np.ndarray:
    """A deterministic valid arrangement used only if optimization fails."""
    return np.array(
        (
            (0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0),
            (0.154, 0.143), (0.492, 0.107), (0.814, 0.204),
            (0.105, 0.455), (0.518, 0.416), (0.867, 0.569),
            (0.225, 0.798), (0.568, 0.851), (0.748, 0.724),
        ),
        dtype=np.float64,
    )


def heilbronn_convex13() -> np.ndarray:
    """
    Deterministically construct thirteen points in the unit square.

    Returns
    -------
    np.ndarray
        Array of shape (13, 2). The four first points are square corners;
        every remaining point is strictly inside the same convex unit square.
    """
    global _CACHE

    if _CACHE is not None:
        return _CACHE.copy()

    rng = np.random.default_rng(13031957)

    try:
        winner = None
        winner_score = -np.inf

        for restart in range(8):
            candidate, areas = _run_restart(rng, restart)
            score = float(areas.min())

            if score > winner_score:
                winner = candidate
                winner_score = score

        if (
            winner is None
            or winner.shape != (_N, 2)
            or not np.all(np.isfinite(winner))
            or np.any(winner < -_EPS)
            or np.any(winner > 1.0 + _EPS)
            or _double_areas(winner).min() <= 0.0
        ):
            winner = _fallback()

    except (FloatingPointError, ValueError, RuntimeError):
        winner = _fallback()

    _CACHE = np.asarray(winner, dtype=np.float64)
    return _CACHE.copy()


# EVOLVE-BLOCK-END
