# EVOLVE-BLOCK-START
import numpy as np


_SQRT3_OVER_2 = np.sqrt(3.0) * 0.5
_N_POINTS = 11
_N_FREE = 8

# Affine coordinates (u, v) map to Cartesian coordinates:
# (u + v/2, sqrt(3)*v/2), with u >= 0, v >= 0, u + v <= 1.
_FIXED_CORNERS = np.array(
    [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
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
    """Project affine points into the closed unit triangular simplex."""
    p = np.asarray(points, dtype=float).copy()
    p = np.maximum(p, 0.0)
    totals = p[..., 0] + p[..., 1]
    outside = totals > 1.0
    if np.any(outside):
        p[outside] /= totals[outside, None]
    return p


def _all_affine(free_points: np.ndarray) -> np.ndarray:
    """Append the three fixed enclosing-triangle corners."""
    if free_points.ndim == 2:
        return np.vstack((_FIXED_CORNERS, free_points))
    return np.concatenate(
        (
            np.broadcast_to(_FIXED_CORNERS, (free_points.shape[0], 3, 2)),
            free_points,
        ),
        axis=1,
    )


def _determinants(population: np.ndarray) -> np.ndarray:
    """Signed normalized areas for every triple of each population member."""
    points = _all_affine(population)
    a = points[:, _TRIPLES[:, 0]]
    b = points[:, _TRIPLES[:, 1]]
    c = points[:, _TRIPLES[:, 2]]
    return (
        (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1])
        - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])
    )


def _normalized_triangle_areas(population: np.ndarray) -> np.ndarray:
    """Normalized areas, where 1 equals the enclosing triangle's area."""
    return np.abs(_determinants(population))


def _min_areas(population: np.ndarray) -> np.ndarray:
    return np.min(_normalized_triangle_areas(population), axis=1)


def _fitness(population: np.ndarray) -> np.ndarray:
    """
    Lexicographic-like maximin fitness.

    The minimum area overwhelmingly dominates.  The mean of the next active
    constraints resolves otherwise nearly identical candidates more smoothly.
    """
    areas = _normalized_triangle_areas(population)
    low = np.partition(areas, 11, axis=1)[:, :12]
    return low[:, 0] + 0.0015 * np.mean(low[:, 1:], axis=1)


def _project_direction(point: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Remove outward components when a point lies on a simplex boundary."""
    d = direction.copy()
    eps = 2.0e-10
    if point[0] <= eps and d[0] < 0.0:
        d[0] = 0.0
    if point[1] <= eps and d[1] < 0.0:
        d[1] = 0.0
    if point[0] + point[1] >= 1.0 - eps and d[0] + d[1] > 0.0:
        excess = 0.5 * (d[0] + d[1])
        d -= excess
    return d


def _active_gradient(layout: np.ndarray) -> np.ndarray:
    """
    Gradient ascent direction for the currently tight determinant constraints.

    For each of the low-area triples, signed determinant gradients are
    accumulated for its movable vertices.  Exponentially decaying weights
    retain information from several nearly active constraints rather than
    overfitting to a single limiting triple.
    """
    points = _all_affine(layout)
    a = points[_TRIPLES[:, 0]]
    b = points[_TRIPLES[:, 1]]
    c = points[_TRIPLES[:, 2]]

    det = (
        (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
        - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    )
    areas = np.abs(det)
    minimum = float(np.min(areas))

    # A small active band is important near an optimum, where many triples
    # should be nearly equal.
    band = max(0.0022, minimum * 0.16)
    active = areas <= minimum + band
    weights = np.exp(-(areas[active] - minimum) / max(band * 0.42, 1.0e-8))
    signs = np.where(det[active] >= 0.0, 1.0, -1.0)

    triples = _TRIPLES[active]
    aa = a[active]
    bb = b[active]
    cc = c[active]

    grad_a = np.column_stack((bb[:, 1] - cc[:, 1], cc[:, 0] - bb[:, 0]))
    grad_b = np.column_stack((cc[:, 1] - aa[:, 1], aa[:, 0] - cc[:, 0]))
    grad_c = np.column_stack((aa[:, 1] - bb[:, 1], bb[:, 0] - aa[:, 0]))

    direction = np.zeros((_N_FREE, 2), dtype=float)
    for vertex_indices, gradients in (
        (triples[:, 0], grad_a),
        (triples[:, 1], grad_b),
        (triples[:, 2], grad_c),
    ):
        movable = vertex_indices >= 3
        if np.any(movable):
            np.add.at(
                direction,
                vertex_indices[movable] - 3,
                gradients[movable] * (weights[movable] * signs[movable])[:, None],
            )

    for i in range(_N_FREE):
        direction[i] = _project_direction(layout[i], direction[i])

    norm = float(np.max(np.linalg.norm(direction, axis=1)))
    if norm > 1.0e-14:
        direction /= norm
    return direction


def _polish_layout(layout: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Use active-triple determinant gradients and batched local proposals."""
    current = _project_to_simplex(layout)
    current_score = float(_fitness(current[None, ...])[0])
    current_min = float(_min_areas(current[None, ...])[0])

    schedules = (
        (0.026, 80),
        (0.012, 110),
        (0.0055, 140),
        (0.0022, 160),
        (0.00085, 170),
    )

    for step, rounds in schedules:
        for _ in range(rounds):
            direction = _active_gradient(current)
            batch = 72
            trials = np.broadcast_to(current, (batch, _N_FREE, 2)).copy()

            # Deterministic directed ascent steps of several lengths.
            directed_count = 18
            alphas = np.linspace(-0.30 * step, 2.6 * step, directed_count)
            trials[:directed_count] += alphas[:, None, None] * direction

            # Per-point active-gradient variants handle conflicting local
            # constraints better than a single fully coupled displacement.
            for q in range(18, 42):
                p = (q - 18) % _N_FREE
                trials[q, p] += direction[p] * step * (1.1 + 0.18 * ((q - 18) // 8))
                if q % 3 == 0:
                    other = (p + 1 + q) % _N_FREE
                    trials[q, other] += direction[other] * step * 0.55

            # Small noisy directed proposals prevent stagnation at nonsmooth
            # intersections of active area constraints.
            noise = rng.normal(0.0, step * 0.34, size=(batch - 42, _N_FREE, 2))
            masks = rng.random((batch - 42, _N_FREE, 1)) < 0.34
            trials[42:] += masks * noise
            trials[42:] += direction[None, :, :] * rng.uniform(
                0.15 * step, 1.55 * step, size=(batch - 42, 1, 1)
            )

            trials = _project_to_simplex(trials)
            trial_min = _min_areas(trials)
            trial_score = _fitness(trials)

            best = int(np.argmax(trial_score))
            # Permit secondary-score tie breaking only at numerical equality
            # of the actual limiting area.
            if (
                trial_min[best] > current_min + 2.0e-12
                or (
                    trial_min[best] >= current_min - 2.0e-12
                    and trial_score[best] > current_score + 1.0e-13
                )
            ):
                current = trials[best]
                current_min = float(trial_min[best])
                current_score = float(trial_score[best])

    return current


def _epigraph_cell_polish(start: np.ndarray) -> np.ndarray:
    """
    Maximize the actual minimum normalized area in the local orientation cell.

    The stochastic stages locate a favorable combinatorial layout.  Freezing
    determinant signs makes every absolute-area condition locally smooth:
    sign(det_i) * det_i >= t.  SLSQP can then balance the tight triangles much
    more accurately than sampled displacement proposals.
    """
    try:
        from scipy.optimize import minimize
    except Exception:
        return start

    best = _project_to_simplex(start)
    best_value = float(_min_areas(best[None, ...])[0])

    for phase in range(3):
        signed = _determinants(best[None, ...])[0]
        signs = np.where(signed >= 0.0, 1.0, -1.0)
        initial = np.concatenate((best.ravel(), [best_value]))

        def constraints(w: np.ndarray) -> np.ndarray:
            free = w[:-1].reshape(_N_FREE, 2)
            det = _determinants(free[None, ...])[0]
            return np.concatenate(
                (
                    signs * det - w[-1],
                    free.ravel(),
                    1.0 - np.sum(free, axis=1),
                )
            )

        try:
            result = minimize(
                lambda w: -w[-1],
                initial,
                method="SLSQP",
                bounds=[(0.0, 1.0)] * (2 * _N_FREE) + [(0.0, 0.2)],
                constraints={"type": "ineq", "fun": constraints},
                options={"maxiter": 2200, "ftol": 1.0e-13, "disp": False},
            )
        except Exception:
            break

        if not result.success or not np.all(np.isfinite(result.x)):
            break

        candidate = _project_to_simplex(result.x[:-1].reshape(_N_FREE, 2))
        value = float(_min_areas(candidate[None, ...])[0])
        if value > best_value + 1.0e-10:
            best = candidate
            best_value = value
        else:
            break

    return best


def _seed_population(rng: np.random.Generator, count: int) -> np.ndarray:
    """Produce deterministic diverse triangular layouts."""
    population = np.empty((count, _N_FREE, 2), dtype=float)
    base = np.array(
        [
            [0.11, 0.07],
            [0.38, 0.055],
            [0.69, 0.07],
            [0.08, 0.34],
            [0.34, 0.25],
            [0.62, 0.23],
            [0.14, 0.63],
            [0.40, 0.46],
        ],
        dtype=float,
    )

    population[0] = base
    for i in range(1, count):
        if i % 5 == 0:
            raw = rng.exponential(1.0, size=(_N_FREE, 3))
            bary = raw / np.sum(raw, axis=1, keepdims=True)
            population[i] = bary[:, 1:3]
        else:
            population[i] = _project_to_simplex(
                base + rng.normal(0.0, 0.115, size=(_N_FREE, 2))
            )
    return population


def _evolve_layout() -> np.ndarray:
    """Deterministic global search followed by directed active-set polishing."""
    rng = np.random.default_rng(11031987)
    population_size = 240
    elite_count = 32
    generations = 680

    population = _seed_population(rng, population_size)
    scores = _fitness(population)
    best = population[int(np.argmax(scores))].copy()
    best_score = float(np.max(scores))

    for generation in range(generations):
        order = np.argsort(scores)[::-1]
        elites = population[order[:elite_count]]
        children = np.empty_like(population)
        children[:elite_count] = elites

        t = generation / max(1, generations - 1)
        mutation = 0.085 * (1.0 - t) + 0.0045

        for i in range(elite_count, population_size):
            parent = elites[rng.integers(elite_count)].copy()

            if rng.random() < 0.48:
                ia, ib, ic = rng.integers(elite_count, size=3)
                parent = elites[ia] + 0.62 * (elites[ib] - elites[ic])

            changed = rng.random(_N_FREE) < (0.58 - 0.22 * t)
            if not np.any(changed):
                changed[rng.integers(_N_FREE)] = True

            parent[changed] += rng.normal(
                0.0, mutation, size=(int(np.sum(changed)), 2)
            )
            children[i] = _project_to_simplex(parent)

        population = children
        scores = _fitness(population)
        index = int(np.argmax(scores))
        if scores[index] > best_score:
            best_score = float(scores[index])
            best = population[index].copy()

    # Polish several top layouts: occasionally a slightly lower evolutionary
    # score lies in a more favorable active-constraint basin.
    final_order = np.argsort(scores)[::-1]
    candidates = [best]
    candidates.extend(population[i].copy() for i in final_order[:5])

    polished = [_polish_layout(candidate, rng) for candidate in candidates]
    values = np.array([_min_areas(x[None, ...])[0] for x in polished])

    # Spend the more expensive exact constrained solve only on the strongest
    # distinct lower-tail basins.  Ranking here is strictly by raw bottleneck,
    # rather than the evolutionary surrogate, so the final decision directly
    # targets the evaluation metric.
    strongest = np.argsort(values)[::-1][:3]
    for index in strongest:
        candidate = _epigraph_cell_polish(polished[int(index)])
        candidate_value = float(_min_areas(candidate[None, ...])[0])
        if candidate_value > values[int(index)]:
            polished[int(index)] = candidate
            values[int(index)] = candidate_value

    return polished[int(np.argmax(values))]


def _affine_to_cartesian(affine_points: np.ndarray) -> np.ndarray:
    result = np.empty_like(affine_points)
    result[:, 0] = affine_points[:, 0] + 0.5 * affine_points[:, 1]
    result[:, 1] = _SQRT3_OVER_2 * affine_points[:, 1]
    return result


def _fallback_layout() -> np.ndarray:
    affine = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.11, 0.07],
            [0.38, 0.055],
            [0.69, 0.07],
            [0.08, 0.34],
            [0.34, 0.25],
            [0.62, 0.23],
            [0.14, 0.63],
            [0.40, 0.46],
        ],
        dtype=float,
    )
    return _affine_to_cartesian(affine)


def _build_cached_layout() -> np.ndarray:
    """Build once and retain a validated deterministic Cartesian arrangement."""
    try:
        free = _evolve_layout()
        affine = np.vstack((_FIXED_CORNERS, _project_to_simplex(free)))
        result = _affine_to_cartesian(affine)

        if (
            result.shape != (11, 2)
            or not np.all(np.isfinite(result))
            or np.min(_normalized_triangle_areas(free[None, ...])) <= 0.0
        ):
            return _fallback_layout()
        return result
    except Exception:
        return _fallback_layout()


_CACHED_POINTS = _build_cached_layout()


def heilbronn_triangle11() -> np.ndarray:
    """
    Return a deterministic 11-point arrangement in the specified equilateral
    triangle. The returned array has Cartesian shape (11, 2).
    """
    return _CACHED_POINTS.copy()


# EVOLVE-BLOCK-END