# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """
    Deterministically construct eleven points in the reference equilateral
    triangle.  Internal coordinates are normalized simplex coordinates:

        (u, v) -> u * (1, 0) + v * (1/2, sqrt(3)/2)

    Hence every determinant in (u, v) coordinates is exactly the associated
    triangle area normalized by the enclosing triangle area.
    """
    n = 11
    height = np.sqrt(3.0) * 0.5
    rng = np.random.default_rng(381104729)

    triples = np.array(
        [(i, j, k)
         for i in range(n)
         for j in range(i + 1, n)
         for k in range(j + 1, n)],
        dtype=np.intp,
    )
    affected = [
        np.flatnonzero(np.any(triples == point, axis=1))
        for point in range(n)
    ]

    anchors = np.array(
        ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
        dtype=float,
    )

    def determinants(points: np.ndarray) -> np.ndarray:
        a = points[triples[:, 0]]
        b = points[triples[:, 1]]
        c = points[triples[:, 2]]
        return ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))

    def all_areas(points: np.ndarray) -> np.ndarray:
        return np.abs(determinants(points))

    def changed_areas(points: np.ndarray, indices: np.ndarray) -> np.ndarray:
        t = triples[indices]
        a = points[t[:, 0]]
        b = points[t[:, 1]]
        c = points[t[:, 2]]
        return np.abs(
            (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
            - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        )

    def rank_value(values: np.ndarray, progress: float) -> tuple[float, float]:
        """
        Adaptive tail ranking.  Early broad tails encourage globally robust
        spacing; late narrow tails resolve the active maximin constraints.
        """
        count = int(round(28.0 - 20.0 * progress))
        count = max(7, min(count, len(values)))
        low = np.partition(values, count - 1)[:count]
        minimum = float(low[0])
        weight = 0.028 + 0.070 * progress
        return minimum, float(minimum + weight * low.mean())

    def project_weights(weights: np.ndarray) -> np.ndarray:
        weights = np.maximum(weights, 1.0e-10)
        return weights / weights.sum()

    def initial_points() -> np.ndarray:
        points = np.empty((n, 2), dtype=float)
        points[:3] = anchors
        bary = rng.dirichlet((0.95, 0.95, 0.95), size=8)
        points[3:, 0] = bary[:, 1]
        points[3:, 1] = bary[:, 2]
        return points

    best = None
    best_minimum = -1.0
    best_rank = -1.0

    # Moderate independent trajectories plus a stronger final epigraph solve
    # are more efficient than spending the entire budget in random walks.
    restarts = 14
    iterations = 23000

    for restart in range(restarts):
        points = initial_points()
        area_values = all_areas(points)
        current_minimum, current_rank = rank_value(area_values, 0.0)

        if current_minimum > best_minimum:
            best = points.copy()
            best_minimum = current_minimum
            best_rank = current_rank

        for iteration in range(iterations):
            progress = iteration / float(iterations - 1)
            move_scale = 0.102 * (1.0 - progress) ** 1.72 + 0.00032
            temperature = 0.0048 * (1.0 - progress) ** 2.15 + 1.0e-7

            if rng.random() < 0.74:
                active_count = int(round(22.0 - 16.0 * progress))
                active_count = max(5, active_count)
                active = np.argpartition(area_values, active_count - 1)[:active_count]
                triangle = triples[int(active[rng.integers(active_count)])]
                movable = triangle[triangle >= 3]
                if len(movable):
                    point_index = int(movable[rng.integers(len(movable))])
                else:
                    point_index = int(rng.integers(3, n))
            else:
                point_index = int(rng.integers(3, n))

            changed = affected[point_index]
            old_position = points[point_index].copy()
            old_values = area_values[changed].copy()
            old_weights = np.array(
                (1.0 - old_position[0] - old_position[1],
                 old_position[0],
                 old_position[1]),
                dtype=float,
            )

            if progress < 0.60 and rng.random() < 0.12:
                target = rng.dirichlet((1.0, 1.0, 1.0))
                blend = 0.16 * (1.0 - progress) + 0.012
                weights = project_weights((1.0 - blend) * old_weights + blend * target)
            else:
                weights = project_weights(
                    old_weights + rng.normal(0.0, move_scale, size=3)
                )

            points[point_index] = weights[1:]
            area_values[changed] = changed_areas(points, changed)
            candidate_minimum, candidate_rank = rank_value(area_values, progress)
            delta = candidate_rank - current_rank

            if delta >= 0.0 or rng.random() < np.exp(max(-55.0, delta / temperature)):
                current_minimum = candidate_minimum
                current_rank = candidate_rank
                if (candidate_minimum > best_minimum + 1.0e-14 or
                        (abs(candidate_minimum - best_minimum) <= 1.0e-14 and
                         candidate_rank > best_rank)):
                    best = points.copy()
                    best_minimum = candidate_minimum
                    best_rank = candidate_rank
            else:
                points[point_index] = old_position
                area_values[changed] = old_values

    # Strict local tightening before the smooth constrained polish.
    points = best.copy()
    area_values = all_areas(points)
    current_minimum, current_rank = rank_value(area_values, 1.0)

    for iteration in range(24000):
        progress = iteration / 23999.0
        if rng.random() < 0.78:
            active = np.argpartition(area_values, 8)[:9]
            triangle = triples[int(active[rng.integers(len(active))])]
            movable = triangle[triangle >= 3]
            point_index = (int(movable[rng.integers(len(movable))])
                           if len(movable) else int(rng.integers(3, n)))
        else:
            point_index = int(rng.integers(3, n))

        changed = affected[point_index]
        old_position = points[point_index].copy()
        old_values = area_values[changed].copy()
        old_weights = np.array(
            (1.0 - old_position[0] - old_position[1],
             old_position[0],
             old_position[1]),
            dtype=float,
        )

        scale = 0.0060 * (1.0 - progress) ** 1.45 + 0.000025
        weights = project_weights(old_weights + rng.normal(0.0, scale, size=3))
        points[point_index] = weights[1:]
        area_values[changed] = changed_areas(points, changed)
        candidate_minimum, candidate_rank = rank_value(area_values, 1.0)

        if (candidate_minimum > current_minimum + 1.0e-13 or
                (abs(candidate_minimum - current_minimum) <= 1.0e-13 and
                 candidate_rank > current_rank + 1.0e-14)):
            current_minimum = candidate_minimum
            current_rank = candidate_rank
        else:
            points[point_index] = old_position
            area_values[changed] = old_values

    if current_minimum > best_minimum:
        best = points.copy()
        best_minimum = current_minimum

    # In a fixed orientation cell abs(det) is smooth: maximize an epigraph
    # variable subject to sign(det_ijk) * det_ijk >= t.
    try:
        from scipy.optimize import minimize

        signs = np.sign(determinants(best))
        signs[signs == 0.0] = 1.0

        def unpack(w: np.ndarray) -> np.ndarray:
            return np.vstack((anchors, w[:16].reshape(8, 2)))

        def signed_constraints(w: np.ndarray) -> np.ndarray:
            return signs * determinants(unpack(w)) - w[-1]

        def signed_jacobian(w: np.ndarray) -> np.ndarray:
            p = unpack(w)
            a = p[triples[:, 0]]
            b = p[triples[:, 1]]
            c = p[triples[:, 2]]

            jac = np.zeros((len(triples), 17), dtype=float)
            gradients = (
                np.column_stack((b[:, 1] - c[:, 1], c[:, 0] - b[:, 0])),
                np.column_stack((c[:, 1] - a[:, 1], a[:, 0] - c[:, 0])),
                np.column_stack((a[:, 1] - b[:, 1], b[:, 0] - a[:, 0])),
            )

            for position in range(3):
                ids = triples[:, position]
                mask = ids >= 3
                rows = np.nonzero(mask)[0]
                cols = 2 * (ids[mask] - 3)
                g = gradients[position][mask] * signs[mask, None]
                jac[rows, cols] = g[:, 0]
                jac[rows, cols + 1] = g[:, 1]

            jac[:, -1] = -1.0
            return jac

        def simplex_constraints(w: np.ndarray) -> np.ndarray:
            q = w[:16].reshape(8, 2)
            return 1.0 - q[:, 0] - q[:, 1]

        def simplex_jacobian(w: np.ndarray) -> np.ndarray:
            jac = np.zeros((8, 17), dtype=float)
            rows = np.arange(8)
            jac[rows, 2 * rows] = -1.0
            jac[rows, 2 * rows + 1] = -1.0
            return jac

        start = np.concatenate((best[3:].ravel(), [best_minimum * (1.0 - 1.0e-10)]))
        constraints = [
            {"type": "ineq", "fun": signed_constraints, "jac": signed_jacobian},
            {"type": "ineq", "fun": simplex_constraints, "jac": simplex_jacobian},
        ]

        for _ in range(2):
            solution = minimize(
                lambda w: -w[-1],
                start,
                jac=lambda w: np.r_[np.zeros(16), -1.0],
                method="SLSQP",
                bounds=[(0.0, 1.0)] * 16 + [(0.0, 1.0)],
                constraints=constraints,
                options={"maxiter": 1800, "ftol": 1.0e-13, "disp": False},
            )

            if not np.isfinite(solution.x).all():
                break

            trial = unpack(solution.x)
            trial[3:] = np.maximum(trial[3:], 0.0)
            sums = trial[3:].sum(axis=1)
            outside = sums > 1.0
            if np.any(outside):
                trial[3:][outside] /= sums[outside, None]

            value = float(np.min(all_areas(trial)))
            if value > best_minimum + 1.0e-12:
                best = trial
                best_minimum = value
                signs = np.sign(determinants(best))
                signs[signs == 0.0] = 1.0
                start = np.concatenate(
                    (best[3:].ravel(), [best_minimum * (1.0 - 1.0e-10)])
                )
            else:
                break
    except Exception:
        pass

    result = np.empty((n, 2), dtype=float)
    result[:, 0] = best[:, 0] + 0.5 * best[:, 1]
    result[:, 1] = height * best[:, 1]
    return result


# EVOLVE-BLOCK-END