# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct eleven points in the requested equilateral triangle.

    Internal coordinates are reference-simplex coordinates (u, v):
        u >= 0, v >= 0, u + v <= 1.

    The affine map
        (u, v) -> (u + v / 2, sqrt(3) * v / 2)
    maps this simplex to the requested equilateral triangle.  A determinant
    in reference coordinates is exactly the corresponding normalized area.
    """
    n = 11
    free_start = 3
    rng = np.random.default_rng(381104729)
    height = np.sqrt(3.0) * 0.5

    triples = np.array(
        [(i, j, k)
         for i in range(n)
         for j in range(i + 1, n)
         for k in range(j + 1, n)],
        dtype=np.intp,
    )

    anchors = np.array(
        [[0.0, 0.0],
         [1.0, 0.0],
         [0.0, 1.0]],
        dtype=float,
    )

    affected = [
        np.flatnonzero(np.any(triples == point, axis=1))
        for point in range(n)
    ]

    def determinants(points: np.ndarray) -> np.ndarray:
        a = points[triples[:, 0]]
        b = points[triples[:, 1]]
        c = points[triples[:, 2]]
        return (
            (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
            - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        )

    def areas(points: np.ndarray) -> np.ndarray:
        return np.abs(determinants(points))

    def local_areas(points: np.ndarray, rows: np.ndarray) -> np.ndarray:
        t = triples[rows]
        a = points[t[:, 0]]
        b = points[t[:, 1]]
        c = points[t[:, 2]]
        return np.abs(
            (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
            - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        )

    def ranking(values: np.ndarray) -> tuple[float, float]:
        """Lexicographic maximin ranking with a low-tail tie breaker."""
        tail = np.partition(values, 13)[:14]
        minimum = float(tail[0])
        return minimum, float(minimum + 0.045 * tail.mean())

    def simplex_weights(position: np.ndarray) -> np.ndarray:
        return np.array(
            [1.0 - position[0] - position[1], position[0], position[1]],
            dtype=float,
        )

    def normalize_weights(weights: np.ndarray) -> np.ndarray:
        weights = np.maximum(weights, 1.0e-9)
        return weights / weights.sum()

    def structured_seed(kind: int) -> np.ndarray:
        """
        Produce several geometrically distinct starting arrangements.  The
        jitter makes each island explore a nearby, but reproducible, cell.
        """
        templates = (
            np.array([
                [0.18, 0.07], [0.49, 0.06], [0.78, 0.08],
                [0.06, 0.31], [0.34, 0.25], [0.65, 0.23],
                [0.10, 0.63], [0.38, 0.49],
            ]),
            np.array([
                [0.25, 0.04], [0.55, 0.07], [0.83, 0.05],
                [0.05, 0.26], [0.29, 0.31], [0.61, 0.27],
                [0.08, 0.57], [0.33, 0.56],
            ]),
            np.array([
                [0.13, 0.13], [0.47, 0.09], [0.76, 0.12],
                [0.08, 0.42], [0.37, 0.29], [0.63, 0.20],
                [0.18, 0.68], [0.46, 0.43],
            ]),
        )

        points = np.empty((n, 2), dtype=float)
        points[:3] = anchors

        if kind < len(templates):
            q = templates[kind] + rng.normal(0.0, 0.026, size=(8, 2))
            for row in range(8):
                points[row + 3] = normalize_weights(
                    simplex_weights(q[row])
                )[1:]
        else:
            alpha = 0.76 if kind % 2 else 1.12
            bary = rng.dirichlet((alpha, alpha, alpha), size=8)
            points[3:] = bary[:, 1:]

        return points

    def anneal_island(initial: np.ndarray, iterations: int) -> tuple[np.ndarray, float]:
        """
        Incremental maximin search.  Only the 45 triangles containing the
        moved point are updated after each proposal.
        """
        points = initial.copy()
        value_array = areas(points)
        current_min, current_rank = ranking(value_array)
        local_best = points.copy()
        local_best_value = current_min

        for step in range(iterations):
            fraction = step / float(iterations - 1)
            scale = 0.105 * (1.0 - fraction) ** 1.65 + 0.00065
            temperature = 0.0034 * (1.0 - fraction) ** 2.15 + 1.0e-6

            if rng.random() < 0.72:
                count = 9 if fraction < 0.55 else 5
                low_rows = np.argpartition(value_array, count)[:count]
                triangle = triples[int(low_rows[rng.integers(count)])]
                movable = triangle[triangle >= free_start]
                point = (
                    int(rng.choice(movable))
                    if len(movable)
                    else int(rng.integers(free_start, n))
                )
            else:
                point = int(rng.integers(free_start, n))

            previous = points[point].copy()
            rows = affected[point]
            previous_values = value_array[rows].copy()
            old_w = simplex_weights(previous)

            if rng.random() < 0.11 and fraction < 0.72:
                destination = rng.dirichlet((0.9, 0.9, 0.9))
                blend = 0.18 * (1.0 - fraction) + 0.018
                new_w = normalize_weights((1.0 - blend) * old_w + blend * destination)
            else:
                new_w = normalize_weights(
                    old_w + rng.normal(0.0, scale, size=3)
                )

            points[point] = new_w[1:]
            value_array[rows] = local_areas(points, rows)
            candidate_min, candidate_rank = ranking(value_array)
            delta = candidate_rank - current_rank

            if delta >= 0.0 or rng.random() < np.exp(max(-55.0, delta / temperature)):
                current_min = candidate_min
                current_rank = candidate_rank
                if candidate_min > local_best_value:
                    local_best = points.copy()
                    local_best_value = candidate_min
            else:
                points[point] = previous
                value_array[rows] = previous_values

        return local_best, local_best_value

    def epigraph_polish(initial: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Optimize within the orientation cell discovered by annealing.  Once
        determinant signs are fixed, signed area constraints are smooth.
        """
        try:
            from scipy.optimize import minimize
        except Exception:
            return initial, float(np.min(areas(initial)))

        best = initial.copy()
        best_value = float(np.min(areas(best)))

        for _ in range(3):
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
                derivatives = (
                    np.column_stack((b[:, 1] - c[:, 1], c[:, 0] - b[:, 0])),
                    np.column_stack((c[:, 1] - a[:, 1], a[:, 0] - c[:, 0])),
                    np.column_stack((a[:, 1] - b[:, 1], b[:, 0] - a[:, 0])),
                )

                for slot in range(3):
                    ids = triples[:, slot]
                    mask = ids >= free_start
                    rows = np.flatnonzero(mask)
                    columns = 2 * (ids[mask] - free_start)
                    grad = derivatives[slot][mask] * signs[mask, None]
                    jac[rows, columns] = grad[:, 0]
                    jac[rows, columns + 1] = grad[:, 1]

                jac[:, -1] = -1.0
                return jac

            def simplex_constraint(w: np.ndarray) -> np.ndarray:
                q = w[:16].reshape(8, 2)
                return 1.0 - q[:, 0] - q[:, 1]

            def simplex_jacobian(w: np.ndarray) -> np.ndarray:
                jac = np.zeros((8, 17), dtype=float)
                rows = np.arange(8)
                jac[rows, 2 * rows] = -1.0
                jac[rows, 2 * rows + 1] = -1.0
                return jac

            start = np.concatenate((
                best[3:].ravel(),
                [best_value * (1.0 - 1.0e-10)],
            ))

            try:
                result = minimize(
                    lambda w: -w[-1],
                    start,
                    jac=lambda w: np.r_[np.zeros(16), -1.0],
                    method="SLSQP",
                    bounds=[(0.0, 1.0)] * 16 + [(0.0, 1.0)],
                    constraints=[
                        {
                            "type": "ineq",
                            "fun": signed_constraints,
                            "jac": signed_jacobian,
                        },
                        {
                            "type": "ineq",
                            "fun": simplex_constraint,
                            "jac": simplex_jacobian,
                        },
                    ],
                    options={"maxiter": 1800, "ftol": 1.0e-13, "disp": False},
                )
            except Exception:
                break

            if not np.isfinite(result.x).all():
                break

            candidate = unpack(result.x)
            candidate[3:] = np.maximum(candidate[3:], 0.0)
            totals = candidate[3:].sum(axis=1)
            outside = totals > 1.0
            if np.any(outside):
                candidate[3:][outside] /= totals[outside, None]

            candidate_value = float(np.min(areas(candidate)))
            if candidate_value > best_value + 1.0e-12:
                best = candidate
                best_value = candidate_value
            else:
                break

        return best, best_value

    # Search islands are deliberately shorter than one monolithic walk: this
    # covers substantially more combinatorial arrangements at similar cost.
    finalists = []
    for island in range(11):
        start = structured_seed(island % 5)
        candidate, value = anneal_island(start, 10500)
        finalists.append((value, candidate))

    finalists.sort(key=lambda item: item[0], reverse=True)

    best_points = finalists[0][1].copy()
    best_value = finalists[0][0]

    # The top cells are polished independently; a failed local solve cannot
    # damage an already valid candidate.
    for _, candidate in finalists[:4]:
        polished, value = epigraph_polish(candidate)
        if value > best_value:
            best_points = polished
            best_value = value

    result = np.empty((n, 2), dtype=float)
    result[:, 0] = best_points[:, 0] + 0.5 * best_points[:, 1]
    result[:, 1] = height * best_points[:, 1]
    return result


# EVOLVE-BLOCK-END