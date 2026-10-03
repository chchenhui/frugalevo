# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_convex13() -> np.ndarray:
    """
    Construct a deterministic thirteen-point maximin configuration in the
    unit square.  The four square corners fix a convex hull of area one, so
    the raw minimum triangle area is also the normalized objective.
    """
    rng = np.random.default_rng(13051957)
    n = 13

    corners = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=float,
    )

    triples = np.array(
        [(i, j, k)
         for i in range(n)
         for j in range(i + 1, n)
         for k in range(j + 1, n)],
        dtype=np.intp,
    )

    affected = [
        np.flatnonzero((triples == index).any(axis=1)).astype(np.intp)
        for index in range(n)
    ]

    def triangle_areas(points: np.ndarray, ids: np.ndarray = None) -> np.ndarray:
        if ids is None:
            ids = np.arange(triples.shape[0], dtype=np.intp)
        t = triples[ids]
        a = points[t[:, 0]]
        b = points[t[:, 1]]
        c = points[t[:, 2]]
        return 0.5 * np.abs(
            (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
            - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        )

    def objective(values: np.ndarray, blend: float) -> tuple[float, float]:
        # Averaging several active constraints makes early annealing much less
        # likely to overfit one triangle at the cost of nearby tight ones.
        low = np.partition(values, 15)[:16]
        minimum = float(low.min())
        return minimum + blend * float(low.mean()), minimum

    best_points = None
    best_minimum = -1.0

    # Six deterministic starts: alternating stratified and uniform layouts.
    # Stratification gives a substantially better initial covering, while the
    # uniform starts retain the ability to discover asymmetric arrangements.
    for restart in range(6):
        if restart % 2 == 0:
            gx, gy = np.meshgrid(
                np.array([0.20, 0.50, 0.80]),
                np.array([0.20, 0.50, 0.80]),
            )
            interior = np.column_stack((gx.ravel(), gy.ravel()))
            interior += rng.normal(0.0, 0.095, size=(9, 2))
            interior = np.clip(interior, 0.055, 0.945)
            interior = interior[rng.permutation(9)]
        else:
            interior = rng.uniform(0.075, 0.925, size=(9, 2))

        points = np.vstack((corners, interior))
        values = triangle_areas(points)
        current_score, current_minimum = objective(values, 0.085)

        if current_minimum > best_minimum:
            best_points = points.copy()
            best_minimum = current_minimum

        iterations = 36000
        for iteration in range(iterations):
            progress = iteration / (iterations - 1.0)
            blend = 0.085 * (1.0 - progress) ** 1.25
            step = 0.112 * (1.0 - progress) ** 1.65 + 0.00075

            index = int(rng.integers(4, 13))
            old_position = points[index].copy()

            # A small fraction of proposals use axis-biased moves.  They are
            # useful for resolving constraints induced by square boundaries.
            if rng.random() < 0.28:
                displacement = np.zeros(2)
                displacement[int(rng.integers(0, 2))] = rng.normal(0.0, step)
            else:
                displacement = rng.normal(0.0, step, size=2)

            points[index] = np.clip(old_position + displacement, 0.038, 0.962)

            ids = affected[index]
            candidate_values = values.copy()
            candidate_values[ids] = triangle_areas(points, ids)
            candidate_score, candidate_minimum = objective(candidate_values, blend)

            temperature = 0.0025 * (1.0 - progress) ** 2 + 0.0000015
            if (candidate_score >= current_score or
                    rng.random() < np.exp(
                        (candidate_score - current_score) / temperature
                    )):
                values = candidate_values
                current_score = candidate_score
                current_minimum = candidate_minimum
            else:
                points[index] = old_position

            if current_minimum > best_minimum:
                best_points = points.copy()
                best_minimum = current_minimum

    # Deterministic maximin-only coordinate pattern search.  Unlike annealing,
    # this stage never accepts a decrease in the true primary metric.
    directions = np.array(
        [
            [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0],
            [1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0],
        ],
        dtype=float,
    )
    directions[4:] /= np.sqrt(2.0)

    best_values = triangle_areas(best_points)
    step = 0.014
    for _ in range(9):
        changed = True
        while changed:
            changed = False
            for index in range(4, 13):
                ids = affected[index]
                old_position = best_points[index].copy()

                for direction in directions:
                    candidate_position = np.clip(
                        old_position + step * direction, 0.038, 0.962
                    )
                    best_points[index] = candidate_position

                    candidate_values = best_values.copy()
                    candidate_values[ids] = triangle_areas(best_points, ids)
                    candidate_minimum = float(candidate_values.min())

                    if candidate_minimum > best_minimum + 1.0e-14:
                        best_values = candidate_values
                        best_minimum = candidate_minimum
                        old_position = candidate_position
                        changed = True
                    else:
                        best_points[index] = old_position

        step *= 0.53

    return best_points


# EVOLVE-BLOCK-END