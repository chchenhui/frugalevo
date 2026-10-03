# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_convex13() -> np.ndarray:
    """
    Deterministically optimize thirteen points in an equilateral unit convex
    region.  The three hull vertices are fixed, while the other points are
    searched in barycentric coordinates, guaranteeing feasibility.
    """
    rng = np.random.default_rng(seed=1847)

    # A fixed triangular hull makes maximizing the raw minimum determinant
    # equivalent to maximizing the evaluator's hull-normalized area.
    hull = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]],
        dtype=float,
    )
    triples = np.array(
        [(i, j, k) for i in range(13) for j in range(i + 1, 13)
         for k in range(j + 1, 13)],
        dtype=np.intp,
    )

    # The ten free points are stored as three nonnegative barycentric weights.
    weights = rng.dirichlet(np.ones(3), size=10)
    weights = 0.02 + 0.94 * weights

    def make_points(w: np.ndarray) -> np.ndarray:
        return np.vstack((hull, w @ hull))

    def triangle_doubles(points: np.ndarray) -> np.ndarray:
        a = points[triples[:, 0]]
        b = points[triples[:, 1]]
        c = points[triples[:, 2]]
        return np.abs(
            (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
            - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        )

    points = make_points(weights)
    areas = triangle_doubles(points)
    value = float(areas.min())

    # Moving a member of the active worst triangle focuses work on the
    # bottleneck; occasional global moves avoid locking into one active set.
    iterations = 90000
    for step in range(iterations):
        worst = triples[int(np.argmin(areas))]
        movable = worst[worst >= 3] - 3
        if len(movable) and rng.random() < 0.88:
            index = int(movable[rng.integers(len(movable))])
        else:
            index = int(rng.integers(10))

        fraction = step / iterations
        scale = 0.090 * (1.0 - fraction) ** 1.7 + 0.0012
        candidate = weights.copy()
        candidate[index] += rng.normal(0.0, scale, size=3)
        candidate[index] = np.maximum(candidate[index], 1.0e-4)
        candidate[index] /= candidate[index].sum()

        candidate_points = make_points(candidate)
        candidate_areas = triangle_doubles(candidate_points)
        candidate_value = float(candidate_areas.min())

        temperature = 0.0045 * (1.0 - fraction) ** 2 + 1.0e-6
        if (candidate_value >= value or
                rng.random() < np.exp((candidate_value - value) / temperature)):
            weights = candidate
            points = candidate_points
            areas = candidate_areas
            value = candidate_value

    # A reproducible final pattern search removes small residual defects
    # without accepting any deterioration of the actual maximin objective.
    directions = np.array(
        [[1.0, -1.0, 0.0], [1.0, 0.0, -1.0], [0.0, 1.0, -1.0],
         [-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0], [0.0, -1.0, 1.0]]
    )
    for scale in (0.010, 0.004, 0.0015, 0.0005):
        improved = True
        while improved:
            improved = False
            for index in range(10):
                for direction in directions:
                    candidate = weights.copy()
                    candidate[index] = np.maximum(
                        candidate[index] + scale * direction, 1.0e-5
                    )
                    candidate[index] /= candidate[index].sum()
                    candidate_points = make_points(candidate)
                    candidate_areas = triangle_doubles(candidate_points)
                    candidate_value = float(candidate_areas.min())
                    if candidate_value > value + 1.0e-12:
                        weights = candidate
                        points = candidate_points
                        areas = candidate_areas
                        value = candidate_value
                        improved = True

    return points


# EVOLVE-BLOCK-END