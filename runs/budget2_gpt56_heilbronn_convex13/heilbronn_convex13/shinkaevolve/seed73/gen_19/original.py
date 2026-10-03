# EVOLVE-BLOCK-START
import numpy as np


# The hull is the reference triangle (0,0), (1,0), (0,1).  In this
# normalization a triangle's area divided by the hull area is simply the
# absolute two-dimensional determinant.
_TRIANGLES_13 = np.array(
    [(i, j, k) for i in range(13) for j in range(i + 1, 13)
     for k in range(j + 1, 13)],
    dtype=np.intp,
)


def _project_to_reference_triangle(p: np.ndarray) -> np.ndarray:
    """Project a point to x >= 0, y >= 0, x + y <= 1."""
    p = np.maximum(p, 0.0)
    s = float(p[0] + p[1])
    if s > 1.0:
        p /= s
    return p


def _minimum_normalized_area(points: np.ndarray) -> tuple[float, np.ndarray]:
    triples = points[_TRIANGLES_13]
    determinants = np.abs(
        (triples[:, 1, 0] - triples[:, 0, 0])
        * (triples[:, 2, 1] - triples[:, 0, 1])
        - (triples[:, 1, 1] - triples[:, 0, 1])
        * (triples[:, 2, 0] - triples[:, 0, 0])
    )
    return float(np.min(determinants)), determinants


def _lower_tail_quality(areas: np.ndarray) -> float:
    """A smooth surrogate emphasizing all currently tight constraints."""
    tail = np.partition(areas, 15)[:16]
    weights = np.linspace(3.0, 1.0, len(tail))
    return float(np.dot(tail, weights) / np.sum(weights))


def heilbronn_convex13() -> np.ndarray:
    """
    Deterministically search for thirteen points in a triangular convex hull.

    Fixing three hull vertices removes affine degrees of freedom: since the
    evaluator normalizes by hull area, this loses no scale information while
    making every candidate automatically feasible.
    """
    rng = np.random.default_rng(13031957)
    hull = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    best_points = None
    best_value = -1.0

    # Several independently jittered starts are substantially more reliable
    # than one long run for this highly nonsmooth max-min objective.
    for restart in range(6):
        # Reflection gives an unbiased uniform distribution on the full
        # triangle; unlike coordinate sorting it does not concentrate starts
        # in one geometrically correlated strip.
        interior = rng.random((10, 2))
        outside = np.sum(interior, axis=1) > 1.0
        interior[outside] = 1.0 - interior[outside]
        points = np.vstack((hull, interior))

        value, areas = _minimum_normalized_area(points)
        quality = _lower_tail_quality(areas)
        local_best = points.copy()
        local_best_value = value

        for iteration in range(14000):
            fraction = iteration / 14000.0
            temperature = 0.0018 * (1.0 - fraction) ** 2 + 0.00001
            step = 0.11 * (1.0 - fraction) + 0.002

            # Select an interior member of a worst (or nearly worst) triple.
            cutoff = np.partition(areas, min(7, len(areas) - 1))[min(7, len(areas) - 1)]
            critical = np.flatnonzero(areas <= cutoff)
            triple = _TRIANGLES_13[critical[rng.integers(len(critical))]]
            movable = triple[triple >= 3]
            index = int(movable[rng.integers(len(movable))])

            candidate = points.copy()
            candidate[index] = _project_to_reference_triangle(
                candidate[index] + rng.normal(0.0, step, size=2)
            )
            candidate_value, candidate_areas = _minimum_normalized_area(candidate)
            candidate_quality = _lower_tail_quality(candidate_areas)

            # The lower-tail surrogate lets active constraints exchange roles
            # instead of freezing whenever the identity of the minimum changes.
            delta = candidate_quality - quality
            if delta >= 0.0 or rng.random() < np.exp(delta / temperature):
                points, value, areas, quality = (
                    candidate, candidate_value, candidate_areas, candidate_quality
                )

            if value > local_best_value:
                local_best_value = value
                local_best = points.copy()

        if local_best_value > best_value:
            best_value = local_best_value
            best_points = local_best

    return best_points


# EVOLVE-BLOCK-END