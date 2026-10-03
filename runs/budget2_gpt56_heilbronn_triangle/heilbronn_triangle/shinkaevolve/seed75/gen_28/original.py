# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct a deterministic maximin configuration for eleven points in the
    unit equilateral triangle.
    """
    n = 11
    height = np.sqrt(3.0) / 2.0
    triples = np.array(
        [(i, j, k) for i in range(n - 2) for j in range(i + 1, n - 1)
         for k in range(j + 1, n)],
        dtype=np.intp,
    )

    def areas(points: np.ndarray) -> np.ndarray:
        a = points[triples[:, 0]]
        b = points[triples[:, 1]]
        c = points[triples[:, 2]]
        return 0.5 * np.abs(
            (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
            - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        )

    def minimum_area(simplex_points: np.ndarray) -> float:
        # simplex coordinates (u, v) map to (u + v/2, height*v).
        cartesian = np.empty_like(simplex_points)
        cartesian[:, 0] = simplex_points[:, 0] + 0.5 * simplex_points[:, 1]
        cartesian[:, 1] = height * simplex_points[:, 1]
        return float(np.min(areas(cartesian)))

    def random_simplex(rng: np.random.Generator, count: int) -> np.ndarray:
        barycentric = rng.dirichlet((1.0, 1.0, 1.0), size=count)
        return barycentric[:, :2]

    def project_to_simplex(point: np.ndarray) -> np.ndarray:
        point = np.maximum(point, 0.0)
        total = float(point[0] + point[1])
        if total > 1.0:
            point /= total
        return point

    rng = np.random.default_rng(11031991)
    vertices = np.array(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    best = None
    best_value = -1.0

    # The points in the smallest triangles are the active constraints of the
    # maximin problem.  Moving one of their incident points is much more
    # productive than uniformly perturbing a point which does not currently
    # affect the objective.  Uniform moves remain part of the proposal kernel
    # so that inactive points can become active after a larger rearrangement.
    for restart in range(34):
        current = np.vstack((vertices, random_simplex(rng, n - 3)))
        current_value = minimum_area(current)

        for step in range(5350):
            fraction = step / 5349.0
            scale = 0.125 * (0.0075 / 0.125) ** fraction
            temperature = 0.0028 * (0.000008 / 0.0028) ** fraction

            cartesian = np.empty_like(current)
            cartesian[:, 0] = current[:, 0] + 0.5 * current[:, 1]
            cartesian[:, 1] = height * current[:, 1]
            current_areas = areas(cartesian)

            candidate = current.copy()
            if rng.random() < 0.24:
                index = int(rng.integers(3, n))
            else:
                # Sample from several active constraints, not solely the
                # single smallest one, to avoid oscillating around one face.
                active_count = min(12, len(current_areas))
                active = np.argpartition(current_areas, active_count - 1)[:active_count]
                triangle = triples[int(active[rng.integers(active_count)])]
                movable = triangle[triangle >= 3]
                index = int(
                    movable[rng.integers(len(movable))]
                    if len(movable) else rng.integers(3, n)
                )

            candidate[index] = project_to_simplex(
                candidate[index] + rng.normal(0.0, scale, size=2)
            )

            # Coordinated small moves help pass narrow local bottlenecks where
            # no individual point can move without briefly reducing the area.
            if rng.random() < 0.18:
                other = int(rng.integers(3, n - 1))
                if other >= index:
                    other += 1
                candidate[other] = project_to_simplex(
                    candidate[other] + rng.normal(0.0, 0.62 * scale, size=2)
                )

            candidate_value = minimum_area(candidate)

            if candidate_value >= current_value or rng.random() < np.exp(
                (candidate_value - current_value) / max(temperature, 1e-12)
            ):
                current = candidate
                current_value = candidate_value

            if current_value > best_value:
                best = current.copy()
                best_value = current_value

    points = np.empty((n, 2))
    points[:, 0] = best[:, 0] + 0.5 * best[:, 1]
    points[:, 1] = height * best[:, 1]
    return points


# EVOLVE-BLOCK-END