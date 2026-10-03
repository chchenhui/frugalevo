# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """Deterministically anneal simplex coordinates, then polish the best max-min arrangement."""
    rng = np.random.default_rng(11031991)
    n = 11

    # Work in the unit reference simplex:
    # u >= 0, v >= 0, u + v <= 1.
    # In these coordinates abs(det) is the area normalized by the area of
    # the enclosing equilateral triangle.
    triples = np.array(
        [(i, j, k) for i in range(n) for j in range(i + 1, n)
         for k in range(j + 1, n)],
        dtype=np.intp,
    )

    def simplex_random(count: int) -> np.ndarray:
        z = rng.random((count, 2))
        outside = z[:, 0] + z[:, 1] > 1.0
        z[outside] = 1.0 - z[outside]
        return z

    def triangle_areas(points: np.ndarray) -> np.ndarray:
        """Return normalized determinant areas for every point triple."""
        a = points[triples[:, 1]] - points[triples[:, 0]]
        b = points[triples[:, 2]] - points[triples[:, 0]]
        return np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])

    best_points = None
    best_value = -1.0

    # Edge seeds give the optimizer access to useful boundary configurations
    # without fixing corners, which would create unavoidable collinearities.
    for _ in range(24):
        points = np.empty((n, 2), dtype=float)
        points[:3] = np.array([[0.5, 0.0], [0.0, 0.5], [0.5, 0.5]])
        points[3:] = simplex_random(n - 3)

        areas = triangle_areas(points)
        current = float(np.min(areas))
        steps = 55000

        for step in range(steps):
            fraction = step / (steps - 1)
            sigma = 0.115 * (1.0 - fraction) ** 1.65 + 0.00018
            temperature = 0.0090 * (1.0 - fraction) ** 2.15 + 0.000001

            # Bias most moves toward a vertex of the currently limiting
            # triangle.  Some proposals explicitly increase that triangle's
            # determinant; the annealing acceptance test still protects all
            # other constraints and permits basin-to-basin exploration.
            limiting = triples[int(np.argmin(areas))]
            if rng.random() < 0.68:
                index = int(limiting[rng.integers(0, 3)])
            else:
                index = int(rng.integers(n))

            previous = points[index].copy()

            if rng.random() < 0.30 * (1.0 - fraction) ** 1.5 + 0.015:
                s = float(np.clip(rng.normal(0.5, 0.28), 0.015, 0.985))
                edge = int(rng.integers(3))
                if edge == 0:
                    proposal = np.array([s, 0.0])
                elif edge == 1:
                    proposal = np.array([0.0, s])
                else:
                    proposal = np.array([s, 1.0 - s])
            else:
                proposal = None
                if index in limiting and rng.random() < 0.24:
                    other = limiting[limiting != index]
                    q, r = points[other[0]], points[other[1]]
                    determinant = ((q[0] - previous[0]) * (r[1] - previous[1])
                                   - (q[1] - previous[1]) * (r[0] - previous[0]))
                    direction = np.array([q[1] - r[1], r[0] - q[0]])
                    direction *= 1.0 if determinant >= 0.0 else -1.0
                    direction /= max(np.linalg.norm(direction), 1e-15)
                    candidate = previous + direction * sigma * (0.25 + rng.random())
                    if (candidate[0] >= 0.0 and candidate[1] >= 0.0
                            and candidate[0] + candidate[1] <= 1.0):
                        proposal = candidate

                if proposal is None:
                    for _ in range(10):
                        candidate = previous + rng.normal(0.0, sigma, size=2)
                        if (candidate[0] >= 0.0 and candidate[1] >= 0.0
                                and candidate[0] + candidate[1] <= 1.0):
                            proposal = candidate
                            break
                if proposal is None:
                    proposal = simplex_random(1)[0]

            points[index] = proposal
            candidate_areas = triangle_areas(points)
            proposed_value = float(np.min(candidate_areas))

            if (proposed_value >= current or
                    rng.random() < np.exp((proposed_value - current) / temperature)):
                areas = candidate_areas
                current = proposed_value
            else:
                points[index] = previous

            if current > best_value:
                best_value = current
                best_points = points.copy()

    # The stochastic phase locates a good basin, but its finite final step
    # size can leave small deterministic improvements available.  A compact
    # direct-search pass tests axial and diagonal directions at progressively
    # smaller scales.  Every accepted move strictly improves the actual
    # max-min objective, so this polish cannot degrade the stored result.
    directions = np.vstack((
        np.column_stack((
            np.cos(np.arange(16) * np.pi / 8.0),
            np.sin(np.arange(16) * np.pi / 8.0),
        )),
        np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float),
    ))

    def full_minimum(p: np.ndarray) -> float:
        """Return the exact minimum normalized determinant over all triples."""
        return float(np.min(triangle_areas(p)))

    polished_value = full_minimum(best_points)
    for scale in 0.006 * 0.5 ** np.arange(13):
        improved = True
        # Re-sweeping after an improvement lets coupled active constraints
        # settle without materially affecting the overall run time.
        while improved:
            improved = False
            for index in range(n):
                original = best_points[index].copy()
                trial_best = polished_value
                trial_point = original
                for direction in directions:
                    candidate = original + scale * direction
                    if (candidate[0] < 0.0 or candidate[1] < 0.0
                            or candidate[0] + candidate[1] > 1.0):
                        continue
                    best_points[index] = candidate
                    value = full_minimum(best_points)
                    if value > trial_best + 1e-14:
                        trial_best = value
                        trial_point = candidate.copy()
                best_points[index] = trial_point
                if trial_best > polished_value + 1e-14:
                    polished_value = trial_best
                    improved = True
                else:
                    best_points[index] = original

    # Map reference-simplex coordinates to the requested equilateral triangle.
    result = np.empty_like(best_points)
    result[:, 0] = best_points[:, 0] + 0.5 * best_points[:, 1]
    result[:, 1] = 0.5 * np.sqrt(3.0) * best_points[:, 1]
    return result


# EVOLVE-BLOCK-END
