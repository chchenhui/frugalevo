# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_convex13() -> np.ndarray:
    """Deterministically optimize 13 simplex-contained points using targeted bottleneck-triangle annealing and exact incremental area updates."""
    # Optimization is performed in the reference simplex
    # {(x, y): x >= 0, y >= 0, x + y <= 1}.  Its area is 1/2, so a
    # reference-simplex triangle area is normalized by multiplying by 2.
    #
    # The three simplex corners are retained as hull vertices.  This makes
    # normalization well-defined and guarantees every generated point remains
    # in a single convex region throughout the search.
    rng = np.random.default_rng(18713)
    n = 13
    tri = np.array(
        [(i, j, k) for i in range(n - 2) for j in range(i + 1, n - 1)
         for k in range(j + 1, n)],
        dtype=np.intp,
    )

    def project_simplex(p: np.ndarray) -> np.ndarray:
        """Project one point into the closed reference triangle."""
        q = np.maximum(p, 1.0e-7)
        total = q[0] + q[1]
        if total >= 1.0 - 1.0e-7:
            q *= (1.0 - 1.0e-7) / total
        return q

    def areas(p: np.ndarray, rows=None) -> np.ndarray:
        """Return exact unsigned areas for all triangles or a selected row set."""
        t = tri if rows is None else tri[rows]
        a = p[t[:, 0]]
        b = p[t[:, 1]]
        c = p[t[:, 2]]
        return 0.5 * np.abs(
            (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
            - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        )

    def soft_score(a: np.ndarray, tau: float) -> float:
        """Return a stable temperature-controlled soft minimum of all triangle areas."""
        m = float(np.min(a))
        return m - tau * np.log(np.exp(-(a - m) / tau).sum())

    def random_configuration() -> np.ndarray:
        """Sample ten independent uniform points in the reference simplex."""
        p = np.empty((n, 2), dtype=float)
        p[:3] = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
        u = rng.random((10, 2))
        swap = u[:, 0] + u[:, 1] > 1.0
        u[swap] = 1.0 - u[swap]
        p[3:] = u
        return p

    # A one-point proposal changes only C(12, 2) = 66 of the 286 constraints.
    # Reusing the other exact values permits many more independent basins
    # without changing the optimization objective or random trajectory.
    incident = [
        np.flatnonzero(np.any(tri == index, axis=1))
        for index in range(n)
    ]

    # Search independent deterministic basins.  The broad initial soft
    # objective raises groups of nearly degenerate triangles together, while
    # the final low-temperature stage approaches the literal max-min target.
    best = None
    best_min = -np.inf
    # More independent basins and longer cooling substantially improve the
    # probability of reaching a high-quality max-min configuration while
    # remaining below the evaluation time limit.
    restarts = 24
    iterations = 250000

    for _ in range(restarts):
        current = random_configuration()
        current_a = areas(current)
        current_score = soft_score(current_a, 1.4e-3)

        for _ in range(23):
            candidate = random_configuration()
            candidate_a = areas(candidate)
            candidate_score = soft_score(candidate_a, 1.4e-3)
            if candidate_score > current_score:
                current, current_a, current_score = (
                    candidate, candidate_a, candidate_score
                )

        for it in range(iterations):
            fraction = it / (iterations - 1)
            tau = 1.4e-3 * (1.0 - fraction) ** 1.8 + 7.0e-5
            step = 0.155 * (1.0 - fraction) ** 1.7 + 5.0e-4
            temperature = 1.25e-3 * (1.0 - fraction) ** 2.7 + 8.0e-8

            candidate = current.copy()
            if it > iterations // 4 and rng.random() < 0.78:
                # Select an actual movable point from the active bottleneck
                # triangle.  Sampling all three vertices first would often
                # discard the intended target when a fixed simplex corner is
                # selected.
                active = tri[int(np.argmin(current_a))]
                movable = active[active >= 3]
                if movable.size:
                    index = int(movable[rng.integers(movable.size)])
                else:
                    index = int(rng.integers(3, n))
            else:
                index = int(rng.integers(3, n))

            candidate[index] = project_simplex(
                candidate[index] + rng.normal(0.0, step, size=2)
            )
            changed = incident[index]
            candidate_a = current_a.copy()
            candidate_a[changed] = areas(candidate, changed)
            candidate_score = soft_score(candidate_a, tau)
            delta = candidate_score - soft_score(current_a, tau)

            if delta >= 0.0 or rng.random() < np.exp(delta / temperature):
                current, current_a = candidate, candidate_a

            candidate_min = float(np.min(candidate_a))
            if candidate_min > best_min:
                best = candidate.copy()
                best_min = candidate_min

    # Deterministic literal-objective polishing cannot worsen the incumbent.
    directions = np.array(
        ((1., 0.), (-1., 0.), (0., 1.), (0., -1.),
         (1., 1.), (1., -1.), (-1., 1.), (-1., -1.)),
        dtype=float,
    )
    directions[4:] /= np.sqrt(2.0)
    polished = best.copy()
    polished_min = best_min

    for radius in (0.018, 0.010, 0.0055, 0.003, 0.0016, 0.0008,
                   0.0004, 0.0002, 0.0001):
        improved = True
        while improved:
            improved = False
            for index in range(3, n):
                for direction in directions:
                    candidate = polished.copy()
                    candidate[index] = project_simplex(
                        candidate[index] + radius * direction
                    )
                    candidate_min = float(np.min(areas(candidate)))
                    if candidate_min > polished_min + 1.0e-13:
                        polished = candidate
                        polished_min = candidate_min
                        improved = True

    best = polished

    # Affinely map the reference simplex to an equilateral triangle of area 1.
    # Affine scaling doubles all areas, exactly matching the normalization
    # factor for the reference hull of area 1/2.
    side = np.sqrt(4.0 / np.sqrt(3.0))
    height = 0.5 * np.sqrt(3.0) * side
    result = np.empty_like(best)
    result[:, 0] = side * (best[:, 0] + 0.5 * best[:, 1])
    result[:, 1] = height * best[:, 1]
    return result


# EVOLVE-BLOCK-END
