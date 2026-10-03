# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_convex13() -> np.ndarray:
    """Deterministically optimize 13 simplex points by annealing one-point moves with incremental triangle-area updates."""
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
    # A move of one point can only alter triangles containing that point.
    # There are 66 such triangles rather than all 286, which makes longer
    # deterministic annealing schedules practical.
    incident = tuple(
        np.flatnonzero(np.any(tri == i, axis=1)).astype(np.intp)
        for i in range(n)
    )

    def project_simplex(p: np.ndarray) -> np.ndarray:
        """Project one point into the closed reference triangle."""
        q = np.maximum(p, 1.0e-7)
        total = q[0] + q[1]
        if total >= 1.0 - 1.0e-7:
            q *= (1.0 - 1.0e-7) / total
        return q

    def areas(p: np.ndarray, rows: np.ndarray | None = None) -> np.ndarray:
        """Return unsigned areas for all triangles, or only the requested triangle rows."""
        t = tri if rows is None else tri[rows]
        a = p[t[:, 0]]
        b = p[t[:, 1]]
        c = p[t[:, 2]]
        return 0.5 * np.abs(
            (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
            - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        )

    def soft_score(a: np.ndarray, tau: float = 6.0e-4) -> float:
        """Return a stable temperature-controlled soft approximation to min(area)."""
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

    # Independent deterministic restarts are substantially more reliable than
    # one very long trajectory for this highly non-convex max-min problem.
    # Each run starts from the best member of a small random population, then
    # gradually changes from a broad soft minimum to an almost literal minimum.
    best = None
    best_min = -np.inf
    # Incremental updates below make these extra independent basins affordable.
    restarts = 8
    iterations = 150000

    for restart in range(restarts):
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
            # Continuation in tau avoids optimizing an overly averaged
            # objective in the final refinement stages.
            tau = 1.4e-3 * (1.0 - fraction) ** 1.8 + 7.0e-5
            step = 0.155 * (1.0 - fraction) ** 1.7 + 5.0e-4
            temperature = 1.25e-3 * (1.0 - fraction) ** 2.7 + 8.0e-8

            candidate = current.copy()
            if it > iterations // 4 and rng.random() < 0.78:
                active = tri[int(np.argmin(current_a))]
                index = int(active[rng.integers(3)])
                if index < 3:
                    index = int(rng.integers(3, n))
            else:
                index = int(rng.integers(3, n))

            candidate[index] = project_simplex(
                candidate[index] + rng.normal(0.0, step, size=2)
            )
            changed = incident[index]
            candidate_a = current_a.copy()
            candidate_a[changed] = areas(candidate, changed)
            current_score = soft_score(current_a, tau)
            candidate_score = soft_score(candidate_a, tau)
            delta = candidate_score - current_score

            if delta >= 0.0 or rng.random() < np.exp(delta / temperature):
                current, current_a = candidate, candidate_a

            candidate_min = float(np.min(candidate_a))
            if candidate_min > best_min:
                best = candidate.copy()
                best_min = candidate_min

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
