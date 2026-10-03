# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct eleven points using a deterministic maximin search.

    The search is performed in the unit barycentric simplex.  If a point has
    simplex coordinates (u, v), its Cartesian position is
    (u + v / 2, sqrt(3) * v / 2).  Consequently, the absolute determinant of
    three (u, v) pairs is exactly that triangle's area normalized by the area
    of the enclosing equilateral triangle.
    """
    n = 11
    sqrt3_over_2 = np.sqrt(3.0) / 2.0
    triples = np.array(
        [(i, j, k) for i in range(n - 2)
         for j in range(i + 1, n - 1)
         for k in range(j + 1, n)],
        dtype=np.intp,
    )

    def normalize_simplex(p: np.ndarray) -> np.ndarray:
        """Project a proposed barycentric pair into u >= 0, v >= 0, u + v <= 1."""
        p = np.maximum(p, 0.0)
        total = float(p[0] + p[1])
        if total > 1.0:
            p = p / total
        return p

    def areas_and_score(p: np.ndarray) -> tuple[np.ndarray, float]:
        a = p[triples[:, 0]]
        b = p[triples[:, 1]]
        c = p[triples[:, 2]]
        areas = np.abs(
            (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
            - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        )
        return areas, float(np.min(areas))

    rng = np.random.default_rng(11031987)
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    best = np.vstack((vertices, np.full((8, 2), 1.0 / 3.0)))
    _, best_score = areas_and_score(best)

    # Several moderate restarts are much more reliable than one long walk for
    # this nonsmooth objective.  The fixed seed keeps the returned design
    # reproducible.
    for restart in range(8):
        interior_bary = rng.dirichlet((1.15, 1.15, 1.15), size=8)
        current = np.vstack((vertices, interior_bary[:, 1:]))
        _, current_score = areas_and_score(current)

        iterations = 30000
        for iteration in range(iterations):
            areas, score = areas_and_score(current)
            worst = triples[int(np.argmin(areas))]

            # Most moves directly resolve a currently limiting triangle;
            # occasional unrestricted moves prevent persistent local traps.
            if rng.random() < 0.88:
                movable = worst[worst >= 3]
                index = int(rng.choice(movable)) if len(movable) else int(rng.integers(3, n))
            else:
                index = int(rng.integers(3, n))

            progress = iteration / iterations
            step = 0.090 * (1.0 - progress) ** 1.35 + 0.0012
            candidate = current.copy()
            candidate[index] = normalize_simplex(
                candidate[index] + rng.normal(0.0, step, size=2)
            )
            _, candidate_score = areas_and_score(candidate)

            temperature = 0.0025 * (1.0 - progress) ** 2 + 1.0e-7
            if (candidate_score >= current_score or
                    rng.random() < np.exp((candidate_score - current_score) / temperature)):
                current = candidate
                current_score = candidate_score

            if current_score > best_score:
                best = current.copy()
                best_score = current_score

    # Deterministic small-step polishing of the surviving bottlenecks.
    for step in (0.012, 0.006, 0.003, 0.0015, 0.0007):
        improved = True
        while improved:
            improved = False
            areas, score = areas_and_score(best)
            worst = triples[int(np.argmin(areas))]
            for index in worst:
                if index < 3:
                    continue
                for dx, dy in ((step, 0.0), (-step, 0.0), (0.0, step), (0.0, -step),
                               (step, -step), (-step, step)):
                    candidate = best.copy()
                    candidate[index] = normalize_simplex(
                        candidate[index] + np.array([dx, dy])
                    )
                    _, candidate_score = areas_and_score(candidate)
                    if candidate_score > best_score:
                        best = candidate
                        best_score = candidate_score
                        improved = True

    points = np.empty((n, 2), dtype=float)
    points[:, 0] = best[:, 0] + 0.5 * best[:, 1]
    points[:, 1] = sqrt3_over_2 * best[:, 1]
    return points


# EVOLVE-BLOCK-END