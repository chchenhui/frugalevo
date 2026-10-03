# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    rng = np.random.default_rng(seed=137)

    # Indices of every triangle are fixed once, making objective evaluation
    # compact and completely reproducible.
    triples = np.array(
        [(i, j, k) for i in range(n) for j in range(i + 1, n)
         for k in range(j + 1, n)],
        dtype=np.intp,
    )

    def hull_double_area(p: np.ndarray) -> float:
        """Twice the area of the convex hull (Andrew monotone chain)."""
        q = p[np.lexsort((p[:, 1], p[:, 0]))]

        def cross(o, a, b):
            return ((a[0] - o[0]) * (b[1] - o[1])
                    - (a[1] - o[1]) * (b[0] - o[0]))

        lower = []
        for v in q:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], v) <= 0.0:
                lower.pop()
            lower.append(v)
        upper = []
        for v in q[::-1]:
            while len(upper) >= 2 and cross(upper[-2], upper[-1], v) <= 0.0:
                upper.pop()
            upper.append(v)
        h = np.asarray(lower[:-1] + upper[:-1])
        return abs(np.dot(h[:, 0], np.roll(h[:, 1], -1))
                   - np.dot(h[:, 1], np.roll(h[:, 0], -1)))

    def quality(p: np.ndarray) -> tuple[float, float]:
        a = p[triples[:, 0]]
        b = p[triples[:, 1]]
        c = p[triples[:, 2]]
        double_areas = np.abs(
            (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
            - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        )
        hull_area2 = hull_double_area(p)
        if hull_area2 <= 1.0e-12:
            return 0.0, 0.0
        values = double_areas / hull_area2
        smallest = np.partition(values, 11)[:12]
        # The tail term is only a search guide; selection below remains
        # strictly based on the requested maximin quantity.
        return float(values.min()), float(values.min() + 0.16 * smallest.mean())

    best = None
    best_primary = -1.0

    # Circular boundary candidates avoid the many initially tiny triangles
    # produced by an unconstrained uniform cloud.  Interior points are then
    # free to move, and the hull itself defines the admissible convex region.
    for restart in range(7):
        angles = (np.arange(9) + 0.23 * rng.normal(size=9)) * (2.0 * np.pi / 9.0)
        radii = 0.455 + 0.025 * rng.normal(size=9)
        boundary = np.column_stack((0.5 + radii * np.cos(angles),
                                    0.5 + radii * np.sin(angles)))
        inner_angles = rng.uniform(0.0, 2.0 * np.pi, size=4)
        inner_radii = rng.uniform(0.10, 0.31, size=4)
        current = np.vstack((
            boundary,
            np.column_stack((0.5 + inner_radii * np.cos(inner_angles),
                             0.5 + inner_radii * np.sin(inner_angles))),
        ))
        current = np.clip(current, 0.0, 1.0)
        primary, score = quality(current)

        # Coordinate annealing is deliberately modest: all expensive geometry
        # is vectorized, and the decreasing step scale gives a final local
        # maximin refinement after the broader exploratory phase.
        for iteration in range(26000):
            fraction = iteration / 25999.0
            step = 0.070 * (1.0 - fraction) + 0.0018
            temperature = 0.0025 * (1.0 - fraction) ** 2 + 0.000015
            proposal = current.copy()
            if iteration % 701 == 700:
                # Rare collective shake preserves a route out of broadly
                # different hull and interior-point arrangements.
                proposal += rng.normal(0.0, step * 1.8, size=proposal.shape)
            else:
                # Most limiting constraints are nearly collinear triples.
                # Target their vertices rather than spending most proposals on
                # points which cannot improve the current minimum.
                a = current[triples[:, 0]]
                b = current[triples[:, 1]]
                c = current[triples[:, 2]]
                current_areas = np.abs(
                    (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                    - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
                )
                worst = np.argpartition(current_areas, 7)[:8]
                active_vertices = np.unique(triples[worst].ravel())

                if rng.random() < 0.18 and active_vertices.size >= 2:
                    # Coordinated displacements can improve a bottleneck when
                    # moving either endpoint independently activates another
                    # small triangle.
                    chosen = rng.choice(active_vertices, size=2, replace=False)
                    proposal[chosen] += rng.normal(
                        0.0, step * 0.72, size=(2, 2)
                    )
                elif rng.random() < 0.82:
                    index = int(rng.choice(active_vertices))
                    proposal[index] += rng.normal(0.0, step, size=2)
                else:
                    index = int(rng.integers(n))
                    proposal[index] += rng.normal(0.0, step, size=2)
            proposal = np.clip(proposal, 0.0, 1.0)

            new_primary, new_score = quality(proposal)
            if (new_score >= score
                    or rng.random() < np.exp((new_score - score) / temperature)):
                current, primary, score = proposal, new_primary, new_score

            if primary > best_primary:
                best_primary = primary
                best = current.copy()

    # A translation does not affect any area; retaining coordinates in the
    # unit square is convenient while the convex hull is the actual region.
    return best


# EVOLVE-BLOCK-END