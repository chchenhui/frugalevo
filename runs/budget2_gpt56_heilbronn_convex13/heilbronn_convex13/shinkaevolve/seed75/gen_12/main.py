# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_convex13() -> np.ndarray:
    """
    Deterministically search for a 13-point Heilbronn configuration.

    The containing region is the triangle with vertices (0,0), (1,0), and
    (0,1).  Its area is immaterial because the evaluator normalizes by the
    convex-hull area; keeping these three vertices fixed makes that
    normalization constant and guarantees a convex containing region.
    """
    n = 13
    rng = np.random.default_rng(184729)
    triples = np.array(
        [(i, j, k) for i in range(n - 2) for j in range(i + 1, n - 1)
         for k in range(j + 1, n)],
        dtype=np.intp,
    )

    def triangle_quality(p: np.ndarray) -> tuple[float, float, np.ndarray]:
        """Return true floor, a smooth bottleneck score, and critical triples."""
        a = p[triples[:, 0]]
        b = p[triples[:, 1]]
        c = p[triples[:, 2]]
        twice_areas = np.abs(
            (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
            - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        )
        floor = float(twice_areas.min())

        # Unlike the literal minimum, this gives useful improvement signal to
        # all triangles close to being limiting.  The subtraction is the
        # standard stable log-sum-exp implementation of a soft minimum.
        beta = 850.0
        soft_floor = floor - np.log(
            np.exp(-beta * (twice_areas - floor)).sum()
        ) / beta
        critical = np.argpartition(twice_areas, 18)[:18]
        # The outer triangle has determinant one, hence determinants are
        # precisely areas normalized by the fixed hull area.
        return floor, float(soft_floor), critical

    def random_triangle_points() -> np.ndarray:
        p = np.empty((n, 2), dtype=float)
        p[:3] = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
        uv = rng.random((n - 3, 2))
        outside = uv.sum(axis=1) > 1.0
        uv[outside] = 1.0 - uv[outside]
        p[3:] = uv
        return p

    best = None
    best_value = -1.0
    # Independent starts are useful because the minimum-area objective is
    # piecewise linear and has many distinct local optima.
    for restart in range(18):
        points = random_triangle_points()
        hard_value, value, critical = triangle_quality(points)
        if hard_value > best_value:
            best = points.copy()
            best_value = hard_value
        temperature = 0.0035

        for step in range(5500):
            fraction = step / 5499.0
            scale = 0.115 * (1.0 - fraction) ** 1.7 + 0.0015

            # Most proposals repair a vertex of one of the presently
            # smallest triangles.  Occasional global choices avoid locking
            # the search into a fixed critical-triangle combinatorics.
            if step % 7:
                active = triples[critical].ravel()
                active = active[active >= 3]
                index = int(rng.choice(active)) if active.size else 3 + rng.integers(n - 3)
            else:
                index = 3 + rng.integers(n - 3)
            candidate = points.copy()

            displacement = rng.normal(0.0, scale, size=2)
            # A sparse long move early in a run helps exchange bottleneck
            # triangles instead of only polishing the initial sample.
            if step < 1800 and step % 173 == 0:
                displacement = rng.normal(0.0, 0.24, size=2)

            q = candidate[index] + displacement
            q = np.maximum(q, 0.0)
            total = q[0] + q[1]
            if total > 1.0:
                q /= total
            candidate[index] = q

            candidate_hard, candidate_value, candidate_critical = triangle_quality(candidate)
            if candidate_hard > best_value:
                best = candidate.copy()
                best_value = candidate_hard

            cooling = temperature * (1.0 - fraction) + 0.000015
            if candidate_value >= value or rng.random() < np.exp(
                (candidate_value - value) / cooling
            ):
                points = candidate
                hard_value = candidate_hard
                value = candidate_value
                critical = candidate_critical

    return best


# EVOLVE-BLOCK-END