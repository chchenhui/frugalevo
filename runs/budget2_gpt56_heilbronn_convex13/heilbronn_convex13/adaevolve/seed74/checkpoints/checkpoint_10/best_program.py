# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_convex13() -> np.ndarray:
    """Deterministically optimize 13 points in a unit right-triangular convex hull.

    Three points are fixed at the triangle's vertices and ten interior points are
    refined by multi-start simulated annealing.  The objective is the minimum
    triangle area with a small lower-tail bonus, which makes the nonsmooth
    maximin objective substantially more stable than optimizing only one
    currently smallest triangle.
    """
    n = 13
    rng = np.random.default_rng(13031957)

    # The fixed vertices give a known convex hull of area 1/2.  Every generated
    # point is projected into x >= 0, y >= 0, x + y <= 1.
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    tri = np.array(
        [(i, j, k) for i in range(n - 2)
         for j in range(i + 1, n - 1)
         for k in range(j + 1, n)],
        dtype=np.intp,
    )

    def project_simplex(p):
        """Reflect a point into the reference triangle without rejection loops."""
        p = np.abs(p)
        s = p[0] + p[1]
        if s > 1.0:
            p = 1.0 - p
            p = np.abs(p)
            s = p[0] + p[1]
            if s > 1.0:
                p /= s
        return p

    def areas(pts):
        a = pts[tri[:, 0]]
        b = pts[tri[:, 1]]
        c = pts[tri[:, 2]]
        return 0.5 * np.abs(
            (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
            - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        )

    def quality(pts):
        ar = areas(pts)
        # Several small triangles are optimized together.  This avoids accepting
        # a move that improves one bottleneck while immediately creating another.
        low = np.partition(ar, 11)[:12]
        return low[0] + 0.12 * low.mean(), low[0]

    best_points = None
    best_minimum = -1.0

    # Independent deterministic starts are useful because the maximin landscape
    # contains many shallow local optima.
    for restart in range(12):
        interior = rng.dirichlet((1.15, 1.15, 1.15), size=10)[:, 1:]
        points = np.vstack((corners, interior))
        score, current_min = quality(points)

        for iteration in range(18000):
            fraction = iteration / 17999.0
            # Large moves discover different combinatorial arrangements; the
            # final small moves accurately balance active triangle constraints.
            step = 0.115 * (1.0 - fraction) ** 1.65 + 0.0012
            temperature = 0.0018 * (1.0 - fraction) ** 2.4 + 0.000002

            index = int(rng.integers(3, n))
            old = points[index].copy()
            proposal = project_simplex(old + rng.normal(0.0, step, size=2))
            points[index] = proposal

            new_score, new_min = quality(points)
            delta = new_score - score
            if delta >= 0.0 or rng.random() < np.exp(delta / temperature):
                score, current_min = new_score, new_min
            else:
                points[index] = old

            if current_min > best_minimum:
                best_minimum = current_min
                best_points = points.copy()

    # This fallback is only defensive; under normal operation best_points is
    # always assigned during the first annealing iteration.
    if best_points is None or not np.all(np.isfinite(best_points)):
        return np.vstack((corners, rng.dirichlet((1.0, 1.0, 1.0), size=10)[:, 1:]))
    return best_points


# EVOLVE-BLOCK-END
