# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 11.

    Returns:
        points: np.ndarray of shape (11,2) with the x,y coordinates of the points.
    """
    # Keep the affine hull extremities fixed.  In barycentric coordinates
    # (b, c), the physical point is b*(1, 0) + c*(1/2, sqrt(3)/2), and the
    # normalized area of a triangle is simply its determinant in (b, c).
    if hasattr(heilbronn_triangle11, "_cached_points"):
        return heilbronn_triangle11._cached_points.copy()

    h = np.sqrt(3.0) / 2.0
    triples = np.asarray(
        [(i, j, k) for i in range(11) for j in range(i + 1, 11)
         for k in range(j + 1, 11)],
        dtype=int,
    )

    def barycentric_from_unit(z: np.ndarray) -> np.ndarray:
        """Map independent unit-square variables into the unit simplex."""
        q = np.empty((11, 2), dtype=float)
        q[:3] = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
        u = z[0::2]
        q[3:, 0] = u
        q[3:, 1] = (1.0 - u) * z[1::2]
        return q

    def minimum_area(q: np.ndarray) -> float:
        a = q[triples[:, 0]]
        b = q[triples[:, 1]]
        c = q[triples[:, 2]]
        return float(np.min(np.abs(
            (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
            - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        )))

    def lower_tail_score(q: np.ndarray) -> float:
        """
        Lexicographic surrogate for global exploration.

        The minimum determinant is the primary quantity.  The deliberately
        tiny second term only breaks practically equal minimum-area ties in
        favor of configurations whose other near-active triples are also
        large.  Such basins are substantially more amenable to the subsequent
        exact oriented maximin polish than isolated raw-minimum plateaus.
        """
        a = q[triples[:, 0]]
        b = q[triples[:, 1]]
        c = q[triples[:, 2]]
        areas = np.abs(
            (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
            - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        )
        smallest = np.partition(areas, 11)[:12]
        return float(np.min(smallest) + 1.0e-4 * np.mean(smallest))

    # This is also a valid deterministic answer in minimal installations.
    # The irrational-looking sequence avoids the many collinearities of a
    # triangular lattice.
    fallback_z = np.mod(
        np.arange(1, 17, dtype=float) * np.array(
            [0.6180339887498949, 0.4142135623730950] * 8
        ),
        1.0,
    )

    try:
        from scipy.optimize import differential_evolution, minimize

        result = differential_evolution(
            lambda z: -lower_tail_score(barycentric_from_unit(z)),
            bounds=[(0.0, 1.0)] * 16,
            seed=11031987,
            popsize=22,
            maxiter=1050,
            tol=2.0e-8,
            atol=1.0e-10,
            mutation=(0.45, 1.15),
            recombination=0.82,
            polish=False,
            updating="immediate",
        )
        q = barycentric_from_unit(result.x)

        # Once DE has selected orientations for the close triples, SLSQP is
        # an effective deterministic local maximin polish in direct simplex
        # coordinates.  Retaining those orientations prevents an accidental
        # crossing through a zero-area configuration.
        signs_a = q[triples[:, 0]]
        signs_b = q[triples[:, 1]]
        signs_c = q[triples[:, 2]]
        signs = np.sign(
            (signs_b[:, 0] - signs_a[:, 0]) * (signs_c[:, 1] - signs_a[:, 1])
            - (signs_b[:, 1] - signs_a[:, 1]) * (signs_c[:, 0] - signs_a[:, 0])
        )
        signs[signs == 0.0] = 1.0

        def polish_objective(w: np.ndarray) -> float:
            return -w[-1]

        def polish_constraints(w: np.ndarray) -> np.ndarray:
            r = np.empty((11, 2), dtype=float)
            r[:3] = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
            r[3:] = w[:-1].reshape(8, 2)
            aa, bb, cc = r[triples[:, 0]], r[triples[:, 1]], r[triples[:, 2]]
            oriented = signs * (
                (bb[:, 0] - aa[:, 0]) * (cc[:, 1] - aa[:, 1])
                - (bb[:, 1] - aa[:, 1]) * (cc[:, 0] - aa[:, 0])
            )
            return np.concatenate((oriented - w[-1], 1.0 - r[3:].sum(axis=1)))

        start = np.r_[q[3:].ravel(), minimum_area(q)]
        polished = minimize(
            polish_objective,
            start,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * 16 + [(0.0, 1.0)],
            constraints={"type": "ineq", "fun": polish_constraints},
            options={"maxiter": 2500, "ftol": 1.0e-11},
        )
        if polished.success:
            candidate = np.empty((11, 2), dtype=float)
            candidate[:3] = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
            candidate[3:] = polished.x[:-1].reshape(8, 2)
            if minimum_area(candidate) >= minimum_area(q) - 1.0e-10:
                q = candidate
    except Exception:
        q = barycentric_from_unit(fallback_z)

    points = np.empty((11, 2), dtype=float)
    points[:, 0] = q[:, 0] + 0.5 * q[:, 1]
    points[:, 1] = h * q[:, 1]
    heilbronn_triangle11._cached_points = points.copy()
    return points


# EVOLVE-BLOCK-END