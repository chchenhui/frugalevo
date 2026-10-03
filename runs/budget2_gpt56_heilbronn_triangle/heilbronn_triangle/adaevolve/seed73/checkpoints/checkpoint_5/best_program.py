# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """
    Use deterministic differential-evolution searches in barycentric coordinates,
    followed by constrained SLSQP polishing, to maximize the minimum normalized
    area of all 165 triangles while fixing the three container vertices.
    """
    # In coordinates (u, v), a point represents
    # (u + v/2, sqrt(3)*v/2).  The containing triangle is simply
    # u >= 0, v >= 0, u + v <= 1, and the absolute determinant of
    # three (u, v) points is exactly its area normalized by the area
    # of the containing equilateral triangle.
    root3 = np.sqrt(3.0)
    fixed = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    triples = np.array(
        [(i, j, k) for i in range(11) for j in range(i + 1, 11)
         for k in range(j + 1, 11)],
        dtype=int,
    )

    def unpack(z):
        q = np.asarray(z, dtype=float).reshape(8, 2)
        # This parameterization makes every evolutionary candidate feasible.
        u = q[:, 0]
        v = (1.0 - u) * q[:, 1]
        return np.vstack((fixed, np.column_stack((u, v))))

    def determinants(p):
        a = p[triples[:, 0]]
        b = p[triples[:, 1]]
        c = p[triples[:, 2]]
        return ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))

    def minimum_area(p):
        return float(np.min(np.abs(determinants(p))))

    # A nonzero fallback is also useful if scipy is unavailable in a restricted
    # execution environment.
    fallback = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.18, 0.07], [0.47, 0.10], [0.76, 0.06],
        [0.09, 0.31], [0.35, 0.29], [0.62, 0.23],
        [0.16, 0.58], [0.39, 0.43],
    ])
    best = fallback
    best_value = minimum_area(best)

    try:
        from scipy.optimize import differential_evolution, minimize

        def polish(candidate):
            """Maximize the signed local epigraph objective around one basin."""
            value = minimum_area(candidate)
            signs = np.where(determinants(candidate) >= 0.0, 1.0, -1.0)
            x0 = np.concatenate((candidate[3:].ravel(), [value]))

            def local_points(x):
                return np.vstack((fixed, x[:-1].reshape(8, 2)))

            def local_constraints(x):
                p = local_points(x)
                signed_area = signs * determinants(p) - x[-1]
                inside = 1.0 - p[3:, 0] - p[3:, 1]
                return np.concatenate((signed_area, inside))

            result = minimize(
                lambda x: -x[-1],
                x0,
                method="SLSQP",
                bounds=[(0.0, 1.0)] * 16 + [(0.0, 0.5)],
                constraints={"type": "ineq", "fun": local_constraints},
                options={"maxiter": 4000, "ftol": 1.0e-13, "disp": False},
            )
            if result.success:
                refined = local_points(result.x)
                if minimum_area(refined) > value:
                    return refined
            return candidate

        # The objective is nonsmooth, so longer independent searches are more
        # useful than relying on one large population.  Each basin is polished:
        # its raw DE value need not predict its final local epigraph optimum.
        for seed in (173, 947, 2719):
            result = differential_evolution(
                lambda z: -minimum_area(unpack(z)),
                bounds=[(0.0, 1.0)] * 16,
                seed=seed,
                popsize=20,
                maxiter=3000,
                tol=1.0e-9,
                atol=1.0e-11,
                mutation=(0.45, 1.0),
                recombination=0.82,
                polish=False,
                updating="immediate",
            )
            candidate = polish(unpack(result.x))
            value = minimum_area(candidate)
            if value > best_value:
                best, best_value = candidate, value
    except Exception:
        # Return the deterministic feasible fallback rather than failing if an
        # optimizer backend is absent or unexpectedly terminates.
        pass

    # Convert barycentric planar coordinates to the requested Cartesian frame.
    return np.column_stack((
        best[:, 0] + 0.5 * best[:, 1],
        0.5 * root3 * best[:, 1],
    ))


# EVOLVE-BLOCK-END
