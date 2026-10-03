# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """
    Search deterministic differential-evolution orientation basins in barycentric
    coordinates, then polish each with an exact-Jacobian signed-area epigraph NLP.
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
            """Locally maximize one fixed-orientation signed-area epigraph."""
            value = minimum_area(candidate)
            signs = np.where(determinants(candidate) >= 0.0, 1.0, -1.0)
            x0 = np.concatenate((candidate[3:].ravel(), [value]))

            def local_points(x):
                return np.vstack((fixed, x[:-1].reshape(8, 2)))

            def local_constraints(x):
                p = local_points(x)
                return np.concatenate((
                    signs * determinants(p) - x[-1],
                    1.0 - p[3:, 0] - p[3:, 1],
                ))

            def constraint_jacobian(x):
                """Exact derivatives of all determinant and boundary slacks."""
                p = local_points(x)
                a, b, c = p[triples[:, 0]], p[triples[:, 1]], p[triples[:, 2]]
                jac = np.zeros((173, 17))
                rows = np.arange(len(triples))

                # Gradients of det(b-a, c-a), for its a, b, and c vertices.
                gradients = (
                    np.column_stack((b[:, 1] - c[:, 1], c[:, 0] - b[:, 0])),
                    np.column_stack((c[:, 1] - a[:, 1], a[:, 0] - c[:, 0])),
                    np.column_stack((a[:, 1] - b[:, 1], b[:, 0] - a[:, 0])),
                )
                for role, gradient in enumerate(gradients):
                    indices = triples[:, role]
                    movable = indices >= 3
                    jac[rows[movable], 2 * (indices[movable] - 3)] = (
                        signs[movable] * gradient[movable, 0]
                    )
                    jac[rows[movable], 2 * (indices[movable] - 3) + 1] = (
                        signs[movable] * gradient[movable, 1]
                    )

                jac[:165, -1] = -1.0
                boundary_rows = np.arange(165, 173)
                jac[boundary_rows, 0:16:2] = -1.0
                jac[boundary_rows, 1:16:2] = -1.0
                return jac

            result = minimize(
                lambda x: -x[-1],
                x0,
                jac=lambda x: np.r_[np.zeros(16), -1.0],
                method="SLSQP",
                bounds=[(0.0, 1.0)] * 16 + [(0.0, 0.5)],
                constraints={
                    "type": "ineq",
                    "fun": local_constraints,
                    "jac": constraint_jacobian,
                },
                options={"maxiter": 4000, "ftol": 1.0e-13, "disp": False},
            )
            refined = local_points(result.x)
            # SLSQP occasionally labels a feasible epigraph improvement as an
            # unsuccessful line-search termination, so judge the returned point.
            if (np.min(local_constraints(result.x)) >= -1.0e-9
                    and minimum_area(refined) > value):
                return refined
            return candidate

        # The objective has many orientation basins.  Six moderately long,
        # independent searches provide substantially better basin coverage than
        # three nearly-converged runs at essentially the same evaluation budget.
        # Every terminal basin is subsequently optimized by the signed epigraph
        # NLP, which is much more effective than late DE generations.
        for seed in (173, 947, 2719, 5113, 7919, 10427, 13873, 16927,
                     20117, 23743):
            result = differential_evolution(
                lambda z: -minimum_area(unpack(z)),
                bounds=[(0.0, 1.0)] * 16,
                seed=seed,
                popsize=20,
                maxiter=1500,
                tol=1.0e-7,
                atol=1.0e-10,
                mutation=(0.35, 1.0),
                recombination=0.88,
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
