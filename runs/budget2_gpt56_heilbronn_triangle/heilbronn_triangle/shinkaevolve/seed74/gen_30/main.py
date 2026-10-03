# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct a reproducible maximin arrangement of eleven points in the
    equilateral triangle with vertices (0,0), (1,0), and (.5,sqrt(3)/2).

    Internal coordinates are barycentric affine coordinates (u, v), where
    u >= 0, v >= 0, u + v <= 1.  A determinant in these coordinates equals
    the corresponding triangle area normalized by the outer triangle area.
    """
    rng = np.random.default_rng(11031991)
    n = 11
    triples = np.array(
        [(i, j, k) for i in range(n) for j in range(i + 1, n)
         for k in range(j + 1, n)],
        dtype=np.intp,
    )

    def determinants(p: np.ndarray) -> np.ndarray:
        a = p[triples[:, 1]] - p[triples[:, 0]]
        b = p[triples[:, 2]] - p[triples[:, 0]]
        return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]

    def merit(p: np.ndarray) -> tuple[float, float]:
        values = np.abs(determinants(p))
        tail = np.partition(values, 15)[:16]
        minimum = float(values.min())
        return minimum, minimum + 0.055 * float(tail.mean())

    def project_simplex(weights: np.ndarray) -> np.ndarray:
        weights = np.maximum(weights, 1.0e-8)
        weights /= weights.sum()
        return weights[1:]

    best = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]] +
        [[1.0 / 3.0, 1.0 / 3.0]] * 8,
        dtype=float,
    )
    best_min = -1.0

    # Global search: independently seeded restarts and decreasing-scale
    # barycentric mutations efficiently explore different order types.
    for restart in range(10):
        bary = rng.dirichlet((1.08, 1.08, 1.08), size=8)
        current = np.empty((n, 2), dtype=float)
        current[:3] = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
        current[3:, 0] = bary[:, 1]
        current[3:, 1] = bary[:, 2]
        current_min, current_merit = merit(current)

        for iteration in range(15000):
            fraction = iteration / 14999.0
            step = 0.115 * (1.0 - fraction) ** 1.7 + 0.0008
            thermal = 0.0065 * (1.0 - fraction) ** 2.1 + 0.00001
            # Bias proposals toward free points occurring in the current
            # active low-area triples.  Uniform choices remain useful for
            # changing order type during the hotter part of the search.
            active_count = 18 if fraction < 0.55 else 9
            active_rows = np.argpartition(
                np.abs(determinants(current)), active_count - 1
            )[:active_count]
            active_points = triples[active_rows].ravel()
            active_free = active_points[active_points >= 3]
            if active_free.size and rng.random() < 0.84:
                index = int(rng.choice(active_free))
            else:
                index = int(rng.integers(3, n))

            candidate = current.copy()
            old = candidate[index]
            weights = np.array([1.0 - old[0] - old[1], old[0], old[1]])

            # A small number of broad early proposals avoids persistent
            # near-collinear configurations.
            if iteration < 3500 and rng.random() < 0.035:
                weights = rng.dirichlet((1.0, 1.0, 1.0))
            else:
                weights += rng.normal(0.0, step, size=3)

            candidate[index] = project_simplex(weights)

            # Coupled active-point motions can improve intersecting tight
            # triples that no feasible one-point displacement can relax.
            if (active_free.size > 1 and rng.random() < 0.16):
                partners = active_free[active_free != index]
                if partners.size:
                    index2 = int(rng.choice(partners))
                    old2 = candidate[index2]
                    weights2 = np.array(
                        [1.0 - old2[0] - old2[1], old2[0], old2[1]]
                    )
                    weights2 += rng.normal(0.0, 0.68 * step, size=3)
                    candidate[index2] = project_simplex(weights2)
            candidate_min, candidate_merit = merit(candidate)
            delta = candidate_merit - current_merit

            if delta >= 0.0 or rng.random() < np.exp(delta / thermal):
                current = candidate
                current_min = candidate_min
                current_merit = candidate_merit

            if current_min > best_min:
                best_min = current_min
                best = current.copy()

    # Local maximin polish.  For a fixed nonzero orientation pattern,
    # |det| is represented by signed determinant constraints.  SLSQP can
    # directly maximize an explicit lower-bound variable for every triple.
    try:
        from scipy.optimize import minimize

        def polish(seed: np.ndarray) -> np.ndarray:
            p0 = seed.copy()

            for _ in range(3):
                signed = determinants(p0)
                signs = np.where(signed >= 0.0, 1.0, -1.0)
                x0 = np.empty(17, dtype=float)
                x0[:16] = p0[3:].ravel()
                x0[16] = max(1.0e-8, float(np.abs(signed).min()))

                def unpack(x: np.ndarray) -> np.ndarray:
                    p = np.empty((n, 2), dtype=float)
                    p[:3] = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
                    p[3:] = x[:16].reshape(8, 2)
                    return p

                def area_constraints(x: np.ndarray) -> np.ndarray:
                    return signs * determinants(unpack(x)) - x[16]

                def area_jacobian(x: np.ndarray) -> np.ndarray:
                    p = unpack(x)
                    jac = np.zeros((len(triples), 17), dtype=float)

                    for row, (i, j, k) in enumerate(triples):
                        gi = np.array([
                            p[j, 1] - p[k, 1],
                            p[k, 0] - p[j, 0],
                        ])
                        gj = np.array([
                            p[k, 1] - p[i, 1],
                            p[i, 0] - p[k, 0],
                        ])
                        gk = np.array([
                            p[i, 1] - p[j, 1],
                            p[j, 0] - p[i, 0],
                        ])
                        for point, gradient in ((i, gi), (j, gj), (k, gk)):
                            if point >= 3:
                                start = 2 * (point - 3)
                                jac[row, start:start + 2] = signs[row] * gradient
                        jac[row, 16] = -1.0
                    return jac

                def simplex_constraints(x: np.ndarray) -> np.ndarray:
                    q = x[:16].reshape(8, 2)
                    return 1.0 - q[:, 0] - q[:, 1]

                def simplex_jacobian(x: np.ndarray) -> np.ndarray:
                    jac = np.zeros((8, 17), dtype=float)
                    for row in range(8):
                        jac[row, 2 * row:2 * row + 2] = -1.0
                    return jac

                result = minimize(
                    fun=lambda x: -x[16],
                    x0=x0,
                    jac=lambda x: np.r_[np.zeros(16), -1.0],
                    method="SLSQP",
                    bounds=[(0.0, 1.0)] * 16 + [(0.0, 1.0)],
                    constraints=[
                        {"type": "ineq", "fun": area_constraints,
                         "jac": area_jacobian},
                        {"type": "ineq", "fun": simplex_constraints,
                         "jac": simplex_jacobian},
                    ],
                    options={"maxiter": 900, "ftol": 1.0e-11, "disp": False},
                )

                if not result.success:
                    break

                candidate = unpack(result.x)
                candidate_min, _ = merit(candidate)
                old_min, _ = merit(p0)
                if candidate_min <= old_min + 1.0e-10:
                    break
                p0 = candidate

            return p0

        polished = polish(best)
        polished_min, _ = merit(polished)
        if polished_min > best_min:
            best = polished
            best_min = polished_min
    except Exception:
        # The annealed arrangement is always valid, including environments
        # without SciPy or rare optimizer failures.
        pass

    points = np.empty((n, 2), dtype=float)
    points[:, 0] = best[:, 0] + 0.5 * best[:, 1]
    points[:, 1] = (np.sqrt(3.0) / 2.0) * best[:, 1]
    return points


# EVOLVE-BLOCK-END