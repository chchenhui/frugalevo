# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """
    Deterministically construct eleven points in an equilateral triangle.

    Internally points use simplex coordinates (u, v):
        u >= 0, v >= 0, u + v <= 1.
    The determinant of three such points is their area normalized by the
    containing triangle area.  A global search identifies a useful oriented
    matroid cell, after which a smooth signed-determinant epigraph program
    polishes the configuration.
    """
    n = 11
    height = np.sqrt(3.0) / 2.0
    rng = np.random.default_rng(110271)

    triples = np.array(
        [(i, j, k)
         for i in range(n)
         for j in range(i + 1, n)
         for k in range(j + 1, n)],
        dtype=np.intp,
    )

    anchors = np.array(
        [[0.0, 0.0],
         [1.0, 0.0],
         [0.0, 1.0]],
        dtype=float,
    )

    def decode_stick(z: np.ndarray) -> np.ndarray:
        """Map eight pairs in [0,1]^2 into the closed reference simplex."""
        q = np.asarray(z, dtype=float).reshape(8, 2)
        # u=(1-b)a, v=b is a bijective stick-breaking parameterization.
        return np.vstack((anchors, np.column_stack((q[:, 0] * (1.0 - q[:, 1]),
                                                     q[:, 1]))))

    def determinants(p: np.ndarray) -> np.ndarray:
        a = p[triples[:, 0]]
        b = p[triples[:, 1]]
        c = p[triples[:, 2]]
        return ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))

    def minimum_area(p: np.ndarray) -> float:
        return float(np.min(np.abs(determinants(p))))

    def repair_simplex(x: np.ndarray) -> np.ndarray:
        """Numerically restore simplex feasibility after a local optimizer."""
        q = np.asarray(x, dtype=float).reshape(8, 2).copy()
        q = np.maximum(q, 0.0)
        totals = q.sum(axis=1)
        mask = totals > 1.0
        if np.any(mask):
            q[mask] /= totals[mask, None]
        return q.ravel()

    # A small deterministic fallback is retained for installations without
    # scipy.  It is deliberately feasible at every step.
    def fallback_search() -> np.ndarray:
        z = rng.random(16)
        best_p = decode_stick(z)
        best_value = minimum_area(best_p)
        current = z.copy()
        current_value = best_value

        for iteration in range(70000):
            fraction = iteration / 69999.0
            scale = 0.12 * (1.0 - fraction) ** 1.7 + 0.001
            trial = current.copy()
            changed = rng.choice(16, size=2, replace=False)
            trial[changed] = np.clip(
                trial[changed] + rng.normal(0.0, scale, size=2), 0.0, 1.0
            )
            value = minimum_area(decode_stick(trial))
            temperature = 0.0015 * (1.0 - fraction) ** 2 + 1.0e-8
            if value >= current_value or rng.random() < np.exp(
                (value - current_value) / temperature
            ):
                current, current_value = trial, value
            if value > best_value:
                z, best_value = trial.copy(), value
                best_p = decode_stick(z)
        return best_p

    try:
        from scipy.optimize import differential_evolution, minimize

        # Global search remains in stick coordinates, where ordinary box
        # bounds imply geometric feasibility without penalties.  In addition
        # to Dirichlet points, use several staggered templates and every
        # permutation of their barycentric coordinates.  These expose cells
        # with edge-near points that uniform box starts rarely discover.
        population_count = 23 * 16
        initial_population = np.empty((population_count, 16), dtype=float)

        def simplex_to_stick(q: np.ndarray) -> np.ndarray:
            """Convert reference-simplex pairs to stable stick coordinates."""
            v = np.clip(q[:, 1], 0.0, 1.0 - 1.0e-12)
            a = np.divide(q[:, 0], 1.0 - v,
                          out=np.zeros_like(v), where=(1.0 - v) > 1.0e-12)
            return np.column_stack((np.clip(a, 0.0, 1.0), v)).ravel()

        templates = (
            np.array([
                [0.10, 0.10], [0.43, 0.055], [0.76, 0.075],
                [0.055, 0.36], [0.32, 0.29], [0.65, 0.22],
                [0.13, 0.68], [0.43, 0.47],
            ]),
            np.array([
                [0.17, 0.055], [0.49, 0.075], [0.80, 0.055],
                [0.055, 0.28], [0.31, 0.34], [0.61, 0.28],
                [0.075, 0.60], [0.34, 0.55],
            ]),
            np.array([
                [0.13, 0.15], [0.46, 0.09], [0.75, 0.12],
                [0.075, 0.43], [0.38, 0.28], [0.66, 0.20],
                [0.19, 0.66], [0.47, 0.42],
            ]),
        )
        permutations = (
            (0, 1, 2), (0, 2, 1), (1, 0, 2),
            (1, 2, 0), (2, 0, 1), (2, 1, 0),
        )
        seeded = 0
        for template in templates:
            for permutation in permutations:
                for jitter in (0.0, 0.018):
                    q = template + rng.normal(0.0, jitter, size=template.shape)
                    weights = np.column_stack((
                        1.0 - q[:, 0] - q[:, 1], q[:, 0], q[:, 1],
                    ))
                    weights = np.maximum(weights, 1.0e-8)
                    weights /= weights.sum(axis=1, keepdims=True)
                    weights = weights[:, permutation]
                    initial_population[seeded] = simplex_to_stick(weights[:, 1:])
                    seeded += 1

        # Concentrations below one deliberately retain additional candidates
        # close to the three sides, while the remaining samples are broad.
        alpha_choices = (0.62, 0.82, 1.12)
        while seeded < population_count:
            alpha = alpha_choices[seeded % len(alpha_choices)]
            bary = rng.dirichlet((alpha, alpha, alpha), size=8)
            initial_population[seeded] = simplex_to_stick(bary[:, 1:])
            seeded += 1

        def global_merit(z: np.ndarray) -> float:
            values = np.abs(determinants(decode_stick(z)))
            minimum = float(values.min())
            # This only breaks near-ties during global exploration; the final
            # configuration is still chosen and polished by true maximin area.
            low_tail = np.partition(values, 17)[:18].mean()
            return minimum + 0.018 * float(low_tail)

        result = differential_evolution(
            lambda z: -global_merit(z),
            bounds=[(0.0, 1.0)] * 16,
            init=initial_population,
            seed=110271,
            maxiter=520,
            tol=8.0e-9,
            atol=1.0e-11,
            polish=False,
            updating="immediate",
            workers=1,
        )

        # DE ranks with the tail merit above, but the orientation cell sent to
        # SLSQP must be the strongest true maximin member of its population.
        final_pool = np.vstack((result.population, result.x[None, :]))
        final_values = np.array([
            minimum_area(decode_stick(np.clip(z, 0.0, 1.0)))
            for z in final_pool
        ])
        initial = decode_stick(np.clip(final_pool[int(np.argmax(final_values))], 0.0, 1.0))
        best = initial.copy()
        best_value = minimum_area(best)

        # Variables are now direct simplex coordinates plus epigraph height t.
        # Fixing determinant signs makes every area constraint differentiable:
        #     sign(det_ijk) * det_ijk >= t.
        x0 = initial[3:].ravel()
        signs = np.sign(determinants(initial))
        signs[signs == 0.0] = 1.0

        def unpack(w: np.ndarray) -> np.ndarray:
            return np.vstack((anchors, w[:16].reshape(8, 2)))

        def signed_constraints(w: np.ndarray) -> np.ndarray:
            return signs * determinants(unpack(w)) - w[-1]

        def signed_jacobian(w: np.ndarray) -> np.ndarray:
            """
            Analytic derivatives of all oriented determinants.

            For D(a,b,c):
              dD/da = (b_y-c_y, c_x-b_x)
              dD/db = (c_y-a_y, a_x-c_x)
              dD/dc = (a_y-b_y, b_x-a_x).
            """
            p = unpack(w)
            a = p[triples[:, 0]]
            b = p[triples[:, 1]]
            c = p[triples[:, 2]]

            jac = np.zeros((len(triples), 17), dtype=float)
            gradients = (
                np.column_stack((b[:, 1] - c[:, 1], c[:, 0] - b[:, 0])),
                np.column_stack((c[:, 1] - a[:, 1], a[:, 0] - c[:, 0])),
                np.column_stack((a[:, 1] - b[:, 1], b[:, 0] - a[:, 0])),
            )

            for position in range(3):
                point_ids = triples[:, position]
                free = point_ids >= 3
                if np.any(free):
                    cols = 2 * (point_ids[free] - 3)
                    rows = np.nonzero(free)[0]
                    g = gradients[position][free] * signs[free, None]
                    jac[rows, cols] = g[:, 0]
                    jac[rows, cols + 1] = g[:, 1]

            jac[:, -1] = -1.0
            return jac

        def simplex_constraints(w: np.ndarray) -> np.ndarray:
            q = w[:16].reshape(8, 2)
            return 1.0 - q[:, 0] - q[:, 1]

        def simplex_jacobian(w: np.ndarray) -> np.ndarray:
            jac = np.zeros((8, 17), dtype=float)
            rows = np.arange(8)
            jac[rows, 2 * rows] = -1.0
            jac[rows, 2 * rows + 1] = -1.0
            return jac

        start_t = best_value * (1.0 - 1.0e-9)
        start = np.concatenate((x0, [start_t]))
        constraints = [
            {"type": "ineq", "fun": signed_constraints, "jac": signed_jacobian},
            {"type": "ineq", "fun": simplex_constraints, "jac": simplex_jacobian},
        ]

        # Repeating the smooth epigraph solve is inexpensive and helps SLSQP
        # finish resolving tightly active determinant constraints.
        for _ in range(3):
            polished = minimize(
                lambda w: -w[-1],
                start,
                jac=lambda w: np.r_[np.zeros(16), -1.0],
                method="SLSQP",
                bounds=[(0.0, 1.0)] * 16 + [(0.0, 1.0)],
                constraints=constraints,
                options={"maxiter": 2200, "ftol": 2.0e-13, "disp": False},
            )

            if not np.isfinite(polished.x).all():
                break

            repaired = repair_simplex(polished.x[:16])
            candidate = np.vstack((anchors, repaired.reshape(8, 2)))
            value = minimum_area(candidate)

            if value > best_value + 1.0e-13:
                best = candidate
                best_value = value
                start = np.concatenate((repaired, [value * (1.0 - 1.0e-10)]))
                # A valid positive-area point cannot change orientation within
                # this cell, but recomputing signs avoids roundoff ambiguity.
                signs = np.sign(determinants(best))
                signs[signs == 0.0] = 1.0
            else:
                break

    except Exception:
        best = fallback_search()

    # Reference-simplex affine map to the requested equilateral triangle.
    points = np.empty((11, 2), dtype=float)
    points[:, 0] = best[:, 0] + 0.5 * best[:, 1]
    points[:, 1] = height * best[:, 1]
    return points


# EVOLVE-BLOCK-END