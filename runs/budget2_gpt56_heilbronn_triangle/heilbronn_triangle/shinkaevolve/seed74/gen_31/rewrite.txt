# EVOLVE-BLOCK-START
import numpy as np


_CACHED_HEILBRONN11 = None


def heilbronn_triangle11() -> np.ndarray:
    """
    Deterministically construct eleven points in the unit-side equilateral
    triangle.  Internal coordinates are reference-simplex coordinates (u, v):
        u >= 0, v >= 0, u + v <= 1.

    Under (u, v) -> (u + v/2, sqrt(3)*v/2), every determinant in reference
    coordinates equals the corresponding triangle area normalized by the
    containing triangle area.
    """
    global _CACHED_HEILBRONN11
    if _CACHED_HEILBRONN11 is not None:
        return _CACHED_HEILBRONN11.copy()

    n = 11
    free_n = 8
    dims = 16
    height = np.sqrt(3.0) * 0.5
    rng = np.random.default_rng(381104729)

    triples = np.array(
        [(i, j, k)
         for i in range(n)
         for j in range(i + 1, n)
         for k in range(j + 1, n)],
        dtype=np.intp,
    )
    anchors = np.array(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=float)

    def project(q: np.ndarray) -> np.ndarray:
        """Project one or many simplex-coordinate pairs into the simplex."""
        q = np.maximum(q, 0.0)
        total = q[..., 0] + q[..., 1]
        outside = total > 1.0
        if np.any(outside):
            q[outside] /= total[outside, None]
        return q

    def determinants(points: np.ndarray) -> np.ndarray:
        a = points[triples[:, 0]]
        b = points[triples[:, 1]]
        c = points[triples[:, 2]]
        return ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))

    def areas(points: np.ndarray) -> np.ndarray:
        return np.abs(determinants(points))

    def population_areas(population: np.ndarray) -> np.ndarray:
        count = population.shape[0]
        points = np.empty((count, n, 2), dtype=float)
        points[:, :3] = anchors
        points[:, 3:] = population.reshape(count, free_n, 2)
        a = points[:, triples[:, 0]]
        b = points[:, triples[:, 1]]
        c = points[:, triples[:, 2]]
        return np.abs(
            (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1])
            - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])
        )

    def score_population(population: np.ndarray, progress: float):
        """
        Phase-dependent low-tail surrogate.  The exact minimum remains
        available independently for archival and final elite selection.
        """
        value = population_areas(population)
        minimum = value.min(axis=1)

        tail_count = int(round(30.0 - 20.0 * progress))
        tail_count = max(10, min(tail_count, value.shape[1]))
        low = np.partition(value, tail_count - 1, axis=1)[:, :tail_count]

        # Broad early tail finds robust cells; stronger late weight equalizes
        # the final active constraints before signed epigraph refinement.
        weight = 0.018 + 0.072 * progress
        merit = minimum + weight * low.mean(axis=1)
        return minimum, merit

    def point_from_vector(x: np.ndarray) -> np.ndarray:
        p = np.empty((n, 2), dtype=float)
        p[:3] = anchors
        p[3:] = x.reshape(free_n, 2)
        return p

    def fallback() -> np.ndarray:
        """Dependency-free deterministic local search fallback."""
        bary = rng.dirichlet((0.9, 0.9, 0.9), size=free_n)
        current = bary[:, 1:].copy()
        current_points = point_from_vector(current.ravel())
        current_value = float(areas(current_points).min())
        best = current.copy()
        best_value = current_value

        for iteration in range(60000):
            f = iteration / 59999.0
            candidate = current.copy()
            index = int(rng.integers(free_n))
            weights = np.array(
                [1.0 - candidate[index, 0] - candidate[index, 1],
                 candidate[index, 0], candidate[index, 1]]
            )
            weights += rng.normal(0.0, 0.08 * (1.0 - f) ** 1.7 + 0.00008, 3)
            weights = np.maximum(weights, 1.0e-10)
            weights /= weights.sum()
            candidate[index] = weights[1:]

            value = float(areas(point_from_vector(candidate.ravel())).min())
            if value >= current_value:
                current, current_value = candidate, value
            if value > best_value:
                best, best_value = candidate.copy(), value

        return point_from_vector(best.ravel())

    try:
        # ------------------------------------------------------------------
        # Vectorized phase-dependent differential evolution.
        # ------------------------------------------------------------------
        population_size = 96
        generations = 760
        elite_count = 10

        raw = rng.dirichlet((0.93, 0.93, 0.93),
                            size=(population_size, free_n))
        population = raw[:, :, 1:].reshape(population_size, dims)

        # Structured seeds supplement broad Dirichlet starts.
        template = np.array([
            [0.15, 0.10], [0.46, 0.07], [0.76, 0.08],
            [0.08, 0.36], [0.36, 0.29], [0.66, 0.21],
            [0.16, 0.66], [0.44, 0.46],
        ])
        for row in range(18):
            trial = template + rng.normal(0.0, 0.045, template.shape)
            population[row] = project(trial).ravel()

        minima, merits = score_population(population, 0.0)
        archive_vector = population[int(np.argmax(minima))].copy()
        archive_minimum = float(np.max(minima))

        for generation in range(generations):
            progress = generation / float(generations - 1)
            minima, merits = score_population(population, progress)

            best_index = int(np.argmax(merits))
            best = population[best_index].copy()

            # Random distinct parents for each vector.
            parent_ids = np.empty((population_size, 3), dtype=np.intp)
            indices = np.arange(population_size)
            for i in range(population_size):
                available = np.delete(indices, i)
                parent_ids[i] = rng.choice(available, size=3, replace=False)

            a = population[parent_ids[:, 0]]
            b = population[parent_ids[:, 1]]
            c = population[parent_ids[:, 2]]

            scale = (0.78 - 0.31 * progress
                     + rng.uniform(-0.055, 0.055, population_size))
            mutant = a + scale[:, None] * (b - c)

            # Best-directed component is delayed until cells have been found.
            if progress > 0.20:
                pull = 0.08 + 0.27 * (progress - 0.20) / 0.80
                mutant += pull * (best - mutant)

            crossover = rng.random((population_size, dims)) < (
                0.91 - 0.20 * progress
            )
            crossover[np.arange(population_size),
                      rng.integers(0, dims, population_size)] = True
            trial = np.where(crossover, mutant, population)
            trial = project(trial.reshape(population_size, free_n, 2)).reshape(
                population_size, dims
            )

            trial_minima, trial_merits = score_population(trial, progress)
            # Surrogate selection permits tail repair; retain a modest
            # tolerance so an insignificant minimum loss can improve a broad
            # cluster of near-active constraints.
            allowance = 0.00085 * (1.0 - progress) ** 2
            accept = ((trial_merits > merits + 1.0e-15)
                      & (trial_minima >= minima - allowance))
            population[accept] = trial[accept]

            candidate_index = int(np.argmax(trial_minima))
            if trial_minima[candidate_index] > archive_minimum:
                archive_vector = trial[candidate_index].copy()
                archive_minimum = float(trial_minima[candidate_index])

            # Preserve exact-minimum elites against surrogate drift.
            if generation % 38 == 37:
                updated_minima, _ = score_population(population, progress)
                order = np.argsort(updated_minima)
                keep = population[order[-4:]].copy()
                population[order[:4]] = keep

        final_minima, final_merits = score_population(population, 1.0)
        order = np.lexsort((final_merits, final_minima))[::-1]
        finalists = [archive_vector] + [population[i].copy() for i in order[:elite_count]]

        # ------------------------------------------------------------------
        # Smooth signed-determinant epigraph optimization in several cells.
        # ------------------------------------------------------------------
        from scipy.optimize import minimize

        def polish(vector: np.ndarray):
            best = point_from_vector(vector)
            best_value = float(areas(best).min())

            for _ in range(2):
                signs = np.sign(determinants(best))
                signs[signs == 0.0] = 1.0

                def unpack(w):
                    return np.vstack((anchors, w[:16].reshape(free_n, 2)))

                def constraint(w):
                    return signs * determinants(unpack(w)) - w[-1]

                def jacobian(w):
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
                    for slot in range(3):
                        ids = triples[:, slot]
                        mask = ids >= 3
                        rows = np.flatnonzero(mask)
                        cols = 2 * (ids[mask] - 3)
                        grad = gradients[slot][mask] * signs[mask, None]
                        jac[rows, cols] = grad[:, 0]
                        jac[rows, cols + 1] = grad[:, 1]
                    jac[:, -1] = -1.0
                    return jac

                def simplex(w):
                    q = w[:16].reshape(free_n, 2)
                    return 1.0 - q[:, 0] - q[:, 1]

                def simplex_jac(w):
                    out = np.zeros((free_n, 17), dtype=float)
                    r = np.arange(free_n)
                    out[r, 2 * r] = -1.0
                    out[r, 2 * r + 1] = -1.0
                    return out

                start = np.r_[best[3:].ravel(), best_value * (1.0 - 1.0e-10)]
                result = minimize(
                    lambda w: -w[-1],
                    start,
                    jac=lambda w: np.r_[np.zeros(16), -1.0],
                    method="SLSQP",
                    bounds=[(0.0, 1.0)] * 16 + [(0.0, 1.0)],
                    constraints=[
                        {"type": "ineq", "fun": constraint, "jac": jacobian},
                        {"type": "ineq", "fun": simplex, "jac": simplex_jac},
                    ],
                    options={"maxiter": 1500, "ftol": 8.0e-14, "disp": False},
                )

                if not np.isfinite(result.x).all():
                    break

                q = project(result.x[:16].reshape(free_n, 2))
                candidate = np.vstack((anchors, q))
                value = float(areas(candidate).min())
                if value > best_value + 1.0e-12:
                    best, best_value = candidate, value
                else:
                    break

            return best, best_value

        best = point_from_vector(archive_vector)
        best_value = float(areas(best).min())

        for vector in finalists:
            candidate, value = polish(vector)
            if value > best_value:
                best, best_value = candidate, value

    except Exception:
        best = fallback()

    result = np.empty((n, 2), dtype=float)
    result[:, 0] = best[:, 0] + 0.5 * best[:, 1]
    result[:, 1] = height * best[:, 1]

    _CACHED_HEILBRONN11 = result.copy()
    return result.copy()


# EVOLVE-BLOCK-END