# EVOLVE-BLOCK-START
import numpy as np


def _triangle_indices(n: int = 13) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    triples = np.array(
        [(i, j, k) for i in range(n - 2)
         for j in range(i + 1, n - 1)
         for k in range(j + 1, n)],
        dtype=np.intp,
    )
    return triples[:, 0], triples[:, 1], triples[:, 2]


def _fold_simplex(uv: np.ndarray) -> np.ndarray:
    """Map arbitrary unit-square coordinates into the unit simplex."""
    q = np.mod(uv, 2.0)
    q = np.where(q > 1.0, 2.0 - q, q)
    over = (q[..., 0] + q[..., 1]) > 1.0
    q = q.copy()
    q[over] = 1.0 - q[over]
    return q


def heilbronn_convex13() -> np.ndarray:
    """
    Deterministically construct thirteen points in a unit-area equilateral
    triangle.  The three fixed hull vertices have area one, so raw triangle
    areas are already the evaluator's normalized objective.
    """
    rng = np.random.default_rng(9132027)

    # Equilateral triangle with area exactly one:
    # side^2 * sqrt(3) / 4 = 1.
    side = np.sqrt(4.0 / np.sqrt(3.0))
    height = np.sqrt(3.0) * side / 2.0
    hull = np.array(
        [[0.0, 0.0], [side, 0.0], [0.5 * side, height]],
        dtype=np.float64,
    )

    ia, ib, ic = _triangle_indices(13)
    population_size = 132
    free_count = 10
    generations = 390

    def points_from_uv(uv: np.ndarray) -> np.ndarray:
        q = _fold_simplex(uv)
        result = np.empty((q.shape[0], 13, 2), dtype=np.float64)
        result[:, :3] = hull
        result[:, 3:] = (
            hull[0][None, None, :]
            + q[..., 0, None] * (hull[1] - hull[0])
            + q[..., 1, None] * (hull[2] - hull[0])
        )
        return result

    def evaluate_uv(uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pts = points_from_uv(uv)
        a = pts[:, ia]
        b = pts[:, ib]
        c = pts[:, ic]
        areas = 0.5 * np.abs(
            (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1])
            - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])
        )
        minimum = np.min(areas, axis=1)
        tail = np.partition(areas, 17, axis=1)[:, :18].mean(axis=1)
        return minimum, tail

    def score(minimum: np.ndarray, tail: np.ndarray, weight: float) -> np.ndarray:
        return minimum + weight * tail

    # Uniform simplex candidates plus a stratified family.
    population = rng.random((population_size, free_count, 2))
    population = _fold_simplex(population)

    # Nine broad cells plus a central point give useful initial spacing.
    lattice = np.array(
        [
            (0.10, 0.10), (0.34, 0.08), (0.62, 0.08),
            (0.08, 0.35), (0.34, 0.31), (0.58, 0.28),
            (0.08, 0.62), (0.29, 0.55), (0.48, 0.45),
            (1.0 / 3.0, 1.0 / 3.0),
        ],
        dtype=np.float64,
    )
    for row in range(28):
        candidate = lattice + rng.normal(0.0, 0.045 + 0.0015 * row,
                                         size=(free_count, 2))
        population[row] = _fold_simplex(candidate)

    # Three-fold-inspired seeds.  They are intentionally perturbed so the
    # optimizer is not trapped by exact symmetry or radial collinearities.
    center = np.array([0.5 * side, height / 3.0])
    basis = np.array([[side, 0.0], [0.5 * side, height]], dtype=np.float64)
    inverse_basis = np.linalg.inv(basis.T)
    for row in range(28, 54):
        coords = [center]
        radii = (0.18, 0.31, 0.43)
        phases = rng.uniform(-0.24, 0.24, size=3)
        for ring, radius in enumerate(radii):
            for turn in range(3):
                angle = np.pi / 2.0 + phases[ring] + turn * 2.0 * np.pi / 3.0
                coords.append(
                    center + radius * side * np.array([np.cos(angle), np.sin(angle)])
                )
        coords = np.asarray(coords)
        uv = (coords @ inverse_basis.T)
        population[row] = _fold_simplex(
            uv + rng.normal(0.0, 0.010, size=uv.shape)
        )

    minimum, tail = evaluate_uv(population)

    best_index = int(np.argmax(minimum))
    best_uv = population[best_index].copy()
    best_min = float(minimum[best_index])
    best_tail = float(tail[best_index])

    # Global current-to-elite differential evolution.
    for generation in range(generations):
        progress = generation / float(generations - 1)
        tail_weight = 0.19 * (1.0 - progress) ** 1.55 + 0.012
        values = score(minimum, tail, tail_weight)

        rank = np.argsort(values)
        elite_count = 14
        elite = rank[-elite_count:]
        saved = population[elite].copy()
        saved_min = minimum[elite].copy()
        saved_tail = tail[elite].copy()

        pbest = elite[rng.integers(elite_count, size=population_size)]
        r1 = rng.integers(population_size, size=population_size)
        r2 = rng.integers(population_size, size=population_size)
        r2 = (r2 + (r2 == r1)) % population_size

        factor = 0.78 - 0.34 * progress
        donor = (
            population
            + factor * (population[pbest] - population)
            + factor * (population[r1] - population[r2])
        )

        trial = population.copy()
        cross_probability = 0.92 - 0.16 * progress
        cross = rng.random((population_size, free_count, 2)) < cross_probability
        forced = rng.integers(free_count * 2, size=population_size)
        cross.reshape(population_size, -1)[np.arange(population_size), forced] = True
        trial = np.where(cross, donor, trial)
        trial = _fold_simplex(trial)

        trial_min, trial_tail = evaluate_uv(trial)
        trial_value = score(trial_min, trial_tail, tail_weight)
        accept = trial_value >= values

        population[accept] = trial[accept]
        minimum[accept] = trial_min[accept]
        tail[accept] = trial_tail[accept]

        # Preserve the generation's best patterns exactly.
        current_values = score(minimum, tail, tail_weight)
        weakest = np.argsort(current_values)[:elite_count]
        population[weakest] = saved
        minimum[weakest] = saved_min
        tail[weakest] = saved_tail

        generation_best = int(np.argmax(minimum))
        if (
            minimum[generation_best] > best_min + 1.0e-14
            or (
                abs(minimum[generation_best] - best_min) <= 1.0e-14
                and tail[generation_best] > best_tail
            )
        ):
            best_uv = population[generation_best].copy()
            best_min = float(minimum[generation_best])
            best_tail = float(tail[generation_best])

        # Controlled re-diversification while broad DE moves are still useful.
        if generation in (122, 238):
            current_values = score(minimum, tail, tail_weight)
            replace = np.argsort(current_values)[:population_size // 9]
            fresh = rng.random((len(replace), free_count, 2))
            population[replace] = _fold_simplex(fresh)
            minimum[replace], tail[replace] = evaluate_uv(population[replace])

    # Select only a few genuine maximin finalists for active local polishing.
    final_order = np.lexsort((tail, minimum))
    finalists = population[final_order[-4:]].copy()

    for candidate_start in finalists:
        current = candidate_start.copy()
        current_min, current_tail = evaluate_uv(current[None])
        current_min = float(current_min[0])
        current_tail = float(current_tail[0])

        for step in range(620):
            fraction = step / 619.0
            pts = points_from_uv(current[None])[0]
            a = pts[ia]
            b = pts[ib]
            c = pts[ic]
            areas = 0.5 * np.abs(
                (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
            )

            active = np.argpartition(areas, 11)[:12]
            involved = np.unique(
                np.concatenate((ia[active], ib[active], ic[active]))
            )
            free_active = involved[involved >= 3] - 3
            if free_active.size == 0:
                free_active = np.arange(free_count)

            batch_size = 24
            proposals = np.repeat(current[None], batch_size, axis=0)
            sigma = 0.030 * (1.0 - fraction) ** 1.65 + 0.00038

            for q in range(batch_size):
                if q < 18:
                    idx = int(free_active[rng.integers(free_active.size)])
                    proposals[q, idx] += rng.normal(0.0, sigma, size=2)
                else:
                    count = min(2, free_active.size)
                    ids = rng.choice(free_active, size=count, replace=False)
                    proposals[q, ids] += rng.normal(
                        0.0, 0.68 * sigma, size=(count, 2)
                    )

            proposals = _fold_simplex(proposals)
            prop_min, prop_tail = evaluate_uv(proposals)

            # Early lower-tail guidance permits active-set changes; the latter
            # half is a strict exact maximin hill climb.
            if fraction < 0.58:
                prop_value = prop_min + 0.026 * prop_tail
                cur_value = current_min + 0.026 * current_tail
                winner = int(np.argmax(prop_value))
                slack = 0.000045 * (1.0 - fraction) ** 2
                accept = prop_value[winner] >= cur_value - slack
            else:
                winner = int(np.argmax(prop_min))
                accept = (
                    prop_min[winner] > current_min + 1.0e-14
                    or (
                        abs(prop_min[winner] - current_min) <= 1.0e-14
                        and prop_tail[winner] > current_tail
                    )
                )

            if accept:
                current = proposals[winner]
                current_min = float(prop_min[winner])
                current_tail = float(prop_tail[winner])

            if (
                current_min > best_min + 1.0e-14
                or (
                    abs(current_min - best_min) <= 1.0e-14
                    and current_tail > best_tail
                )
            ):
                best_uv = current.copy()
                best_min = current_min
                best_tail = current_tail

    return points_from_uv(best_uv[None])[0]


# EVOLVE-BLOCK-END