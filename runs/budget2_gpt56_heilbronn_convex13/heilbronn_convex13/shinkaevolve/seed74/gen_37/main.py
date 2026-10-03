# EVOLVE-BLOCK-START
import numpy as np


def _triangle_arrays(n: int = 13) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ia, ib, ic = [], [], []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                ia.append(i)
                ib.append(j)
                ic.append(k)
    return (
        np.asarray(ia, dtype=np.intp),
        np.asarray(ib, dtype=np.intp),
        np.asarray(ic, dtype=np.intp),
    )


def _reflect_unit(x: np.ndarray) -> np.ndarray:
    """Reflect arbitrary real coordinates into the closed unit interval."""
    x = np.mod(x, 2.0)
    return np.where(x > 1.0, 2.0 - x, x)


def heilbronn_convex13() -> np.ndarray:
    """
    Deterministically construct thirteen points in the unit square.

    Four square corners are fixed throughout optimization.  Their presence
    guarantees a convex hull of area exactly one, so raw triangle area is the
    same as the normalized objective used by the evaluator.
    """
    rng = np.random.default_rng(13031957)

    n = 13
    free_count = 9
    population_size = 224
    generations = 430

    ia, ib, ic = _triangle_arrays(n)
    corners = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=np.float64,
    )

    def layouts_from_free(free: np.ndarray) -> np.ndarray:
        count = free.shape[0]
        result = np.empty((count, n, 2), dtype=np.float64)
        result[:, :4] = corners
        result[:, 4:] = free
        return result

    def area_batch_from_free(free: np.ndarray) -> np.ndarray:
        points = layouts_from_free(free)
        a = points[:, ia]
        b = points[:, ib]
        c = points[:, ic]
        return 0.5 * np.abs(
            (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1])
            - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])
        )

    def signature(free: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        areas = area_batch_from_free(free)
        low = np.partition(areas, 17, axis=1)[:, :18]
        return low[:, 0], np.mean(low, axis=1), areas

    def objective(minimum: np.ndarray,
                  tail: np.ndarray,
                  weight: float) -> np.ndarray:
        return minimum + weight * tail

    # Stratified starts provide broad coverage without exact lattice lines.
    cells = np.array(
        [(x, y) for y in range(3) for x in range(3)],
        dtype=np.float64,
    )
    population = np.empty((population_size, free_count, 2), dtype=np.float64)

    for row in range(population_size):
        order = rng.permutation(free_count)
        jitter = rng.uniform(0.075, 0.925, size=(free_count, 2))
        population[row] = (cells[order] + jitter) / 3.0

    # Preserve some fully independent basins in addition to grid-derived ones.
    random_rows = population_size // 5
    population[-random_rows:] = rng.uniform(
        0.012, 0.988, size=(random_rows, free_count, 2)
    )

    # A few mildly asymmetric seeds avoid over-commitment to square symmetry.
    base = np.array(
        [
            [0.16, 0.15], [0.48, 0.13], [0.82, 0.20],
            [0.13, 0.49], [0.51, 0.46], [0.87, 0.53],
            [0.21, 0.81], [0.53, 0.86], [0.79, 0.77],
        ],
        dtype=np.float64,
    )
    for row in range(18):
        population[row] = _reflect_unit(
            base + rng.normal(0.0, 0.020 + 0.002 * row, size=(free_count, 2))
        )

    minimum, tail, _ = signature(population)

    # Current-to-pbest DE combines global pattern recombination with the
    # tail-aware selection needed before final exact maximin polishing.
    for generation in range(generations):
        fraction = generation / float(generations - 1)
        tail_weight = 0.22 * (1.0 - fraction) ** 1.55 + 0.012
        scores = objective(minimum, tail, tail_weight)

        elite_count = 16
        ranked = np.argsort(scores)
        elite = ranked[-elite_count:]

        pbest = elite[rng.integers(0, elite_count, size=population_size)]
        r1 = rng.integers(0, population_size, size=population_size)
        r2 = rng.integers(0, population_size, size=population_size)
        r2 = (r2 + (r1 == r2)) % population_size

        factor = 0.76 - 0.31 * fraction
        donor = (
            population
            + factor * (population[pbest] - population)
            + factor * (population[r1] - population[r2])
        )

        # Late small isotropic variation is useful when DE differences become
        # nearly aligned with a limited active-triangle set.
        donor += rng.normal(
            0.0,
            0.011 * (1.0 - fraction) ** 2 + 0.00045,
            size=donor.shape,
        )
        donor = _reflect_unit(donor)

        crossover_rate = 0.93 - 0.12 * fraction
        mask = rng.random((population_size, free_count, 2)) < crossover_rate
        forced = rng.integers(0, free_count * 2, size=population_size)
        mask.reshape(population_size, -1)[np.arange(population_size), forced] = True

        trial = np.where(mask, donor, population)
        trial_min, trial_tail, _ = signature(trial)
        trial_score = objective(trial_min, trial_tail, tail_weight)

        accepted = trial_score >= scores
        population[accepted] = trial[accepted]
        minimum[accepted] = trial_min[accepted]
        tail[accepted] = trial_tail[accepted]

        # Periodic deterministic immigrants preserve diversity cheaply.
        if generation in (105, 220, 320):
            scores = objective(minimum, tail, tail_weight)
            replace = np.argsort(scores)[:population_size // 9]
            for row in replace:
                order = rng.permutation(free_count)
                jitter = rng.uniform(0.06, 0.94, size=(free_count, 2))
                population[row] = (cells[order] + jitter) / 3.0
            minimum[replace], tail[replace], _ = signature(population[replace])

    # Retain several genuinely distinct high-minimum basins for local search.
    final_score = objective(minimum, tail, 0.018)
    ordered = np.argsort(final_score)[::-1]
    finalists = []
    for index in ordered:
        candidate = population[index]
        if all(np.max(np.abs(candidate - previous)) > 0.035
               for previous in finalists):
            finalists.append(candidate.copy())
        if len(finalists) == 7:
            break
    while len(finalists) < 7:
        finalists.append(population[ordered[len(finalists)]].copy())

    best_free = finalists[0].copy()
    best_min, best_tail, _ = signature(best_free[None])
    best_min = float(best_min[0])
    best_tail = float(best_tail[0])

    # Active-constraint stochastic pattern search.  The early acceptance
    # temperature is inherited from annealing approaches; it permits active
    # triangle sets to change before strict maximin hill climbing takes over.
    for start in finalists:
        current = start.copy()
        cur_min_arr, cur_tail_arr, current_areas_batch = signature(current[None])
        cur_min = float(cur_min_arr[0])
        cur_tail = float(cur_tail_arr[0])
        current_areas = current_areas_batch[0]

        for step in range(1120):
            progress = step / 1119.0
            sigma = 0.033 * (1.0 - progress) ** 1.72 + 0.00038

            active = np.argpartition(current_areas, 15)[:16]
            ids = np.unique(np.concatenate((ia[active], ib[active], ic[active])))
            free_ids = ids[ids >= 4] - 4
            if free_ids.size == 0:
                free_ids = np.arange(free_count)

            batch_size = 18
            proposals = np.repeat(current[None], batch_size, axis=0)

            # Single active-point moves dominate, but coordinated moves resolve
            # competing bottlenecks that cannot improve independently.
            for q in range(batch_size):
                if q < 13 or free_ids.size == 1:
                    point = int(free_ids[rng.integers(free_ids.size)])
                    proposals[q, point] += rng.normal(0.0, sigma, size=2)
                else:
                    chosen = rng.choice(
                        free_ids, size=min(2, free_ids.size), replace=False
                    )
                    proposals[q, chosen] += rng.normal(
                        0.0, sigma * 0.70, size=(len(chosen), 2)
                    )

            proposals = _reflect_unit(proposals)
            prop_min, prop_tail, prop_areas = signature(proposals)

            # Tail preference prevents cycling among arrangements with equal
            # numerical minima but many additional weak constraints.
            prop_value = prop_min + 0.028 * prop_tail
            cur_value = cur_min + 0.028 * cur_tail
            chosen = int(np.argmax(prop_value))

            temperature = 0.000095 * (1.0 - progress) ** 2
            delta = float(prop_value[chosen] - cur_value)
            take = delta >= 0.0
            if not take and temperature > 1.0e-10:
                take = rng.random() < np.exp(delta / temperature)

            if take:
                current = proposals[chosen]
                cur_min = float(prop_min[chosen])
                cur_tail = float(prop_tail[chosen])
                current_areas = prop_areas[chosen]

            if (
                cur_min > best_min + 1.0e-13
                or (
                    abs(cur_min - best_min) <= 1.0e-13
                    and cur_tail > best_tail
                )
            ):
                best_free = current.copy()
                best_min = cur_min
                best_tail = cur_tail

    result = np.empty((n, 2), dtype=np.float64)
    result[:4] = corners
    result[4:] = best_free
    return result


# EVOLVE-BLOCK-END