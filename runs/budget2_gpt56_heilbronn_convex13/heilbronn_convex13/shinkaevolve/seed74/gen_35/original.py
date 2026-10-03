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


def _areas(points: np.ndarray,
           ia: np.ndarray,
           ib: np.ndarray,
           ic: np.ndarray) -> np.ndarray:
    a = points[..., ia, :]
    b = points[..., ib, :]
    c = points[..., ic, :]
    return 0.5 * np.abs(
        (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1])
        - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])
    )


def _reflect_unit(x: np.ndarray) -> np.ndarray:
    """Repeated reflection is robust to the occasional large DE donor."""
    x = np.mod(x, 2.0)
    return np.where(x > 1.0, 2.0 - x, x)


def heilbronn_convex13() -> np.ndarray:
    """
    Deterministically construct thirteen points in the unit square.

    The four fixed corners ensure that every returned configuration has
    convex-hull area one.  Consequently raw triangle areas used internally
    are exactly the requested hull-normalized triangle areas.
    """
    n = 13
    free_count = 9
    population_size = 192
    generations = 460
    rng = np.random.default_rng(13031957)

    ia, ib, ic = _triangle_arrays(n)
    corners = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=float,
    )

    # Nine-cell stratification gives each initial candidate broad coverage
    # without introducing exact grid collinearities.
    cells = np.array([(x, y) for y in range(3) for x in range(3)],
                     dtype=float)
    population = np.empty((population_size, n, 2), dtype=float)
    population[:, :4] = corners
    for p in range(population_size):
        permutation = rng.permutation(9)
        jitter = rng.uniform(0.11, 0.89, size=(free_count, 2))
        population[p, 4:] = (cells[permutation] + jitter) / 3.0

    # Independent starts prevent excessive bias toward a perturbed lattice.
    random_count = population_size // 4
    population[-random_count:, 4:] = rng.random((random_count, free_count, 2))

    def signature_batch(configs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = _areas(configs, ia, ib, ic)
        low = np.partition(values, 19, axis=1)[:, :20]
        return low[:, 0], np.mean(low, axis=1)

    def surrogate(configs: np.ndarray, tail_weight: float) -> np.ndarray:
        minimum, tail = signature_batch(configs)
        # Early generations protect a broad lower tail; this weight is reduced
        # later so final DE selection is almost exact maximin selection.
        return minimum + tail_weight * tail

    scores = surrogate(population, 0.25)

    # Current-to-elite differential evolution is less disruptive than
    # rand/1 for this small, highly constrained maximin problem.
    for generation in range(generations):
        progress = generation / float(generations - 1)
        # A broad tail objective avoids prematurely committing to a layout
        # having one acceptable bottleneck but many nearly degenerate triples.
        # The final coefficient leaves the exact minimum dominant.
        tail_weight = 0.23 * (1.0 - progress) ** 1.45 + 0.02
        scores = surrogate(population, tail_weight)

        rank = np.argsort(scores)
        elite_count = 12
        elite_indices = rank[-elite_count:]
        saved_elites = population[elite_indices].copy()
        saved_scores = scores[elite_indices].copy()

        pbest = elite_indices[rng.integers(elite_count, size=population_size)]
        r1 = rng.integers(population_size, size=population_size)
        r2 = rng.integers(population_size, size=population_size)
        r2 = (r2 + (r2 == r1)) % population_size

        factor = 0.82 - 0.34 * progress
        donor_free = (
            population[:, 4:]
            + factor * (population[pbest, 4:] - population[:, 4:])
            + factor * (population[r1, 4:] - population[r2, 4:])
        )
        donor_free = _reflect_unit(donor_free)

        trial = population.copy()
        cross = rng.random((population_size, free_count, 2)) < (
            0.91 - 0.13 * progress
        )
        force = rng.integers(free_count * 2, size=population_size)
        cross.reshape(population_size, -1)[np.arange(population_size), force] = True
        trial[:, 4:] = np.where(cross, donor_free, population[:, 4:])

        trial_scores = surrogate(trial, tail_weight)
        accepted = trial_scores >= scores
        population[accepted] = trial[accepted]
        scores[accepted] = trial_scores[accepted]

        # Preserve the pre-generation elite exactly.
        worst = np.argsort(scores)[:elite_count]
        population[worst] = saved_elites
        scores[worst] = saved_scores

        # A small deterministic immigration event in the early global phase.
        if generation in (105, 205):
            worst = np.argsort(scores)[:population_size // 10]
            for row in worst:
                perm = rng.permutation(9)
                jitter = rng.uniform(0.08, 0.92, size=(free_count, 2))
                population[row, :4] = corners
                population[row, 4:] = (cells[perm] + jitter) / 3.0
            scores[worst] = surrogate(population[worst], tail_weight)

    # Choose finalists using the true primary objective, not merely surrogate.
    mins, tails = signature_batch(population)
    ordering = np.lexsort((tails, mins))
    finalists = population[ordering[-10:]].copy()

    best = finalists[0].copy()
    best_min, best_tail = signature_batch(best[None, ...])
    best_min = float(best_min[0])
    best_tail = float(best_tail[0])

    # Active-constraint batch polish.  Each batch supplies alternatives around
    # points found in the currently worst triangles, avoiding slow blind moves.
    for start_id, start in enumerate(finalists):
        current = start.copy()
        cur_min_arr, cur_tail_arr = signature_batch(current[None, ...])
        cur_min = float(cur_min_arr[0])
        cur_tail = float(cur_tail_arr[0])
        cur_score = cur_min + 0.035 * cur_tail

        for step in range(1550):
            fraction = step / 1549.0
            sigma = 0.038 * (1.0 - fraction) ** 1.65 + 0.00055
            current_areas = _areas(current, ia, ib, ic)
            active = np.argpartition(current_areas, 13)[:14]
            active_points = np.unique(
                np.concatenate((ia[active], ib[active], ic[active]))
            )
            active_free = active_points[active_points >= 4]
            if active_free.size == 0:
                active_free = np.arange(4, 13)

            batch_size = 16
            proposals = np.repeat(current[None, :, :], batch_size, axis=0)
            for q in range(batch_size):
                # Most proposals alter one active point; a minority jointly
                # resolve two competing active constraints.
                if q < 12:
                    point_id = int(active_free[rng.integers(active_free.size)])
                    proposals[q, point_id] += rng.normal(0.0, sigma, size=2)
                else:
                    ids = rng.choice(active_free,
                                     size=min(2, active_free.size),
                                     replace=False)
                    proposals[q, ids] += rng.normal(
                        0.0, sigma * 0.72, size=(len(ids), 2)
                    )

            proposals[:, 4:] = _reflect_unit(proposals[:, 4:])
            prop_min, prop_tail = signature_batch(proposals)
            prop_score = prop_min + 0.035 * prop_tail
            chosen = int(np.argmax(prop_score))

            # A small early tolerance allows changes of active constraint sets;
            # final iterations are strict maximin hill climbing.
            tolerance = 0.00010 * (1.0 - fraction) ** 2
            if prop_score[chosen] >= cur_score - tolerance:
                current = proposals[chosen]
                cur_min = float(prop_min[chosen])
                cur_tail = float(prop_tail[chosen])
                cur_score = float(prop_score[chosen])

            if (
                cur_min > best_min + 1e-13
                or (
                    abs(cur_min - best_min) <= 1e-13
                    and cur_tail > best_tail
                )
            ):
                best = current.copy()
                best_min = cur_min
                best_tail = cur_tail

    best[:4] = corners
    return np.asarray(best, dtype=float)


# EVOLVE-BLOCK-END