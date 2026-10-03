# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_convex13() -> np.ndarray:
    """
    Deterministically construct thirteen points in a unit-area equilateral
    triangle using a 3-fold symmetric ring parameterization.

    The layout consists of the three hull vertices, their centroid, and three
    independently phased three-point rings.  Ring radii are restricted to the
    inscribed disk, guaranteeing containment in the convex hull.
    """
    rng = np.random.default_rng(20260315)

    root3 = np.sqrt(3.0)
    side = np.sqrt(4.0 / root3)
    height = 0.5 * root3 * side
    hull = np.array(
        [[0.0, 0.0], [side, 0.0], [0.5 * side, height]],
        dtype=np.float64,
    )
    center = np.array([0.5 * side, height / 3.0], dtype=np.float64)

    inradius = height / 3.0
    radius_lo = 0.075 * inradius
    radius_hi = 0.978 * inradius
    period = 2.0 * np.pi / 3.0
    offsets = np.array([0.0, period, 2.0 * period], dtype=np.float64)

    triples = np.array(
        [
            (i, j, k)
            for i in range(13)
            for j in range(i + 1, 13)
            for k in range(j + 1, 13)
        ],
        dtype=np.intp,
    )

    def reflect_unit(x: np.ndarray) -> np.ndarray:
        y = np.mod(x, 2.0)
        return np.where(y > 1.0, 2.0 - y, y)

    def canonical(z: np.ndarray) -> np.ndarray:
        q = np.array(z, dtype=np.float64, copy=True)
        q[..., :3] = reflect_unit(q[..., :3])
        q[..., 3:] = np.mod(q[..., 3:], 1.0)
        return q

    def layouts(z: np.ndarray) -> np.ndarray:
        z = canonical(z)
        count = z.shape[0]
        p = np.empty((count, 13, 2), dtype=np.float64)
        p[:, :3] = hull
        p[:, 3] = center

        radii = radius_lo + (radius_hi - radius_lo) * z[:, :3]
        phases = period * z[:, 3:]
        for ring in range(3):
            theta = phases[:, ring, None] + offsets[None, :]
            start = 4 + 3 * ring
            p[:, start:start + 3, 0] = (
                center[0] + radii[:, ring, None] * np.cos(theta)
            )
            p[:, start:start + 3, 1] = (
                center[1] + radii[:, ring, None] * np.sin(theta)
            )
        return p

    def area_values(z: np.ndarray) -> np.ndarray:
        p = layouts(z)
        a = p[:, triples[:, 0]]
        b = p[:, triples[:, 1]]
        c = p[:, triples[:, 2]]
        return 0.5 * np.abs(
            (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1])
            - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])
        )

    def evaluate(z: np.ndarray, tail_weight: float = 0.0):
        values = area_values(z)
        low = np.partition(values, 19, axis=1)[:, :20]
        minimum = low[:, 0]
        if tail_weight == 0.0:
            return minimum, minimum
        return minimum + tail_weight * low.mean(axis=1), minimum

    # Latin-hypercube coverage is complemented with radial/phase seeds chosen
    # to avoid phase locking between neighboring rings.
    population_size = 336
    generations = 650
    population = np.empty((population_size, 6), dtype=np.float64)
    for dimension in range(6):
        values = np.arange(population_size, dtype=np.float64) + rng.random(population_size)
        rng.shuffle(values)
        population[:, dimension] = values / population_size

    seeds = np.array(
        [
            [0.13, 0.39, 0.73, 0.03, 0.37, 0.79],
            [0.18, 0.48, 0.82, 0.11, 0.53, 0.91],
            [0.22, 0.56, 0.88, 0.21, 0.66, 0.07],
            [0.29, 0.63, 0.93, 0.34, 0.76, 0.15],
            [0.10, 0.45, 0.79, 0.28, 0.62, 0.96],
            [0.34, 0.67, 0.90, 0.06, 0.46, 0.84],
            [0.16, 0.51, 0.85, 0.42, 0.80, 0.19],
            [0.26, 0.59, 0.95, 0.17, 0.58, 0.89],
        ],
        dtype=np.float64,
    )
    population[:len(seeds)] = seeds
    population = canonical(population)

    _, minima = evaluate(population)
    best_index = int(np.argmax(minima))
    best = population[best_index].copy()
    best_value = float(minima[best_index])

    elite_count = 28
    for generation in range(generations):
        progress = generation / float(generations - 1)
        tail_weight = 0.115 * (1.0 - progress) ** 1.9
        merits, minima = evaluate(population, tail_weight)

        rank = np.argsort(merits)
        elite = population[rank[-elite_count:]]
        elite_choice = elite[rng.integers(0, elite_count, population_size)]

        ia = rng.permutation(population_size)
        ib = rng.permutation(population_size)
        ic = rng.permutation(population_size)

        factor = 0.78 - 0.48 * progress
        pull = 0.10 + 0.53 * progress

        rand_donor = population[ia] + factor * (population[ib] - population[ic])
        current_donor = (
            population
            + pull * (elite_choice - population)
            + factor * (population[ia] - population[ib])
        )
        elite_donor = elite_choice + (
            0.50 - 0.29 * progress
        ) * (population[ia] - population[ib])

        style = rng.random(population_size)
        donor = np.where(
            (style < 0.38)[:, None],
            rand_donor,
            np.where((style < 0.84)[:, None], current_donor, elite_donor),
        )
        donor = canonical(donor)

        crossover = rng.random((population_size, 6)) < (0.88 - 0.30 * progress)
        crossover[np.arange(population_size), rng.integers(0, 6, population_size)] = True
        trial = canonical(np.where(crossover, donor, population))

        trial_merit, trial_minimum = evaluate(trial, tail_weight)
        accepted = trial_merit >= merits
        population[accepted] = trial[accepted]

        if trial_minimum.size:
            trial_best = int(np.argmax(trial_minimum))
            if trial_minimum[trial_best] > best_value:
                best_value = float(trial_minimum[trial_best])
                best = trial[trial_best].copy()

        _, minima = evaluate(population)
        current = int(np.argmax(minima))
        if minima[current] > best_value:
            best_value = float(minima[current])
            best = population[current].copy()

        if generation in (210, 420):
            weak = np.argsort(minima)[:population_size // 5]
            population[weak] = rng.random((len(weak), 6))

    # Active-constraint local search.  Each ring controls three layout points:
    # ring r corresponds to parameter r (radius) and r+3 (phase).  Parameter
    # probabilities are recomputed from the currently tight triangle bundle.
    for iteration in range(360):
        progress = iteration / 359.0
        values = area_values(best[None])[0]
        tight_ids = np.argpartition(values, 27)[:28]
        tight_triples = triples[tight_ids]

        weights = np.full(6, 0.18, dtype=np.float64)
        for ring in range(3):
            ring_points = np.arange(4 + 3 * ring, 7 + 3 * ring)
            participation = np.isin(tight_triples, ring_points).sum()
            weights[ring] += participation
            weights[ring + 3] += participation
        weights /= weights.sum()

        radial_scale = 0.040 * (0.0010 / 0.040) ** progress
        phase_scale = 0.105 * (0.0014 / 0.105) ** progress
        trial_count = 112

        trials = np.repeat(best[None, :], trial_count, axis=0)
        for row in range(trial_count):
            # Most moves target one active ring; some coordinated moves permit
            # two simultaneously tight rings to separate without sacrificing
            # an already balanced third ring.
            count = 2 if rng.random() < 0.31 else 1
            chosen = rng.choice(6, size=count, replace=False, p=weights)
            delta = rng.normal(0.0, 1.0, size=count)
            for coordinate, amount in zip(chosen, delta):
                scale = radial_scale if coordinate < 3 else phase_scale
                trials[row, coordinate] += amount * scale

            if rng.random() < 0.24:
                ring = int(rng.choice(3, p=weights[:3] / weights[:3].sum()))
                trials[row, ring] += rng.normal(0.0, radial_scale)
                trials[row, ring + 3] += rng.normal(0.0, phase_scale)

        trials[0] = best
        trials = canonical(trials)
        _, trial_values = evaluate(trials)
        winner = int(np.argmax(trial_values))
        if trial_values[winner] > best_value + 1e-14:
            best_value = float(trial_values[winner])
            best = trials[winner].copy()

    # Small deterministic coordinate/radius-phase pattern search closes the
    # final nonsmooth gap without accepting surrogate-objective regressions.
    directions = np.array(
        [
            [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0],
            [1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0],
        ],
        dtype=np.float64,
    )
    directions[4:] /= np.sqrt(2.0)

    for scale in (0.012, 0.006, 0.0028, 0.0012):
        candidates = []
        for ring in range(3):
            for direction in directions:
                candidate = best.copy()
                candidate[ring] += scale * direction[0]
                candidate[ring + 3] += scale * direction[1]
                candidates.append(candidate)
        candidates = canonical(np.asarray(candidates))
        _, candidate_values = evaluate(candidates)
        winner = int(np.argmax(candidate_values))
        if candidate_values[winner] > best_value + 1e-14:
            best_value = float(candidate_values[winner])
            best = candidates[winner].copy()

    return layouts(best[None])[0]


# EVOLVE-BLOCK-END