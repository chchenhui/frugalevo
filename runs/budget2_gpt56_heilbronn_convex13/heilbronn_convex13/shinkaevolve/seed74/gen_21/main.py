# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_convex13() -> np.ndarray:
    """
    Construct a deterministic 13-point maximin arrangement inside an
    equilateral convex region of area exactly one.

    Three hull vertices, the centroid, and three independently phased
    3-fold rings give a compact six-parameter family.  All ring points are
    confined to the centroid's inscribed disk, so containment is guaranteed.
    """
    rng = np.random.default_rng(20260314)

    root3 = np.sqrt(3.0)

    # Equilateral triangle with side length chosen so area is exactly one.
    side = np.sqrt(4.0 / root3)
    height = 0.5 * root3 * side
    hull = np.array(
        [[0.0, 0.0], [side, 0.0], [0.5 * side, height]],
        dtype=float,
    )
    center = np.array([0.5 * side, height / 3.0], dtype=float)

    # Its inradius is height / 3.  Staying slightly inside it makes every
    # complete circular ring safely contained in the triangular hull.
    inradius = height / 3.0
    radius_lo = 0.10 * inradius
    radius_hi = 0.965 * inradius
    angle_period = 2.0 * np.pi / 3.0
    offsets = np.array([0.0, angle_period, 2.0 * angle_period])

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
        """Reflect real coordinates into [0,1], avoiding boundary clipping."""
        y = np.mod(x, 2.0)
        return np.where(y > 1.0, 2.0 - y, y)

    def canonical(z: np.ndarray) -> np.ndarray:
        """
        First three normalized coordinates are radial variables and are
        reflected; final coordinates are phase variables and are periodic.
        """
        q = z.copy()
        q[..., :3] = reflect_unit(q[..., :3])
        q[..., 3:] = np.mod(q[..., 3:], 1.0)
        return q

    def layouts(z: np.ndarray) -> np.ndarray:
        """Convert a batch of normalized six-vectors into 13-point layouts."""
        z = canonical(z)
        count = z.shape[0]
        result = np.empty((count, 13, 2), dtype=float)
        result[:, :3] = hull
        result[:, 3] = center

        radii = radius_lo + (radius_hi - radius_lo) * z[:, :3]
        phases = angle_period * z[:, 3:]

        for ring in range(3):
            theta = phases[:, ring, None] + offsets[None, :]
            begin = 4 + 3 * ring
            result[:, begin:begin + 3, 0] = (
                center[0] + radii[:, ring, None] * np.cos(theta)
            )
            result[:, begin:begin + 3, 1] = (
                center[1] + radii[:, ring, None] * np.sin(theta)
            )
        return result

    def triangle_values(z: np.ndarray) -> np.ndarray:
        p = layouts(z)
        a = p[:, triples[:, 0]]
        b = p[:, triples[:, 1]]
        c = p[:, triples[:, 2]]
        # Hull area is exactly one, hence these are normalized areas already.
        return 0.5 * np.abs(
            (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1])
            - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])
        )

    def score_batch(z: np.ndarray, tail_weight: float = 0.0):
        values = triangle_values(z)
        low = np.partition(values, 15, axis=1)[:, :16]
        minimum = low.min(axis=1)
        merit = minimum + tail_weight * low.mean(axis=1)
        return merit, minimum

    # Latin-hypercube initialization gives coverage in both radial and phase
    # dimensions, while several structured seeds preserve useful ring spacing.
    population_size = 300
    generations = 570
    population = np.empty((population_size, 6), dtype=float)
    for coordinate in range(6):
        strata = (np.arange(population_size) + rng.random(population_size))
        rng.shuffle(strata)
        population[:, coordinate] = strata / population_size

    seeds = np.array(
        [
            [0.18, 0.49, 0.83, 0.04, 0.37, 0.84],
            [0.24, 0.57, 0.91, 0.15, 0.59, 0.96],
            [0.12, 0.44, 0.77, 0.31, 0.67, 0.08],
            [0.30, 0.63, 0.88, 0.08, 0.48, 0.77],
            [0.20, 0.54, 0.86, 0.23, 0.71, 0.93],
            [0.36, 0.69, 0.95, 0.42, 0.81, 0.16],
        ],
        dtype=float,
    )
    population[:len(seeds)] = seeds
    population = canonical(population)

    _, minima = score_batch(population)
    best_index = int(np.argmax(minima))
    best = population[best_index].copy()
    best_value = float(minima[best_index])

    for generation in range(generations):
        progress = generation / float(generations - 1)
        tail_weight = 0.105 * (1.0 - progress) ** 1.7
        merits, minima = score_batch(population, tail_weight)

        ranking = np.argsort(merits)
        elite_count = 24
        elite = population[ranking[-elite_count:]]
        elite_pick = elite[rng.integers(0, elite_count, population_size)]

        r1 = rng.integers(0, population_size, population_size)
        r2 = rng.integers(0, population_size, population_size)
        r3 = rng.integers(0, population_size, population_size)
        r2 = (r2 + (r2 == r1)) % population_size
        r3 = (r3 + (r3 == r1) + (r3 == r2)) % population_size

        f = 0.72 - 0.43 * progress
        donor_rand = population[r1] + f * (population[r2] - population[r3])
        donor_current = (
            population
            + (0.12 + 0.47 * progress) * (elite_pick - population)
            + f * (population[r1] - population[r2])
        )
        donor_elite = elite_pick + (
            0.42 - 0.20 * progress
        ) * (population[r1] - population[r2])

        # A per-member mutation portfolio retains global DE diversity while
        # applying elite guidance only to part of the population.
        mode = rng.random(population_size)
        donor = np.where(
            (mode < 0.46)[:, None],
            donor_rand,
            np.where((mode < 0.80)[:, None], donor_current, donor_elite),
        )
        donor = canonical(donor)

        crossover_rate = 0.83 - 0.27 * progress
        cross = rng.random((population_size, 6)) < crossover_rate
        cross[np.arange(population_size), rng.integers(0, 6, population_size)] = True
        trial = canonical(np.where(cross, donor, population))

        trial_merits, trial_minima = score_batch(trial, tail_weight)
        accept = trial_merits >= merits
        population[accept] = trial[accept]

        if trial_minima[accept].size:
            idx = int(np.argmax(trial_minima))
            if trial_minima[idx] > best_value:
                best_value = float(trial_minima[idx])
                best = trial[idx].copy()

        # Preserve the strongest current true-minimum candidate independently
        # of the temporary lower-tail surrogate.
        _, minima = score_batch(population)
        current = int(np.argmax(minima))
        if minima[current] > best_value:
            best_value = float(minima[current])
            best = population[current].copy()

        # Controlled refresh avoids a fully phase-locked population.
        if generation in (190, 380):
            worst = np.argsort(minima)[:population_size // 6]
            population[worst] = rng.random((len(worst), 6))

    # Deterministic active-constraint bundle polishing.  Soft-min gradients
    # aggregate several limiting triangle constraints, unlike a one-triangle
    # finite-difference search which tends to oscillate at nonsmooth ties.
    def smooth_merit(z: np.ndarray, temperature: float) -> float:
        values = triangle_values(canonical(z[None, :]))[0]
        shifted = values - values.min()
        return float(
            values.min()
            - temperature * np.log(np.exp(-shifted / temperature).sum())
        )

    for stage, temperature in enumerate((0.0018, 0.0010, 0.00055, 0.00028)):
        h = 0.0014 / (1.0 + 0.35 * stage)
        base_step = 0.115 / (1.0 + 0.40 * stage)

        for _ in range(42):
            gradient = np.empty(6, dtype=float)
            for coordinate in range(6):
                plus = best.copy()
                minus = best.copy()
                plus[coordinate] += h
                minus[coordinate] -= h
                gradient[coordinate] = (
                    smooth_merit(plus, temperature)
                    - smooth_merit(minus, temperature)
                ) / (2.0 * h)

            norm = float(np.linalg.norm(gradient))
            if norm < 1e-14:
                break

            direction = gradient / norm
            improved = False
            for multiplier in (1.0, 0.50, 0.24, 0.10, -0.08):
                candidate = canonical(best + multiplier * base_step * direction)
                _, candidate_value = score_batch(candidate[None, :])
                if candidate_value[0] > best_value + 1e-14:
                    best = candidate
                    best_value = float(candidate_value[0])
                    improved = True
                    break
            if not improved:
                base_step *= 0.68
                if base_step < 0.002:
                    break

    # Sparse deterministic local proposals provide a final nonsmooth polish.
    for iteration in range(110):
        scale = 0.030 * (0.0015 / 0.030) ** (iteration / 109.0)
        trials = np.repeat(best[None, :], 40, axis=0)
        mask = rng.random((40, 6)) < 0.32
        mask[0, rng.integers(0, 6)] = True
        trials = canonical(trials + mask * rng.normal(0.0, scale, size=(40, 6)))
        _, values = score_batch(trials)
        winner = int(np.argmax(values))
        if values[winner] > best_value:
            best_value = float(values[winner])
            best = trials[winner].copy()

    return layouts(best[None, :])[0]


# EVOLVE-BLOCK-END