# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_convex13() -> np.ndarray:
    """
    Construct a deterministic 13-point maximin configuration in the unit
    square.  The four square corners are fixed, so the convex hull has unit
    area and the minimum triangle area is already normalized.
    """
    rng = np.random.default_rng(seed=13051957)

    corners = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=float,
    )
    triples = np.array(
        [(i, j, k) for i in range(13) for j in range(i + 1, 13)
         for k in range(j + 1, 13)],
        dtype=np.intp,
    )

    def triangle_areas_batch(interior: np.ndarray) -> np.ndarray:
        """Return all triangle areas for a batch of nine-point interiors."""
        count = interior.shape[0]
        layouts = np.empty((count, 13, 2), dtype=float)
        layouts[:, :4] = corners
        layouts[:, 4:] = interior

        a = layouts[:, triples[:, 0]]
        b = layouts[:, triples[:, 1]]
        c = layouts[:, triples[:, 2]]
        return 0.5 * np.abs(
            (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1])
            - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])
        )

    def score_batch(interior: np.ndarray, blend: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        values = triangle_areas_batch(interior)
        minima = values.min(axis=1)
        if blend <= 0.0:
            return minima, minima

        tight = np.partition(values, 11, axis=1)[:, :12]
        merit = minima + blend * tight.mean(axis=1)
        return merit, minima

    population_size = 112
    generations = 760
    population = rng.uniform(0.025, 0.975, size=(population_size, 9, 2))

    grid = np.array(
        [[x, y] for y in (0.18, 0.50, 0.82) for x in (0.18, 0.50, 0.82)],
        dtype=float,
    )
    for index in range(28):
        scale = 0.014 + 0.0024 * index
        population[index] = np.clip(
            grid + rng.normal(0.0, scale, size=(9, 2)),
            0.008,
            0.992,
        )

    _, minima = score_batch(population)
    best_index = int(np.argmax(minima))
    best = population[best_index].copy()
    best_score = float(minima[best_index])

    for generation in range(generations):
        fraction = generation / float(generations - 1)
        blend = 0.075 * (1.0 - fraction) ** 1.6
        step = 0.105 * (0.018 / 0.105) ** fraction

        noise = rng.normal(0.0, step, size=(population_size, 2, 9, 2))
        attraction = rng.uniform(
            0.0, 0.14 * (1.0 - 0.45 * fraction),
            size=(population_size, 2, 1, 1),
        )
        local = np.clip(
            population[:, None] + noise
            + attraction * (best[None, None] - population[:, None]),
            0.0,
            1.0,
        )

        donor_a = rng.integers(0, population_size, size=population_size)
        donor_b = rng.integers(0, population_size, size=population_size)
        differential = np.clip(
            population
            + (0.62 - 0.39 * fraction)
            * (population[donor_a] - population[donor_b])
            + rng.uniform(0.0, 0.12, size=(population_size, 1, 1))
            * (best[None] - population),
            0.0,
            1.0,
        )

        candidates = np.concatenate(
            (population[:, None], local, differential[:, None]),
            axis=1,
        )
        flat = candidates.reshape(-1, 9, 2)
        merits, candidate_minima = score_batch(flat, blend)
        merits = merits.reshape(population_size, 4)
        candidate_minima = candidate_minima.reshape(population_size, 4)

        selected = np.argmax(merits, axis=1)
        population = candidates[np.arange(population_size), selected]
        minima = candidate_minima[np.arange(population_size), selected]

        current = int(np.argmax(minima))
        if minima[current] > best_score:
            best_score = float(minima[current])
            best = population[current].copy()

        if generation in (250, 500):
            worst = np.argsort(minima)[: population_size // 5]
            population[worst] = rng.uniform(
                0.02, 0.98, size=(len(worst), 9, 2)
            )

    # Small stochastic moves selectively perturb points, preserving most
    # non-active triangle constraints during fine maximin refinement.
    for iteration in range(360):
        fraction = iteration / 359.0
        step = 0.028 * (0.0018 / 0.028) ** fraction
        trial_count = 40
        trials = np.repeat(best[None], trial_count, axis=0)
        mask = rng.random((trial_count, 9, 1)) < 0.17
        mask[0, rng.integers(0, 9), 0] = True
        trials = np.clip(
            trials + mask * rng.normal(0.0, step, size=(trial_count, 9, 2)),
            0.0,
            1.0,
        )
        _, values = score_batch(trials)
        index = int(np.argmax(values))
        if values[index] > best_score:
            best_score = float(values[index])
            best = trials[index].copy()

    # Deterministic pattern search removes residual random-search noise.
    directions = np.array(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0],
         [1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]],
        dtype=float,
    )
    directions[4:] /= np.sqrt(2.0)

    step = 0.010
    for _ in range(6):
        improved = True
        while improved:
            improved = False
            for point_index in range(9):
                trials = np.repeat(best[None], 8, axis=0)
                trials[:, point_index] = np.clip(
                    trials[:, point_index] + step * directions,
                    0.0,
                    1.0,
                )
                _, values = score_batch(trials)
                index = int(np.argmax(values))
                if values[index] > best_score + 1e-14:
                    best_score = float(values[index])
                    best = trials[index].copy()
                    improved = True
        step *= 0.52

    return np.vstack((corners, best))


# EVOLVE-BLOCK-END