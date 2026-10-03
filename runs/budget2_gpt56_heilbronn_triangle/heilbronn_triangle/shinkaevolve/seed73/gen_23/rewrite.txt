# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """
    Deterministically construct eleven points in the reference equilateral
    triangle.  Optimization is performed in simplex coordinates:
        (x, y) = (u + v/2, sqrt(3)*v/2),
    where u >= 0, v >= 0, u + v <= 1.

    The first three points are the container vertices.  Three additional
    points are explicitly placed on the three sides, and five are free
    barycentric points.
    """
    rng = np.random.default_rng(11031987)
    h = np.sqrt(3.0) * 0.5
    n = 11
    pop_size = 256
    dimensions = 13

    vertices_uv = np.array(
        ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
        dtype=float,
    )
    triples = np.array(
        [(i, j, k) for i in range(n) for j in range(i + 1, n)
         for k in range(j + 1, n)],
        dtype=np.intp,
    )

    def reflect_unit(x: np.ndarray) -> np.ndarray:
        """Reflect parameters into [0,1] without clipping accumulation."""
        x = np.mod(x, 2.0)
        return np.where(x > 1.0, 2.0 - x, x)

    def decode(params: np.ndarray) -> np.ndarray:
        """
        Decode 13 parameters:
          0..2    side positions on base, right side, left side
          3..12   (q,v) coordinates for five simplex interior points.
        """
        m = len(params)
        p = np.empty((m, n, 2), dtype=float)
        p[:, :3] = vertices_uv

        t0 = params[:, 0]
        t1 = params[:, 1]
        t2 = params[:, 2]

        # Base, right edge from (1,0) to (0,1), left edge from (0,1) to (0,0).
        p[:, 3, 0] = t0
        p[:, 3, 1] = 0.0
        p[:, 4, 0] = 1.0 - t1
        p[:, 4, 1] = t1
        p[:, 5, 0] = 0.0
        p[:, 5, 1] = t2

        z = params[:, 3:].reshape(m, 5, 2)
        v = z[:, :, 1]
        u = z[:, :, 0] * (1.0 - v)
        p[:, 6:, 0] = u
        p[:, 6:, 1] = v
        return p

    def determinant_areas(params: np.ndarray) -> np.ndarray:
        p = decode(params)
        a = p[:, triples[:, 0]]
        b = p[:, triples[:, 1]]
        c = p[:, triples[:, 2]]
        return np.abs(
            (b[:, :, 0] - a[:, :, 0]) * (c[:, :, 1] - a[:, :, 1])
            - (b[:, :, 1] - a[:, :, 1]) * (c[:, :, 0] - a[:, :, 0])
        )

    def evaluate(params: np.ndarray, temperature: float):
        """
        Return exact max-min quality and a soft-min ranking objective.
        The soft objective helps separate configurations with several nearly
        active constraints during the exploratory phase.
        """
        ar = determinant_areas(params)
        minimum = ar.min(axis=1)
        if temperature <= 0.0:
            return minimum, minimum
        soft = minimum - temperature * np.log(
            np.exp(-(ar - minimum[:, None]) / temperature).sum(axis=1)
        )
        return minimum, soft

    # Structured initial population.  Explicit side support points reduce the
    # difficulty of finding useful hull configurations through random drift.
    population = rng.random((pop_size, dimensions))
    population[:, :3] = np.array((0.22, 0.53, 0.76)) + rng.normal(
        0.0, 0.19, size=(pop_size, 3)
    )

    # Several diverse quasi-grid-like starts, followed by noise.
    seeds = np.array(
        (
            (0.17, 0.16), (0.48, 0.12), (0.79, 0.16),
            (0.14, 0.47), (0.42, 0.38), (0.68, 0.31),
            (0.16, 0.73), (0.39, 0.57), (0.61, 0.20),
            (0.30, 0.27),
        ),
        dtype=float,
    )
    for r in range(64):
        population[r, :3] = np.array((0.19, 0.52, 0.81)) + rng.normal(
            0.0, 0.075, size=3
        )
        chosen = seeds[rng.choice(len(seeds), 5, replace=False)]
        uv = chosen + rng.normal(0.0, 0.075, size=(5, 2))
        uv = np.maximum(uv, 0.0)
        total = uv.sum(axis=1)
        over = total > 0.96
        uv[over] /= (total[over, None] / 0.96)
        population[r, 3::2] = uv[:, 0] / np.maximum(1.0 - uv[:, 1], 1e-12)
        population[r, 4::2] = uv[:, 1]

    population = reflect_unit(population)
    exact, rank = evaluate(population, 0.0045)
    best_i = int(np.argmax(exact))
    best = population[best_i].copy()
    best_value = float(exact[best_i])

    member = np.arange(pop_size)
    generations = 1900

    for generation in range(generations):
        progress = generation / float(generations - 1)
        temperature = 0.0048 * (1.0 - progress) + 0.00045

        exact, rank = evaluate(population, temperature)
        order = np.argsort(rank)
        elite_count = max(12, int(pop_size * (0.28 - 0.13 * progress)))
        pbest = order[rng.integers(pop_size - elite_count, pop_size, size=pop_size)]

        r1 = rng.permutation(member)
        r2 = rng.permutation(member)
        bad = (r1 == member) | (r2 == member) | (r1 == r2)
        while np.any(bad):
            r1[bad] = rng.integers(0, pop_size, np.count_nonzero(bad))
            r2[bad] = rng.integers(0, pop_size, np.count_nonzero(bad))
            bad = (r1 == member) | (r2 == member) | (r1 == r2)

        scale = (0.82 - 0.35 * progress) + rng.normal(0.0, 0.055, pop_size)
        scale = np.clip(scale, 0.32, 0.90)[:, None]
        donor = (
            population
            + scale * (population[pbest] - population)
            + scale * (population[r1] - population[r2])
        )
        donor = reflect_unit(donor)

        crossover_rate = 0.93 - 0.19 * progress
        take = rng.random((pop_size, dimensions)) < crossover_rate
        take[member, rng.integers(0, dimensions, pop_size)] = True
        trial = np.where(take, donor, population)

        trial_exact, trial_rank = evaluate(trial, temperature)

        # Smooth selection early; exact quality becomes dominant late.
        if generation < 1350:
            accepted = trial_rank >= rank
        else:
            accepted = trial_exact >= exact

        population[accepted] = trial[accepted]
        exact[accepted] = trial_exact[accepted]

        candidate = int(np.argmax(exact))
        if exact[candidate] > best_value:
            best_value = float(exact[candidate])
            best = population[candidate].copy()

        # Exact incumbent is preserved and restarted neighborhoods maintain
        # useful diversity after the differential population contracts.
        if generation > 0 and generation % 260 == 0:
            worst = np.argsort(exact)[:pop_size // 4]
            restart = np.repeat(best[None, :], len(worst), axis=0)
            sigma = 0.080 * (1.0 - progress) + 0.014
            restart += rng.normal(0.0, sigma, size=restart.shape)
            restart = reflect_unit(restart)
            population[worst] = restart
            population[0] = best

    # Active-set hill refinement.  Only coordinates belonging to points in
    # the currently tight triangle constraints receive the strongest noise.
    step = 0.030
    for iteration in range(560):
        ar = determinant_areas(best[None, :])[0]
        cutoff = np.partition(ar, 7)[7] + 1e-12
        tight_triples = triples[ar <= cutoff]
        active_points = np.unique(tight_triples.ravel())

        active_dims = []
        for point in active_points:
            if point == 3:
                active_dims.append(0)
            elif point == 4:
                active_dims.append(1)
            elif point == 5:
                active_dims.append(2)
            elif point >= 6:
                base = 3 + 2 * (point - 6)
                active_dims.extend((base, base + 1))

        batch = 144
        candidates = np.repeat(best[None, :], batch, axis=0)
        candidates += rng.normal(0.0, step * 0.24, size=candidates.shape)
        if active_dims:
            candidates[:, active_dims] += rng.normal(
                0.0, step, size=(batch, len(active_dims))
            )
        candidates = reflect_unit(candidates)

        candidate_values, _ = evaluate(candidates, 0.0)
        winner = int(np.argmax(candidate_values))
        if candidate_values[winner] > best_value:
            best = candidates[winner]
            best_value = float(candidate_values[winner])
            step = min(0.060, step * 1.018)
        else:
            step = max(0.00065, step * 0.988)

    uv = decode(best[None, :])[0]
    points = np.empty((n, 2), dtype=float)
    points[:, 0] = uv[:, 0] + 0.5 * uv[:, 1]
    points[:, 1] = h * uv[:, 1]
    return points


# EVOLVE-BLOCK-END