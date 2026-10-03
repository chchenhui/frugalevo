# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_convex13() -> np.ndarray:
    """
    Construct a deterministic 13-point Heilbronn-style configuration.

    The containing convex region is an equilateral triangle.  Its three
    vertices are retained as hull points; the remaining ten points consist
    of the centroid and three independently phased 3-fold rings.  Since the
    maximum ring radius is smaller than the triangle inradius, every point
    is guaranteed to lie inside the hull.

    Returns
    -------
    np.ndarray
        Array of shape (13, 2).
    """
    rng = np.random.default_rng(20260313)

    root3 = np.sqrt(3.0)
    hull = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 0.5 * root3],
        ],
        dtype=float,
    )
    center = np.array([0.5, root3 / 6.0], dtype=float)
    hull_area = root3 / 4.0

    triples = np.array(
        [
            (i, j, k)
            for i in range(13)
            for j in range(i + 1, 13)
            for k in range(j + 1, 13)
        ],
        dtype=np.intp,
    )

    # Parameter vector:
    # [ring_radius_0, ring_radius_1, ring_radius_2,
    #  ring_phase_0,  ring_phase_1,  ring_phase_2].
    #
    # A disk of radius inradius=0.288675... about the centroid is wholly
    # contained in the equilateral triangle, so 0.278 is a safe strict bound.
    radius_lo = 0.035
    radius_hi = 0.278
    angle_period = 2.0 * np.pi / 3.0

    def wrap_parameters(x: np.ndarray) -> np.ndarray:
        y = x.copy()
        y[..., :3] = np.clip(y[..., :3], radius_lo, radius_hi)
        y[..., 3:] = np.mod(y[..., 3:], angle_period)
        return y

    def make_layouts(params: np.ndarray) -> np.ndarray:
        """Build a batch of 13-point layouts from ring parameters."""
        count = params.shape[0]
        layouts = np.empty((count, 13, 2), dtype=float)
        layouts[:, :3] = hull
        layouts[:, 3] = center

        radii = params[:, :3]
        phases = params[:, 3:]
        offsets = np.array([0.0, angle_period, 2.0 * angle_period])

        for ring in range(3):
            angles = phases[:, ring, None] + offsets[None, :]
            start = 4 + 3 * ring
            layouts[:, start:start + 3, 0] = (
                center[0] + radii[:, ring, None] * np.cos(angles)
            )
            layouts[:, start:start + 3, 1] = (
                center[1] + radii[:, ring, None] * np.sin(angles)
            )
        return layouts

    def score(params: np.ndarray) -> np.ndarray:
        """Normalized minimum area for a batch of parameter vectors."""
        p = make_layouts(params)
        a = p[:, triples[:, 0]]
        b = p[:, triples[:, 1]]
        c = p[:, triples[:, 2]]
        double_area = np.abs(
            (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1])
            - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])
        )
        return 0.5 * double_area.min(axis=1) / hull_area

    population_size = 384
    generations = 720

    # Stratified initialization gives substantially more radial diversity than
    # plain random starts, which is useful for a nonsmooth maximin objective.
    population = np.empty((population_size, 6), dtype=float)
    for coordinate in range(3):
        strata = (np.arange(population_size) + rng.random(population_size))
        rng.shuffle(strata)
        population[:, coordinate] = radius_lo + (
            radius_hi - radius_lo
        ) * strata / population_size
    population[:, 3:] = rng.uniform(
        0.0, angle_period, size=(population_size, 3)
    )

    # Hand-spaced seeds keep useful ring separations present from generation 0.
    population[0] = np.array(
        [0.090, 0.166, 0.250, 0.070, 0.410, 1.030], dtype=float
    )
    population[1] = np.array(
        [0.112, 0.191, 0.267, 0.225, 0.800, 1.285], dtype=float
    )
    population = wrap_parameters(population)
    values = score(population)

    best_index = int(np.argmax(values))
    best = population[best_index].copy()
    best_value = float(values[best_index])

    for generation in range(generations):
        progress = generation / float(generations - 1)

        # Current-to-best differential evolution: broad early exploration,
        # then increasingly conservative moves around active constraints.
        factor = 0.82 - 0.48 * progress
        attraction = 0.16 + 0.42 * progress
        crossover_rate = 0.78 - 0.22 * progress

        order_a = rng.permutation(population_size)
        order_b = rng.permutation(population_size)
        donor = (
            population
            + attraction * (best[None, :] - population)
            + factor * (population[order_a] - population[order_b])
        )
        donor = wrap_parameters(donor)

        mask = rng.random((population_size, 6)) < crossover_rate
        mask[np.arange(population_size), rng.integers(0, 6, population_size)] = True
        trial = np.where(mask, donor, population)
        trial = wrap_parameters(trial)

        trial_values = score(trial)
        accepted = trial_values >= values
        population[accepted] = trial[accepted]
        values[accepted] = trial_values[accepted]

        generation_best = int(np.argmax(values))
        if values[generation_best] > best_value:
            best_value = float(values[generation_best])
            best = population[generation_best].copy()

        # Deterministic diversity injection prevents all rings from becoming
        # phase-locked into the same local arrangement.
        if generation in (230, 470):
            worst = np.argsort(values)[: population_size // 5]
            population[worst, :3] = rng.uniform(
                radius_lo, radius_hi, size=(len(worst), 3)
            )
            population[worst, 3:] = rng.uniform(
                0.0, angle_period, size=(len(worst), 3)
            )
            values[worst] = score(population[worst])

    # Batch local refinement.  Unlike coordinate annealing, each round tests
    # many correlated radius/phase changes and accepts only true improvements.
    for iteration in range(260):
        fraction = iteration / 259.0
        radial_step = 0.018 * (0.0012 / 0.018) ** fraction
        phase_step = 0.160 * (0.0020 / 0.160) ** fraction

        candidates = np.repeat(best[None, :], 96, axis=0)
        perturb = rng.normal(size=(96, 6))
        sparse = rng.random((96, 6)) < 0.48
        sparse[0, rng.integers(0, 6)] = True
        perturb *= sparse
        perturb[:, :3] *= radial_step
        perturb[:, 3:] *= phase_step
        candidates = wrap_parameters(candidates + perturb)

        candidate_values = score(candidates)
        candidate_index = int(np.argmax(candidate_values))
        if candidate_values[candidate_index] > best_value:
            best_value = float(candidate_values[candidate_index])
            best = candidates[candidate_index].copy()

    return make_layouts(best[None, :])[0]


# EVOLVE-BLOCK-END