# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_convex13() -> np.ndarray:
    """
    Deterministic 13-point Heilbronn search in the unit square.

    The four corners are fixed, hence the convex hull always has unit area and
    the returned configuration's raw minimum triangle area is its normalized
    objective value.
    """
    rng = np.random.default_rng(13031957)

    n = 13
    movable_count = 9
    corners = np.array(
        [[0.0, 0.0],
         [1.0, 0.0],
         [1.0, 1.0],
         [0.0, 1.0]],
        dtype=float,
    )

    tri = np.array(
        [(i, j, k)
         for i in range(n - 2)
         for j in range(i + 1, n - 1)
         for k in range(j + 1, n)],
        dtype=np.intp,
    )
    triangle_count = len(tri)
    incident = [np.flatnonzero(np.any(tri == p, axis=1)) for p in range(n)]

    def areas_batch(population: np.ndarray) -> np.ndarray:
        """Triangle areas for shape (population_size, 13, 2)."""
        a = population[:, tri[:, 0], :]
        b = population[:, tri[:, 1], :]
        c = population[:, tri[:, 2], :]
        return 0.5 * np.abs(
            (b[:, :, 0] - a[:, :, 0]) * (c[:, :, 1] - a[:, :, 1])
            - (b[:, :, 1] - a[:, :, 1]) * (c[:, :, 0] - a[:, :, 0])
        )

    def quality(area_values: np.ndarray, temperature: float) -> np.ndarray:
        """
        Smooth lower-tail value used only during global exploration.

        Subtracting the row minimum makes this numerically stable.  A small
        lower-tail reward distinguishes configurations having identical weak
        triangles without replacing max-min optimization by an average-area
        objective.
        """
        minimum = np.min(area_values, axis=1)
        if temperature <= 0.0:
            low = np.partition(area_values, 15, axis=1)[:, :16].mean(axis=1)
            return minimum + 1e-7 * low
        z = np.exp(-(area_values - minimum[:, None]) / temperature).sum(axis=1)
        return minimum - temperature * np.log(z)

    def rank_key(area_values: np.ndarray) -> np.ndarray:
        """Strict bottleneck plus tiny deterministic lower-tail tie-break."""
        minimum = np.min(area_values, axis=1)
        tail = np.partition(area_values, 17, axis=1)[:, :18].mean(axis=1)
        return minimum + 1e-9 * tail

    # Diverse seeds deliberately include staggered, diagonal and irregular
    # layouts rather than repeatedly perturbing a single Cartesian lattice.
    bases = np.array(
        [
            [[.18, .19], [.50, .15], [.82, .21],
             [.16, .49], [.51, .48], [.84, .53],
             [.21, .80], [.50, .84], [.79, .78]],

            [[.28, .15], [.61, .20], [.84, .35],
             [.15, .38], [.47, .47], [.77, .57],
             [.28, .72], [.57, .83], [.84, .77]],

            [[.18, .28], [.48, .15], [.78, .24],
             [.27, .51], [.57, .43], [.84, .58],
             [.14, .78], [.49, .83], [.76, .77]],

            [[.23, .16], [.69, .25], [.82, .48],
             [.15, .40], [.49, .50], [.76, .68],
             [.22, .76], [.49, .84], [.78, .82]],
        ],
        dtype=float,
    )

    population_size = 28
    population = np.empty((population_size, n, 2), dtype=float)
    population[:, :4, :] = corners

    for q in range(population_size):
        base = bases[q % len(bases)]
        scale = 0.045 + 0.010 * (q % 4)
        candidate = base + rng.uniform(-scale, scale, size=(movable_count, 2))
        if q >= 20:
            # A few broad seeds give differential evolution routes into
            # non-lattice combinatorial regimes.
            candidate = 0.60 * candidate + 0.40 * rng.uniform(
                0.10, 0.90, size=(movable_count, 2)
            )
        population[q, 4:, :] = np.clip(candidate, 0.025, 0.975)

    areas = areas_batch(population)
    generations = 380

    # Population search: each accepted trial replaces exactly one member.
    # This is an active-constraint exchange mechanism rather than a sequence
    # of independent coordinate hill-climbs.
    for generation in range(generations):
        fraction = generation / float(generations - 1)
        temperature = 0.0085 * max(0.0, 1.0 - fraction) ** 2.4

        order = rng.permutation(population_size)
        trial = population.copy()

        for target in order:
            pool = np.arange(population_size)
            pool = pool[pool != target]
            parents = rng.choice(pool, size=3, replace=False)
            a, b, c = population[parents, 4:, :]

            differential_weight = 0.72 - 0.27 * fraction
            donor = a + differential_weight * (b - c)

            # Late generations increasingly exchange elite structure rather
            # than using only random differences.
            if generation > generations // 2 and rng.random() < 0.35:
                elite = np.argsort(rank_key(areas))[-6:]
                guide = population[elite[rng.integers(len(elite))], 4:, :]
                donor = donor + 0.22 * (guide - population[target, 4:, :])

            donor = np.clip(donor, 0.018, 0.982)
            crossover = rng.random((movable_count, 2)) < (0.76 - 0.22 * fraction)
            crossover[rng.integers(movable_count), rng.integers(2)] = True

            child = population[target, 4:, :].copy()
            child[crossover] = donor[crossover]

            # Small annealed isotropic mutation keeps population members from
            # sharing an accidental nearly-collinear pattern.
            if rng.random() < 0.30:
                child += rng.normal(
                    scale=0.018 * (1.0 - fraction) + 0.001,
                    size=child.shape,
                )
            trial[target, 4:, :] = np.clip(child, 0.015, 0.985)

        trial_areas = areas_batch(trial)

        if generation < int(0.72 * generations):
            old_score = quality(areas, temperature)
            new_score = quality(trial_areas, temperature)
        else:
            old_score = rank_key(areas)
            new_score = rank_key(trial_areas)

        accept = new_score >= old_score
        population[accept] = trial[accept]
        areas[accept] = trial_areas[accept]

    # Strict coordinate polishing of several independent surviving basins.
    survivor_count = 6
    survivor_indices = np.argsort(rank_key(areas))[-survivor_count:]
    best_points = None
    best_value = -np.inf
    best_tail = -np.inf

    angles = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    ring = np.column_stack((np.cos(angles), np.sin(angles)))

    for source in survivor_indices:
        points = population[source].copy()
        current_areas = areas[source].copy()

        for level, step in enumerate((0.014, 0.007, 0.003, 0.0012, 0.00045)):
            for round_number in range(3):
                point_order = np.roll(np.arange(4, n), (level + round_number) % 9)

                for p in point_order:
                    current = points[p].copy()
                    phase = (p * 11 + level * 7 + round_number * 3) % len(ring)
                    directions = np.roll(ring, phase, axis=0)

                    proposals = np.empty((1 + 2 * len(directions), 2), dtype=float)
                    proposals[0] = current
                    proposals[1:1 + len(directions)] = current + step * directions
                    proposals[1 + len(directions):] = (
                        current + 0.42 * step * directions
                    )
                    proposals = np.clip(proposals, 0.0, 1.0)

                    local_tri = tri[incident[p]]
                    work = np.broadcast_to(points, (len(proposals), n, 2)).copy()
                    work[:, p, :] = proposals
                    aa = work[:, local_tri[:, 0], :]
                    bb = work[:, local_tri[:, 1], :]
                    cc = work[:, local_tri[:, 2], :]
                    local = 0.5 * np.abs(
                        (bb[:, :, 0] - aa[:, :, 0]) *
                        (cc[:, :, 1] - aa[:, :, 1])
                        - (bb[:, :, 1] - aa[:, :, 1]) *
                        (cc[:, :, 0] - aa[:, :, 0])
                    )

                    fixed_mask = np.ones(triangle_count, dtype=bool)
                    fixed_mask[incident[p]] = False
                    fixed = current_areas[fixed_mask]

                    candidate_values = np.empty(
                        (len(proposals), triangle_count), dtype=float
                    )
                    candidate_values[:, fixed_mask] = fixed
                    candidate_values[:, incident[p]] = local

                    candidate_min = np.min(candidate_values, axis=1)
                    old_min = float(np.min(current_areas))
                    feasible = candidate_min >= old_min - 1e-14

                    lower_tail = np.partition(
                        candidate_values, 17, axis=1
                    )[:, :18].mean(axis=1)
                    score = np.where(
                        feasible,
                        candidate_min + 1e-9 * lower_tail,
                        -np.inf,
                    )
                    choice = int(np.argmax(score))

                    if feasible[choice]:
                        points[p] = proposals[choice]
                        current_areas[incident[p]] = local[choice]

        value = float(np.min(current_areas))
        tail = float(np.partition(current_areas, 17)[:18].mean())
        if (value > best_value + 1e-15 or
                (abs(value - best_value) <= 1e-15 and tail > best_tail)):
            best_value = value
            best_tail = tail
            best_points = points.copy()

    if best_points is None or not np.all(np.isfinite(best_points)):
        return np.vstack((
            corners,
            np.array(
                [[.18, .19], [.50, .16], [.82, .20],
                 [.16, .49], [.50, .49], [.84, .50],
                 [.20, .80], [.50, .84], [.80, .80]],
                dtype=float,
            ),
        ))

    return best_points


# EVOLVE-BLOCK-END