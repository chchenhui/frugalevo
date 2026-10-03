# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_convex13() -> np.ndarray:
    """
    Deterministic island differential-evolution construction for thirteen
    points in the unit square.  The four fixed corners make the hull area one,
    hence raw and normalized minimum triangle areas coincide.
    """
    rng = np.random.default_rng(13031957)

    n = 13
    d = 18
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
    t0, t1, t2 = tri[:, 0], tri[:, 1], tri[:, 2]

    incident = [
        np.flatnonzero(np.any(tri == point, axis=1))
        for point in range(n)
    ]

    def population_areas(x):
        """Return all triangle areas for a population of flattened interiors."""
        m = len(x)
        pts = np.empty((m, n, 2), dtype=float)
        pts[:, :4] = corners
        pts[:, 4:] = x.reshape(m, 9, 2)
        a = pts[:, t0]
        b = pts[:, t1]
        c = pts[:, t2]
        return 0.5 * np.abs(
            (b[:, :, 0] - a[:, :, 0]) * (c[:, :, 1] - a[:, :, 1])
            - (b[:, :, 1] - a[:, :, 1]) * (c[:, :, 0] - a[:, :, 0])
        )

    def quality(areas, phase):
        """
        A continuation ranking: early generations reward an entire low-area
        tail, while the final generations increasingly rank by strict min.
        """
        low = np.partition(areas, 11, axis=1)[:, :12]
        mn = low[:, 0]
        tail = np.mean(low, axis=1)
        # The tail term helps populations move through nonsmooth bottlenecks,
        # but is reduced to essentially a tie-breaker at the end.
        weight = 0.34 * (1.0 - phase) ** 2 + 0.012
        return mn + weight * tail, mn

    # Four distinct deterministic initial geometry families.  Their points are
    # subsequently represented only by their 18 interior coordinates.
    square = np.array(
        [[.20, .20], [.50, .20], [.80, .20],
         [.20, .50], [.50, .50], [.80, .50],
         [.20, .80], [.50, .80], [.80, .80]]
    )
    stagger = np.array(
        [[.24, .16], [.51, .18], [.78, .16],
         [.14, .43], [.49, .43], [.86, .43],
         [.25, .71], [.52, .70], [.76, .73]]
    )
    skew = np.array(
        [[.17, .25], [.48, .14], [.81, .27],
         [.28, .47], [.65, .42], [.15, .67],
         [.51, .64], [.85, .60], [.54, .88]]
    )
    radial = np.array(
        [[.50, .15], [.78, .27], [.84, .57],
         [.68, .82], [.38, .84], [.16, .68],
         [.17, .37], [.38, .28], [.57, .54]]
    )
    bases = (square, stagger, skew, radial)

    islands = 4
    popsize = 36
    generations = 390
    lo, hi = 0.018, 0.982

    population = np.empty((islands, popsize, d), dtype=float)
    for h, base in enumerate(bases):
        jitter = rng.uniform(-0.135, 0.135, size=(popsize, 9, 2))
        cloud = np.clip(base[None, :, :] + jitter, lo, hi)
        # A few broad members prevent all islands inheriting a grid adjacency.
        cloud[-7:] = rng.uniform(.055, .945, size=(7, 9, 2))
        population[h] = cloud.reshape(popsize, d)

    best_x = None
    best_min = -np.inf

    for generation in range(generations):
        phase = generation / float(generations - 1)

        for island in range(islands):
            x = population[island]
            areas = population_areas(x)
            score, strict = quality(areas, phase)

            local_best = int(np.argmax(strict))
            if strict[local_best] > best_min:
                best_min = float(strict[local_best])
                best_x = x[local_best].copy()

            order = np.argsort(score)
            elite_count = max(3, popsize // 5)
            elite = order[-elite_count:]

            # JADE-like current-to-elite differential mutation.  All complete
            # configurations move together, permitting coordinated repairs of
            # active small triangles.
            r1 = rng.integers(0, popsize, size=popsize)
            r2 = rng.integers(0, popsize, size=popsize)
            for i in range(popsize):
                while r1[i] == i:
                    r1[i] = rng.integers(popsize)
                while r2[i] == i or r2[i] == r1[i]:
                    r2[i] = rng.integers(popsize)

            pbest = x[elite[rng.integers(0, elite_count, size=popsize)]]
            f = 0.48 + 0.36 * rng.random((popsize, 1))
            mutant = x + f * (pbest - x) + f * (x[r1] - x[r2])

            # Periodic stronger jumps retain basin-to-basin mobility.
            if generation < 190:
                jumpers = rng.random(popsize) < 0.10
                mutant[jumpers] += rng.normal(
                    0.0, 0.075 * (1.0 - phase), size=(np.sum(jumpers), d)
                )

            mutant = np.clip(mutant, lo, hi)
            cross_rate = 0.82 - 0.28 * phase
            mask = rng.random((popsize, d)) < cross_rate
            mask[np.arange(popsize), rng.integers(0, d, size=popsize)] = True
            trial = np.where(mask, mutant, x)

            trial_areas = population_areas(trial)
            trial_score, trial_strict = quality(trial_areas, phase)

            # In the last third strict area is dominant.  Earlier, the smooth
            # lower-tail rank admits useful rearrangements of several tight
            # triangles at once.
            if generation > 250:
                accept = trial_strict >= strict - 2e-7
                close = np.abs(trial_strict - strict) < 2e-7
                accept |= close & (trial_score >= score)
            else:
                accept = trial_score >= score

            x[accept] = trial[accept]
            population[island] = x

        # Ring migration keeps the independently seeded islands diverse while
        # distributing configurations which have discovered good active sets.
        if generation % 39 == 38:
            champions = []
            for island in range(islands):
                aa = population_areas(population[island])
                champions.append(population[island][int(np.argmax(np.min(aa, axis=1)))].copy())
            for island in range(islands):
                aa = population_areas(population[island])
                worst = int(np.argmin(np.min(aa, axis=1)))
                population[island, worst] = champions[(island - 1) % islands]

    # Obtain the best strict population member after all migration.
    for island in range(islands):
        aa = population_areas(population[island])
        q = int(np.argmax(np.min(aa, axis=1)))
        val = float(np.min(aa[q]))
        if val > best_min:
            best_min = val
            best_x = population[island, q].copy()

    # Small deterministic direct-search polish, now using the exact objective.
    points = np.vstack((corners, best_x.reshape(9, 2)))
    angles = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    directions = np.column_stack((np.cos(angles), np.sin(angles)))

    for radius in (0.012, 0.006, 0.0025, 0.0010):
        for repeat in range(3):
            for p in range(4, n):
                candidates = np.vstack((
                    points[p],
                    np.clip(points[p] + radius * directions, lo, hi)
                ))
                work = np.broadcast_to(points, (len(candidates), n, 2)).copy()
                work[:, p] = candidates
                a = work[:, t0]
                b = work[:, t1]
                c = work[:, t2]
                aa = 0.5 * np.abs(
                    (b[:, :, 0] - a[:, :, 0]) * (c[:, :, 1] - a[:, :, 1])
                    - (b[:, :, 1] - a[:, :, 1]) * (c[:, :, 0] - a[:, :, 0])
                )
                values = np.min(aa, axis=1)
                choice = int(np.argmax(values))
                if values[choice] >= best_min - 1e-15:
                    points[p] = candidates[choice]
                    best_min = float(values[choice])

    if not np.all(np.isfinite(points)) or points.shape != (13, 2):
        return np.vstack((corners, np.full((9, 2), 0.5, dtype=float)))
    return points


# EVOLVE-BLOCK-END