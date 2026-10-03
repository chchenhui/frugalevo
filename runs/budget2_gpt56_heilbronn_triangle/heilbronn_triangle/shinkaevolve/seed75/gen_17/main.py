# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct 11 points in the equilateral triangle with vertices
    (0, 0), (1, 0), and (0.5, sqrt(3)/2).

    The internal search is deterministic and works in barycentric
    coordinates.  In these coordinates, the absolute determinant of
    three points is exactly their area normalized by the area of the
    containing equilateral triangle.
    """
    try:
        rng = np.random.default_rng(11031991)

        n = 11
        population_size = 72
        generations = 850

        # All point triples.
        triples = np.array(
            [(i, j, k) for i in range(n) for j in range(i + 1, n)
             for k in range(j + 1, n)],
            dtype=np.intp,
        )

        def project_to_simplex(a: np.ndarray) -> np.ndarray:
            """Fast reflection/clamping projection into u>=0, v>=0, u+v<=1."""
            a = np.clip(a, 0.0, 1.0)
            mask = (a[..., 0] + a[..., 1]) > 1.0
            if np.any(mask):
                reflected = a[mask]
                a[mask] = 1.0 - reflected[:, ::-1]
            return np.clip(a, 0.0, 1.0)

        def areas_batch(configs: np.ndarray) -> np.ndarray:
            p0 = configs[:, triples[:, 0], :]
            p1 = configs[:, triples[:, 1], :]
            p2 = configs[:, triples[:, 2], :]
            det = (
                (p1[..., 0] - p0[..., 0]) * (p2[..., 1] - p0[..., 1])
                - (p1[..., 1] - p0[..., 1]) * (p2[..., 0] - p0[..., 0])
            )
            return np.abs(det)

        def evaluate(configs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            a = areas_batch(configs)
            minimum = np.min(a, axis=1)

            # A small lower-tail contribution gives evolutionary search useful
            # information before a strict minimum-area improvement is found.
            low = np.partition(a, 11, axis=1)[:, :12]
            fitness = minimum + 0.045 * np.mean(low, axis=1)
            return minimum, fitness

        # Uniform random points in the unit barycentric simplex.
        raw = rng.exponential(1.0, size=(population_size, n, 3))
        raw /= np.sum(raw, axis=2, keepdims=True)
        population = raw[:, :, 1:3]

        # Seed several geometrically diverse candidates, including configurations
        # with boundary support points, which are frequently useful in this problem.
        for q in range(12):
            t = rng.random((n, 2))
            t = project_to_simplex(t)
            t[0] = (0.0, 0.0)
            t[1] = (1.0, 0.0)
            t[2] = (0.0, 1.0)
            for j in range(3, n):
                edge = (q + j) % 3
                s = rng.random()
                if edge == 0:
                    t[j] = (s, 0.0)
                elif edge == 1:
                    t[j] = (0.0, s)
                else:
                    t[j] = (s, 1.0 - s)
            population[q] = t

        minimum, fitness = evaluate(population)
        best_index = int(np.argmax(minimum))
        best = population[best_index].copy()
        best_minimum = float(minimum[best_index])

        for generation in range(generations):
            # Differential mutation strength gradually shifts toward local search.
            progress = generation / max(1, generations - 1)
            differential_weight = 0.82 - 0.35 * progress
            crossover_rate = 0.88 - 0.28 * progress

            choices = rng.integers(0, population_size, size=(population_size, 3))
            for i in range(population_size):
                while (
                    choices[i, 0] == i or choices[i, 1] == i or choices[i, 2] == i
                    or choices[i, 0] == choices[i, 1]
                    or choices[i, 0] == choices[i, 2]
                    or choices[i, 1] == choices[i, 2]
                ):
                    choices[i] = rng.integers(0, population_size, size=3)

            a = population[choices[:, 0]]
            b = population[choices[:, 1]]
            c = population[choices[:, 2]]

            # Current-to-best differential evolution retains diversity while
            # continuously directing the population toward the best basin.
            donor = (
                population
                + 0.36 * (best[None, :, :] - population)
                + differential_weight * (a - b)
                + 0.16 * (b - c)
            )

            cross = rng.random((population_size, n, 2)) < crossover_rate
            forced = rng.integers(0, n * 2, size=population_size)
            cross.reshape(population_size, -1)[np.arange(population_size), forced] = True
            trial = np.where(cross, donor, population)
            trial = project_to_simplex(trial)

            trial_minimum, trial_fitness = evaluate(trial)

            # Preserve strict minimum-area gains; use the smooth lower-tail
            # score only when the minima are effectively tied.
            accept = (
                (trial_minimum > minimum + 1.0e-12)
                | (
                    (trial_minimum >= minimum - 2.5e-5)
                    & (trial_fitness > fitness)
                )
            )

            population[accept] = trial[accept]
            minimum[accept] = trial_minimum[accept]
            fitness[accept] = trial_fitness[accept]

            candidate = int(np.argmax(minimum))
            if minimum[candidate] > best_minimum:
                best_minimum = float(minimum[candidate])
                best = population[candidate].copy()

        # Batched active-set refinement.  Near an optimum, several small
        # determinants are normally coupled: improving one with a one-point
        # move tends to damage another.  Repair all points occurring in the
        # currently active triples together, while retaining some ordinary
        # local moves for diversification.
        current = best.copy()
        current_value = best_minimum
        current_fitness = float(evaluate(current[None, :, :])[1][0])
        elite = current.copy()
        elite_value = current_value

        refinement_rounds = 780
        batch_size = 30
        for step in range(refinement_rounds):
            progress = step / max(1, refinement_rounds - 1)
            scale = 0.022 * (1.0 - progress) ** 1.35 + 0.00018

            current_areas = areas_batch(current[None, :, :])[0]
            active_count = 12 if progress < 0.70 else 20
            active_ids = np.argpartition(current_areas, active_count - 1)[:active_count]
            active_points = np.unique(triples[active_ids].ravel())
            active_mask = np.zeros(n, dtype=bool)
            active_mask[active_points] = True

            proposals = np.repeat(current[None, :, :], batch_size, axis=0)

            # Individual moves remain useful for constraints with a single
            # clearly responsible point.
            for q in range(8):
                point_id = int(rng.choice(active_points))
                proposals[q, point_id] += rng.normal(0.0, scale, size=2)

            # The remaining proposals are explicit coupled active-set repairs.
            # Each has a common displacement component, allowing an entire
            # active cluster to move coherently, plus point-specific motion.
            count = int(np.sum(active_mask))
            for q in range(8, batch_size):
                common = rng.normal(0.0, scale * 0.42, size=2)
                local_scale = scale * (0.30 + 0.55 * rng.random())
                proposals[q, active_mask] += common
                proposals[q, active_mask] += rng.normal(
                    0.0, local_scale, size=(count, 2)
                )

                # Periodically include a weak global adjustment, which helps
                # when all active triples share nearly all their vertices.
                if q % 4 == 0:
                    proposals[q] += rng.normal(0.0, scale * 0.10, size=(n, 2))

            proposals = project_to_simplex(proposals)
            candidate_minimum, candidate_fitness = evaluate(proposals)

            candidate = int(np.argmax(candidate_minimum))
            if candidate_minimum[candidate] > elite_value:
                elite_value = float(candidate_minimum[candidate])
                elite = proposals[candidate].copy()

            # During the earlier repair rounds, permit a very small loss in
            # the strict minimum if the lower tail becomes substantially more
            # balanced.  This is a deterministic threshold-annealing rule,
            # not an uncontrolled random restart.
            tolerance = 0.0016 * (1.0 - progress) ** 2
            admissible = candidate_minimum >= current_value - tolerance
            if np.any(admissible):
                adjusted = candidate_fitness.copy()
                adjusted[~admissible] = -np.inf
                chosen = int(np.argmax(adjusted))
                chosen_value = float(candidate_minimum[chosen])
                chosen_fitness = float(candidate_fitness[chosen])
                if (
                    chosen_value > current_value + 1.0e-12
                    or chosen_fitness > current_fitness + 0.00015 * (1.0 - progress)
                ):
                    current = proposals[chosen].copy()
                    current_value = chosen_value
                    current_fitness = chosen_fitness

            if current_value > elite_value:
                elite_value = current_value
                elite = current.copy()

        current = elite

        # Convert simplex coordinates (u, v) to Cartesian coordinates:
        # p = (1-u-v)*(0,0) + u*(1,0) + v*(1/2,sqrt(3)/2).
        result = np.empty((n, 2), dtype=float)
        result[:, 0] = current[:, 0] + 0.5 * current[:, 1]
        result[:, 1] = (np.sqrt(3.0) * 0.5) * current[:, 1]
        return result

    except Exception:
        # A valid deterministic fallback arrangement inside the triangle.
        v = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, np.sqrt(3.0) / 2.0],
                [0.25, 0.0],
                [0.75, 0.0],
                [0.125, np.sqrt(3.0) / 8.0],
                [0.375, np.sqrt(3.0) / 8.0],
                [0.625, np.sqrt(3.0) / 8.0],
                [0.875, np.sqrt(3.0) / 8.0],
                [0.375, 3.0 * np.sqrt(3.0) / 8.0],
                [0.625, 3.0 * np.sqrt(3.0) / 8.0],
            ],
            dtype=float,
        )
        return v


# EVOLVE-BLOCK-END