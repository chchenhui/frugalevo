# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """
    Deterministically construct eleven points in the prescribed equilateral
    triangle.  Search is performed in affine simplex coordinates (u, v):
        u >= 0, v >= 0, u + v <= 1.
    In these coordinates, absolute 2D determinants are exactly triangle
    areas normalized by the containing triangle area.
    """
    try:
        rng = np.random.default_rng(11031991)

        n = 11
        free = 8
        height = np.sqrt(3.0) * 0.5
        corners = np.array(
            ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
            dtype=np.float64,
        )
        triples = np.array(
            [
                (i, j, k)
                for i in range(n - 2)
                for j in range(i + 1, n - 1)
                for k in range(j + 1, n)
            ],
            dtype=np.intp,
        )

        def project_batch(x: np.ndarray) -> np.ndarray:
            """Reflection projection onto u>=0, v>=0, u+v<=1."""
            y = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0).copy()
            outside = (y[..., 0] + y[..., 1]) > 1.0
            if np.any(outside):
                z = y[outside]
                y[outside] = 1.0 - z[:, ::-1]
            return np.clip(y, 0.0, 1.0)

        def random_simplex(count: int) -> np.ndarray:
            x = rng.random((count, free, 2))
            outside = x.sum(axis=2) > 1.0
            x[outside] = 1.0 - x[outside]
            return x

        def all_areas(population: np.ndarray) -> np.ndarray:
            count = population.shape[0]
            fixed = np.broadcast_to(corners, (count, 3, 2))
            points = np.concatenate((fixed, population), axis=1)
            a = points[:, triples[:, 0]]
            b = points[:, triples[:, 1]]
            c = points[:, triples[:, 2]]
            return np.abs(
                (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1])
                - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])
            )

        def rank_population(population: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            values = all_areas(population)
            # The strict minimum remains dominant.  The lower-tail mean helps
            # discriminate geometrically balanced configurations on near ties.
            tail = np.partition(values, 17, axis=1)[:, :18]
            minimum = tail[:, 0]
            score = minimum + 0.018 * np.mean(tail[:, 1:], axis=1)
            return minimum, score

        template = np.array(
            [
                [0.115, 0.055],
                [0.405, 0.050],
                [0.735, 0.060],
                [0.075, 0.300],
                [0.335, 0.235],
                [0.635, 0.225],
                [0.105, 0.600],
                [0.385, 0.430],
            ],
            dtype=np.float64,
        )

        def initial_population(size: int, archive_seed: np.ndarray | None) -> np.ndarray:
            population = random_simplex(size)

            structured = size // 3
            population[:structured] = project_batch(
                template[None, :, :]
                + rng.normal(0.0, 0.095, size=(structured, free, 2))
            )

            if archive_seed is not None:
                seeded = size * 3 // 5
                population[:seeded] = project_batch(
                    archive_seed[None, :, :]
                    + rng.normal(0.0, 0.038, size=(seeded, free, 2))
                )

                # A few stronger structure-preserving perturbations prevent
                # archive starts from collapsing into a single local basin.
                wide_start = seeded // 2
                population[wide_start:seeded] = project_batch(
                    archive_seed[None, :, :]
                    + rng.normal(0.0, 0.075, size=(seeded - wide_start, free, 2))
                )

                # Replace one free point in several archive descendants.
                for row in range(min(18, seeded)):
                    point = int(rng.integers(free))
                    replacement = random_simplex(1)[0, point]
                    population[row, point] = replacement

            return population

        def run_island(archive_seed: np.ndarray | None) -> np.ndarray:
            population_size = 120
            generations = 340
            population = initial_population(population_size, archive_seed)
            minimum, score = rank_population(population)

            best_index = int(np.argmax(minimum))
            best = population[best_index].copy()
            best_value = float(minimum[best_index])

            for generation in range(generations):
                fraction = generation / max(1, generations - 1)
                differential_weight = 0.78 - 0.39 * fraction
                crossover = 0.90 - 0.30 * fraction

                ranking = np.argsort(score)[::-1]
                pbest_pool = ranking[:max(12, 30 - generation // 22)]
                pbest = population[
                    pbest_pool[rng.integers(len(pbest_pool), size=population_size)]
                ]

                pa = rng.permutation(population_size)
                pb = rng.permutation(population_size)
                pc = rng.permutation(population_size)

                donor = (
                    population
                    + 0.44 * (pbest - population)
                    + differential_weight * (population[pa] - population[pb])
                    + 0.11 * (population[pb] - population[pc])
                )

                mask = rng.random((population_size, free, 2)) < crossover
                forced = rng.integers(0, free * 2, size=population_size)
                mask.reshape(population_size, -1)[
                    np.arange(population_size), forced
                ] = True

                trial = project_batch(np.where(mask, donor, population))
                trial_minimum, trial_score = rank_population(trial)

                # Tail score is used only in a narrow tolerance band.
                accept = (
                    (trial_minimum > minimum + 1.0e-12)
                    | (
                        (trial_minimum >= minimum - 1.5e-5)
                        & (trial_score > score + 2.0e-8)
                    )
                )

                population[accept] = trial[accept]
                minimum[accept] = trial_minimum[accept]
                score[accept] = trial_score[accept]

                candidate = int(np.argmax(minimum))
                if minimum[candidate] > best_value:
                    best_value = float(minimum[candidate])
                    best = population[candidate].copy()

            return best

        # Three fresh islands establish diverse basins.  The later islands are
        # archive seeded, transferring useful active-constraint geometry.
        archive: list[np.ndarray] = []
        for island in range(6):
            if island < 3 or not archive:
                seed = None
            else:
                archive_values, _ = rank_population(np.asarray(archive))
                seed = archive[int(np.argmax(archive_values))]
            archive.append(run_island(seed))

        archive_array = np.asarray(archive)
        archive_minimum, _ = rank_population(archive_array)
        current = archive_array[int(np.argmax(archive_minimum))].copy()
        current_minimum = float(np.max(archive_minimum))
        _, current_score_array = rank_population(current[None, :, :])
        current_score = float(current_score_array[0])

        elite = current.copy()
        elite_value = current_minimum

        # Batched active-set local repair.  It supports both single-point
        # movements and coherent movements of all points incident to currently
        # restrictive triples.
        rounds = 760
        batch_size = 176

        for step in range(rounds):
            fraction = step / max(1, rounds - 1)
            scale = 0.027 * (1.0 - fraction) ** 1.48 + 0.00020

            current_areas = all_areas(current[None, :, :])[0]
            active_count = 14 if fraction < 0.45 else 18
            active_ids = np.argpartition(current_areas, active_count - 1)[:active_count]
            active_triangles = triples[active_ids]
            incident = np.unique(active_triangles.ravel())
            movable_incident = incident[incident >= 3] - 3
            if len(movable_incident) == 0:
                movable_incident = np.arange(free)

            proposals = np.repeat(current[None, :, :], batch_size, axis=0)

            for row in range(batch_size):
                mode = row % 5
                if mode < 2:
                    point = int(movable_incident[rng.integers(len(movable_incident))])
                    proposals[row, point] += rng.normal(0.0, scale, size=2)
                elif mode == 2:
                    point = int(movable_incident[rng.integers(len(movable_incident))])
                    other = int(rng.integers(free))
                    proposals[row, point] += rng.normal(0.0, scale, size=2)
                    if other != point:
                        proposals[row, other] += rng.normal(0.0, 0.58 * scale, size=2)
                else:
                    chosen = movable_incident[
                        rng.random(len(movable_incident)) < 0.72
                    ]
                    if len(chosen) == 0:
                        chosen = movable_incident[:1]
                    common = rng.normal(0.0, 0.32 * scale, size=2)
                    proposals[row, chosen] += common
                    proposals[row, chosen] += rng.normal(
                        0.0, 0.48 * scale, size=(len(chosen), 2)
                    )

            proposals = project_batch(proposals)
            candidate_minimum, candidate_score = rank_population(proposals)

            strict_winner = int(np.argmax(candidate_minimum))
            if candidate_minimum[strict_winner] > elite_value:
                elite_value = float(candidate_minimum[strict_winner])
                elite = proposals[strict_winner].copy()

            # Early controlled downhill acceptance permits balancing several
            # nearly active determinants before the final strict polishing.
            tolerance = 0.00070 * (1.0 - fraction) ** 2
            admissible = candidate_minimum >= current_minimum - tolerance
            if np.any(admissible):
                adjusted = candidate_score.copy()
                adjusted[~admissible] = -np.inf
                winner = int(np.argmax(adjusted))
                if (
                    candidate_minimum[winner] > current_minimum + 1.0e-12
                    or candidate_score[winner] > current_score + 2.0e-6
                ):
                    current = proposals[winner].copy()
                    current_minimum = float(candidate_minimum[winner])
                    current_score = float(candidate_score[winner])

            if current_minimum > elite_value:
                elite_value = current_minimum
                elite = current.copy()

        affine = np.vstack((corners, elite))
        result = np.empty((n, 2), dtype=np.float64)
        result[:, 0] = affine[:, 0] + 0.5 * affine[:, 1]
        result[:, 1] = height * affine[:, 1]

        if result.shape != (11, 2) or not np.all(np.isfinite(result)):
            raise ValueError("invalid optimization result")
        return result

    except Exception:
        # Guaranteed feasible deterministic fallback.
        h = np.sqrt(3.0) * 0.5
        return np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, h],
                [0.16, 0.06],
                [0.43, 0.05],
                [0.76, 0.05],
                [0.22, 0.27],
                [0.51, 0.22],
                [0.74, 0.20],
                [0.35, 0.54],
                [0.56, 0.40],
            ],
            dtype=np.float64,
        )


# EVOLVE-BLOCK-END