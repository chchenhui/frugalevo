# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """
    Deterministic maximin construction for eleven points in the prescribed
    equilateral triangle.  The search is performed in affine simplex
    coordinates (u,v), u>=0, v>=0, u+v<=1.  In these coordinates absolute
    determinants are exactly areas normalized by the containing triangle.
    """
    try:
        rng = np.random.default_rng(11031991)

        n = 11
        free = 8
        height = np.sqrt(3.0) * 0.5
        corners = np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64
        )
        triples = np.array(
            [(i, j, k)
             for i in range(n - 2)
             for j in range(i + 1, n - 1)
             for k in range(j + 1, n)],
            dtype=np.intp,
        )

        def project(x: np.ndarray) -> np.ndarray:
            """Reflect samples into the two-dimensional unit simplex."""
            y = np.asarray(x, dtype=np.float64).copy()
            y = np.abs(y)
            y = np.mod(y, 2.0)
            y = np.where(y > 1.0, 2.0 - y, y)
            outside = y[..., 0] + y[..., 1] > 1.0
            if np.any(outside):
                z = y[outside].copy()
                y[outside, 0] = 1.0 - z[:, 1]
                y[outside, 1] = 1.0 - z[:, 0]
            return np.clip(y, 0.0, 1.0)

        def random_simplex(count: int) -> np.ndarray:
            x = rng.random((count, free, 2))
            outside = x.sum(axis=2) > 1.0
            x[outside] = 1.0 - x[outside]
            return x

        def all_areas(pop: np.ndarray) -> np.ndarray:
            count = pop.shape[0]
            points = np.empty((count, n, 2), dtype=np.float64)
            points[:, :3] = corners
            points[:, 3:] = pop
            a = points[:, triples[:, 0]]
            b = points[:, triples[:, 1]]
            c = points[:, triples[:, 2]]
            return np.abs(
                (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1])
                - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])
            )

        def values(pop: np.ndarray, tail_weight: float):
            ar = all_areas(pop)
            low = np.partition(ar, 13, axis=1)[:, :14]
            minimum = low[:, 0]
            # The tail has useful gradient-like information when candidates
            # have equally poor bottlenecks, but is deliberately weak late.
            score = minimum + tail_weight * np.mean(low[:, 1:], axis=1)
            return minimum, score

        # A separated triangular seed is substantially better than entirely
        # uniform starts, while noise and random members retain pattern diversity.
        template = np.array(
            [
                [0.12, 0.070], [0.43, 0.065], [0.76, 0.070],
                [0.070, 0.330], [0.350, 0.255], [0.650, 0.235],
                [0.140, 0.610], [0.405, 0.455],
            ],
            dtype=np.float64,
        )

        best = None
        best_value = -1.0
        pop_size = 144
        generations = 420
        elite_keep = 18

        # Independent vectorized islands are cheaper than many scalar
        # annealing restarts and explore distinct combinatorial arrangements.
        for island in range(3):
            population = random_simplex(pop_size)
            seeded = 108
            population[:seeded] = project(
                template[None] + rng.normal(0.0, 0.115, (seeded, free, 2))
            )

            minimum, score = values(population, 0.055)
            idx = int(np.argmax(minimum))
            if minimum[idx] > best_value:
                best_value = float(minimum[idx])
                best = population[idx].copy()

            for generation in range(generations):
                frac = generation / float(generations - 1)

                # Strong lower-tail balancing early; near the end this is
                # effectively lexicographic raw-minimum selection.
                tail_weight = 0.055 * max(0.0, 1.0 - frac / 0.55) + 2.0e-5
                minimum, score = values(population, tail_weight)
                order = np.argsort(score)[::-1]

                if frac > 0.55:
                    # Raw min is primary in the final phase.  The small score
                    # term resolves only numerically indistinguishable minima.
                    rank = np.lexsort((score, minimum))[::-1]
                    order = rank

                elites = population[order[:elite_keep]]
                count = pop_size - elite_keep

                r1 = rng.integers(0, pop_size, size=count)
                r2 = rng.integers(0, pop_size, size=count)
                r3 = rng.integers(0, pop_size, size=count)
                for q in range(count):
                    while r2[q] == r1[q]:
                        r2[q] = rng.integers(pop_size)
                    while r3[q] == r1[q] or r3[q] == r2[q]:
                        r3[q] = rng.integers(pop_size)

                base = population[r1]
                differential = 0.84 - 0.42 * frac
                donor = base + differential * (population[r2] - population[r3])

                # Blend a portion toward leading configurations.  This is
                # annealed so the late population performs local DE repair.
                leaders = population[order[rng.integers(elite_keep, size=count)]]
                donor += (0.20 * (1.0 - frac) + 0.055) * (leaders - base)

                parent_indices = order[elite_keep:]
                parent = population[parent_indices]
                cross_rate = 0.88 - 0.28 * frac
                mask = rng.random((count, free, 2)) < cross_rate
                forced = rng.integers(0, free * 2, size=count)
                mask.reshape(count, -1)[np.arange(count), forced] = True
                offspring = project(np.where(mask, donor, parent))

                off_min, off_score = values(offspring, tail_weight)
                par_min, par_score = values(parent, tail_weight)

                if frac < 0.55:
                    accept = off_score >= par_score
                else:
                    accept = (
                        (off_min > par_min + 1.0e-12)
                        | (
                            (np.abs(off_min - par_min) <= 1.0e-12)
                            & (off_score > par_score)
                        )
                    )

                next_population = np.empty_like(population)
                next_population[:elite_keep] = elites
                next_population[elite_keep:] = parent
                next_population[elite_keep:][accept] = offspring[accept]

                # Rare late isotropic perturbations avoid a completely frozen
                # coordinate direction without destroying the elite set.
                if generation % 41 == 0 and frac < 0.70:
                    slots = rng.integers(elite_keep, pop_size, size=8)
                    next_population[slots] = project(
                        next_population[slots]
                        + rng.normal(0.0, 0.035 * (1.0 - frac),
                                     (len(slots), free, 2))
                    )

                population = next_population
                minimum, _ = values(population, 2.0e-5)
                idx = int(np.argmax(minimum))
                if minimum[idx] > best_value:
                    best_value = float(minimum[idx])
                    best = population[idx].copy()

        # Strict active-set polish.  The incumbent is retained unless a batch
        # candidate improves its actual minimum determinant.
        current = best.copy()
        current_min, current_score = values(current[None], 2.0e-5)
        current_min = float(current_min[0])
        current_score = float(current_score[0])

        rounds = 520
        batch_size = 128
        for step in range(rounds):
            frac = step / float(rounds - 1)
            scale = 0.020 * (1.0 - frac) ** 1.55 + 0.00016

            ar = all_areas(current[None])[0]
            active_ids = np.argpartition(ar, 17)[:18]
            active = triples[active_ids]
            candidates = np.repeat(current[None], batch_size, axis=0)

            for q in range(batch_size):
                tri = active[rng.integers(len(active))]
                movable = tri[tri >= 3] - 3
                point = int(
                    movable[rng.integers(len(movable))]
                    if len(movable) else rng.integers(free)
                )
                candidates[q, point] += rng.normal(0.0, scale, 2)

                # Coupled repairs address constraints sharing several points.
                if q % 3 == 0:
                    other = int(rng.integers(free))
                    if other != point:
                        candidates[q, other] += rng.normal(0.0, 0.56 * scale, 2)
                if q % 11 == 0:
                    candidates[q] += rng.normal(0.0, 0.13 * scale, (free, 2))

            candidates = project(candidates)
            cand_min, cand_score = values(candidates, 2.0e-5)
            winner = int(np.lexsort((cand_score, cand_min))[-1])

            if (
                cand_min[winner] > current_min + 1.0e-12
                or (
                    abs(cand_min[winner] - current_min) <= 1.0e-12
                    and cand_score[winner] > current_score
                )
            ):
                current = candidates[winner].copy()
                current_min = float(cand_min[winner])
                current_score = float(cand_score[winner])

        affine = np.vstack((corners, current))
        result = np.empty((n, 2), dtype=np.float64)
        result[:, 0] = affine[:, 0] + 0.5 * affine[:, 1]
        result[:, 1] = height * affine[:, 1]
        return result

    except Exception:
        # Deterministic feasible fallback.
        return np.array(
            [
                [0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0],
                [0.155, 0.060], [0.465, 0.055], [0.805, 0.060],
                [0.205, 0.285], [0.500, 0.255], [0.795, 0.285],
                [0.325, 0.555], [0.675, 0.555],
            ],
            dtype=np.float64,
        )


# EVOLVE-BLOCK-END