# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct eleven reproducible maximin points in an equilateral triangle.

    Internal coordinates are (u, v) in the unit simplex, corresponding to
    u * (1, 0) + v * (1/2, sqrt(3)/2).  Determinants in these coordinates
    equal triangle areas normalized by the enclosing triangle area.
    """
    n = 11
    sqrt3_over_2 = np.sqrt(3.0) / 2.0
    rng = np.random.default_rng(11031987)

    triples = np.array(
        [(i, j, k) for i in range(n - 2)
         for j in range(i + 1, n - 1)
         for k in range(j + 1, n)],
        dtype=np.intp,
    )
    vertices = np.array(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))

    def areas(p: np.ndarray) -> np.ndarray:
        a = p[triples[:, 1]] - p[triples[:, 0]]
        b = p[triples[:, 2]] - p[triples[:, 0]]
        return np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])

    def merit(value: np.ndarray) -> tuple[float, float]:
        minimum = float(value.min())
        tail = np.partition(value, 11)[:12]
        return minimum, float(minimum + 0.085 * tail.mean())

    def annealing_merit(value: np.ndarray, progress: float) -> float:
        """
        During basin discovery reward a healthy broad lower tail; near the
        end concentrate on the genuinely active constraints.  This is less
        brittle than optimizing either the raw minimum or log-sum-exp alone.
        """
        minimum = float(value.min())
        count = max(10, min(len(value), int(round(30.0 - 20.0 * progress))))
        low = np.partition(value, count - 1)[:count]
        tail_mean = float(low.mean())

        scale = 0.0090 * (1.0 - progress) ** 1.25 + 0.00030
        shifted = (low - minimum) / scale
        soft_min = minimum - scale * np.log(np.exp(-shifted).mean())

        tail_weight = 0.60 * (1.0 - progress) ** 1.45 + 0.08
        return float(
            (1.0 - tail_weight) * minimum
            + tail_weight * (0.55 * tail_mean + 0.45 * soft_min)
        )

    def project_weights(weights: np.ndarray) -> np.ndarray:
        weights = np.maximum(weights, 1.0e-9)
        return weights / weights.sum()

    best = np.vstack((vertices, np.full((8, 2), 1.0 / 3.0)))
    best_values = areas(best)
    best_min, best_merit = merit(best_values)
    elite: list[tuple[float, float, np.ndarray]] = []

    # Lower-tail simulated annealing finds several distinct oriented-matroid
    # cells.  Barycentric perturbations preserve feasibility even near edges.
    iterations = 22000
    for restart in range(9):
        bary = rng.dirichlet((1.15, 1.15, 1.15), size=8)
        current = np.empty((n, 2), dtype=float)
        current[:3] = vertices
        current[3:] = bary[:, 1:]

        current_values = areas(current)
        current_min, current_merit = merit(current_values)

        for iteration in range(iterations):
            progress = iteration / (iterations - 1.0)
            scale = 0.0105 * (1.0 - progress) + 0.00055
            current_energy = annealing_merit(current_values, progress)

            if rng.random() < 0.88:
                cutoff = current_min + max(2.7 * scale, 0.0018)
                restrictive = triples[current_values <= cutoff]
                chosen = restrictive[int(rng.integers(len(restrictive)))]
                movable = chosen[chosen >= 3]
                index = (int(rng.choice(movable)) if len(movable)
                         else int(rng.integers(3, n)))
            else:
                index = int(rng.integers(3, n))

            step = 0.100 * (1.0 - progress) ** 1.52 + 0.0010
            old_weights = np.array((
                1.0 - current[index, 0] - current[index, 1],
                current[index, 0],
                current[index, 1],
            ))
            new_weights = project_weights(
                old_weights + rng.normal(0.0, step, size=3)
            )

            candidate = current.copy()
            candidate[index] = new_weights[1:]
            candidate_values = areas(candidate)
            candidate_min, candidate_merit = merit(candidate_values)
            candidate_energy = annealing_merit(candidate_values, progress)

            temperature = 0.0022 * (1.0 - progress) ** 2 + 2.0e-6
            delta = candidate_energy - current_energy
            if delta >= 0.0 or rng.random() < np.exp(delta / temperature):
                current = candidate
                current_values = candidate_values
                current_min = candidate_min
                current_merit = candidate_merit

            if (current_min > best_min + 1.0e-14 or
                    (abs(current_min - best_min) <= 1.0e-14 and
                     current_merit > best_merit)):
                best = current.copy()
                best_min = current_min
                best_merit = current_merit

        elite.append((current_min, current_merit, current.copy()))
        elite.sort(key=lambda item: (item[0], item[1]), reverse=True)
        del elite[5:]

    elite.append((best_min, best_merit, best.copy()))
    elite.sort(key=lambda item: (item[0], item[1]), reverse=True)
    del elite[5:]

    # Refine multiple good basins.  This is useful because a slightly worse
    # annealing endpoint can have considerably better local maximin potential.
    stencil = (
        np.array((1.0, -1.0, 0.0)),
        np.array((-1.0, 1.0, 0.0)),
        np.array((1.0, 0.0, -1.0)),
        np.array((-1.0, 0.0, 1.0)),
        np.array((0.0, 1.0, -1.0)),
        np.array((0.0, -1.0, 1.0)),
    )

    for _, _, seed in elite:
        seed_values = areas(seed)
        seed_min, seed_merit = merit(seed_values)

        for radius, passes in ((0.012, 7), (0.005, 8)):
            for _ in range(passes):
                changed = False
                for index in rng.permutation(np.arange(3, n)):
                    original = np.array((
                        1.0 - seed[index, 0] - seed[index, 1],
                        seed[index, 0],
                        seed[index, 1],
                    ))
                    local = seed
                    local_min, local_merit = seed_min, seed_merit

                    directions = list(stencil)
                    for _ in range(8):
                        direction = rng.normal(size=3)
                        directions.append(direction - direction.mean())

                    for direction in directions:
                        direction = direction / np.linalg.norm(direction)
                        trial = seed.copy()
                        trial[index] = project_weights(
                            original + radius * direction
                        )[1:]
                        trial_values = areas(trial)
                        trial_min, trial_merit = merit(trial_values)
                        if (trial_min > local_min + 1.0e-12 or
                                (abs(trial_min - local_min) <= 1.0e-12 and
                                 trial_merit > local_merit + 1.0e-13)):
                            local = trial
                            local_min, local_merit = trial_min, trial_merit

                    if local is not seed:
                        seed = local
                        seed_min, seed_merit = local_min, local_merit
                        changed = True
                if not changed:
                    break

        if (seed_min > best_min + 1.0e-14 or
                (abs(seed_min - best_min) <= 1.0e-14 and
                 seed_merit > best_merit)):
            best = seed.copy()
            best_min, best_merit = seed_min, seed_merit

    # Final active-set coordinate refinement combines deterministic simplex
    # directions with directions suggested by the currently tight triples.
    for radius in (0.010, 0.005, 0.0025, 0.0012, 0.00055):
        for _ in range(18):
            value = areas(best)
            score, tail_score = merit(value)
            active = triples[value <= score + max(2.8 * radius, 2.0e-5)]
            indices = np.unique(active[active >= 3])
            changed = False

            for index in indices:
                original = np.array((
                    1.0 - best[index, 0] - best[index, 1],
                    best[index, 0],
                    best[index, 1],
                ))
                directions = list(stencil)

                for triple in active:
                    if index in triple:
                        partners = triple[triple != index]
                        vector = best[partners].mean(axis=0) - best[index]
                        length = float(np.linalg.norm(vector))
                        if length > 1.0e-12:
                            uv = vector / length
                            directions.extend((
                                np.array((-uv[0] - uv[1], uv[0], uv[1])),
                                np.array((uv[0] + uv[1], -uv[0], -uv[1])),
                            ))

                local = best
                local_score, local_tail = score, tail_score
                for direction in directions:
                    direction = direction / np.linalg.norm(direction)
                    trial = best.copy()
                    trial[index] = project_weights(
                        original + radius * direction
                    )[1:]
                    trial_values = areas(trial)
                    trial_score, trial_tail = merit(trial_values)
                    if (trial_score > local_score + 1.0e-12 or
                            (abs(trial_score - local_score) <= 1.0e-12 and
                             trial_tail > local_tail + 1.0e-13)):
                        local = trial
                        local_score, local_tail = trial_score, trial_tail

                if local is not best:
                    best = local
                    best_min, best_merit = local_score, local_tail
                    changed = True

            if not changed:
                break

    points = np.empty((n, 2), dtype=float)
    points[:, 0] = best[:, 0] + 0.5 * best[:, 1]
    points[:, 1] = sqrt3_over_2 * best[:, 1]
    return points


# EVOLVE-BLOCK-END