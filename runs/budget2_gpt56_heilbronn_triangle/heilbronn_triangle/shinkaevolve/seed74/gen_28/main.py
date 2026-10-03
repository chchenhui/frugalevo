# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """
    Deterministically construct eleven points in the equilateral triangle.

    Internal coordinates are simplex coordinates (u, v), representing
    u * (1, 0) + v * (1/2, sqrt(3)/2).  Determinants in these coordinates
    are triangle areas normalized by the enclosing triangle area.
    """
    n = 11
    rng = np.random.default_rng(11032011)
    sqrt3_over_2 = np.sqrt(3.0) * 0.5

    triples = np.array(
        [(i, j, k)
         for i in range(n - 2)
         for j in range(i + 1, n - 1)
         for k in range(j + 1, n)],
        dtype=np.intp,
    )
    vertices = np.array(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))

    def determinant_areas(p: np.ndarray) -> np.ndarray:
        a = p[triples[:, 0]]
        b = p[triples[:, 1]]
        c = p[triples[:, 2]]
        return np.abs(
            (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
            - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        )

    def barycentric_pair(weights: np.ndarray) -> np.ndarray:
        """Normalize nonnegative three-weight barycentric coordinates."""
        weights = np.maximum(weights, 1.0e-9)
        weights /= weights.sum()
        return weights[1:].copy()

    def point_weights(p: np.ndarray, index: int) -> np.ndarray:
        return np.array(
            (1.0 - p[index, 0] - p[index, 1], p[index, 0], p[index, 1]),
            dtype=float,
        )

    def phased_merit(values: np.ndarray, progress: float) -> tuple[float, float]:
        """
        Broad lower-tail merit initially, increasingly exact-minimum focused
        near the end of every annealing trajectory.
        """
        minimum = float(values.min())
        count = int(round(30.0 - 20.0 * progress))
        count = max(10, min(count, len(values)))
        low = np.partition(values, count - 1)[:count]
        tail_mean = float(low.mean())

        scale = 0.0085 * (1.0 - progress) ** 1.25 + 0.00028
        shifted = (low - minimum) / scale
        soft_min = minimum - scale * np.log(np.exp(-shifted).mean())

        # Early phases explore configurations with a broad healthy lower tail.
        # Late phases equalize genuinely active determinant constraints.
        tail_weight = 0.62 * (1.0 - progress) ** 1.45 + 0.10
        surrogate = (
            (1.0 - tail_weight) * minimum
            + tail_weight * (0.58 * tail_mean + 0.42 * soft_min)
        )
        return minimum, float(surrogate)

    best = np.vstack((vertices, np.full((8, 2), 1.0 / 3.0)))
    best_values = determinant_areas(best)
    best_score = float(best_values.min())
    elite: list[tuple[float, np.ndarray]] = []

    restarts = 11
    iterations = 20000

    for restart in range(restarts):
        # Alternate ordinary and edge-seeking starts.  Good eleven-point
        # designs frequently use several points close to distinct boundaries.
        concentration = (1.10, 1.10, 1.10) if restart % 3 else (0.72, 0.72, 0.72)
        initial = rng.dirichlet(concentration, size=8)
        current = np.vstack((vertices, initial[:, 1:]))
        current_values = determinant_areas(current)
        current_score, current_merit = phased_merit(current_values, 0.0)

        for iteration in range(iterations):
            progress = iteration / float(iterations - 1)
            current_score, current_merit = phased_merit(current_values, progress)

            # Most moves address a member of a dangerous lower-tail triple.
            # The retained uniform component permits structural reorganization.
            if rng.random() < 0.84:
                scale = 0.0085 * (1.0 - progress) ** 1.25 + 0.00028
                cutoff = current_score + max(2.7 * scale, 0.0011)
                dangerous = triples[current_values <= cutoff]
                chosen = dangerous[int(rng.integers(len(dangerous)))]
                movable = chosen[chosen >= 3]
                index = (
                    int(rng.choice(movable))
                    if len(movable)
                    else int(rng.integers(3, n))
                )
            else:
                index = int(rng.integers(3, n))

            step = 0.115 * (1.0 - progress) ** 1.55 + 0.00065
            candidate = current.copy()

            # Barycentric perturbations avoid the directional bias introduced
            # by separately clipping u and v Cartesian-simplex coordinates.
            weights = point_weights(candidate, index)
            displacement = rng.normal(0.0, step, size=3)
            displacement -= displacement.mean()
            candidate[index] = barycentric_pair(weights + displacement)

            # Coordinated moves are useful during basin discovery, but are
            # suppressed during active-constraint equalization.
            if progress < 0.55 and rng.random() < 0.13:
                second = int(rng.integers(3, n - 1))
                if second >= index:
                    second += 1
                weights = point_weights(candidate, second)
                displacement = rng.normal(0.0, 0.58 * step, size=3)
                displacement -= displacement.mean()
                candidate[second] = barycentric_pair(weights + displacement)

            candidate_values = determinant_areas(candidate)
            candidate_score, candidate_merit = phased_merit(
                candidate_values, progress
            )

            temperature = 0.0029 * (1.0 - progress) ** 2.15 + 1.5e-6
            delta = candidate_merit - current_merit
            if delta >= 0.0 or rng.random() < np.exp(
                max(-700.0, delta / temperature)
            ):
                current = candidate
                current_values = candidate_values
                current_score = candidate_score

            if current_score > best_score + 1.0e-14:
                best = current.copy()
                best_values = current_values.copy()
                best_score = current_score

        elite.append((current_score, current.copy()))
        elite.append((best_score, best.copy()))
        elite.sort(key=lambda item: item[0], reverse=True)
        del elite[6:]

    # Refine all strong terminal basins.  This avoids relying on the raw
    # annealing winner when another near-best cell has superior local potential.
    for _, seed in elite:
        seed_values = determinant_areas(seed)
        seed_score = float(seed_values.min())

        for radius in (0.010, 0.0045):
            for _ in range(10):
                active = triples[
                    seed_values <= seed_score + max(2.5 * radius, 1.5e-4)
                ]
                indices = np.unique(active[active >= 3])
                changed = False

                for index in indices:
                    original = point_weights(seed, int(index))
                    local = seed
                    local_values = seed_values
                    local_score = seed_score

                    for _ in range(14):
                        direction = rng.normal(size=3)
                        direction -= direction.mean()
                        direction *= radius / np.linalg.norm(direction)

                        trial = seed.copy()
                        trial[index] = barycentric_pair(original + direction)
                        trial_values = determinant_areas(trial)
                        trial_score = float(trial_values.min())

                        if trial_score > local_score + 1.0e-12:
                            local = trial
                            local_values = trial_values
                            local_score = trial_score

                    if local is not seed:
                        seed = local
                        seed_values = local_values
                        seed_score = local_score
                        changed = True

                if not changed:
                    break

        if seed_score > best_score + 1.0e-14:
            best = seed.copy()
            best_values = seed_values.copy()
            best_score = seed_score

    # Active-set coordinate polishing with deterministic simplex stencils.
    for radius in (0.008, 0.0035, 0.0015, 0.00065, 0.00025):
        for _ in range(18):
            values = determinant_areas(best)
            score = float(values.min())
            tail_count = 12
            tail = float(np.partition(values, tail_count - 1)[:tail_count].mean())
            active = triples[values <= score + max(2.6 * radius, 3.0e-5)]
            indices = np.unique(active[active >= 3])
            changed = False

            for index in indices:
                original = point_weights(best, int(index))
                directions = [
                    np.array((1.0, -1.0, 0.0)),
                    np.array((1.0, 0.0, -1.0)),
                    np.array((0.0, 1.0, -1.0)),
                    np.array((-1.0, 1.0, 0.0)),
                    np.array((-1.0, 0.0, 1.0)),
                    np.array((0.0, -1.0, 1.0)),
                ]

                for _ in range(10):
                    direction = rng.normal(size=3)
                    directions.append(direction - direction.mean())

                local = best
                local_score = score
                local_tail = tail

                for direction in directions:
                    direction = direction * (radius / np.linalg.norm(direction))
                    trial = best.copy()
                    trial[index] = barycentric_pair(original + direction)
                    trial_values = determinant_areas(trial)
                    trial_score = float(trial_values.min())
                    trial_tail = float(
                        np.partition(trial_values, tail_count - 1)[:tail_count].mean()
                    )

                    if (
                        trial_score > local_score + 1.0e-12
                        or (
                            abs(trial_score - local_score) <= 1.0e-12
                            and trial_tail > local_tail + 1.0e-13
                        )
                    ):
                        local = trial
                        local_score = trial_score
                        local_tail = trial_tail

                if local is not best:
                    best = local
                    changed = True

            if not changed:
                break

    points = np.empty((n, 2), dtype=float)
    points[:, 0] = best[:, 0] + 0.5 * best[:, 1]
    points[:, 1] = sqrt3_over_2 * best[:, 1]
    return points


# EVOLVE-BLOCK-END