# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """
    Deterministically construct eleven maximin points in an equilateral
    triangle.  Search coordinates are (u, v) in the unit simplex, where
    Cartesian coordinates are (u + v/2, sqrt(3)*v/2).
    """
    n = 11
    sqrt3_over_2 = np.sqrt(3.0) * 0.5
    rng = np.random.default_rng(11031987)

    triples = np.array(
        [(i, j, k)
         for i in range(n - 2)
         for j in range(i + 1, n - 1)
         for k in range(j + 1, n)],
        dtype=np.intp,
    )
    vertices = np.array(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    involved = [np.flatnonzero(np.any(triples == i, axis=1)) for i in range(n)]

    def all_areas(p: np.ndarray) -> np.ndarray:
        a = p[triples[:, 0]]
        b = p[triples[:, 1]]
        c = p[triples[:, 2]]
        return np.abs(
            (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
            - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        )

    def updated_areas(p: np.ndarray, old: np.ndarray, index: int) -> np.ndarray:
        """Update only the 45 determinants involving a moved point."""
        result = old.copy()
        rows = involved[index]
        t = triples[rows]
        a = p[t[:, 0]]
        b = p[t[:, 1]]
        c = p[t[:, 2]]
        result[rows] = np.abs(
            (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
            - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        )
        return result

    def weights(p: np.ndarray, index: int) -> np.ndarray:
        return np.array(
            (1.0 - p[index, 0] - p[index, 1], p[index, 0], p[index, 1]),
            dtype=float,
        )

    def barycentric_pair(w: np.ndarray) -> np.ndarray:
        w = np.maximum(w, 1.0e-10)
        w /= w.sum()
        return w[1:]

    def measure(values: np.ndarray, progress: float) -> tuple[float, float]:
        minimum = float(values.min())
        count = max(10, int(round(26.0 - 14.0 * progress)))
        low = np.partition(values, count - 1)[:count]
        scale = 0.0090 * (1.0 - progress) ** 1.2 + 0.00038
        soft = minimum - scale * np.log(
            np.exp(-(low - minimum) / scale).mean()
        )
        tail_weight = 0.46 * (1.0 - progress) ** 1.4 + 0.08
        merit = (1.0 - tail_weight) * minimum + tail_weight * (
            0.55 * float(low.mean()) + 0.45 * soft
        )
        return minimum, float(merit)

    best = np.vstack((vertices, np.full((8, 2), 1.0 / 3.0)))
    best_values = all_areas(best)
    best_score = float(best_values.min())
    elite: list[tuple[float, np.ndarray]] = []

    # Incremental area updates make a larger collection of moderate,
    # reproducible trajectories affordable.
    restarts = 10
    iterations = 28000

    for restart in range(restarts):
        concentration = (0.76, 0.76, 0.76) if restart % 3 == 0 else (1.12, 1.12, 1.12)
        initial = rng.dirichlet(concentration, size=8)
        current = np.vstack((vertices, initial[:, 1:]))
        current_values = all_areas(current)
        current_score, current_merit = measure(current_values, 0.0)

        for iteration in range(iterations):
            progress = iteration / float(iterations - 1)
            current_score, current_merit = measure(current_values, progress)
            scale = 0.0090 * (1.0 - progress) ** 1.2 + 0.00038

            if rng.random() < 0.88:
                active_rows = np.flatnonzero(
                    current_values <= current_score + max(2.8 * scale, 0.0010)
                )
                chosen = triples[int(active_rows[rng.integers(len(active_rows))])]
                movable = chosen[chosen >= 3]
                index = (
                    int(movable[rng.integers(len(movable))])
                    if len(movable) else int(rng.integers(3, n))
                )
            else:
                index = int(rng.integers(3, n))

            step = 0.106 * (1.0 - progress) ** 1.5 + 0.0007
            candidate = current.copy()
            displacement = rng.normal(0.0, step, size=3)
            displacement -= displacement.mean()
            candidate[index] = barycentric_pair(weights(candidate, index) + displacement)
            candidate_values = updated_areas(candidate, current_values, index)

            # Early coordinated perturbations help cross combinatorial cells.
            if progress < 0.42 and rng.random() < 0.09:
                second = int(rng.integers(3, n - 1))
                if second >= index:
                    second += 1
                displacement = rng.normal(0.0, 0.52 * step, size=3)
                displacement -= displacement.mean()
                candidate[second] = barycentric_pair(
                    weights(candidate, second) + displacement
                )
                candidate_values = updated_areas(candidate, candidate_values, second)

            candidate_score, candidate_merit = measure(candidate_values, progress)
            temperature = 0.00225 * (1.0 - progress) ** 2.1 + 1.5e-6
            delta = candidate_merit - current_merit

            if delta >= 0.0 or rng.random() < np.exp(max(-700.0, delta / temperature)):
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
        del elite[5:]

    stencil = (
        np.array((1.0, -1.0, 0.0)),
        np.array((-1.0, 1.0, 0.0)),
        np.array((1.0, 0.0, -1.0)),
        np.array((-1.0, 0.0, 1.0)),
        np.array((0.0, 1.0, -1.0)),
        np.array((0.0, -1.0, 0.0)),
    )

    # Locally improve several strong terminal cells before choosing the winner.
    for _, seed in elite:
        values = all_areas(seed)
        score = float(values.min())

        for radius in (0.009, 0.0035):
            for _ in range(8):
                active = triples[values <= score + max(2.6 * radius, 1.0e-4)]
                indices = np.unique(active[active >= 3])
                changed = False

                for index in indices:
                    index = int(index)
                    original = weights(seed, index)
                    local = seed
                    local_values = values
                    local_score = score

                    directions = list(stencil)
                    for _ in range(8):
                        direction = rng.normal(size=3)
                        directions.append(direction - direction.mean())

                    for direction in directions:
                        direction = radius * direction / np.linalg.norm(direction)
                        trial = seed.copy()
                        trial[index] = barycentric_pair(original + direction)
                        trial_values = all_areas(trial)
                        trial_score = float(trial_values.min())
                        if trial_score > local_score + 1.0e-12:
                            local = trial
                            local_values = trial_values
                            local_score = trial_score

                    if local is not seed:
                        seed, values, score = local, local_values, local_score
                        changed = True

                if not changed:
                    break

        if score > best_score + 1.0e-14:
            best, best_values, best_score = seed.copy(), values.copy(), score

    # Exact active-set coordinate polishing of the selected arrangement.
    for radius in (0.006, 0.0025, 0.0010, 0.0004):
        for _ in range(14):
            values = all_areas(best)
            score = float(values.min())
            tail = float(np.partition(values, 11)[:12].mean())
            active = triples[values <= score + max(2.7 * radius, 2.0e-5)]
            indices = np.unique(active[active >= 3])
            changed = False

            for index in indices:
                index = int(index)
                original = weights(best, index)
                local = best
                local_score = score
                local_tail = tail

                for direction in stencil:
                    trial = best.copy()
                    trial[index] = barycentric_pair(
                        original + radius * direction / np.linalg.norm(direction)
                    )
                    trial_values = all_areas(trial)
                    trial_score = float(trial_values.min())
                    trial_tail = float(np.partition(trial_values, 11)[:12].mean())

                    if (
                        trial_score > local_score + 1.0e-12
                        or (
                            abs(trial_score - local_score) <= 1.0e-12
                            and trial_tail > local_tail + 1.0e-13
                        )
                    ):
                        local, local_score, local_tail = trial, trial_score, trial_tail

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