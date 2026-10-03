# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_convex13() -> np.ndarray:
    """
    Deterministically construct thirteen points in an equilateral triangular
    convex region.  Three points are fixed hull vertices and the other ten
    points are optimized in barycentric coordinates.

    The hull is fixed, so maximizing the minimum triangle determinant is
    exactly equivalent to maximizing the hull-normalized minimum area.
    """
    rng = np.random.default_rng(4815162342)

    hull = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, np.sqrt(3.0) * 0.5],
        ],
        dtype=float,
    )
    triples = np.array(
        [
            (i, j, k)
            for i in range(13)
            for j in range(i + 1, 13)
            for k in range(j + 1, 13)
        ],
        dtype=np.intp,
    )

    def normalize_weights(w: np.ndarray) -> np.ndarray:
        """Project positive barycentric candidates back into the simplex."""
        w = np.maximum(w, 2.0e-5)
        return w / w.sum(axis=-1, keepdims=True)

    def evaluate(weights: np.ndarray, need_worst: bool = False):
        """
        Score a batch of layouts using twice their triangle areas.

        Since every layout has the same triangular hull, this quantity has
        exactly the same ordering as normalized triangle area.
        """
        batch = weights.shape[0]
        points = np.empty((batch, 13, 2), dtype=float)
        points[:, :3] = hull
        points[:, 3:] = weights @ hull

        a = points[:, triples[:, 0]]
        b = points[:, triples[:, 1]]
        c = points[:, triples[:, 2]]
        determinants = np.abs(
            (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1])
            - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])
        )
        worst = np.argmin(determinants, axis=1)
        values = determinants[np.arange(batch), worst]
        if need_worst:
            return values, worst
        return values

    population_size = 112
    free_count = 10

    population = rng.dirichlet(np.ones(3), size=(population_size, free_count))

    # Include several low-discrepancy-like, radially balanced starts.  Small
    # deterministic offsets deliberately avoid the collinear triples caused
    # by an exact triangular lattice.
    for seed_index in range(12):
        q = np.empty((free_count, 3), dtype=float)
        for i in range(free_count):
            u = ((i + 0.5 + 0.173 * seed_index) / free_count) % 1.0
            v = ((i * 0.61803398875 + 0.117 * seed_index) % 1.0)
            x = min(u, 1.0 - 1.0e-4)
            y = (1.0 - x) * v
            q[i] = (1.0 - x - y, x, y)
        population[seed_index] = normalize_weights(
            q + rng.normal(0.0, 0.035, size=(free_count, 3))
        )

    scores, worst_ids = evaluate(population, need_worst=True)
    elite_index = int(np.argmax(scores))
    best = population[elite_index].copy()
    best_score = float(scores[elite_index])

    generations = 760
    for generation in range(generations):
        progress = generation / (generations - 1.0)
        step = 0.115 * (0.009 / 0.115) ** progress

        # Candidate 1: full-layout correlated local perturbation.
        local = normalize_weights(
            population + rng.normal(0.0, step, size=population.shape)
        )

        # Candidate 2: move a point participating in the active bottleneck.
        targeted = population.copy()
        for row in range(population_size):
            bottleneck = triples[worst_ids[row]]
            movable = bottleneck[bottleneck >= 3] - 3
            if len(movable):
                point_index = int(movable[rng.integers(len(movable))])
            else:
                point_index = int(rng.integers(free_count))
            targeted[row, point_index] += rng.normal(0.0, step * 1.45, size=3)
            if rng.random() < 0.20:
                other = int(rng.integers(free_count))
                targeted[row, other] += rng.normal(0.0, step * 0.55, size=3)
        targeted = normalize_weights(targeted)

        # Candidate 3: differential move.  It remains valuable late in the
        # search because it transports successful geometric relationships.
        ia = rng.integers(population_size, size=population_size)
        ib = rng.integers(population_size, size=population_size)
        differential = normalize_weights(
            population
            + 0.62 * (population[ia] - population[ib])
            + rng.normal(0.0, step * 0.14, size=population.shape)
        )

        candidates = np.stack((population, local, targeted, differential), axis=1)
        candidate_scores = evaluate(
            candidates.reshape(-1, free_count, 3)
        ).reshape(population_size, 4)
        choices = np.argmax(candidate_scores, axis=1)
        population = candidates[np.arange(population_size), choices]
        scores = candidate_scores[np.arange(population_size), choices]

        current = int(np.argmax(scores))
        if scores[current] > best_score:
            best_score = float(scores[current])
            best = population[current].copy()

        # Preserve multiple basins rather than allowing repeated elitist
        # replacement to collapse the population around a near-duplicate.
        if generation in (150, 300, 455, 600):
            order = np.argsort(scores)
            replace = order[:24]
            fresh = rng.dirichlet(np.ones(3), size=(len(replace), free_count))

            # Half fresh random members; the other half are broad descendants
            # of the elite, giving both global and local diversity.
            split = len(replace) // 2
            fresh[split:] = normalize_weights(
                best[None]
                + rng.normal(0.0, 0.115, size=(len(replace) - split, free_count, 3))
            )
            population[replace] = fresh
            scores[replace], worst_ids[replace] = evaluate(
                population[replace], need_worst=True
            )

        scores, worst_ids = evaluate(population, need_worst=True)
        current = int(np.argmax(scores))
        if scores[current] > best_score:
            best_score = float(scores[current])
            best = population[current].copy()

    # Final active-set polish.  Each batch contains targeted one-point moves,
    # sparse two-point moves, and a few broad moves; only true improvements
    # are accepted.
    polish_steps = (0.020, 0.010, 0.0045, 0.0018, 0.0007)
    for step_size in polish_steps:
        stalled = 0
        for _ in range(150):
            value, worst = evaluate(best[None], need_worst=True)
            bottleneck = triples[int(worst[0])]
            movable = bottleneck[bottleneck >= 3] - 3
            trials = np.repeat(best[None], 48, axis=0)

            for trial in range(48):
                if len(movable):
                    index = int(movable[trial % len(movable)])
                else:
                    index = trial % free_count
                trials[trial, index] += rng.normal(0.0, step_size, size=3)

                if trial >= 24:
                    second = int(rng.integers(free_count))
                    trials[trial, second] += rng.normal(
                        0.0, step_size * 0.42, size=3
                    )

            trials = normalize_weights(trials)
            trial_scores = evaluate(trials)
            chosen = int(np.argmax(trial_scores))
            if trial_scores[chosen] > best_score + 1.0e-13:
                best = trials[chosen].copy()
                best_score = float(trial_scores[chosen])
                stalled = 0
            else:
                stalled += 1
                if stalled >= 18:
                    break

    return np.vstack((hull, best @ hull))


# EVOLVE-BLOCK-END