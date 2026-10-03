# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """
    Deterministically optimize eight free points with differential evolution,
    batched Gaussian maximin polishing, and archive-difference refinement.
    """
    rng = np.random.default_rng(11031987)
    h = np.sqrt(3.0) / 2.0
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, h]])
    triples = np.array(
        [(i, j, k) for i in range(11) for j in range(i + 1, 11)
         for k in range(j + 1, 11)],
        dtype=np.intp,
    )

    def project(p):
        """Project Cartesian candidate points onto the closed equilateral triangle."""
        q = np.asarray(p, dtype=float).copy()
        q[..., 1] = np.clip(q[..., 1], 0.0, h)
        left = q[..., 1] / (2.0 * h)
        q[..., 0] = np.clip(q[..., 0], left, 1.0 - left)
        return q

    def minimum_areas(population):
        """Return the minimum triangle area for every configuration in a population."""
        a = population[:, triples[:, 0]]
        b = population[:, triples[:, 1]]
        c = population[:, triples[:, 2]]
        areas = 0.5 * np.abs(
            (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1])
            - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])
        )
        return areas.min(axis=1)

    # Uniform barycentric samples, with all population members sharing the
    # useful and usually optimal corner points.
    pop_size = 48
    population = np.empty((pop_size, 11, 2), dtype=float)
    population[:, :3] = vertices
    uv = rng.random((pop_size, 8, 2))
    mask = uv.sum(axis=2) > 1.0
    uv[mask] = 1.0 - uv[mask]
    population[:, 3:, 0] = uv[..., 0] + 0.5 * uv[..., 1]
    population[:, 3:, 1] = h * uv[..., 1]
    scores = minimum_areas(population)

    # Differential evolution supplies large coordinated moves, which are much
    # more effective than moving one point at a time from a random start.
    for generation in range(700):
        trial = population.copy()
        scale = 0.72 - 0.22 * generation / 699.0
        for i in range(pop_size):
            choices = rng.choice(pop_size - 1, size=3, replace=False)
            choices += choices >= i
            a, b, c = population[choices]
            donor = a + scale * (b - c)
            donor[:3] = vertices

            cross = rng.random((11, 2)) < 0.72
            cross[:3] = False
            # Ensure that each trial actually inherits some donor coordinates.
            point = rng.integers(3, 11)
            cross[point, rng.integers(0, 2)] = True
            trial[i] = np.where(cross, donor, population[i])
            trial[i, :3] = vertices
            trial[i, 3:] = project(trial[i, 3:])

        trial_scores = minimum_areas(trial)
        accepted = trial_scores >= scores
        population[accepted] = trial[accepted]
        scores[accepted] = trial_scores[accepted]

    best = population[np.argmax(scores)].copy()
    best_score = float(np.max(scores))

    # The nonsmooth minimum-area objective commonly needs two or three points
    # to move together before an active constraint can be released.  Evaluate
    # independent perturbations in vectorized batches: this is substantially
    # cheaper than Python-level single-candidate evaluations and greedily keeps
    # the strongest non-worsening proposal from every batch.
    batch_size = 64
    for step, batches, moved in (
        (0.045, 160, 1), (0.035, 220, 3), (0.016, 300, 2),
        (0.007, 380, 2), (0.003, 500, 1), (0.0011, 550, 1),
        (0.00035, 400, 1),
    ):
        for _ in range(batches):
            candidates = np.repeat(best[None, :, :], batch_size, axis=0)
            # Choose exactly ``moved`` distinct free points in each candidate.
            order = np.argpartition(
                rng.random((batch_size, 8)), moved - 1, axis=1
            )[:, :moved]
            rows = np.arange(batch_size)[:, None]
            cols = order + 3
            candidates[rows, cols] += rng.normal(
                scale=step, size=(batch_size, moved, 2)
            )
            candidates[:, 3:] = project(candidates[:, 3:])
            values = minimum_areas(candidates)
            winner = int(np.argmax(values))
            if values[winner] >= best_score:
                best = candidates[winner]
                best_score = float(values[winner])

    # The final DE population is a useful archive of geometrically distinct
    # arrangements.  Difference vectors between its members provide coherent
    # multi-point directions that are unavailable to independent local noise.
    # This phase remains monotone, so it can only preserve or improve the
    # incumbent found above.
    elite_count = 16
    elite = population[np.argsort(scores)[-elite_count:]]
    for scale, batches in ((0.22, 180), (0.11, 220), (0.050, 260), (0.020, 280)):
        for _ in range(batches):
            choices = rng.integers(elite_count, size=(batch_size, 2))
            candidates = np.repeat(best[None, :, :], batch_size, axis=0)
            direction = elite[choices[:, 0]] - elite[choices[:, 1]]
            # A point-wise mask preserves useful point correlations while
            # allowing only part of a differential direction to be used.
            take = rng.random((batch_size, 8)) < 0.58
            take[np.arange(batch_size), rng.integers(8, size=batch_size)] = True
            candidates[:, 3:] += scale * direction[:, 3:] * take[:, :, None]
            candidates[:, 3:] = project(candidates[:, 3:])
            values = minimum_areas(candidates)
            winner = int(np.argmax(values))
            if values[winner] >= best_score:
                best = candidates[winner]
                best_score = float(values[winner])

    # Finish the archive phase with fine coupled perturbations around its best
    # result.  These moves are particularly useful after a differential move
    # has changed the set of active minimum-area triples.
    for step, batches, moved in ((0.0020, 350, 2), (0.00065, 450, 1)):
        for _ in range(batches):
            candidates = np.repeat(best[None, :, :], batch_size, axis=0)
            order = np.argpartition(
                rng.random((batch_size, 8)), moved - 1, axis=1
            )[:, :moved]
            rows = np.arange(batch_size)[:, None]
            candidates[rows, order + 3] += rng.normal(
                scale=step, size=(batch_size, moved, 2)
            )
            candidates[:, 3:] = project(candidates[:, 3:])
            values = minimum_areas(candidates)
            winner = int(np.argmax(values))
            if values[winner] >= best_score:
                best = candidates[winner]
                best_score = float(values[winner])

    return best


# EVOLVE-BLOCK-END
