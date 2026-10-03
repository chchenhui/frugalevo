# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct eleven points by maximizing the minimum normalized triangle area.

    Optimization is performed in the reference simplex
    {(u, v): u >= 0, v >= 0, u + v <= 1}.  Its affine image under
    (u, v) -> (u + v / 2, sqrt(3) * v / 2) is the requested triangle.
    """
    rng = np.random.default_rng(11031989)
    h = np.sqrt(3.0) * 0.5
    anchors = np.array(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    triples = np.array(
        [(i, j, k) for i in range(11) for j in range(i + 1, 11)
         for k in range(j + 1, 11)],
        dtype=np.intp,
    )

    def project_simplex(values: np.ndarray) -> np.ndarray:
        """Project approximately onto the two-dimensional probability simplex."""
        values = np.maximum(values, 0.0)
        totals = values.sum(axis=-1, keepdims=True)
        return values / np.maximum(totals, 1.0)

    def score(interior: np.ndarray) -> np.ndarray:
        """Minimum determinant; this is area normalized by enclosing area."""
        all_points = np.concatenate(
            (np.broadcast_to(anchors, (interior.shape[0], 3, 2)), interior),
            axis=1,
        )
        a = all_points[:, triples[:, 0]]
        b = all_points[:, triples[:, 1]]
        c = all_points[:, triples[:, 2]]
        determinants = np.abs(
            (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1])
            - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])
        )
        return determinants.min(axis=1)

    # Uniform simplex samples are obtained by folding the upper half of a square.
    population_size = 96
    population = rng.random((population_size, 8, 2))
    over = population.sum(axis=2) > 1.0
    population[over] = 1.0 - population[over]
    fitness = score(population)

    # Differential evolution supplies global exploration and naturally discovers
    # useful boundary points through simplex projection.
    for generation in range(2500):
        order_a = rng.permutation(population_size)
        order_b = rng.permutation(population_size)
        order_c = rng.permutation(population_size)
        mutant = population[order_a] + 0.78 * (
            population[order_b] - population[order_c]
        )
        mutant = project_simplex(mutant)

        crossover = rng.random((population_size, 8, 2)) < 0.82
        forced = rng.integers(0, 16, size=population_size)
        crossover.reshape(population_size, -1)[np.arange(population_size), forced] = True
        trial = np.where(crossover, mutant, population)
        trial_fitness = score(trial)
        accepted = trial_fitness >= fitness
        population[accepted] = trial[accepted]
        fitness[accepted] = trial_fitness[accepted]

        # A small decreasing jitter prevents stagnation on determinant plateaux.
        if generation % 125 == 124:
            elite = population[np.argmax(fitness)]
            jitter = 0.035 * (1.0 - generation / 2700.0)
            injected = project_simplex(
                elite + rng.normal(0.0, jitter, size=(12, 8, 2))
            )
            injected_fitness = score(injected)
            worst = np.argsort(fitness)[:12]
            population[worst] = injected
            fitness[worst] = injected_fitness

    best = population[np.argmax(fitness)].copy()
    best_value = float(score(best[None, ...])[0])

    # Batched coordinate perturbations polish the nonsmooth maximin solution.
    for iteration in range(2200):
        batch_size = 40
        candidates = np.broadcast_to(best, (batch_size, 8, 2)).copy()
        changed = rng.integers(0, 8, size=batch_size)
        scale = 0.030 * (0.10 ** (iteration / 2200.0))
        candidates[np.arange(batch_size), changed] += rng.normal(
            0.0, scale, size=(batch_size, 2)
        )
        candidates = project_simplex(candidates)
        candidate_values = score(candidates)
        winner = int(np.argmax(candidate_values))
        if candidate_values[winner] > best_value:
            best = candidates[winner]
            best_value = float(candidate_values[winner])

    reference_points = np.vstack((anchors, best))
    points = np.empty_like(reference_points)
    points[:, 0] = reference_points[:, 0] + 0.5 * reference_points[:, 1]
    points[:, 1] = h * reference_points[:, 1]

    if not np.all(np.isfinite(points)):
        return np.array(
            ((0.0, 0.0), (1.0, 0.0), (0.5, h), (0.25, 0.0),
             (0.75, 0.0), (0.125, 0.25 * h), (0.625, 0.25 * h),
             (0.25, 0.5 * h), (0.75, 0.5 * h), (0.375, 0.75 * h),
             (0.5, 0.25 * h)),
            dtype=float,
        )
    return points


# EVOLVE-BLOCK-END