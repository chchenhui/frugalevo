# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct eleven points in the equilateral triangle with vertices
    (0, 0), (1, 0), and (1/2, sqrt(3)/2).

    Search is performed in reference-simplex coordinates (u, v), where
    u >= 0, v >= 0, u + v <= 1.  Determinants in these coordinates are
    triangle areas normalized by the containing triangle area.
    """
    rng = np.random.default_rng(11031987)
    h = np.sqrt(3.0) * 0.5
    anchors = np.array(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=float)

    triples = np.array(
        [(i, j, k) for i in range(11) for j in range(i + 1, 11)
         for k in range(j + 1, 11)],
        dtype=np.intp,
    )

    def project_simplex(values: np.ndarray) -> np.ndarray:
        """Safely project coordinate pairs onto u,v >= 0, u+v <= 1."""
        result = np.maximum(values, 0.0)
        total = result.sum(axis=-1, keepdims=True)
        result /= np.maximum(total, 1.0)
        return result

    def score(interior: np.ndarray) -> np.ndarray:
        """Return the minimum normalized area for every candidate layout."""
        count = interior.shape[0]
        layouts = np.empty((count, 11, 2), dtype=float)
        layouts[:, :3] = anchors
        layouts[:, 3:] = interior

        a = layouts[:, triples[:, 0]]
        b = layouts[:, triples[:, 1]]
        c = layouts[:, triples[:, 2]]
        determinants = np.abs(
            (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1])
            - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])
        )
        return determinants.min(axis=1)

    population_size = 128

    # Folded square samples are uniform in the reference simplex.
    population = rng.random((population_size, 8, 2))
    outside = population.sum(axis=2) > 1.0
    population[outside] = 1.0 - population[outside]
    fitness = score(population)

    # Differential evolution supplies large-scale geometric rearrangements.
    generations = 2800
    for generation in range(generations):
        ia = rng.permutation(population_size)
        ib = rng.permutation(population_size)
        ic = rng.permutation(population_size)

        # A modestly varying differential weight avoids a fixed-scale plateau.
        differential_weight = 0.70 + 0.12 * rng.random()
        mutant = population[ia] + differential_weight * (
            population[ib] - population[ic]
        )
        mutant = project_simplex(mutant)

        mask = rng.random((population_size, 8, 2)) < 0.84
        forced = rng.integers(0, 16, size=population_size)
        mask.reshape(population_size, 16)[np.arange(population_size), forced] = True
        trial = np.where(mask, mutant, population)
        trial_fitness = score(trial)

        accepted = trial_fitness >= fitness
        population[accepted] = trial[accepted]
        fitness[accepted] = trial_fitness[accepted]

        # Preserve exploration by replacing weak members with perturbed elites.
        if generation % 140 == 139:
            elite = population[int(np.argmax(fitness))]
            scale = 0.040 * (1.0 - 0.65 * generation / generations)
            injected = project_simplex(
                elite + rng.normal(0.0, scale, size=(16, 8, 2))
            )
            injected_fitness = score(injected)
            weakest = np.argsort(fitness)[:16]
            population[weakest] = injected
            fitness[weakest] = injected_fitness

    best = population[int(np.argmax(fitness))].copy()
    best_value = float(score(best[None, ...])[0])

    # Batched local maximin search is inexpensive and handles final active
    # determinant constraints more effectively than another population phase.
    for iteration in range(1800):
        batch_size = 48
        candidates = np.broadcast_to(best, (batch_size, 8, 2)).copy()
        changed = rng.integers(0, 8, size=batch_size)
        scale = 0.024 * (0.08 ** (iteration / 1800.0))
        candidates[np.arange(batch_size), changed] += rng.normal(
            0.0, scale, size=(batch_size, 2)
        )
        candidates = project_simplex(candidates)
        values = score(candidates)
        winner = int(np.argmax(values))
        if values[winner] > best_value:
            best = candidates[winner]
            best_value = float(values[winner])

    # A compact deterministic stencil removes reproducibility-sensitive noise.
    directions = np.array(
        ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
         (0.707106781, 0.707106781), (0.707106781, -0.707106781),
         (-0.707106781, 0.707106781), (-0.707106781, -0.707106781))
    )
    for step in (0.004, 0.002, 0.001):
        for _ in range(3):
            improved = False
            for point_index in range(8):
                candidates = np.broadcast_to(best, (8, 8, 2)).copy()
                candidates[:, point_index] += step * directions
                candidates = project_simplex(candidates)
                values = score(candidates)
                winner = int(np.argmax(values))
                if values[winner] > best_value:
                    best = candidates[winner]
                    best_value = float(values[winner])
                    improved = True
            if not improved:
                break

    reference = np.vstack((anchors, best))
    points = np.empty((11, 2), dtype=float)
    points[:, 0] = reference[:, 0] + 0.5 * reference[:, 1]
    points[:, 1] = h * reference[:, 1]

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