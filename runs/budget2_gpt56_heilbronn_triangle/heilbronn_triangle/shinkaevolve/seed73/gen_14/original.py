# EVOLVE-BLOCK-START
import itertools

import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct a deterministic high-quality 11-point Heilbronn configuration.

    The optimization is carried out in barycentric coordinates (u, v).  In
    these coordinates the determinant of a point triple is its area divided
    by the area of the enclosing equilateral triangle.
    """
    n = 11
    rng = np.random.default_rng(11031957)
    triples = np.asarray(list(itertools.combinations(range(n), 3)), dtype=int)
    population_size = 128
    elite_count = 16
    generations = 6500

    # u, v, and 1-u-v are nonnegative barycentric weights.  Keeping the
    # enclosing triangle's corners makes the available region fully used.
    population = rng.dirichlet((1.0, 1.0, 1.0), size=(population_size, n))
    population = population[..., 1:]
    population[:, 0] = (0.0, 0.0)
    population[:, 1] = (1.0, 0.0)
    population[:, 2] = (0.0, 1.0)

    def project_simplex(uv: np.ndarray) -> np.ndarray:
        """Project coordinate pairs to u>=0, v>=0, u+v<=1."""
        weights = np.empty(uv.shape[:-1] + (3,), dtype=float)
        weights[..., 0] = 1.0 - uv[..., 0] - uv[..., 1]
        weights[..., 1:] = uv
        np.maximum(weights, 1.0e-10, out=weights)
        weights /= weights.sum(axis=-1, keepdims=True)
        return weights[..., 1:]

    def quality(configurations: np.ndarray) -> np.ndarray:
        selected = configurations[:, triples]
        vectors1 = selected[:, :, 1] - selected[:, :, 0]
        vectors2 = selected[:, :, 2] - selected[:, :, 0]
        areas = np.abs(vectors1[..., 0] * vectors2[..., 1]
                       - vectors1[..., 1] * vectors2[..., 0])
        ordered = np.partition(areas, 7, axis=1)[:, :8]
        # The minimum determinant is the evaluation objective.  A very small
        # tail bonus only resolves nearly equal minimum-area candidates.
        return ordered[:, 0] + 0.004 * ordered.mean(axis=1)

    def minimum_area(configurations: np.ndarray) -> np.ndarray:
        """Return the true normalized minimum triangle area."""
        selected = configurations[:, triples]
        vectors1 = selected[:, :, 1] - selected[:, :, 0]
        vectors2 = selected[:, :, 2] - selected[:, :, 0]
        areas = np.abs(vectors1[..., 0] * vectors2[..., 1]
                       - vectors1[..., 1] * vectors2[..., 0])
        return areas.min(axis=1)

    scores = quality(population)
    best = population[np.argmax(minimum_area(population))].copy()
    best_score = minimum_area(best[None, ...])[0]

    for generation in range(generations):
        ranking = np.argsort(scores)[::-1]
        elites = population[ranking[:elite_count]]
        elite_minima = minimum_area(elites)
        best_elite_index = np.argmax(elite_minima)
        if elite_minima[best_elite_index] > best_score:
            best_score = elite_minima[best_elite_index]
            best = elites[best_elite_index].copy()

        parents = elites[rng.integers(elite_count, size=population_size)]
        candidates = parents.copy()

        progress = generation / (generations - 1)
        sigma = 0.105 * (1.0 - progress) ** 1.65 + 0.0012
        changed = rng.random((population_size, n - 3, 1)) < (0.32 - 0.18 * progress)
        noise = rng.normal(0.0, sigma, size=(population_size, n - 3, 2))
        candidates[:, 3:] += noise * changed
        candidates[:, 3:] = project_simplex(candidates[:, 3:])

        # Preserve a small exact elite set each generation.
        candidates[:elite_count] = elites
        candidate_scores = quality(candidates)
        population = candidates
        scores = candidate_scores

    final_minima = minimum_area(population)
    candidate = population[np.argmax(final_minima)]
    if final_minima.max() > best_score:
        best = candidate

    # Convert barycentric (u, v) coordinates to Cartesian coordinates.
    height = np.sqrt(3.0) / 2.0
    return np.column_stack((best[:, 0] + 0.5 * best[:, 1], height * best[:, 1]))


# EVOLVE-BLOCK-END