# EVOLVE-BLOCK-START
import itertools
import numpy as np


# Point triples are constant for n = 11 and are prepared once.
_TRIPLES = np.asarray(list(itertools.combinations(range(11), 3)), dtype=np.intp)
_RESULT_CACHE = None


def _project_simplex_coordinates(q: np.ndarray) -> np.ndarray:
    """Project two free barycentric coordinates into b>=0, c>=0, b+c<=1."""
    q = np.asarray(q, dtype=float).copy()
    q = np.maximum(q, 1.0e-5)
    s = q[..., 0] + q[..., 1]
    mask = s > 0.99998
    if np.any(mask):
        q[mask] *= (0.99998 / s[mask])[..., None]
    return q


def _normalized_areas(free_points: np.ndarray) -> np.ndarray:
    """
    Return all normalized triangle areas.

    A point is represented by (b,c), with barycentric coordinates
    (1-b-c,b,c).  The determinant of three barycentric rows is exactly
    the corresponding triangle area divided by the outer triangle area.
    """
    p = np.empty((11, 2), dtype=float)
    p[:3] = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
    p[3:] = free_points

    bary = np.empty((11, 3), dtype=float)
    bary[:, 1:] = p
    bary[:, 0] = 1.0 - p[:, 0] - p[:, 1]

    a = bary[_TRIPLES]
    return np.abs(
        a[:, 0, 0] * (a[:, 1, 1] * a[:, 2, 2] - a[:, 1, 2] * a[:, 2, 1])
        - a[:, 0, 1] * (a[:, 1, 0] * a[:, 2, 2] - a[:, 1, 2] * a[:, 2, 0])
        + a[:, 0, 2] * (a[:, 1, 0] * a[:, 2, 1] - a[:, 1, 1] * a[:, 2, 0])
    )


def _merit(free_points: np.ndarray) -> tuple:
    """
    A lexicographic low-tail merit.

    Using several of the smallest areas, rather than only the single
    smallest one, avoids stagnation on configurations having many almost
    active constraints.
    """
    areas = _normalized_areas(free_points)
    tail = np.partition(areas, 11)[:12]
    tail.sort()
    # The first term remains the true maximin objective; the other terms
    # make exchanges robust when multiple triangles are tied.
    value = tail[0] + 0.18 * tail[1:5].mean() + 0.035 * tail[5:12].mean()
    return value, tail[0], areas


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct 11 points in/on the specified equilateral triangle.

    The calculation is deterministic.  Results are cached so repeated
    calls have negligible cost.
    """
    global _RESULT_CACHE
    if _RESULT_CACHE is not None:
        return _RESULT_CACHE.copy()

    try:
        rng = np.random.default_rng(11031987)

        # Vertices are implicit.  These eight points form a well-spread
        # central seed before the exchange search starts.
        seed = np.array(
            [
                (0.115, 0.235), (0.115, 0.500),
                (0.235, 0.115), (0.235, 0.470),
                (0.410, 0.115), (0.410, 0.355),
                (0.590, 0.115), (0.235, 0.650),
            ],
            dtype=float,
        )

        population_size = 22
        population = np.empty((population_size, 8, 2), dtype=float)
        population[0] = seed
        for k in range(1, population_size):
            noise = rng.normal(0.0, 0.075, size=(8, 2))
            population[k] = _project_simplex_coordinates(seed + noise)

        merits = np.empty(population_size)
        minima = np.empty(population_size)
        for k in range(population_size):
            merits[k], minima[k], _ = _merit(population[k])

        best_index = int(np.argmax(minima))
        best = population[best_index].copy()
        best_minimum = float(minima[best_index])

        # Differential exchange phase.  Candidate replacement uses the
        # smooth low-tail merit, while the best record uses true min area.
        for generation in range(3600):
            scale = 0.72 - 0.30 * (generation / 3599.0)
            cross_probability = 0.78

            for i in range(population_size):
                choices = np.delete(np.arange(population_size), i)
                r1, r2, r3 = rng.choice(choices, size=3, replace=False)
                donor = population[r1] + scale * (population[r2] - population[r3])

                mask = rng.random((8, 2)) < cross_probability
                mask[rng.integers(8), rng.integers(2)] = True
                trial = np.where(mask, donor, population[i])
                trial = _project_simplex_coordinates(trial)

                trial_merit, trial_minimum, _ = _merit(trial)
                if trial_merit >= merits[i]:
                    population[i] = trial
                    merits[i] = trial_merit
                    minima[i] = trial_minimum

                if trial_minimum > best_minimum:
                    best = trial.copy()
                    best_minimum = float(trial_minimum)

        # Targeted deterministic local exchange repair.  Perturbations are
        # gradually reduced, producing a reproducible final polishing step.
        current = best.copy()
        current_merit, current_minimum, _ = _merit(current)
        for step in range(18000):
            sigma = 0.018 * (1.0 - step / 18000.0) + 0.0012
            candidate = current.copy()

            # Most moves alter one point; occasional pair moves escape
            # tightly coupled worst-triangle constraints.
            count = 2 if (step % 11 == 0) else 1
            ids = rng.choice(8, size=count, replace=False)
            candidate[ids] += rng.normal(0.0, sigma, size=(count, 2))
            candidate = _project_simplex_coordinates(candidate)

            candidate_merit, candidate_minimum, _ = _merit(candidate)
            if candidate_merit >= current_merit:
                current = candidate
                current_merit = candidate_merit
                current_minimum = candidate_minimum

            if candidate_minimum > best_minimum:
                best = candidate.copy()
                best_minimum = float(candidate_minimum)

        # Convert (b,c) barycentric coordinates to Cartesian coordinates:
        # x = b + c/2, y = sqrt(3)c/2.
        bary_points = np.empty((11, 2), dtype=float)
        bary_points[:3] = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
        bary_points[3:] = best

        result = np.empty((11, 2), dtype=float)
        result[:, 0] = bary_points[:, 0] + 0.5 * bary_points[:, 1]
        result[:, 1] = 0.5 * np.sqrt(3.0) * bary_points[:, 1]

    except Exception:
        # Safe, valid fallback in the unlikely event of an optimization
        # failure.  It still returns exactly eleven points in the region.
        result = np.array(
            [
                (0.0, 0.0), (1.0, 0.0), (0.5, np.sqrt(3.0) / 2.0),
                (0.20, 0.10), (0.40, 0.08), (0.62, 0.10),
                (0.30, 0.28), (0.50, 0.24), (0.70, 0.27),
                (0.40, 0.48), (0.55, 0.46),
            ],
            dtype=float,
        )

    _RESULT_CACHE = result
    return result.copy()


# EVOLVE-BLOCK-END
