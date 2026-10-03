# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """
    Deterministically construct a high-minimum-area 11-point configuration in
    the equilateral triangle with vertices (0,0), (1,0), and (1/2,sqrt(3)/2).

    Three container vertices and three side support points are explicit.  The
    remaining five points use barycentric coordinates, guaranteeing feasibility.
    """
    h = np.sqrt(3.0) * 0.5
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, h]], dtype=float)
    triples = np.array(
        [(i, j, k) for i in range(11) for j in range(i + 1, 11)
         for k in range(j + 1, 11)],
        dtype=np.intp,
    )
    rng = np.random.default_rng(11031987)

    # Parameters:
    #   0,1,2: one point on base, left side, and right side respectively.
    #   3:13 : five interior points, each represented by (q,v), with
    #          u=q*(1-v), so u,v >= 0 and u+v <= 1.
    dimensions = 13

    def reflect_unit(x: np.ndarray) -> np.ndarray:
        """Reflect arbitrary real values into [0,1] without clipping bias."""
        x = np.mod(x, 2.0)
        return np.where(x > 1.0, 2.0 - x, x)

    def decode(population: np.ndarray) -> np.ndarray:
        m = len(population)
        p = np.empty((m, 11, 2), dtype=float)
        p[:, :3] = vertices

        # One non-vertex point on each container side.
        t0, t1, t2 = population[:, 0], population[:, 1], population[:, 2]
        p[:, 3, 0], p[:, 3, 1] = t0, 0.0
        p[:, 4, 0], p[:, 4, 1] = 0.5 * t1, h * t1
        p[:, 5, 0], p[:, 5, 1] = 1.0 - 0.5 * t2, h * t2

        z = population[:, 3:].reshape(m, 5, 2)
        v = z[:, :, 1]
        u = z[:, :, 0] * (1.0 - v)
        p[:, 6:, 0] = u + 0.5 * v
        p[:, 6:, 1] = h * v
        return p

    def areas(population: np.ndarray) -> np.ndarray:
        p = decode(population)
        a = p[:, triples[:, 0]]
        b = p[:, triples[:, 1]]
        c = p[:, triples[:, 2]]
        return np.abs(
            (b[:, :, 0] - a[:, :, 0]) * (c[:, :, 1] - a[:, :, 1])
            - (b[:, :, 1] - a[:, :, 1]) * (c[:, :, 0] - a[:, :, 0])
        )

    def true_quality(population: np.ndarray) -> np.ndarray:
        return areas(population).min(axis=1)

    def smooth_quality(population: np.ndarray, temperature: float) -> np.ndarray:
        """
        Stable soft-minimum of doubled triangle areas.  Unlike a bare minimum,
        this also improves triangles that are close to becoming active.
        """
        x = areas(population)
        lo = x.min(axis=1)
        return lo - temperature * np.log(
            np.exp(-(x - lo[:, None]) / temperature).sum(axis=1)
        )

    population_size = 288
    population = rng.random((population_size, dimensions))

    # Seed several well-spread configurations, avoiding a wholly random start.
    for r in range(48):
        population[r, :3] = np.array([0.22, 0.49, 0.76]) + rng.normal(0.0, 0.07, 3)
        population[r, 3:] = rng.random(10)
        population[r, 3::2] = np.mod(
            np.linspace(0.10, 0.90, 5) + rng.normal(0.0, 0.10, 5), 1.0
        )
        population[r, 4::2] = np.mod(
            np.linspace(0.78, 0.18, 5) + rng.normal(0.0, 0.10, 5), 1.0
        )
    population = reflect_unit(population)

    actual = true_quality(population)
    best_index = int(np.argmax(actual))
    best = population[best_index].copy()
    best_score = float(actual[best_index])

    for generation in range(1500):
        temp = 0.006 if generation < 850 else 0.0022
        scores = smooth_quality(population, temp)

        order = np.argsort(scores)
        elite = population[order[-1]]
        idx = np.arange(population_size)

        a = rng.permutation(idx)
        b = rng.permutation(idx)
        c = rng.permutation(idx)
        a = np.where(a == idx, (a + 1) % population_size, a)
        b = np.where((b == idx) | (b == a), (b + 7) % population_size, b)
        c = np.where(
            (c == idx) | (c == a) | (c == b),
            (c + 19) % population_size,
            c,
        )

        scale = 0.76 if generation < 950 else 0.49
        mutant = population[a] + scale * (population[b] - population[c])
        mutant += 0.13 * (elite - population[a])
        mutant = reflect_unit(mutant)

        cross_rate = 0.87 if generation < 1000 else 0.76
        cross = rng.random((population_size, dimensions)) < cross_rate
        cross[idx, rng.integers(0, dimensions, population_size)] = True
        trial = np.where(cross, mutant, population)

        trial_scores = smooth_quality(trial, temp)
        accepted = trial_scores >= scores
        population[accepted] = trial[accepted]

        # Preserve the best true max-min design even while the population uses
        # the smoother exploratory objective.
        trial_actual = true_quality(trial)
        candidate = int(np.argmax(trial_actual))
        if trial_actual[candidate] > best_score:
            best_score = float(trial_actual[candidate])
            best = trial[candidate].copy()

    # Deterministic batch hill refinement on the exact objective.
    step = 0.040
    for _ in range(520):
        candidates = reflect_unit(
            best + rng.normal(0.0, step, size=(112, dimensions))
        )
        candidate_scores = true_quality(candidates)
        winner = int(np.argmax(candidate_scores))
        if candidate_scores[winner] > best_score:
            best_score = float(candidate_scores[winner])
            best = candidates[winner]
            step = min(0.075, step * 1.018)
        else:
            step = max(0.0010, step * 0.987)

    return decode(best[None, :])[0]


# EVOLVE-BLOCK-END