# EVOLVE-BLOCK-START
import itertools

import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """
    Deterministic continuation-island search for an 11 point Heilbronn
    configuration in the reference equilateral triangle.

    Internal coordinates are (u, v), with u >= 0, v >= 0, u + v <= 1.
    Their determinant areas equal Cartesian triangle areas normalized by the
    containing equilateral triangle's area.
    """
    n = 11
    free_count = n - 3
    rng = np.random.default_rng(728391104)
    triples = np.asarray(list(itertools.combinations(range(n), 3)), dtype=np.intp)
    ia, ib, ic = triples[:, 0], triples[:, 1], triples[:, 2]

    corners = np.array(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=float)

    def project(uv: np.ndarray) -> np.ndarray:
        """Feasibly project pairs onto the barycentric coordinate triangle."""
        weights = np.empty(uv.shape[:-1] + (3,), dtype=float)
        weights[..., 0] = 1.0 - uv[..., 0] - uv[..., 1]
        weights[..., 1:] = uv
        np.maximum(weights, 1.0e-9, out=weights)
        weights /= weights.sum(axis=-1, keepdims=True)
        return weights[..., 1:]

    def areas(configs: np.ndarray) -> np.ndarray:
        q = configs[:, triples]
        p = q[:, :, 1] - q[:, :, 0]
        r = q[:, :, 2] - q[:, :, 0]
        return np.abs(p[..., 0] * r[..., 1] - p[..., 1] * r[..., 0])

    def objective(configs: np.ndarray, tail: float = 0.0) -> np.ndarray:
        a = areas(configs)
        if tail <= 0.0:
            return a.min(axis=1)
        low = np.partition(a, 9, axis=1)[:, :10]
        return low[:, 0] + tail * low.mean(axis=1)

    def initial_population(count: int) -> np.ndarray:
        weights = rng.dirichlet((0.92, 0.92, 0.92), size=(count, n))
        result = weights[..., 1:].copy()
        result[:, :3] = corners

        # Several perturbed triangular patterns make good coverage of the
        # domain available from the start without assuming exact symmetry.
        template = np.array(
            (
                (0.0, 0.0), (1.0, 0.0), (0.0, 1.0),
                (0.18, 0.09), (0.49, 0.08), (0.79, 0.09),
                (0.10, 0.35), (0.39, 0.31), (0.66, 0.26),
                (0.13, 0.65), (0.39, 0.50),
            ),
            dtype=float,
        )
        for k in range(min(28, count)):
            result[k] = template
            result[k, 3:] += rng.normal(0.0, 0.035 + 0.002 * k,
                                        size=(free_count, 2))
        result[:, 3:] = project(result[:, 3:])
        return result

    # Phase 1: broad island exploration.  The population is intentionally
    # larger than the later polish population.
    broad_size = 192
    population = initial_population(broad_size)
    values = objective(population, 0.040)
    best_index = int(np.argmax(objective(population)))
    best = population[best_index].copy()
    best_value = float(objective(best[None, ...])[0])

    for generation in range(1050):
        progress = generation / 1049.0
        rank = np.argsort(values)[::-1]
        elite_count = 24
        elites = population[rank[:elite_count]]

        true_elite = objective(elites)
        winner = int(np.argmax(true_elite))
        if true_elite[winner] > best_value:
            best_value = float(true_elite[winner])
            best = elites[winner].copy()

        parents = elites[rng.integers(elite_count, size=broad_size)]
        candidate = parents.copy()

        sigma = 0.105 * (1.0 - progress) ** 1.45 + 0.004
        changed = rng.random((broad_size, free_count, 1)) < (
            0.54 - 0.18 * progress
        )
        candidate[:, 3:] += rng.normal(
            0.0, sigma, size=(broad_size, free_count, 2)
        ) * changed

        # Correlated island moves relocate several geometrically related
        # points together, rather than relying on independent random walks.
        count = broad_size // 2
        a = rng.integers(broad_size, size=count)
        b = rng.integers(broad_size, size=count)
        c = rng.integers(broad_size, size=count)
        factor = 0.82 - 0.28 * progress
        donor = population[a, 3:] + factor * (
            population[b, 3:] - population[c, 3:]
        )
        mask = rng.random((count, free_count, 1)) < 0.62
        mask[np.arange(count), rng.integers(free_count, size=count), 0] = True
        candidate[elite_count:elite_count + count, 3:] = np.where(
            mask,
            donor,
            parents[elite_count:elite_count + count, 3:],
        )

        # Reflection about the simplex centroid supplies a deterministic
        # alternative basin-generating move.
        reflected_rows = np.arange(elite_count + count, broad_size)
        if len(reflected_rows):
            source = parents[reflected_rows, 3:]
            candidate[reflected_rows, 3:] = (
                2.0 / 3.0 - source
                + rng.normal(0.0, sigma * 0.35, size=source.shape)
            )

        candidate[:, 3:] = project(candidate[:, 3:])
        candidate[:elite_count] = elites
        population = candidate
        values = objective(population, 0.040 * (1.0 - progress) + 0.006)

    # Phase 2: exact 128/16 survivor refinement, seeded by the strongest
    # distinct broad layouts.
    true_values = objective(population)
    order = np.argsort(true_values)[::-1]
    population = np.empty((128, n, 2), dtype=float)
    population[:64] = population_seed = (
        # retain broad leaders, then give their descendants small separation
        # in the remaining slots
        np.array([population[i] for i in order[:64]])
    )
    population[64:] = population_seed[rng.integers(64, size=64)]
    population[64:, 3:] += rng.normal(0.0, 0.014, size=(64, free_count, 2))
    population[64:, 3:] = project(population[64:, 3:])
    values = objective(population, 0.009)

    for generation in range(2500):
        progress = generation / 2499.0
        ranking = np.argsort(values)[::-1]
        elites = population[ranking[:16]]
        elite_true = objective(elites)
        winner = int(np.argmax(elite_true))
        if elite_true[winner] > best_value:
            best_value = float(elite_true[winner])
            best = elites[winner].copy()

        parents = elites[rng.integers(16, size=128)]
        candidate = parents.copy()
        sigma = 0.030 * (1.0 - progress) ** 1.85 + 0.00045
        probability = 0.31 - 0.17 * progress
        touched = rng.random((128, free_count, 1)) < probability
        candidate[:, 3:] += rng.normal(
            0.0, sigma, size=(128, free_count, 2)
        ) * touched

        # A smaller number of late correlated moves avoids collapsing every
        # survivor into one coordinate-wise local optimum.
        if generation < 1450:
            rows = np.arange(16, 64)
            a = rng.integers(128, size=len(rows))
            b = rng.integers(128, size=len(rows))
            c = rng.integers(128, size=len(rows))
            mutant = population[a, 3:] + (0.52 - 0.16 * progress) * (
                population[b, 3:] - population[c, 3:]
            )
            cross = rng.random((len(rows), free_count, 1)) < 0.48
            cross[np.arange(len(rows)), rng.integers(free_count, size=len(rows)), 0] = True
            candidate[rows, 3:] = np.where(cross, mutant, candidate[rows, 3:])

        candidate[:, 3:] = project(candidate[:, 3:])
        candidate[:16] = elites
        population = candidate
        values = objective(population, 0.009 * (1.0 - progress))

    terminal = objective(population)
    terminal_best = population[int(np.argmax(terminal))]
    if float(terminal.max()) > best_value:
        best = terminal_best.copy()
        best_value = float(terminal.max())

    # Phase 3: active-constraint continuation polish.  The soft-min gradient
    # directs coordinated changes using all nearly limiting triangles.
    for iteration in range(340):
        q = best[triples]
        a, b, c = q[:, 0], q[:, 1], q[:, 2]
        signed = ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                  - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
        absolute = np.abs(signed)
        temperature = max(0.00020, 0.0018 * (1.0 - iteration / 340.0))
        shifted = absolute - absolute.min()
        weight = np.exp(-shifted / temperature)
        weight /= weight.sum()
        coeff = weight * np.where(signed >= 0.0, 1.0, -1.0)

        gradient = np.zeros((n, 2), dtype=float)
        ga = np.column_stack((b[:, 1] - c[:, 1], c[:, 0] - b[:, 0]))
        gb = np.column_stack((c[:, 1] - a[:, 1], a[:, 0] - c[:, 0]))
        gc = np.column_stack((a[:, 1] - b[:, 1], b[:, 0] - a[:, 0]))
        np.add.at(gradient, ia, ga * coeff[:, None])
        np.add.at(gradient, ib, gb * coeff[:, None])
        np.add.at(gradient, ic, gc * coeff[:, None])
        gradient[:3] = 0.0
        norm = np.sqrt((gradient[3:] * gradient[3:]).sum(axis=1, keepdims=True))
        gradient[3:] /= np.maximum(norm, 1.0e-10)

        step = 0.010 * (1.0 - iteration / 340.0) ** 1.5 + 0.00035
        candidates = np.broadcast_to(best, (74, n, 2)).copy()
        candidates[0, 3:] += step * gradient[3:]
        candidates[1, 3:] -= 0.35 * step * gradient[3:]

        noise = rng.normal(0.0, step, size=(72, free_count, 2))
        mask = np.zeros((72, free_count, 1), dtype=bool)
        mask[:42, rng.integers(free_count, size=42), 0] = True
        row = np.arange(42, 72)
        mask[row, rng.integers(free_count, size=30), 0] = True
        mask[row, rng.integers(free_count, size=30), 0] = True
        candidates[2:, 3:] += noise * mask
        candidates[:, 3:] = project(candidates[:, 3:])

        candidate_values = objective(candidates)
        choice = int(np.argmax(candidate_values))
        if candidate_values[choice] > best_value:
            best = candidates[choice]
            best_value = float(candidate_values[choice])

    height = np.sqrt(3.0) * 0.5
    return np.column_stack((best[:, 0] + 0.5 * best[:, 1], height * best[:, 1]))
# EVOLVE-BLOCK-END