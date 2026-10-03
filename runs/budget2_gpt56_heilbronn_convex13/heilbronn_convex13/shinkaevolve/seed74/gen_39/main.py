# EVOLVE-BLOCK-START
import numpy as np


def _triangle_indices(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ia = []
    ib = []
    ic = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                ia.append(i)
                ib.append(j)
                ic.append(k)
    return (
        np.asarray(ia, dtype=np.intp),
        np.asarray(ib, dtype=np.intp),
        np.asarray(ic, dtype=np.intp),
    )


def _all_triangle_areas(
    points: np.ndarray,
    ia: np.ndarray,
    ib: np.ndarray,
    ic: np.ndarray,
) -> np.ndarray:
    """Triangle areas for either (n, 2) or (batch, n, 2) point arrays."""
    a = points[..., ia, :]
    b = points[..., ib, :]
    c = points[..., ic, :]
    return 0.5 * np.abs(
        (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1])
        - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])
    )


def _convex_hull_area(points: np.ndarray) -> float:
    """Monotone-chain convex hull area for a small planar point set."""
    pts = sorted((float(x), float(y)) for x, y in points)

    def cross(o, a, b):
        return (
            (a[0] - o[0]) * (b[1] - o[1])
            - (a[1] - o[1]) * (b[0] - o[0])
        )

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return 0.0

    area2 = 0.0
    for i, p in enumerate(hull):
        q = hull[(i + 1) % len(hull)]
        area2 += p[0] * q[1] - p[1] * q[0]
    return abs(area2) * 0.5


def _normalized_signature(
    points: np.ndarray,
    ia: np.ndarray,
    ib: np.ndarray,
    ic: np.ndarray,
) -> tuple[float, float]:
    """Return minimum normalized area and a stabilizing lower-tail mean."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-14:
        return -np.inf, -np.inf
    values = np.sort(_all_triangle_areas(points, ia, ib, ic) / hull_area)
    return float(values[0]), float(np.mean(values[:12]))


def heilbronn_convex13() -> np.ndarray:
    """
    Deterministically construct 13 points in the unit square.

    Four fixed corners make the convex hull exactly the unit square. Therefore
    every raw triangle area evaluated during search is already the requested
    normalized triangle area.
    """
    n = 13
    population_size = 224
    generations = 500
    elite_count = 12
    rng = np.random.default_rng(20240513)

    ia, ib, ic = _triangle_indices(n)
    corners = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=float,
    )

    # Nine free points are initialized by several complementary distributions.
    # Stratification supplies good separation while random starts retain basins
    # that are not close to a perturbed Cartesian lattice.
    cells = np.array([(x, y) for y in range(3) for x in range(3)], dtype=float)
    population = np.empty((population_size, n, 2), dtype=float)
    population[:, :4] = corners

    stratified_count = population_size * 3 // 5
    for m in range(stratified_count):
        chosen = rng.permutation(9)
        jitter = rng.uniform(0.11, 0.89, size=(9, 2))
        population[m, 4:] = (cells[chosen] + jitter) / 3.0

    random_end = population_size * 4 // 5
    population[stratified_count:random_end, 4:] = rng.uniform(
        0.055, 0.945, size=(random_end - stratified_count, 9, 2)
    )

    # Center-biased starts avoid spending too much initial effort in narrow
    # corner strips, where anchored-corner triangles are necessarily small.
    center_count = population_size - random_end
    population[random_end:, 4:] = (
        0.07 + 0.86 * rng.beta(1.35, 1.35, size=(center_count, 9, 2))
    )

    def reflect_unit(x: np.ndarray) -> np.ndarray:
        """Reflect arbitrary coordinates into [0, 1] without boundary piles."""
        x = np.mod(x, 2.0)
        return np.where(x > 1.0, 2.0 - x, x)

    def score_batch(configs: np.ndarray, tail_weight: float):
        areas = _all_triangle_areas(configs, ia, ib, ic)
        low = np.partition(areas, 15, axis=1)[:, :16]
        minimum = low[:, 0]
        tail = np.mean(low[:, :12], axis=1)
        return minimum + tail_weight * tail, minimum, tail

    scores, _, _ = score_batch(population, 0.24)

    # Hybrid DE combines current-to-best exploitation with rand/1 exploration.
    # Corners are restored after every crossover, so all trial hulls remain
    # identical and objective evaluation is especially inexpensive.
    for generation in range(generations):
        progress = generation / (generations - 1)
        tail_weight = 0.24 * (1.0 - progress) + 0.012 * progress
        scores, _, _ = score_batch(population, tail_weight)

        order = np.argsort(scores)
        elites = population[order[-elite_count:]].copy()

        r1 = rng.integers(population_size, size=population_size)
        r2 = rng.integers(population_size, size=population_size)
        r3 = rng.integers(population_size, size=population_size)
        r2 = (r2 + (r2 == r1)) % population_size
        r3 = (r3 + (r3 == r1) + (r3 == r2)) % population_size

        best_pool = population[order[-max(16, population_size // 8):]]
        guide = best_pool[rng.integers(len(best_pool), size=population_size)]

        f = 0.42 + 0.30 * (1.0 - progress)
        current_donor = (
            population
            + 0.52 * (guide - population)
            + f * (population[r1] - population[r2])
        )
        rand_donor = population[r1] + f * (population[r2] - population[r3])
        use_rand = rng.random(population_size) < (0.34 + 0.18 * (1.0 - progress))
        donor = np.where(use_rand[:, None, None], rand_donor, current_donor)
        donor = reflect_unit(donor)

        crossover_rate = 0.79 - 0.12 * progress
        mask = rng.random((population_size, n, 2)) < crossover_rate
        forced = rng.integers(8, n * 2, size=population_size)
        mask.reshape(population_size, -1)[np.arange(population_size), forced] = True

        trial = np.where(mask, donor, population)
        trial[:, :4] = corners
        trial_scores, _, _ = score_batch(trial, tail_weight)

        accepted = trial_scores > scores
        population[accepted] = trial[accepted]
        scores[accepted] = trial_scores[accepted]

        worst = np.argsort(scores)[:elite_count]
        population[worst] = elites
        scores[worst], _, _ = score_batch(elites, tail_weight)

    # Since the hull is the fixed unit square, raw values below are normalized.
    _, minima, tails = score_batch(population, 0.0)
    rank = np.lexsort((tails, minima))
    chains = population[rank[-10:]].copy()
    chain_min = minima[rank[-10:]].copy()
    chain_tail = tails[rank[-10:]].copy()

    # Batched active-constraint refinement.  Moving a point from a currently
    # small triangle concentrates work on variables that can improve maximin.
    local_steps = 2100
    for step in range(local_steps):
        progress = step / (local_steps - 1)
        sigma = 0.034 * (1.0 - progress) ** 2 + 0.00065

        areas = _all_triangle_areas(chains, ia, ib, ic)
        active = np.argpartition(areas, 15, axis=1)[:, :16]
        chosen_triangle = active[
            np.arange(len(chains)),
            rng.integers(16, size=len(chains)),
        ]

        involved = np.stack(
            (ia[chosen_triangle], ib[chosen_triangle], ic[chosen_triangle]),
            axis=1,
        )
        point_ids = involved[
            np.arange(len(chains)),
            rng.integers(3, size=len(chains)),
        ]

        # Corner positions are intentionally immutable.  If an active triangle
        # selects a corner, choose one of its non-corner participating points.
        for row in range(len(chains)):
            if point_ids[row] < 4:
                free = involved[row][involved[row] >= 4]
                point_ids[row] = (
                    int(free[rng.integers(len(free))])
                    if len(free)
                    else int(rng.integers(4, n))
                )

        proposal = chains.copy()
        proposal[np.arange(len(chains)), point_ids] += rng.normal(
            0.0, sigma, size=(len(chains), 2)
        )
        proposal[:, 4:] = reflect_unit(proposal[:, 4:])
        proposal[:, :4] = corners

        _, proposal_min, proposal_tail = score_batch(proposal, 0.0)
        better = (
            (proposal_min > chain_min + 1e-13)
            | (
                (np.abs(proposal_min - chain_min) <= 1e-13)
                & (proposal_tail > chain_tail + 1e-13)
            )
        )

        chains[better] = proposal[better]
        chain_min[better] = proposal_min[better]
        chain_tail[better] = proposal_tail[better]

    winner = np.lexsort((chain_tail, chain_min))[-1]
    best = chains[winner]

    # Retain the general validator-compatible normalized comparison in case of
    # future changes to the hull anchor layout.
    value, tail = _normalized_signature(best, ia, ib, ic)
    if not np.isfinite(value) or value <= 0.0:
        best = np.vstack((corners, np.full((9, 2), 0.5)))

    return np.clip(np.asarray(best, dtype=float), 0.0, 1.0)


# EVOLVE-BLOCK-END