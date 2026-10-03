# EVOLVE-BLOCK-START
import numpy as np


def _triangle_indices(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return np.triu_indices(n, 1)[0], np.array(
        [k for i in range(n) for j in range(i + 1, n) for k in range(j + 1, n)],
        dtype=np.intp,
    ), np.array(
        [j for i in range(n) for j in range(i + 1, n) for _ in range(j + 1, n)],
        dtype=np.intp,
    )


def _all_triangle_areas(points: np.ndarray,
                        ia: np.ndarray,
                        ib: np.ndarray,
                        ic: np.ndarray) -> np.ndarray:
    """Triangle areas for either (n,2) or (batch,n,2) arrays."""
    a = points[..., ia, :]
    b = points[..., ib, :]
    c = points[..., ic, :]
    return 0.5 * np.abs(
        (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1])
        - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])
    )


def _convex_hull_area(points: np.ndarray) -> float:
    """Monotone-chain convex hull area for the small fixed n=13 case."""
    pts = sorted((float(x), float(y)) for x, y in points)

    def cross(o, a, b):
        return ((a[0] - o[0]) * (b[1] - o[1])
                - (a[1] - o[1]) * (b[0] - o[0]))

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


def _normalized_signature(points: np.ndarray,
                          ia: np.ndarray,
                          ib: np.ndarray,
                          ic: np.ndarray) -> tuple[float, float]:
    """
    Primary value is the requested affine-invariant minimum triangle area.
    The secondary lower-tail average stabilizes local search near ties.
    """
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-14:
        return -np.inf, -np.inf

    values = np.sort(_all_triangle_areas(points, ia, ib, ic) / hull_area)
    return float(values[0]), float(np.mean(values[:12]))


def heilbronn_convex13() -> np.ndarray:
    """
    Deterministically construct 13 points in the unit square.

    The unit square is a convex region of unit area.  The returned array has
    shape (13, 2), and all coordinates are finite values in [0, 1].
    """
    n = 13
    population_size = 160
    generations = 360
    rng = np.random.default_rng(20240513)

    # Explicit triangle index lists, avoiding repeated combinations creation.
    ia = []
    ib = []
    ic = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                ia.append(i)
                ib.append(j)
                ic.append(k)
    ia = np.asarray(ia, dtype=np.intp)
    ib = np.asarray(ib, dtype=np.intp)
    ic = np.asarray(ic, dtype=np.intp)

    # Stratified starts give much better initial separation than iid points.
    cells = np.array([(x, y) for y in range(4) for x in range(4)], dtype=float)
    population = np.empty((population_size, n, 2), dtype=float)
    for m in range(population_size):
        chosen = rng.permutation(16)[:n]
        jitter = rng.uniform(0.12, 0.88, size=(n, 2))
        population[m] = (cells[chosen] + jitter) / 4.0

    # A minority of independent starts maintains global diversity.
    population[-population_size // 4:] = rng.random(
        (population_size // 4, n, 2)
    )

    def batch_score(configs: np.ndarray) -> np.ndarray:
        areas = _all_triangle_areas(configs, ia, ib, ic)
        smallest = np.partition(areas, 11, axis=1)[:, :12]
        # The minimum dominates, while the lower tail prevents one-triangle
        # improvements that create many almost-degenerate triples.
        return smallest[:, 0] + 0.16 * np.mean(smallest, axis=1)

    scores = batch_score(population)

    # Vectorized differential evolution: broad global search with no scipy
    # dependency and deterministic pseudo-random choices.
    for generation in range(generations):
        order = np.argsort(scores)
        elite_count = 10
        elites = population[order[-elite_count:]].copy()

        r1 = rng.integers(0, population_size, size=population_size)
        r2 = rng.integers(0, population_size, size=population_size)
        r3 = rng.integers(0, population_size, size=population_size)

        # Ensure useful donor diversity even if index draws coincide.
        r2 = (r2 + (r2 == r1)) % population_size
        r3 = (r3 + (r3 == r1) + (r3 == r2)) % population_size

        scale = 0.56 + 0.20 * (1.0 - generation / generations)
        donor = population[r1] + scale * (population[r2] - population[r3])

        # Reflection rather than clipping avoids artificial piles on edges.
        donor = np.abs(donor)
        donor = np.where(donor > 1.0, 2.0 - donor, donor)

        crossover = rng.random((population_size, n, 2)) < 0.72
        forced_coordinate = rng.integers(0, n * 2, size=population_size)
        crossover.reshape(population_size, -1)[
            np.arange(population_size), forced_coordinate
        ] = True

        trial = np.where(crossover, donor, population)
        trial_scores = batch_score(trial)
        accept = trial_scores > scores
        population[accept] = trial[accept]
        scores[accept] = trial_scores[accept]

        # Elitism protects rare high-quality maximin configurations.
        worst = np.argsort(scores)[:elite_count]
        population[worst] = elites
        scores[worst] = batch_score(elites)

    # Select several finalists by the true normalized criterion.
    finalists = population[np.argsort(scores)[-12:]]
    best = finalists[0].copy()
    best_value, best_tail = _normalized_signature(best, ia, ib, ic)

    for candidate in finalists:
        value, tail = _normalized_signature(candidate, ia, ib, ic)
        if value > best_value or (value == best_value and tail > best_tail):
            best = candidate.copy()
            best_value, best_tail = value, tail

    # Local direct-search polish on the actual normalized objective.
    # Several restarts permit movement between distinct active-triangle sets.
    for start in finalists[-6:]:
        current = start.copy()
        current_value, current_tail = _normalized_signature(current, ia, ib, ic)

        for step in range(2200):
            progress = step / 2199.0
            sigma = 0.055 * (1.0 - progress) ** 2 + 0.0012

            proposal = current.copy()
            point_id = int(rng.integers(n))
            proposal[point_id] += rng.normal(0.0, sigma, size=2)

            # Reflect at square boundaries.
            proposal = np.abs(proposal)
            proposal = np.where(proposal > 1.0, 2.0 - proposal, proposal)

            value, tail = _normalized_signature(proposal, ia, ib, ic)

            # Primarily maximize the requested metric; lower-tail tie-breaking
            # avoids unstable configurations with many near-active constraints.
            improved = (
                value > current_value + 1e-12
                or (
                    abs(value - current_value) <= 1e-12
                    and tail > current_tail + 1e-12
                )
            )

            if improved:
                current = proposal
                current_value, current_tail = value, tail

            if (
                current_value > best_value + 1e-12
                or (
                    abs(current_value - best_value) <= 1e-12
                    and current_tail > best_tail
                )
            ):
                best = current.copy()
                best_value, best_tail = current_value, current_tail

    # Numerical safety and output contract preservation.
    return np.clip(np.asarray(best, dtype=float), 0.0, 1.0)


# EVOLVE-BLOCK-END
