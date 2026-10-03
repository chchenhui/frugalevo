# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_convex13() -> np.ndarray:
    """
    Deterministically search for a 13-point Heilbronn configuration.

    Three points are fixed as a reference triangle and the other ten remain
    inside it.  Affine invariance makes this fixed-hull normalization natural:
    determinant magnitudes are then already normalized triangle areas.
    """
    n = 13
    rng = np.random.default_rng(13051957)
    triples = np.asarray(
        [(i, j, k) for i in range(n - 2) for j in range(i + 1, n - 1)
         for k in range(j + 1, n)],
        dtype=np.intp,
    )

    def hull_area(p: np.ndarray) -> float:
        """Twice-area monotone-chain hull, with a harmless zero-area guard."""
        ordered = p[np.lexsort((p[:, 1], p[:, 0]))]
        lower = []
        for q in ordered:
            while len(lower) >= 2:
                a, b = lower[-2], lower[-1]
                if (b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0]) > 1e-12:
                    break
                lower.pop()
            lower.append(q)
        upper = []
        for q in ordered[::-1]:
            while len(upper) >= 2:
                a, b = upper[-2], upper[-1]
                if (b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0]) > 1e-12:
                    break
                upper.pop()
            upper.append(q)
        polygon = np.asarray(lower[:-1] + upper[:-1])
        if len(polygon) < 3:
            return 0.0
        return abs(np.dot(polygon[:, 0], np.roll(polygon[:, 1], -1)) -
                   np.dot(polygon[:, 1], np.roll(polygon[:, 0], -1))) * 0.5

    def evaluate(p: np.ndarray) -> tuple[float, float]:
        a, b, c = p[triples[:, 0]], p[triples[:, 1]], p[triples[:, 2]]
        # The reference hull has area 1/2, so area / hull_area is exactly
        # the absolute determinant rather than half the determinant.
        normalized = np.abs(
            (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) -
            (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        )
        # A lower-tail tie breaker lets the search improve several impending
        # bottlenecks instead of merely exchanging the active worst triple.
        tail = np.partition(normalized, 11)[:12]
        return float(normalized.min()), float(normalized.min() + 0.12 * tail.mean())

    best_points = None
    best_score = -np.inf

    # Independent deterministic starts substantially reduce sensitivity to
    # the many local optima of the nonsmooth minimum-area objective.
    for restart in range(6):
        current = np.empty((n, 2), dtype=float)
        current[:3] = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
        interior = rng.random((n - 3, 2))
        reflected = interior.sum(axis=1) > 1.0
        interior[reflected] = 1.0 - interior[reflected]
        current[3:] = interior
        current_score, current_energy = evaluate(current)

        for step in range(14000):
            progress = step / 13999.0
            sigma = 0.115 * (1.0 - progress) ** 1.7 + 0.0025
            candidate = current.copy()
            changed = 1 if rng.random() < 0.82 else 2
            indices = rng.choice(np.arange(3, n), size=changed, replace=False)
            candidate[indices] += rng.normal(0.0, sigma, size=(changed, 2))
            movable = np.maximum(candidate[indices], 0.0002)
            sums = movable.sum(axis=1)
            movable *= np.minimum(1.0, 0.9996 / sums)[:, None]
            candidate[indices] = movable

            candidate_score, candidate_energy = evaluate(candidate)
            temperature = 0.0035 * (1.0 - progress) ** 2 + 0.000015
            if (candidate_energy >= current_energy or
                    rng.random() < np.exp((candidate_energy - current_energy) / temperature)):
                current = candidate
                current_score, current_energy = candidate_score, candidate_energy

            if current_score > best_score:
                best_score = current_score
                best_points = current.copy()

    # A final greedy pass concentrates solely on the metric reported by the
    # evaluator after annealing has identified a promising basin.
    current = best_points.copy()
    current_score, _ = evaluate(current)
    for step in range(5000):
        candidate = current.copy()
        index = int(rng.integers(3, n))
        sigma = 0.012 * (1.0 - step / 5000.0) + 0.0004
        candidate[index] += rng.normal(0.0, sigma, size=2)
        candidate[index] = np.maximum(candidate[index], 0.0002)
        total = float(candidate[index].sum())
        if total > 0.9996:
            candidate[index] *= 0.9996 / total
        candidate_score, _ = evaluate(candidate)
        if candidate_score >= current_score:
            current, current_score = candidate, candidate_score

    return current


# EVOLVE-BLOCK-END