# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """Use deterministic multi-start simulated annealing in barycentric coordinates."""
    rng = np.random.default_rng(110271)
    n = 11
    height = np.sqrt(3.0) / 2.0

    triples = np.array(
        [(i, j, k) for i in range(n) for j in range(i + 1, n)
         for k in range(j + 1, n)],
        dtype=np.intp,
    )
    # A single-point move changes only triangles containing that point.
    incident = [
        np.flatnonzero(np.any(triples == point, axis=1))
        for point in range(n)
    ]

    def random_points(count: int) -> np.ndarray:
        """Generate uniformly reflected points in the unit barycentric simplex."""
        p = rng.random((count, 2))
        outside = p[:, 0] + p[:, 1] > 1.0
        p[outside] = 1.0 - p[outside]
        return p

    def determinants(bary: np.ndarray) -> np.ndarray:
        """Return twice triangle areas, scaled by the enclosing triangle area."""
        xy = np.empty((n, 2), dtype=float)
        xy[:, 0] = bary[:, 0] + 0.5 * bary[:, 1]
        xy[:, 1] = height * bary[:, 1]
        q = xy[triples]
        return np.abs(
            (q[:, 1, 0] - q[:, 0, 0]) * (q[:, 2, 1] - q[:, 0, 1])
            - (q[:, 1, 1] - q[:, 0, 1]) * (q[:, 2, 0] - q[:, 0, 0])
        )

    def changed_determinants(bary: np.ndarray, point: int) -> np.ndarray:
        """Compute areas only for triples affected by one moved point."""
        q = bary[triples[incident[point]]]
        return height * np.abs(
            (q[:, 1, 0] - q[:, 0, 0]) * (q[:, 2, 1] - q[:, 0, 1])
            - (q[:, 1, 1] - q[:, 0, 1]) * (q[:, 2, 0] - q[:, 0, 0])
        )

    def merit(values: np.ndarray) -> float:
        """Smooth the maximin objective using several near-critical triangles."""
        low = np.partition(values, 15)[:16]
        return float(low[0] + 0.016 * np.mean(low[1:]))

    best_bary = None
    best_minimum = -1.0

    # Fixing the three enclosing vertices supplies useful extreme points.
    # The remaining eight points are optimized by independent deterministic runs.
    for restart in range(12):
        bary = np.empty((n, 2), dtype=float)
        bary[0] = (0.0, 0.0)
        bary[1] = (1.0, 0.0)
        bary[2] = (0.0, 1.0)
        bary[3:] = random_points(8)

        areas = determinants(bary)
        value = merit(areas)

        for iteration in range(22000):
            progress = iteration / 22000.0

            # Prefer points participating in the current near-minimum triples.
            critical = np.argpartition(areas, 15)[:16]
            counts = np.bincount(triples[critical].ravel(), minlength=n)[3:]
            if counts.sum() and rng.random() < 0.91:
                point = 3 + rng.choice(8, p=counts / counts.sum())
            else:
                point = int(rng.integers(3, n))

            trial = bary.copy()
            scale = 0.13 * (1.0 - progress) ** 1.35 + 0.0018
            if rng.random() < 0.025 and progress < 0.65:
                trial[point] = random_points(1)[0]
            else:
                trial[point] += rng.normal(0.0, scale, size=2)
                trial[point] = np.maximum(trial[point], 0.0)
                total = trial[point].sum()
                if total > 1.0:
                    trial[point] /= total

            # Reuse unaffected triangle areas instead of evaluating all 165.
            ids = incident[point]
            trial_areas = areas.copy()
            trial_areas[ids] = changed_determinants(trial, point)
            trial_value = merit(trial_areas)
            temperature = 0.0016 * (1.0 - progress) ** 2.4 + 0.0000015

            if (trial_value >= value or
                    rng.random() < np.exp((trial_value - value) / temperature)):
                bary, areas, value = trial, trial_areas, trial_value

            minimum = float(areas.min())
            if minimum > best_minimum:
                best_minimum = minimum
                best_bary = bary.copy()

    # Defensive fallback is never normally reached, but guarantees valid output.
    if best_bary is None:
        best_bary = np.array(
            [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)] +
            [(0.5, 0.25)] * 8,
            dtype=float,
        )

    points = np.empty((n, 2), dtype=float)
    points[:, 0] = best_bary[:, 0] + 0.5 * best_bary[:, 1]
    points[:, 1] = height * best_bary[:, 1]
    return points


# EVOLVE-BLOCK-END
