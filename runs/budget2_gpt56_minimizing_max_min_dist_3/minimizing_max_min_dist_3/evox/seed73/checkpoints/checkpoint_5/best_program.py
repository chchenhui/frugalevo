# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """Build separated spherical starts, then optimize unrestricted 3-D
    coordinates by incremental annealed and greedy diameter-ratio search."""
    n = 14
    rng = np.random.default_rng(918273)

    def normalize_rows(x):
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    def distances(x):
        z = x[:, None, :] - x[None, :, :]
        return np.sum(z * z, axis=2)

    def value_after_move(d, i, newd):
        keep = np.arange(n) != i
        fixed = d[np.ix_(keep, keep)]
        offdiag = ~np.eye(n - 1, dtype=bool)
        return min(np.min(fixed[offdiag]), np.min(newd)) / max(
            np.max(fixed), np.max(newd)
        )

    best_points = None
    best_value = -np.inf

    for restart in range(14):
        points = normalize_rows(rng.normal(size=(n, 3)))

        # A sphere is useful for generating well-separated starts, but is
        # not imposed during the actual Euclidean diameter optimization.
        for it in range(700):
            delta = points[:, None, :] - points[None, :, :]
            q = np.sum(delta * delta, axis=2) + np.eye(n)
            force = np.sum(delta * q[:, :, None] ** -3.0, axis=1)
            force -= np.sum(force * points, axis=1, keepdims=True) * points
            points = normalize_rows(
                points + 0.011 * force / (1.0 + 0.0025 * it)
            )

        d = distances(points)
        current = value_after_move(d, 0, d[0, 1:])

        # Radial freedom permits nonspherical diameter packings.  Scoring
        # only recomputes the thirteen distances incident to one point.
        total = 52000
        for it in range(total):
            i = rng.integers(n)
            frac = it / total
            scale = 0.105 * (1.0 - frac) ** 1.55 + 0.0010
            candidate_point = points[i] + scale * rng.normal(size=3)
            keep = np.arange(n) != i
            newd = np.sum((points[keep] - candidate_point) ** 2, axis=1)
            candidate = value_after_move(d, i, newd)
            temperature = 0.0032 * (1.0 - frac) ** 2 + 0.000008

            if candidate >= current or rng.random() < np.exp(
                (candidate - current) / temperature
            ):
                points[i] = candidate_point
                d[i, keep] = newd
                d[keep, i] = newd
                current = candidate

            # Translation is irrelevant, but recentering prevents random
            # walk of the coordinate origin during unrestricted moves.
            if it % 4000 == 3999:
                points -= np.mean(points, axis=0)

        for it in range(18000):
            i = rng.integers(n)
            scale = 0.008 * (1.0 - it / 18000.0) + 0.00003
            candidate_point = points[i] + scale * rng.normal(size=3)
            keep = np.arange(n) != i
            newd = np.sum((points[keep] - candidate_point) ** 2, axis=1)
            candidate = value_after_move(d, i, newd)

            if candidate > current:
                points[i] = candidate_point
                d[i, keep] = newd
                d[keep, i] = newd
                current = candidate

        if current > best_value:
            best_value = current
            best_points = points.copy()

    best_points -= np.mean(best_points, axis=0)
    return np.asarray(best_points, dtype=float)


# EVOLVE-BLOCK-END
