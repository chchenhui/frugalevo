# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Construct 14 three-dimensional points with a large minimum/maximum
    pairwise-distance ratio.
    """
    n, d = 14, 3
    rng = np.random.default_rng(814729)

    def normalize_and_score(configs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Center configurations, set their diameter to one, and score them."""
        configs = configs - configs.mean(axis=1, keepdims=True)
        differences = configs[:, :, None, :] - configs[:, None, :, :]
        squared = np.sum(differences * differences, axis=-1)

        # The diagonal must not participate in the nearest-neighbour minimum.
        diagonal = np.arange(n)
        squared[:, diagonal, diagonal] = np.inf
        diameter_squared = np.max(np.where(np.isfinite(squared), squared, 0.0),
                                  axis=(1, 2))
        configs = configs / np.sqrt(diameter_squared)[:, None, None]

        differences = configs[:, :, None, :] - configs[:, None, :, :]
        squared = np.sum(differences * differences, axis=-1)
        squared[:, diagonal, diagonal] = np.inf
        # Diameter is one after normalization, so this is precisely
        # (minimum distance / maximum distance)^2.
        return configs, np.min(squared, axis=(1, 2))

    best_score = -np.inf
    best_points = None
    batch_size = 28
    iterations = 1200

    for _ in range(10):
        # A uniform-in-volume ball start includes both boundary and interior
        # points, which is important for finite diameter packings.
        directions = rng.normal(size=(n, d))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        radii = rng.random(n) ** (1.0 / d)
        current, current_score = normalize_and_score(
            (directions * radii[:, None])[None, :, :]
        )
        current = current[0]
        current_score = current_score[0]

        for step in range(iterations):
            # Geometric cooling permits large rearrangements early and precise
            # contact adjustment near the final packing.
            fraction = step / (iterations - 1)
            scale = 0.18 * (0.012 / 0.18) ** fraction

            candidates = np.repeat(current[None, :, :], batch_size, axis=0)
            moved = rng.integers(0, n, size=batch_size)
            candidates[np.arange(batch_size), moved] += (
                rng.normal(size=(batch_size, d)) * scale
            )

            # Occasionally move a second point to escape one-point local traps.
            second_mask = rng.random(batch_size) < 0.18
            second = rng.integers(0, n, size=batch_size)
            candidates[np.arange(batch_size)[second_mask], second[second_mask]] += (
                rng.normal(size=(np.count_nonzero(second_mask), d)) * scale
            )

            candidates, scores = normalize_and_score(candidates)
            choice = int(np.argmax(scores))
            if scores[choice] > current_score:
                current = candidates[choice]
                current_score = scores[choice]

        if current_score > best_score:
            best_score = current_score
            best_points = current

    return np.asarray(best_points, dtype=float)


# EVOLVE-BLOCK-END