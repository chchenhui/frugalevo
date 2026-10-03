# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_convex13() -> np.ndarray:
    """
    Return a reproducibly optimized 13-point configuration in the unit square.

    The four corners keep the convex hull equal to the unit square, so triangle
    areas are already normalized by hull area.  The nine interior points are
    refined from several perturbed lattice starts using a deterministic
    rank-based maximin search.
    """
    rng = np.random.default_rng(1301957)

    # Keeping these four extreme points fixes a convex hull of area exactly one.
    corners = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=float,
    )
    triples = np.array(
        [(i, j, k) for i in range(13) for j in range(i + 1, 13)
         for k in range(j + 1, 13)],
        dtype=np.intp,
    )

    # Emphasize the bottleneck, while also improving its nearest competitors.
    rank_weights = 0.42 ** np.arange(18, dtype=float)
    rank_weights /= rank_weights.sum()

    def areas(configs: np.ndarray) -> np.ndarray:
        """Areas for every triple, for one or a batch of configurations."""
        a = configs[:, triples[:, 0]]
        b = configs[:, triples[:, 1]]
        c = configs[:, triples[:, 2]]
        return 0.5 * np.abs(
            (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1])
            - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])
        )

    lattice = np.array(
        [[x, y] for y in (0.25, 0.50, 0.75) for x in (0.25, 0.50, 0.75)],
        dtype=float,
    )
    best = None
    best_minimum = -np.inf

    # Independent lattice perturbations help avoid the many collinear triples
    # present in an unperturbed grid.
    for restart in range(7):
        interior = lattice + rng.uniform(-0.095, 0.095, size=(9, 2))
        current = np.vstack((corners, np.clip(interior, 0.055, 0.945)))

        for iteration in range(1500):
            # Large moves escape lattice artifacts; the final moves polish the
            # active small-area triangles.
            fraction = iteration / 1499.0
            step = 0.070 * (1.0 - fraction) ** 1.7 + 0.0025
            batch = np.repeat(current[None, :, :], 17, axis=0)

            # Candidate zero is retained, hence the surrogate never worsens.
            indices = rng.integers(0, 9, size=16)
            batch[1:, 4 + indices] += rng.normal(0.0, step, size=(16, 2))

            # Occasional coordinated moves are useful before the fine phase.
            if iteration % 11 == 0:
                batch[1:, 4:] += rng.normal(0.0, step * 0.28, size=(16, 9, 2))

            batch[:, 4:] = np.clip(batch[:, 4:], 0.045, 0.955)
            triangle_areas = areas(batch)
            ordered = np.partition(triangle_areas, 17, axis=1)[:, :18]
            surrogate = ordered @ rank_weights
            chosen = int(np.argmax(surrogate))
            current = batch[chosen]

            candidate_minimum = float(triangle_areas[chosen].min())
            if candidate_minimum > best_minimum:
                best_minimum = candidate_minimum
                best = current.copy()

    # best is always assigned because every generated configuration is finite.
    return best


# EVOLVE-BLOCK-END