# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_convex13() -> np.ndarray:
    """Use deterministic multi-start maximin annealing in a unit square, followed by isotropic coordinate polishing."""
    rng = np.random.default_rng(137)
    n = 13

    # Fixing the four square corners gives a convex hull of exactly unit area.
    # Therefore raw triangle areas are already the evaluator's normalized score.
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    triples = np.array(
        [(i, j, k) for i in range(n) for j in range(i + 1, n)
         for k in range(j + 1, n)],
        dtype=np.intp,
    )

    def triangle_areas(p: np.ndarray) -> np.ndarray:
        q = p[triples]
        return 0.5 * np.abs(
            (q[:, 1, 0] - q[:, 0, 0]) * (q[:, 2, 1] - q[:, 0, 1])
            - (q[:, 1, 1] - q[:, 0, 1]) * (q[:, 2, 0] - q[:, 0, 0])
        )

    def score(p: np.ndarray, tail_weight: float) -> tuple[float, float]:
        a = triangle_areas(p)
        low = np.partition(a, 19)[:20]
        return float(low[0]), float(low[0] + tail_weight * low.mean())

    best = None
    best_min = -1.0

    # Combine random starts with perturbed uniform 3-by-3 layouts.  The latter
    # reach well-distributed basins much more reliably than purely random starts,
    # while the random starts retain diversity against lattice-like local optima.
    lattice = np.array([(x, y) for y in (0.2, 0.5, 0.8) for x in (0.2, 0.5, 0.8)])
    for restart in range(24):
        if restart % 3 == 0:
            interior = np.clip(lattice + rng.normal(0.0, 0.115, (9, 2)), 0.0, 1.0)
        else:
            interior = rng.random((9, 2))
        p = np.vstack((corners, interior))
        current_min, current_score = score(p, 0.18)

        for it in range(48000):
            f = it / 47999.0
            step = 0.11 * (1.0 - f) ** 1.8 + 0.0007
            temperature = 0.0022 * (1.0 - f) ** 2.2 + 0.0000008
            tail_weight = 0.18 * (1.0 - f) + 0.025

            idx = int(rng.integers(4, n))
            old = p[idx].copy()
            p[idx] = np.clip(old + rng.normal(0.0, step, 2), 0.0, 1.0)

            trial_min, trial_score = score(p, tail_weight)
            delta = trial_score - current_score
            if delta >= 0.0 or rng.random() < np.exp(delta / temperature):
                current_min, current_score = trial_min, trial_score
            else:
                p[idx] = old

            if current_min > best_min:
                best_min = current_min
                best = p.copy()

    # Deterministic pattern refinement is particularly effective once the
    # identity of the handful of nearly limiting triangles has stabilized.
    p = best.copy()
    current_min, _ = score(p, 0.0)
    for radius in (0.004, 0.0015, 0.0005, 0.00015):
        improved = True
        while improved:
            improved = False
            for idx in range(4, n):
                old = p[idx].copy()
                # Sixteen directions avoid the strong axis/diagonal bias of a
                # square stencil and better resolve active triangle gradients.
                for angle in np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False):
                    candidate = old + radius * np.array((np.cos(angle), np.sin(angle)))
                    if np.any(candidate < 0.0) or np.any(candidate > 1.0):
                        continue
                    p[idx] = candidate
                    trial_min, _ = score(p, 0.0)
                    if trial_min > current_min + 1e-14:
                        current_min = trial_min
                        old = candidate.copy()
                        improved = True
                    else:
                        p[idx] = old
                p[idx] = old

    return p


# EVOLVE-BLOCK-END
