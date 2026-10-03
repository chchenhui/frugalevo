# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_convex13() -> np.ndarray:
    """Deterministically anneal nine free points in a unit square and maximin-polish them."""
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
        """Return the exact bottleneck and a small-low-tail annealing surrogate."""
        a = triangle_areas(p)
        low = np.partition(a, 19)[:20]
        return float(low[0]), float(low[0] + tail_weight * low.mean())

    best = None
    best_min = -1.0

    # Independent random restarts avoid lattice collinearities and reproduce
    # the previously reliable search basin at a much lower evaluation cost.
    for _ in range(14):
        p = np.vstack((corners, rng.random((9, 2))))
        current_min, current_score = score(p, 0.18)

        for it in range(36000):
            f = it / 35999.0
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
    # Preserve a globally best copy: tail-guided polishing can cross a flat
    # maximin plateau before finding a strictly better active-constraint set.
    polished_best = p.copy()
    polished_min = current_min
    for radius in (0.005, 0.002, 0.0007, 0.0002, 0.00006):
        improved = True
        while improved:
            improved = False
            for idx in range(4, n):
                old = p[idx].copy()
                for dx, dy in (
                    (radius, 0.0), (-radius, 0.0), (0.0, radius), (0.0, -radius),
                    (radius, radius), (radius, -radius),
                    (-radius, radius), (-radius, -radius),
                ):
                    candidate = old + (dx, dy)
                    if np.any(candidate < 0.0) or np.any(candidate > 1.0):
                        continue
                    p[idx] = candidate
                    trial_min, _ = score(p, 0.0)
                    if trial_min > current_min + 1e-14:
                        current_min = trial_min
                        old = candidate.copy()
                        improved = True
                        if current_min > polished_min:
                            polished_min = current_min
                            polished_best = p.copy()
                    else:
                        p[idx] = old
                p[idx] = old

    return polished_best


# EVOLVE-BLOCK-END
