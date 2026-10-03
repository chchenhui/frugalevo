# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_convex13() -> np.ndarray:
    """Search 56 deterministic unit-square annealing basins, retaining the best exact maximin configuration before monotone single- and pair-move polishing."""
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

    # The nonsmooth objective has many distinct local basins.  More fixed-seed
    # restarts improve the retained exact incumbent while remaining reproducible.
    for _ in range(56):
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
    # Retain the coarse schedule, then resolve the final active constraints at
    # substantially finer scales.  Every accepted move strictly raises min area.
    for radius in (0.004, 0.0015, 0.0005, 0.00015, 0.00005, 0.000015):
        improved = True
        while improved:
            improved = False
            for idx in range(4, n):
                old = p[idx].copy()
                for dx, dy in (
                    (radius, 0.0), (-radius, 0.0), (0.0, radius), (0.0, -radius),
                    (radius, radius), (radius, -radius),
                    (-radius, radius), (-radius, -radius),
                    (2.0 * radius, radius), (2.0 * radius, -radius),
                    (-2.0 * radius, radius), (-2.0 * radius, -radius),
                    (radius, 2.0 * radius), (radius, -2.0 * radius),
                    (-radius, 2.0 * radius), (-radius, -2.0 * radius),
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
                    else:
                        p[idx] = old
                p[idx] = old

    # Some active triangle constraints can only be released by moving two
    # nearby interior points together or in opposite directions.  Restricting
    # this to a few fine radii keeps the additional cost small and every
    # accepted proposal improves the exact nonsmoothed minimum.
    for radius in (0.0005, 0.00015, 0.00005, 0.000015):
        improved = True
        while improved:
            improved = False
            for i in range(4, n - 1):
                for j in range(i + 1, n):
                    old_i, old_j = p[i].copy(), p[j].copy()
                    for dx, dy in (
                        (radius, 0.0), (-radius, 0.0),
                        (0.0, radius), (0.0, -radius),
                        (radius, radius), (radius, -radius),
                        (-radius, radius), (-radius, -radius),
                    ):
                        move = np.array((dx, dy))
                        for si, sj in ((1.0, 1.0), (1.0, -1.0)):
                            ci = old_i + si * move
                            cj = old_j + sj * move
                            if (np.any(ci < 0.0) or np.any(ci > 1.0) or
                                    np.any(cj < 0.0) or np.any(cj > 1.0)):
                                continue
                            p[i], p[j] = ci, cj
                            trial_min, _ = score(p, 0.0)
                            if trial_min > current_min + 1e-14:
                                current_min = trial_min
                                old_i, old_j = ci.copy(), cj.copy()
                                improved = True
                            else:
                                p[i], p[j] = old_i, old_j
                    p[i], p[j] = old_i, old_j

    return p


# EVOLVE-BLOCK-END
