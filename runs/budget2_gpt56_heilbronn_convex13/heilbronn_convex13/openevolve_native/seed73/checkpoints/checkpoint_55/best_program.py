# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_convex13() -> np.ndarray:
    """Deterministically anneal square-contained points, then apply one- and two-point exact maximin polishing."""
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
        """Return the exact bottleneck and a small low-tail annealing surrogate."""
        a = triangle_areas(p)
        low = np.partition(a, 19)[:20]
        return float(low[0]), float(low[0] + tail_weight * low.mean())

    best = None
    best_min = -1.0

    # Deterministic multistart search: retain the exact best feasible state
    # across 56 independent annealing basins before exact local polishing.
    # The first 28 trajectories are unchanged; the additional starts can only
    # improve the exact incumbent selected by the maximin objective.
    for _ in range(56):
        p = np.vstack((corners, rng.random((9, 2))))
        current_min, current_score = score(p, 0.18)
        # The initial state is also a feasible candidate for the exact
        # objective, independently of whether later surrogate moves accept it.
        if current_min > best_min:
            best_min = current_min
            best = p.copy()

        for it in range(36000):
            f = it / 35999.0
            step = 0.11 * (1.0 - f) ** 1.8 + 0.0007
            temperature = 0.0022 * (1.0 - f) ** 2.2 + 0.0000008
            tail_weight = 0.18 * (1.0 - f) + 0.025

            # Uniform point selection avoids over-updating a small subset of
            # vertices when several low-area triangles share an endpoint.
            idx = int(rng.integers(4, n))
            old = p[idx].copy()
            p[idx] = np.clip(old + rng.normal(0.0, step, 2), 0.0, 1.0)

            trial_min, trial_score = score(p, tail_weight)

            # The annealing surrogate intentionally rewards a broad low-area
            # tail, but the evaluator uses only the exact minimum.  Keep an
            # exact incumbent before potentially rejecting this proposal:
            # otherwise a genuinely better maximin configuration can be lost
            # merely because its tail surrogate is slightly smaller.
            if trial_min > best_min:
                best_min = trial_min
                best = p.copy()

            delta = trial_score - current_score
            if delta >= 0.0 or rng.random() < np.exp(delta / temperature):
                current_min, current_score = trial_min, trial_score
            else:
                p[idx] = old

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

    # A coordinate optimum can still be blocked by two points which need to
    # move together: improving one alone transiently worsens an active
    # triangle.  Perform small deterministic paired moves using the exact
    # bottleneck objective, retaining every strictly improving configuration.
    p = polished_best.copy()
    current_min, _ = score(p, 0.0)
    directions = (
        (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
        (1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0),
    )
    for radius in (0.0010, 0.00025):
        for i in range(4, n):
            for j in range(i + 1, n):
                old_i = p[i].copy()
                old_j = p[j].copy()
                for dx1, dy1 in directions:
                    for dx2, dy2 in directions:
                        candidate_i = old_i + radius * np.array((dx1, dy1))
                        candidate_j = old_j + radius * np.array((dx2, dy2))
                        if (np.any(candidate_i < 0.0) or np.any(candidate_i > 1.0)
                                or np.any(candidate_j < 0.0) or np.any(candidate_j > 1.0)):
                            continue
                        p[i] = candidate_i
                        p[j] = candidate_j
                        trial_min, _ = score(p, 0.0)
                        if trial_min > current_min + 1e-14:
                            current_min = trial_min
                            old_i = candidate_i.copy()
                            old_j = candidate_j.copy()
                            if current_min > polished_min:
                                polished_min = current_min
                                polished_best = p.copy()
                        else:
                            p[i] = old_i
                            p[j] = old_j
                p[i] = old_i
                p[j] = old_j

    return polished_best


# EVOLVE-BLOCK-END
