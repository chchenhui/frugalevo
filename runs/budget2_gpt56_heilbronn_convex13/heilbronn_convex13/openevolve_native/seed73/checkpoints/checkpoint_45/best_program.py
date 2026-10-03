# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_convex13() -> np.ndarray:
    """Use deterministic multistart maximin annealing in a unit square, followed by one- and two-point pattern polishing."""
    rng = np.random.default_rng(137)
    n = 13

    # The fixed square corners make the convex hull have area exactly one.
    # Hence every raw triangle area is already evaluator-normalized.
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    triples = np.array(
        [(i, j, k) for i in range(n) for j in range(i + 1, n)
         for k in range(j + 1, n)],
        dtype=np.intp,
    )

    def areas(p: np.ndarray) -> np.ndarray:
        """Return areas of all 286 triangles."""
        q = p[triples]
        return 0.5 * np.abs(
            (q[:, 1, 0] - q[:, 0, 0]) * (q[:, 2, 1] - q[:, 0, 1])
            - (q[:, 1, 1] - q[:, 0, 1]) * (q[:, 2, 0] - q[:, 0, 0])
        )

    def score(p: np.ndarray, weight: float) -> tuple[float, float]:
        """Return exact minimum plus a low-tail annealing objective."""
        a = areas(p)
        low = np.partition(a, 19)[:20]
        minimum = float(low[0])
        return minimum, float(minimum + weight * low.mean())

    best = None
    best_min = -1.0

    # More independent basins are valuable for this strongly nonsmooth
    # maximin objective; the fixed seed keeps the selected basin reproducible.
    # Retain the strongest configuration over more deterministic basins.
    # Since `best` is selected by the exact minimum area, this cannot make the
    # pre-polishing maximin result worse than the original 28-start search.
    for _ in range(56):
        p = np.vstack((corners, rng.random((9, 2))))
        current_min, current_score = score(p, 0.18)

        for it in range(36000):
            f = it / 35999.0
            step = 0.11 * (1.0 - f) ** 1.8 + 0.0007
            temperature = 0.0022 * (1.0 - f) ** 2.2 + 0.0000008
            weight = 0.18 * (1.0 - f) + 0.025

            idx = int(rng.integers(4, n))
            old = p[idx].copy()
            p[idx] = np.clip(old + rng.normal(0.0, step, 2), 0.0, 1.0)

            trial_min, trial_score = score(p, weight)
            delta = trial_score - current_score
            if delta >= 0.0 or rng.random() < np.exp(delta / temperature):
                current_min, current_score = trial_min, trial_score
            else:
                p[idx] = old

            if current_min > best_min:
                best_min = current_min
                best = p.copy()

    p = best.copy()
    current_min, _ = score(p, 0.0)
    directions = (
        (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
        (1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0),
    )

    for radius in (0.004, 0.0015, 0.0005, 0.00015):
        improved = True
        while improved:
            improved = False
            for idx in range(4, n):
                old = p[idx].copy()
                for dx, dy in directions:
                    candidate = old + radius * np.array((dx, dy))
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

    # Coordinate moves cannot improve constraints requiring two nearby points
    # to move together.  A short deterministic pair search resolves many such
    # active-constraint locks without changing the unit-square hull.
    pair_directions = (
        (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
        (1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0),
    )
    for radius in (0.0015, 0.0005, 0.00015):
        improved = True
        while improved:
            improved = False
            for i in range(4, n):
                for j in range(i + 1, n):
                    base_i = p[i].copy()
                    base_j = p[j].copy()
                    for dix, diy in pair_directions:
                        candidate_i = base_i + radius * np.array((dix, diy))
                        if np.any(candidate_i < 0.0) or np.any(candidate_i > 1.0):
                            continue
                        p[i] = candidate_i
                        for djx, djy in pair_directions:
                            candidate_j = base_j + radius * np.array((djx, djy))
                            if np.any(candidate_j < 0.0) or np.any(candidate_j > 1.0):
                                continue
                            p[j] = candidate_j
                            trial_min, _ = score(p, 0.0)
                            if trial_min > current_min + 1e-14:
                                current_min = trial_min
                                base_i = candidate_i.copy()
                                base_j = candidate_j.copy()
                                improved = True
                            else:
                                p[j] = base_j
                        p[i] = base_i
                    p[j] = base_j

    return p


# EVOLVE-BLOCK-END
