# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct a reproducible maximin arrangement for eleven points.

    The internal coordinates (u, v) represent u*(1,0) + v*(.5,sqrt(3)/2).
    Thus u >= 0, v >= 0, u + v <= 1, and a 2-by-2 determinant is
    precisely the corresponding area normalized by the outer triangle area.
    """
    rng = np.random.default_rng(11031991)
    triples = np.array(
        [(i, j, k) for i in range(11) for j in range(i + 1, 11)
         for k in range(j + 1, 11)],
        dtype=np.intp,
    )

    def areas(p: np.ndarray) -> np.ndarray:
        a = p[triples[:, 1]] - p[triples[:, 0]]
        b = p[triples[:, 2]] - p[triples[:, 0]]
        return np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])

    def merit(p: np.ndarray) -> tuple[float, float]:
        values = areas(p)
        # The tail term gives useful motion when several constraints tie for
        # the current minimum, while the first component remains dominant.
        tail = np.partition(values, 11)[:12]
        return float(values.min()), float(values.min() + 0.075 * tail.mean())

    # The outer vertices are useful extremal points; the other points are
    # optimized in barycentric coordinates.
    best = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]] +
        [[1.0 / 3.0, 1.0 / 3.0]] * 8,
        dtype=float,
    )
    best_min = -1.0
    best_merit = -1.0
    # Keep several independent terminal basins for a later focused refinement.
    # A marginally inferior annealing endpoint can have substantially better
    # local maximin potential than the nominal global incumbent.
    elite: list[tuple[float, float, np.ndarray]] = []

    for restart in range(9):
        bary = rng.dirichlet((1.15, 1.15, 1.15), size=8)
        current = np.empty((11, 2), dtype=float)
        current[:3] = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
        current[3:, 0] = bary[:, 1]
        current[3:, 1] = bary[:, 2]
        current_min, current_merit = merit(current)

        for iteration in range(18000):
            fraction = iteration / 17999.0
            # Large moves establish global structure; fine moves resolve the
            # many nearly active determinant constraints near the end.
            step = 0.105 * (1.0 - fraction) ** 1.65 + 0.0012
            thermal = 0.006 * (1.0 - fraction) ** 2 + 0.000015
            # Most proposals address a point in a lower-tail triangle.  The
            # residual uniform selection preserves the ability to reorganize
            # non-active points when the current combinatorial cell is poor.
            if rng.random() < 0.86:
                current_values = areas(current)
                cutoff = current_min + max(0.0025, 3.0 * thermal)
                restrictive = triples[current_values <= cutoff]
                chosen = restrictive[int(rng.integers(len(restrictive)))]
                movable = chosen[chosen >= 3]
                index = (int(rng.choice(movable)) if len(movable)
                         else int(rng.integers(3, 11)))
            else:
                index = int(rng.integers(3, 11))

            candidate = current.copy()
            weights = np.array(
                [1.0 - candidate[index, 0] - candidate[index, 1],
                 candidate[index, 0], candidate[index, 1]]
            )
            weights += rng.normal(0.0, step, size=3)
            # Projection through nonnegative barycentric weights guarantees
            # every proposed point remains inside the closed triangle.
            weights = np.maximum(weights, 1.0e-7)
            weights /= weights.sum()
            candidate[index] = weights[1:]

            candidate_min, candidate_merit = merit(candidate)
            delta = candidate_merit - current_merit
            if delta >= 0.0 or rng.random() < np.exp(delta / thermal):
                current = candidate
                current_min, current_merit = candidate_min, candidate_merit

            if (current_min > best_min + 1.0e-14 or
                    (abs(current_min - best_min) <= 1.0e-14 and
                     current_merit > best_merit)):
                best_min = current_min
                best_merit = current_merit
                best = current.copy()

        # Store terminal states, which represent distinct annealing basins.
        elite.append((current_min, current_merit, current.copy()))
        elite.sort(key=lambda item: (item[0], item[1]), reverse=True)
        del elite[6:]

    # Also include the strongest state encountered during every trajectory.
    elite.append((best_min, best_merit, best.copy()))
    elite.sort(key=lambda item: (item[0], item[1]), reverse=True)
    del elite[6:]

    # Briefly refine every elite basin before committing to the expensive
    # final polishing pass.  This reduces sensitivity to which restart first
    # happened to produce the best raw minimum.
    for _, _, seed in elite:
        seed_min, seed_merit = merit(seed)
        for radius in (0.014, 0.006):
            for _ in range(12):
                for index in rng.permutation(np.arange(3, 11)):
                    original = np.array(
                        [1.0 - seed[index, 0] - seed[index, 1],
                         seed[index, 0], seed[index, 1]]
                    )
                    for _ in range(12):
                        displacement = rng.normal(size=3)
                        displacement -= displacement.mean()
                        displacement *= radius / np.linalg.norm(displacement)
                        weights = np.maximum(original + displacement, 1.0e-9)
                        weights /= weights.sum()
                        trial = seed.copy()
                        trial[index] = weights[1:]
                        trial_min, trial_merit = merit(trial)
                        if (trial_min > seed_min + 1.0e-12 or
                                (abs(trial_min - seed_min) <= 1.0e-12 and
                                 trial_merit > seed_merit + 1.0e-13)):
                            seed = trial
                            seed_min, seed_merit = trial_min, trial_merit

        if (seed_min > best_min + 1.0e-14 or
                (abs(seed_min - best_min) <= 1.0e-14 and
                 seed_merit > best_merit)):
            best = seed.copy()
            best_min, best_merit = seed_min, seed_merit

    # Deterministic maximin polishing: annealing identifies a useful
    # oriented-matroid cell, then strictly improving coordinate moves resolve
    # residual slack among its active determinant constraints.
    polished_min, polished_merit = merit(best)
    for radius in (0.020, 0.012, 0.007, 0.004, 0.002):
        for _ in range(36):
            improved = False
            for index in rng.permutation(np.arange(3, 11)):
                original_weights = np.array(
                    [1.0 - best[index, 0] - best[index, 1],
                     best[index, 0], best[index, 1]]
                )
                local_best = best
                local_min = polished_min
                local_merit = polished_merit

                # Tangential barycentric displacements retain sum(weights)=1.
                # First test all coordinate-like simplex directions: these are
                # especially effective for the piecewise-linear active
                # determinant landscape.  Random oblique probes follow.
                stencil = (
                    np.array((1.0, -1.0, 0.0)),
                    np.array((-1.0, 1.0, 0.0)),
                    np.array((1.0, 0.0, -1.0)),
                    np.array((-1.0, 0.0, 1.0)),
                    np.array((0.0, 1.0, -1.0)),
                    np.array((0.0, -1.0, 1.0)),
                )
                directions = list(stencil)
                for _ in range(14):
                    displacement = rng.normal(size=3)
                    directions.append(displacement - displacement.mean())

                for displacement in directions:
                    displacement *= radius / np.linalg.norm(displacement)
                    weights = np.maximum(original_weights + displacement, 1.0e-9)
                    weights /= weights.sum()

                    trial = best.copy()
                    trial[index] = weights[1:]
                    trial_min, trial_merit = merit(trial)
                    if (trial_min > local_min + 1.0e-12 or
                            (abs(trial_min - local_min) <= 1.0e-12 and
                             trial_merit > local_merit + 1.0e-13)):
                        local_best = trial
                        local_min = trial_min
                        local_merit = trial_merit

                if local_best is not best:
                    best = local_best
                    polished_min = local_min
                    polished_merit = local_merit
                    improved = True
            if not improved:
                break

    # Affine conversion from normalized barycentric coordinates to the
    # requested Cartesian equilateral triangle.
    points = np.empty((11, 2), dtype=float)
    points[:, 0] = best[:, 0] + 0.5 * best[:, 1]
    points[:, 1] = (np.sqrt(3.0) / 2.0) * best[:, 1]
    return points


# EVOLVE-BLOCK-END