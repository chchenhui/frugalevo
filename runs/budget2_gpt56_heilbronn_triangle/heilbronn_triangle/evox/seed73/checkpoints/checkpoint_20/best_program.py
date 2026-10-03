# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """
    Optimize eight free points by differential evolution, batched maximin
    perturbations, archive moves, and deterministic coordinate-wise LP polish.
    """
    rng = np.random.default_rng(11031987)
    h = np.sqrt(3.0) / 2.0
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, h]])
    triples = np.array(
        [(i, j, k) for i in range(11) for j in range(i + 1, 11)
         for k in range(j + 1, 11)],
        dtype=np.intp,
    )

    def project(p):
        """Project Cartesian candidate points onto the closed equilateral triangle."""
        q = np.asarray(p, dtype=float).copy()
        q[..., 1] = np.clip(q[..., 1], 0.0, h)
        left = q[..., 1] / (2.0 * h)
        q[..., 0] = np.clip(q[..., 0], left, 1.0 - left)
        return q

    def minimum_areas(population):
        """Return the minimum triangle area for every configuration in a population."""
        a = population[:, triples[:, 0]]
        b = population[:, triples[:, 1]]
        c = population[:, triples[:, 2]]
        areas = 0.5 * np.abs(
            (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1])
            - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])
        )
        return areas.min(axis=1)

    # Uniform barycentric samples, with all population members sharing the
    # useful and usually optimal corner points.
    pop_size = 48
    population = np.empty((pop_size, 11, 2), dtype=float)
    population[:, :3] = vertices
    uv = rng.random((pop_size, 8, 2))
    mask = uv.sum(axis=2) > 1.0
    uv[mask] = 1.0 - uv[mask]
    population[:, 3:, 0] = uv[..., 0] + 0.5 * uv[..., 1]
    population[:, 3:, 1] = h * uv[..., 1]
    scores = minimum_areas(population)

    # Differential evolution supplies large coordinated moves, which are much
    # more effective than moving one point at a time from a random start.
    for generation in range(700):
        trial = population.copy()
        scale = 0.72 - 0.22 * generation / 699.0
        for i in range(pop_size):
            choices = rng.choice(pop_size - 1, size=3, replace=False)
            choices += choices >= i
            a, b, c = population[choices]
            donor = a + scale * (b - c)
            donor[:3] = vertices

            cross = rng.random((11, 2)) < 0.72
            cross[:3] = False
            # Ensure that each trial actually inherits some donor coordinates.
            point = rng.integers(3, 11)
            cross[point, rng.integers(0, 2)] = True
            trial[i] = np.where(cross, donor, population[i])
            trial[i, :3] = vertices
            trial[i, 3:] = project(trial[i, 3:])

        trial_scores = minimum_areas(trial)
        accepted = trial_scores >= scores
        population[accepted] = trial[accepted]
        scores[accepted] = trial_scores[accepted]

    best = population[np.argmax(scores)].copy()
    best_score = float(np.max(scores))

    # The nonsmooth minimum-area objective commonly needs two or three points
    # to move together before an active constraint can be released.  Evaluate
    # independent perturbations in vectorized batches: this is substantially
    # cheaper than Python-level single-candidate evaluations and greedily keeps
    # the strongest non-worsening proposal from every batch.
    batch_size = 64
    for step, batches, moved in (
        (0.045, 160, 1), (0.035, 220, 3), (0.016, 300, 2),
        (0.007, 380, 2), (0.003, 500, 1), (0.0011, 550, 1),
        (0.00035, 400, 1),
    ):
        for _ in range(batches):
            candidates = np.repeat(best[None, :, :], batch_size, axis=0)
            # Choose exactly ``moved`` distinct free points in each candidate.
            order = np.argpartition(
                rng.random((batch_size, 8)), moved - 1, axis=1
            )[:, :moved]
            rows = np.arange(batch_size)[:, None]
            cols = order + 3
            candidates[rows, cols] += rng.normal(
                scale=step, size=(batch_size, moved, 2)
            )
            candidates[:, 3:] = project(candidates[:, 3:])
            values = minimum_areas(candidates)
            winner = int(np.argmax(values))
            if values[winner] >= best_score:
                best = candidates[winner]
                best_score = float(values[winner])

    # The final DE population is a useful archive of geometrically distinct
    # arrangements.  Difference vectors between its members provide coherent
    # multi-point directions that are unavailable to independent local noise.
    # This phase remains monotone, so it can only preserve or improve the
    # incumbent found above.
    elite_count = 16
    elite = population[np.argsort(scores)[-elite_count:]]
    for scale, batches in ((0.22, 180), (0.11, 220), (0.050, 260), (0.020, 280)):
        for _ in range(batches):
            choices = rng.integers(elite_count, size=(batch_size, 2))
            candidates = np.repeat(best[None, :, :], batch_size, axis=0)
            direction = elite[choices[:, 0]] - elite[choices[:, 1]]
            # A point-wise mask preserves useful point correlations while
            # allowing only part of a differential direction to be used.
            take = rng.random((batch_size, 8)) < 0.58
            take[np.arange(batch_size), rng.integers(8, size=batch_size)] = True
            candidates[:, 3:] += scale * direction[:, 3:] * take[:, :, None]
            candidates[:, 3:] = project(candidates[:, 3:])
            values = minimum_areas(candidates)
            winner = int(np.argmax(values))
            if values[winner] >= best_score:
                best = candidates[winner]
                best_score = float(values[winner])

    # Finish the archive phase with fine coupled perturbations around its best
    # result.  These moves are particularly useful after a differential move
    # has changed the set of active minimum-area triples.
    for step, batches, moved in ((0.0020, 350, 2), (0.00065, 450, 1)):
        for _ in range(batches):
            candidates = np.repeat(best[None, :, :], batch_size, axis=0)
            order = np.argpartition(
                rng.random((batch_size, 8)), moved - 1, axis=1
            )[:, :moved]
            rows = np.arange(batch_size)[:, None]
            candidates[rows, order + 3] += rng.normal(
                scale=step, size=(batch_size, moved, 2)
            )
            candidates[:, 3:] = project(candidates[:, 3:])
            values = minimum_areas(candidates)
            winner = int(np.argmax(values))
            if values[winner] >= best_score:
                best = candidates[winner]
                best_score = float(values[winner])

    # With the other ten locations fixed, every oriented determinant involving
    # one selected point is affine in that point.  Thus maximizing its worst
    # triangle area is a three-variable linear program (x, y, lower-bound
    # area).  This deterministic final pass is particularly useful when random
    # perturbations have reached a nonsmooth coordinate-wise local optimum.
    # SciPy is optional: retaining the already validated incumbent is a safe
    # fallback on minimal execution environments.
    try:
        from scipy.optimize import linprog

        def determinant(config, i, j, k):
            a, b, c = config[i], config[j], config[k]
            return ((b[0] - a[0]) * (c[1] - a[1])
                    - (b[1] - a[1]) * (c[0] - a[0]))

        for _ in range(10):
            improved = False
            for point in range(3, 11):
                involved = [ijk for ijk in triples if point in ijk]
                fixed = [ijk for ijk in triples if point not in ijk]
                fixed_limit = min(
                    abs(determinant(best, *ijk)) * 0.5 for ijk in fixed
                )

                constraints = []
                rhs = []
                for ijk in involved:
                    signed_now = determinant(best, *ijk)
                    sign = 1.0 if signed_now >= 0.0 else -1.0

                    # Obtain exact affine determinant coefficients by evaluating
                    # the determinant at the three coordinate basis locations.
                    probe = best.copy()
                    probe[point] = (0.0, 0.0)
                    constant = determinant(probe, *ijk)
                    probe[point] = (1.0, 0.0)
                    coeff_x = determinant(probe, *ijk) - constant
                    probe[point] = (0.0, 1.0)
                    coeff_y = determinant(probe, *ijk) - constant

                    # sign * determinant >= 2 * area_lower_bound.
                    constraints.append(
                        [-sign * coeff_x, -sign * coeff_y, 2.0]
                    )
                    rhs.append(sign * constant)

                # Triangular-domain constraints and triangles unaffected by the
                # currently moved point.
                constraints.extend((
                    [-1.0, 1.0 / (2.0 * h), 0.0],
                    [1.0, 1.0 / (2.0 * h), 0.0],
                    [0.0, 0.0, 1.0],
                ))
                rhs.extend((0.0, 1.0, fixed_limit))

                result = linprog(
                    c=(0.0, 0.0, -1.0),
                    A_ub=np.asarray(constraints),
                    b_ub=np.asarray(rhs),
                    bounds=((0.0, 1.0), (0.0, h), (0.0, None)),
                    method="highs",
                )
                if result.success:
                    candidate = best.copy()
                    candidate[point] = result.x[:2]
                    value = float(minimum_areas(candidate[None])[0])
                    if value > best_score + 1e-13:
                        best, best_score = candidate, value
                        improved = True
            if not improved:
                break
    except Exception:
        # The DE/local-search incumbent remains feasible and deterministic if
        # the optional LP backend is not installed or reports a numerical issue.
        pass

    return best


# EVOLVE-BLOCK-END
