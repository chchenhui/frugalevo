# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """Optimize eight free points by seeded DE, batched maximin search, gradient proposals, and coordinate LP polishing."""
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

    # Retain the three vertices and optimize the remaining eight points.
    # This population size and seed have a demonstrated strong deterministic
    # trajectory for the nonsmooth maximin objective.
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
    # Start with large coordinated moves, since the active small-area triples
    # often share multiple points and cannot be released by a one-point move.
    # The later stages progressively equalize the limiting triangle areas.
    for step, batches, moved in (
        (0.060, 100, 5), (0.045, 140, 4), (0.035, 220, 3),
        (0.024, 220, 3), (0.016, 300, 2), (0.007, 380, 2),
        (0.003, 500, 1), (0.0011, 550, 1), (0.00035, 400, 1),
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

    # Near a maximin arrangement, random single-point noise becomes unlikely to
    # release the limiting constraints.  Use gradients of pairs of almost-tight
    # triangle areas to generate coherent ascent proposals.  Acceptance remains
    # strictly monotone with respect to the true global minimum area.
    for step, batches in ((0.0030, 180), (0.0010, 260), (0.00028, 320)):
        for _ in range(batches):
            a = best[triples[:, 0]]
            b = best[triples[:, 1]]
            c = best[triples[:, 2]]
            det = ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                   - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
            areas = 0.5 * np.abs(det)
            active = np.flatnonzero(areas <= best_score * 1.06 + 1e-14)
            if active.size == 0:
                break

            chosen = active[rng.integers(active.size, size=(batch_size, 2))]
            direction = np.zeros((batch_size, 11, 2))
            rows = np.arange(batch_size)
            for column in range(2):
                ids = chosen[:, column]
                ia, ib, ic = triples[ids].T
                sign = np.sign(det[ids])[:, None]
                direction[rows, ia] += sign * np.column_stack((
                    b[ids, 1] - c[ids, 1], c[ids, 0] - b[ids, 0]
                ))
                direction[rows, ib] += sign * np.column_stack((
                    c[ids, 1] - a[ids, 1], a[ids, 0] - c[ids, 0]
                ))
                direction[rows, ic] += sign * np.column_stack((
                    a[ids, 1] - b[ids, 1], b[ids, 0] - a[ids, 0]
                ))

            direction[:, :3] = 0.0
            norm = np.sqrt(np.sum(direction[:, 3:] ** 2, axis=(1, 2)))
            norm = np.maximum(norm, 1e-15)
            candidates = best[None, :, :] + step * direction / norm[:, None, None]
            candidates[:, 3:] = project(candidates[:, 3:])
            values = minimum_areas(candidates)
            winner = int(np.argmax(values))
            if values[winner] >= best_score:
                best = candidates[winner]
                best_score = float(values[winner])

    # For fixed locations of the other ten points, every signed determinant
    # involving one free point is affine in that point.  A small LP therefore
    # gives an exact coordinate-wise maximin update and is a useful deterministic
    # complement to the stochastic coupled moves above.
    try:
        from scipy.optimize import linprog

        def det(config, i, j, k):
            u, v, w = config[i], config[j], config[k]
            return ((v[0] - u[0]) * (w[1] - u[1])
                    - (v[1] - u[1]) * (w[0] - u[0]))

        for _ in range(6):
            changed = False
            for point in range(3, 11):
                involved = triples[np.any(triples == point, axis=1)]
                fixed = triples[~np.any(triples == point, axis=1)]
                fixed_limit = min(
                    0.5 * abs(det(best, *ijk)) for ijk in fixed
                )

                A, rhs = [], []
                for ijk in involved:
                    old = det(best, *ijk)
                    sign = 1.0 if old >= 0.0 else -1.0
                    probe = best.copy()
                    probe[point] = (0.0, 0.0)
                    constant = det(probe, *ijk)
                    probe[point] = (1.0, 0.0)
                    dx = det(probe, *ijk) - constant
                    probe[point] = (0.0, 1.0)
                    dy = det(probe, *ijk) - constant
                    # sign * determinant >= 2 * lower_area.
                    A.append((-sign * dx, -sign * dy, 2.0))
                    rhs.append(sign * constant)

                # x >= y/(2h), x <= 1-y/(2h), and the unchanged triples.
                A.extend((
                    (-1.0, 1.0 / (2.0 * h), 0.0),
                    (1.0, 1.0 / (2.0 * h), 0.0),
                    (0.0, 0.0, 1.0),
                ))
                rhs.extend((0.0, 1.0, fixed_limit))
                result = linprog(
                    c=(0.0, 0.0, -1.0),
                    A_ub=np.asarray(A),
                    b_ub=np.asarray(rhs),
                    bounds=((0.0, 1.0), (0.0, h), (0.0, None)),
                    method="highs",
                )
                if result.success and np.all(np.isfinite(result.x)):
                    candidate = best.copy()
                    candidate[point] = result.x[:2]
                    value = float(minimum_areas(candidate[None])[0])
                    if value > best_score + 1e-13:
                        best, best_score = candidate, value
                        changed = True
            if not changed:
                break
    except Exception:
        # SciPy is optional; the preceding seeded construction is feasible.
        pass

    return best


# EVOLVE-BLOCK-END
