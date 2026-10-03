# EVOLVE-BLOCK-START
import numpy as np


_HB13_CACHE = None


def heilbronn_convex13() -> np.ndarray:
    """
    Deterministically optimize a 3-fold symmetric 13-point configuration:
    one center point and four rotational orbits of three points, with the
    outer orbit defining an equilateral convex container of unit area.
    """
    global _HB13_CACHE
    if _HB13_CACHE is not None:
        return _HB13_CACHE.copy()

    rng = np.random.default_rng(20260912)
    twopi3 = 2.0 * np.pi / 3.0

    # There are four 3-point rotational orbits.  The final orbit is fixed
    # at the vertices of the containing equilateral triangle; the remaining
    # three orbits have a radius fraction and angular phase as free variables.
    #
    # A ray with angle theta meets the boundary of the unit-circumradius
    # equilateral triangle at radial distance:
    #     1 / (2 max_i cos(theta - normal_i)).
    # Consequently every generated point is guaranteed to stay inside it.
    normals = np.array([np.pi / 3.0, np.pi, 5.0 * np.pi / 3.0])
    triples = np.array(
        [(i, j, k) for i in range(13) for j in range(i + 1, 13)
         for k in range(j + 1, 13)],
        dtype=np.intp,
    )

    def make_points(x):
        """Map population parameters to feasible symmetric point sets."""
        x = np.atleast_2d(x)
        m = x.shape[0]
        pts = np.empty((m, 13, 2), dtype=float)
        pts[:, 0, :] = 0.0

        for ring in range(3):
            frac = x[:, ring]
            phase = x[:, 3 + ring]
            boundary_radius = 0.5 / np.max(
                np.cos(phase[:, None] - normals[None, :]), axis=1
            )
            radius = frac * boundary_radius
            for q in range(3):
                angle = phase + q * twopi3
                pts[:, 1 + 3 * ring + q, 0] = radius * np.cos(angle)
                pts[:, 1 + 3 * ring + q, 1] = radius * np.sin(angle)

        # Outer orbit: equilateral hull vertices, circumradius one.
        for q in range(3):
            angle = q * twopi3
            pts[:, 10 + q, 0] = np.cos(angle)
            pts[:, 10 + q, 1] = np.sin(angle)
        return pts

    def quality(x):
        """Return each candidate's minimum triangle area."""
        pts = make_points(x)
        a = pts[:, triples[:, 0]]
        b = pts[:, triples[:, 1]]
        c = pts[:, triples[:, 2]]
        areas = 0.5 * np.abs(
            (b[:, :, 0] - a[:, :, 0]) * (c[:, :, 1] - a[:, :, 1])
            - (b[:, :, 1] - a[:, :, 1]) * (c[:, :, 0] - a[:, :, 0])
        )
        return np.min(areas, axis=1)

    # Vectorized differential evolution is considerably more reliable than
    # a single random layout for this highly non-smooth max-min objective.
    pop_size = 144
    dim = 6
    lower = np.array([0.015, 0.015, 0.015, 0.0, 0.0, 0.0])
    upper = np.array([0.995, 0.995, 0.995, twopi3, twopi3, twopi3])

    pop = rng.uniform(lower, upper, size=(pop_size, dim))
    # Include several useful radial strata, while retaining random phases.
    pop[:12, :3] = np.array([
        [0.20, 0.48, 0.78],
        [0.25, 0.55, 0.82],
        [0.30, 0.60, 0.88],
    ] * 4)
    values = quality(pop)

    for generation in range(1800):
        order = np.argsort(values)[::-1]
        elite = pop[order[:18]]

        # DE/current-to-best/1: preserves useful ring ordering while still
        # allowing phase changes and non-symmetric radial spacing.
        r1 = rng.integers(0, pop_size, pop_size)
        r2 = rng.integers(0, pop_size, pop_size)
        best_parent = elite[rng.integers(0, len(elite), pop_size)]
        factor = 0.55 + 0.25 * rng.random((pop_size, 1))
        mutant = pop + 0.70 * (best_parent - pop) + factor * (pop[r1] - pop[r2])
        mutant = np.clip(mutant, lower, upper)

        cross = rng.random((pop_size, dim)) < 0.82
        cross[np.arange(pop_size), rng.integers(0, dim, pop_size)] = True
        trial = np.where(cross, mutant, pop)
        trial_values = quality(trial)

        improved = trial_values >= values
        pop[improved] = trial[improved]
        values[improved] = trial_values[improved]

        # Periodic deterministic diversification prevents premature collapse.
        if generation in (500, 1000, 1450):
            worst = np.argsort(values)[:pop_size // 5]
            pop[worst] = rng.uniform(lower, upper, size=(len(worst), dim))
            values[worst] = quality(pop[worst])

    # Short batched local search around the best DE result.  Evaluating a
    # batch at once keeps this inexpensive despite all 286 triangle checks.
    best = pop[np.argmax(values)].copy()
    best_value = quality(best)[0]
    step = np.array([0.025, 0.025, 0.025, 0.045, 0.045, 0.045])
    for _ in range(700):
        candidates = best + rng.normal(size=(32, dim)) * step
        candidates = np.clip(candidates, lower, upper)
        candidate_values = quality(candidates)
        j = int(np.argmax(candidate_values))
        if candidate_values[j] > best_value:
            best = candidates[j]
            best_value = candidate_values[j]
            step *= 1.015
        else:
            step *= 0.994

    points = make_points(best)[0]

    # The outer equilateral triangle above has area 3*sqrt(3)/4.  Scale it
    # so its area is exactly one, satisfying the requested unit-area region.
    points *= np.sqrt(4.0 / (3.0 * np.sqrt(3.0)))
    _HB13_CACHE = points
    return points.copy()


# EVOLVE-BLOCK-END
