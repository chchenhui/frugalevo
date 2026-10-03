# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """Deterministically anneal active triangle constraints, then monotone-polish."""
    rng = np.random.default_rng(11031991)
    n = 11

    # In coordinates (u,v), the reference triangle is
    # u >= 0, v >= 0, u + v <= 1.  Its triangle-area ratio is simply
    # abs(det([p_j-p_i, p_k-p_i])).
    triples = np.array(
        [(i, j, k) for i in range(n) for j in range(i + 1, n)
         for k in range(j + 1, n)],
        dtype=np.intp,
    )
    # Moving point i changes only these triples.  Caching the remaining
    # determinants avoids evaluating all 165 triangles at every proposal.
    affected = [np.flatnonzero(np.any(triples == i, axis=1)) for i in range(n)]
    unaffected = [np.flatnonzero(~np.any(triples == i, axis=1)) for i in range(n)]

    def random_simplex_points(count: int) -> np.ndarray:
        """Draw uniformly distributed points in the reference simplex."""
        z = rng.random((count, 2))
        mask = z.sum(axis=1) > 1.0
        z[mask] = 1.0 - z[mask]
        return z

    def area_values(p: np.ndarray, subset: np.ndarray = None) -> np.ndarray:
        """Return normalized determinant areas, optionally for a triple subset."""
        t = triples if subset is None else triples[subset]
        a = p[t[:, 1]] - p[t[:, 0]]
        b = p[t[:, 2]] - p[t[:, 0]]
        return np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])

    # Do not fix all three corners: doing so forbids every other point from
    # using an edge, since an edge point and that edge's two corners would be
    # collinear.  One seed on each edge gives the search access to boundary
    # arrangements while all points remain movable.
    best = None
    best_value = -1.0

    # Independent deterministic starts are substantially more reliable than a
    # single long trajectory for this nonsmooth max-min problem.  Only the
    # globally best feasible configuration is retained.
    for restart in range(96):
        p = np.empty((n, 2), dtype=float)
        p[:3] = np.array([[0.5, 0.0], [0.0, 0.5], [0.5, 0.5]])
        p[3:] = random_simplex_points(8)
        areas = area_values(p)
        current = float(np.min(areas))

        # Cached proposals are much cheaper, so use the extra budget for
        # additional basin exploration rather than one very long trajectory.
        steps = 55000
        for step in range(steps):
            # Large early moves escape poor random configurations; small late
            # moves accurately separate nearly critical triples.
            t = step / (steps - 1)
            sigma = 0.115 * (1.0 - t) ** 1.65 + 0.00018
            temperature = 0.0090 * (1.0 - t) ** 2.15 + 0.000001

            # Most useful late-stage moves alter a vertex of a currently
            # minimum-area triangle; retain random moves for basin escape.
            limiting = triples[int(np.argmin(areas))]
            if rng.random() < 0.68:
                idx = int(limiting[rng.integers(0, 3)])
            else:
                idx = int(rng.integers(0, n))
            old = p[idx].copy()

            # Boundary points are important in strong constructions, but have
            # probability zero under purely two-dimensional random moves.
            # Exact edge proposals are mixed with ordinary local simplex moves.
            candidate = None
            if rng.random() < 0.30 * (1.0 - t) ** 1.5 + 0.015:
                s = float(np.clip(rng.normal(0.5, 0.28), 0.015, 0.985))
                edge = int(rng.integers(0, 3))
                if edge == 0:
                    candidate = np.array([s, 0.0])
                elif edge == 1:
                    candidate = np.array([0.0, s])
                else:
                    candidate = np.array([s, 1.0 - s])
            else:
                # Occasionally move directly away from the opposite edge of
                # the active triangle, explicitly increasing its determinant.
                candidate = None
                if idx in limiting and rng.random() < 0.24:
                    other = limiting[limiting != idx]
                    q, r = p[other[0]], p[other[1]]
                    determinant = ((q[0] - old[0]) * (r[1] - old[1])
                                   - (q[1] - old[1]) * (r[0] - old[0]))
                    direction = np.array([q[1] - r[1], r[0] - q[0]])
                    direction *= 1.0 if determinant >= 0.0 else -1.0
                    direction /= max(float(np.linalg.norm(direction)), 1e-15)
                    trial = old + direction * sigma * (0.25 + rng.random())
                    if trial[0] >= 0.0 and trial[1] >= 0.0 and trial.sum() <= 1.0:
                        candidate = trial
                if candidate is None:
                    for _ in range(10):
                        q = old + rng.normal(0.0, sigma, size=2)
                        if q[0] >= 0.0 and q[1] >= 0.0 and q[0] + q[1] <= 1.0:
                            candidate = q
                            break
                if candidate is None:
                    candidate = random_simplex_points(1)[0]

            p[idx] = candidate
            changed = affected[idx]
            proposed = area_values(p, changed)
            # All triples not incident to idx retain their cached values.
            value = min(
                float(np.min(areas[unaffected[idx]])),
                float(np.min(proposed)),
            )

            # Standard deterministic-seed Metropolis acceptance.  On
            # acceptance update only the 45 determinants that actually changed.
            if value >= current or rng.random() < np.exp((value - current) / temperature):
                areas[changed] = proposed
                current = value
            else:
                p[idx] = old

            if current > best_value:
                best_value = current
                best = p.copy()

    # A small deterministic pattern search improves the exact objective only.
    # It is deliberately run only once, on the best annealed configuration.
    # Use a denser angular stencil than the annealer: at this stage every
    # accepted move is a strict improvement in the exact objective.
    directions = np.column_stack((
        np.cos(np.arange(32) * np.pi / 16.0),
        np.sin(np.arange(32) * np.pi / 16.0),
    ))

    for scale in 0.006 * 0.5 ** np.arange(13):
        improved = True
        while improved:
            improved = False
            for idx in range(n):
                old = best[idx].copy()
                local_best = best_value
                chosen = old
                for direction in directions:
                    trial = old + scale * direction
                    if trial[0] < 0.0 or trial[1] < 0.0 or trial.sum() > 1.0:
                        continue
                    best[idx] = trial
                    value = float(np.min(area_values(best)))
                    if value > local_best + 1e-14:
                        local_best = value
                        chosen = trial.copy()
                best[idx] = chosen
                if local_best > best_value + 1e-14:
                    best_value = local_best
                    improved = True
                else:
                    best[idx] = old

    # Convert barycentric reference coordinates to the requested Cartesian
    # equilateral triangle: (x,y) = (u + v/2, sqrt(3)*v/2).
    result = np.empty_like(best)
    result[:, 0] = best[:, 0] + 0.5 * best[:, 1]
    result[:, 1] = (np.sqrt(3.0) * 0.5) * best[:, 1]
    return result


# EVOLVE-BLOCK-END
