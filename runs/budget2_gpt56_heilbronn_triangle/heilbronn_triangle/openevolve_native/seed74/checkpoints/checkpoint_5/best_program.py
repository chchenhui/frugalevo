# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """Return 11 points found by deterministic multi-start max-min annealing.

    The search uses reference-simplex coordinates, updates only triangles
    affected by each moved point, and preferentially moves vertices of the
    current minimum-area triangle while retaining occasional global moves.
    """
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

    def random_simplex_points(count: int) -> np.ndarray:
        """Draw uniformly distributed points in the reference simplex."""
        z = rng.random((count, 2))
        mask = z.sum(axis=1) > 1.0
        z[mask] = 1.0 - z[mask]
        return z

    # A move of point i changes exactly C(10, 2) = 45 of the 165 areas.
    # Retaining the other values makes additional independent starts cheap.
    incident = np.array(
        [np.flatnonzero(np.any(triples == i, axis=1)) for i in range(n)],
        dtype=np.intp,
    )
    unaffected = np.ones((n, len(triples)), dtype=bool)
    for i in range(n):
        unaffected[i, incident[i]] = False

    def area_values(p: np.ndarray, subset=None) -> np.ndarray:
        """Return normalized determinant areas for all or selected triples."""
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

    # Several independent starts are much more reliable than one long run for
    # this highly nonsmooth max-min objective.
    for restart in range(24):
        p = np.empty((n, 2), dtype=float)
        p[:3] = np.array([[0.5, 0.0], [0.0, 0.5], [0.5, 0.5]])
        p[3:] = random_simplex_points(8)
        current_areas = area_values(p)
        current = float(np.min(current_areas))

        # Incremental area evaluation allows more independent basins to be
        # explored without the former full-triple evaluation cost.
        steps = 55000
        for step in range(steps):
            # Large early moves escape poor random configurations; small late
            # moves accurately separate nearly critical triples.
            t = step / (steps - 1)
            sigma = 0.115 * (1.0 - t) ** 1.65 + 0.00018
            temperature = 0.0090 * (1.0 - t) ** 2.15 + 0.000001

            # Improving the active constraint requires moving one of its
            # vertices.  Random choices remain necessary to rearrange
            # near-optimal plateaus and escape local structures.
            if rng.random() < 0.68:
                limiting = triples[int(np.argmin(current_areas))]
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
                # Rejecting outside proposals avoids systematically biasing
                # ordinary moves toward the boundary.
                for _ in range(10):
                    q = old + rng.normal(0.0, sigma, size=2)
                    if q[0] >= 0.0 and q[1] >= 0.0 and q[0] + q[1] <= 1.0:
                        candidate = q
                        break
                if candidate is None:
                    candidate = random_simplex_points(1)[0]

            p[idx] = candidate
            changed = area_values(p, incident[idx])
            # Areas of triangles not containing idx are unchanged exactly.
            value = float(min(np.min(current_areas[unaffected[idx]]),
                              np.min(changed)))

            # Standard deterministic-seed Metropolis acceptance.  On an
            # accepted move, update precisely the affected cached entries.
            if value >= current or rng.random() < np.exp((value - current) / temperature):
                current_areas[incident[idx]] = changed
                current = value
            else:
                p[idx] = old

            if current > best_value:
                best_value = current
                best = p.copy()

    # Convert barycentric reference coordinates to the requested Cartesian
    # equilateral triangle: (x,y) = (u + v/2, sqrt(3)*v/2).
    result = np.empty_like(best)
    result[:, 0] = best[:, 0] + 0.5 * best[:, 1]
    result[:, 1] = (np.sqrt(3.0) * 0.5) * best[:, 1]
    return result


# EVOLVE-BLOCK-END
