# EVOLVE-BLOCK-START
import itertools
import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """
    Search eight barycentric points, including triangle edges, with four
    deterministic DE runs and polish distinct signed-area orientation cells
    using fixed-orientation SLSQP maximin optimization.
    """
    try:
        from scipy.optimize import differential_evolution, minimize
    except Exception:
        # A valid deterministic fallback if SciPy is unavailable.
        h = np.sqrt(3.0) / 2.0
        return np.array([
            [0.0, 0.0], [1.0, 0.0], [0.5, h],
            [0.164, 0.105], [0.431, 0.071], [0.742, 0.103],
            [0.096, 0.341], [0.355, 0.286], [0.657, 0.319],
            [0.246, 0.580], [0.524, 0.603],
        ], dtype=float)

    h = np.sqrt(3.0) / 2.0
    triples = np.asarray(list(itertools.combinations(range(11), 3)), dtype=int)
    i0, i1, i2 = triples.T

    # The first three points are the corners.  For each remaining point,
    # u,v in [0,1] are converted to nonnegative barycentric coordinates:
    # (a,b,c) = (1-u, u(1-v), uv).
    def decode(x):
        q = np.asarray(x, dtype=float).reshape(8, 2)
        u = q[:, 0]
        v = q[:, 1]
        b = u * (1.0 - v)
        c = u * v
        interior = np.column_stack((b + 0.5 * c, h * c))
        return np.vstack((
            np.array([[0.0, 0.0], [1.0, 0.0], [0.5, h]]),
            interior,
        ))

    def normalized_signed_areas(points):
        a = points[i1] - points[i0]
        b = points[i2] - points[i0]
        # |cross| / (2 * area of the unit equilateral container)
        return (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) / h

    def quality(x):
        return float(np.min(np.abs(normalized_signed_areas(decode(x)))))

    best_x = None
    best_value = -np.inf
    finalists = []
    # Do not exclude container edges: optimal maximin arrangements can place
    # additional points on an edge.  The area objective prevents duplicates.
    bounds = [(0.0, 1.0)] * 16

    # Long deterministic searches locate substantially better signed-area
    # orientation cells than short DE runs.  Preserve two strong members from
    # every independent population for local maximin polishing.
    for seed in (1701, 2718, 3141, 5772):
        result = differential_evolution(
            lambda x: -quality(x),
            bounds,
            seed=seed,
            strategy="best1bin",
            popsize=24,
            maxiter=1250,
            mutation=(0.35, 1.45),
            recombination=0.90,
            tol=1.0e-7,
            polish=False,
            updating="immediate",
            workers=1,
        )
        value = quality(result.x)
        if value > best_value:
            best_value = value
            best_x = result.x.copy()

        population = np.asarray(result.population)
        energies = np.asarray(result.population_energies)

        # Consecutive low-energy DE members usually belong to one orientation
        # cell.  Keep the strongest representatives from different cells,
        # since each can converge to a different local maximin arrangement.
        seen_signatures = set()
        for index in np.argsort(energies):
            member = population[index]
            signature = np.packbits(
                normalized_signed_areas(decode(member)) >= 0.0
            ).tobytes()
            if signature not in seen_signatures:
                seen_signatures.add(signature)
                finalists.append(member.copy())
            if len(seen_signatures) == 2:
                break

    # Each retained population member may lie in a distinct orientation cell.
    # Fixed-sign SLSQP then optimizes its smooth explicit area lower bound.
    for start in finalists:
        signs = np.sign(normalized_signed_areas(decode(start)))
        signs[signs == 0.0] = 1.0
        t0 = quality(start) * 0.995
        z0 = np.concatenate((start, [t0]))

        def constraints(z, fixed_signs=signs):
            return fixed_signs * normalized_signed_areas(decode(z[:-1])) - z[-1]

        polished = minimize(
            lambda z: -z[-1],
            z0,
            method="SLSQP",
            bounds=bounds + [(0.0, 0.20)],
            constraints={"type": "ineq", "fun": constraints},
            options={"maxiter": 2500, "ftol": 3.0e-13, "disp": False},
        )
        candidate = polished.x[:-1] if polished.x is not None else start
        value = quality(candidate)
        if value > best_value:
            best_value = value
            best_x = candidate.copy()

    # The DE variables deliberately pin the three container vertices, which
    # greatly reduces the difficult global search dimension.  Once a strong
    # orientation cell has been found, however, allow every point (including
    # those vertices) to move within the container.  This cannot replace the
    # incumbent unless its actual unsigned minimum area is larger.
    free_start = decode(best_x)
    free_signs = np.sign(normalized_signed_areas(free_start))
    free_signs[free_signs == 0.0] = 1.0
    free_z0 = np.concatenate((free_start.ravel(), [best_value * 0.995]))

    def free_constraints(z):
        """Enforce the selected orientation cell and triangular container."""
        points = z[:-1].reshape(11, 2)
        oriented = free_signs * normalized_signed_areas(points) - z[-1]
        # y >= 0, y <= sqrt(3)x, and y <= sqrt(3)(1-x).
        container = np.column_stack((
            points[:, 1],
            np.sqrt(3.0) * points[:, 0] - points[:, 1],
            np.sqrt(3.0) * (1.0 - points[:, 0]) - points[:, 1],
        )).ravel()
        return np.concatenate((oriented, container))

    try:
        free = minimize(
            lambda z: -z[-1],
            free_z0,
            method="SLSQP",
            bounds=[(0.0, 1.0), (0.0, h)] * 11 + [(0.0, 0.20)],
            constraints={"type": "ineq", "fun": free_constraints},
            options={"maxiter": 3500, "ftol": 2.0e-13, "disp": False},
        )
        if free.x is not None and np.all(np.isfinite(free.x)):
            candidate_points = free.x[:-1].reshape(11, 2)
            candidate_value = float(
                np.min(np.abs(normalized_signed_areas(candidate_points)))
            )
            # Retain a small feasibility tolerance only for SLSQP roundoff;
            # accepted coordinates themselves are left unmodified.
            if (candidate_value > best_value and
                    np.min(free_constraints(free.x)) >= -1.0e-9):
                free_start = candidate_points
    except Exception:
        # The already-polished pinned configuration remains a valid result.
        pass

    return free_start


# EVOLVE-BLOCK-END
