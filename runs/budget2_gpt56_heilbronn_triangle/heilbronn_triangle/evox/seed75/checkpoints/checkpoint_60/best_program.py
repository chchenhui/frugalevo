# EVOLVE-BLOCK-START
import itertools
import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """
    Fix the three container vertices, globally search eight folded-barycentric
    points with deterministic DE, then maximize a fixed-orientation area bound
    using both folded-coordinate and smooth Cartesian SLSQP refinements.
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

    # The first three points are the corners.  Fold the unit square onto the
    # barycentric simplex: (b,c) is uniformly distributed over b,c >= 0,
    # b+c <= 1 when DE samples its rectangular parameter box uniformly.
    # Within either fold this map is affine, which also gives SLSQP better
    # local conditioning than the former multiplicative stick-breaking map.
    def decode(x):
        q = np.asarray(x, dtype=float).reshape(8, 2)
        b = q[:, 0].copy()
        c = q[:, 1].copy()
        reflected = b + c > 1.0
        b[reflected] = 1.0 - b[reflected]
        c[reflected] = 1.0 - c[reflected]
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
    # Edge-supported points are feasible and can be essential for the best
    # maximin arrangement; duplicate points are already rejected by area zero.
    bounds = [(0.0, 1.0)] * 16

    # Independent long runs explore distinct orientation cells.  The best
    # members of a population need not all converge to the same local maximin
    # arrangement, so retain two from every run for constrained polishing.
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

        # Two numerically best DE members often share one sign cell and hence
        # lead SLSQP to the same local solution.  Preserve strong members from
        # distinct orientation cells instead.
        seen_signatures = set()
        for index in np.argsort(energies):
            member = population[index]
            signature = np.packbits(
                normalized_signed_areas(decode(member)) >= 0.0
            ).tobytes()
            if signature not in seen_signatures:
                seen_signatures.add(signature)
                finalists.append(member.copy())
            # Additional sign cells are cheap to retain relative to the global
            # search and can converge to genuinely different active-constraint
            # patterns under the subsequent maximin refinement.
            if len(seen_signatures) == 5:
                break

    # Optimize an explicit area lower bound in each candidate's fixed signed
    # orientation cell.  The first pass retains the convenient folded global
    # coordinates.  A second pass uses Cartesian coordinates, where all three
    # containment inequalities are linear and no fold discontinuity remains.
    root3 = np.sqrt(3.0)
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, h]])

    for start in finalists:
        signs = np.sign(normalized_signed_areas(decode(start)))
        signs[signs == 0.0] = 1.0
        t0 = quality(start) * 0.995
        z0 = np.concatenate((start, [t0]))

        def folded_constraints(z, fixed_signs=signs):
            return fixed_signs * normalized_signed_areas(decode(z[:-1])) - z[-1]

        polished = minimize(
            lambda z: -z[-1],
            z0,
            method="SLSQP",
            bounds=bounds + [(0.0, 0.20)],
            constraints={"type": "ineq", "fun": folded_constraints},
            options={"maxiter": 2500, "ftol": 3.0e-13, "disp": False},
        )
        folded_x = polished.x[:-1] if polished.x is not None else start
        initial = decode(folded_x)
        cartesian0 = np.concatenate(
            (initial[3:].ravel(), [quality(folded_x) * 0.999])
        )

        def cartesian_constraints(z, fixed_signs=signs):
            points = np.vstack((corners, z[:-1].reshape(8, 2)))
            signed = fixed_signs * normalized_signed_areas(points) - z[-1]
            x = points[3:, 0]
            y = points[3:, 1]
            inside = np.concatenate((
                y, root3 * x - y, root3 * (1.0 - x) - y,
            ))
            return np.concatenate((signed, inside))

        cartesian = minimize(
            lambda z: -z[-1],
            cartesian0,
            method="SLSQP",
            bounds=[(0.0, 1.0), (0.0, h)] * 8 + [(0.0, 0.20)],
            constraints={"type": "ineq", "fun": cartesian_constraints},
            options={"maxiter": 2500, "ftol": 2.0e-13, "disp": False},
        )

        candidate = folded_x
        value = quality(candidate)
        if cartesian.x is not None:
            cartesian_points = np.vstack((corners, cartesian.x[:-1].reshape(8, 2)))
            cartesian_value = float(
                np.min(np.abs(normalized_signed_areas(cartesian_points)))
            )
            if np.isfinite(cartesian_value) and cartesian_value > value:
                candidate = None
                value = cartesian_value
                if value > best_value:
                    best_value = value
                    best_x = cartesian_points

        if candidate is not None and value > best_value:
            best_value = value
            best_x = candidate.copy()

    if best_x.shape == (16,):
        return decode(best_x)
    return best_x


# EVOLVE-BLOCK-END
