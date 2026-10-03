# EVOLVE-BLOCK-START
import itertools
import numpy as np


def heilbronn_convex13() -> np.ndarray:
    """
    Return the best deterministic configuration from a structured maximin search.

    The search compares a regular 12-gon plus its center against regular
    11-gon hulls containing two independently placed interior points.  The
    latter family avoids the especially small consecutive triples of a
    13-gon while retaining enough freedom to improve the bottleneck area.
    The selected configuration is finally scaled to have unit hull area.
    """
    triples = np.asarray(list(itertools.combinations(range(13), 3)), dtype=int)

    def normalized_minimum(points, hull_twice_area):
        u = points[triples[:, 1]] - points[triples[:, 0]]
        v = points[triples[:, 2]] - points[triples[:, 0]]
        return np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]).min() / hull_twice_area

    # This is already substantially better than the regular 13-gon: its
    # bottleneck is the consecutive triple on the 12-vertex boundary.
    a12 = 2.0 * np.pi * np.arange(12, dtype=float) / 12.0
    outer12 = np.column_stack((np.cos(a12), np.sin(a12)))
    best = np.vstack((outer12, [[0.0, 0.0]]))
    best_sides = 12
    best_q = None
    best_value = normalized_minimum(best, 12.0 * np.sin(2.0 * np.pi / 12.0))

    # Search a compact, reproducible two-interior-point family.  Holding the
    # first point's angle to one fundamental 11-gon sector loses no rotational
    # generality.  Batched evaluation keeps this inexpensive.
    a11 = 2.0 * np.pi * np.arange(11, dtype=float) / 11.0
    outer11 = np.column_stack((np.cos(a11), np.sin(a11)))
    radii = np.linspace(0.35, 0.78, 12)
    alpha = np.linspace(0.0, 2.0 * np.pi / 11.0, 9, endpoint=False)
    beta = np.linspace(0.0, 2.0 * np.pi, 22, endpoint=False)
    r1, r2, t1, t2 = np.meshgrid(radii, radii, alpha, beta, indexing="ij")
    params = np.column_stack((r1.ravel(), r2.ravel(), t1.ravel(), t2.ravel()))
    hull11_twice_area = 11.0 * np.sin(2.0 * np.pi / 11.0)

    for start in range(0, len(params), 512):
        q = params[start:start + 512]
        candidates = np.empty((len(q), 13, 2), dtype=float)
        candidates[:, :11] = outer11
        candidates[:, 11] = np.column_stack((q[:, 0] * np.cos(q[:, 2]),
                                               q[:, 0] * np.sin(q[:, 2])))
        candidates[:, 12] = np.column_stack((q[:, 1] * np.cos(q[:, 3]),
                                               q[:, 1] * np.sin(q[:, 3])))
        u = candidates[:, triples[:, 1]] - candidates[:, triples[:, 0]]
        v = candidates[:, triples[:, 2]] - candidates[:, triples[:, 0]]
        values = np.abs(u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]).min(axis=1)
        winner = int(np.argmax(values))
        value = float(values[winner] / hull11_twice_area)
        if value > best_value:
            best_value = value
            best = candidates[winner].copy()
            best_q = q[winner].copy()
            best_sides = 11

    # Refine the best grid candidate by deterministic coordinate pattern
    # search.  The coarse beta grid has spacing about 0.286 radians, so this
    # inexpensive continuous polish is particularly valuable.  The apothem is
    # a strict radial feasibility bound for every interior point of a regular
    # 11-gon.
    if best_q is not None:
        apothem = np.cos(np.pi / 11.0)
        steps = np.array([0.030, 0.030, 0.070, 0.180])
        q = best_q.copy()

        def make_candidate(x):
            return np.vstack((
                outer11,
                [x[0] * np.cos(x[2]), x[0] * np.sin(x[2])],
                [x[1] * np.cos(x[3]), x[1] * np.sin(x[3])],
            ))

        # Each sweep tests both directions for each coordinate, accepting only
        # genuine maximin improvements.  Thus the result remains deterministic
        # and can never be worse than the coarse search result.
        for _ in range(28):
            changed = False
            for coordinate in range(4):
                local_q = q
                local_value = best_value
                for direction in (-1.0, 1.0):
                    trial = q.copy()
                    trial[coordinate] += direction * steps[coordinate]
                    if coordinate < 2:
                        trial[coordinate] = np.clip(trial[coordinate], 0.05, apothem - 1e-10)
                    else:
                        trial[coordinate] %= 2.0 * np.pi
                    trial_value = normalized_minimum(make_candidate(trial), hull11_twice_area)
                    if trial_value > local_value:
                        local_q, local_value = trial, trial_value
                if local_value > best_value:
                    q, best_value, changed = local_q, local_value, True
            steps *= 0.82
            if not changed and steps.max() < 2e-5:
                break

        polished = make_candidate(q)
        polished_value = normalized_minimum(polished, hull11_twice_area)
        if polished_value >= best_value:
            best = polished
            best_sides = 11

    # A radius-one regular m-gon has area m*sin(2*pi/m)/2.
    return best * np.sqrt(2.0 / (best_sides * np.sin(2.0 * np.pi / best_sides)))

    rng = np.random.default_rng(20260912)

    # Fixing these three vertices fixes the convex hull and removes the
    # scale/translation degrees of freedom.  The hull area is sqrt(3)/4.
    h = np.sqrt(3.0) / 2.0
    fixed = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, h]])
    triples = np.asarray(list(itertools.combinations(range(13), 3)), dtype=int)

    def project_simplex(z):
        """Project barycentric (right, top) coordinates into the triangle."""
        z = np.clip(z, 1.0e-6, 1.0 - 1.0e-6)
        s = z[..., 0] + z[..., 1]
        mask = s > 1.0 - 1.0e-6
        z[mask] *= ((1.0 - 1.0e-6) / s[mask])[..., None]
        return z

    def make_points(z):
        """Convert ten pairs of barycentric coordinates into Cartesian points."""
        p = np.empty((z.shape[0], 13, 2), dtype=float)
        p[:, :3] = fixed
        p[:, 3:, 0] = z[:, :, 0] + 0.5 * z[:, :, 1]
        p[:, 3:, 1] = h * z[:, :, 1]
        return p

    def values(z):
        """Return normalized minimum triangle areas for a population."""
        p = make_points(z)
        u = p[:, triples[:, 1]] - p[:, triples[:, 0]]
        v = p[:, triples[:, 2]] - p[:, triples[:, 0]]
        areas = np.abs(u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0])
        # Twice-area divided by twice the equilateral hull area.
        return areas.min(axis=1) / h, areas

    # A population search is substantially more reliable than optimizing a
    # single random arrangement, while still being small enough for fast NumPy
    # evaluation of all 286 triangle constraints.
    population_size = 160
    z = rng.random((population_size, 10, 2))
    reflected = z.sum(axis=2) > 1.0
    z[reflected] = 1.0 - z[reflected]
    z = project_simplex(z)

    best_value = -np.inf
    best_z = None

    for generation in range(3600):
        minimum, all_areas = values(z)

        # Early generations also reward several near-minimal triangles.  This
        # prevents premature convergence to configurations with one good
        # constraint and many almost-degenerate ones.  The weight vanishes as
        # the search becomes a true maximin refinement.
        low = np.partition(all_areas / h, 7, axis=1)[:, :8].mean(axis=1)
        weight = 0.035 * (1.0 - generation / 3600.0)
        fitness = minimum + weight * low

        i = int(np.argmax(minimum))
        if minimum[i] > best_value:
            best_value = float(minimum[i])
            best_z = z[i].copy()

        indices = np.arange(population_size)
        r1 = rng.integers(population_size, size=population_size)
        r2 = rng.integers(population_size, size=population_size)
        r3 = rng.integers(population_size, size=population_size)
        while np.any(r1 == indices):
            bad = r1 == indices
            r1[bad] = rng.integers(population_size, size=bad.sum())
        while np.any((r2 == indices) | (r2 == r1)):
            bad = (r2 == indices) | (r2 == r1)
            r2[bad] = rng.integers(population_size, size=bad.sum())
        while np.any((r3 == indices) | (r3 == r1) | (r3 == r2)):
            bad = (r3 == indices) | (r3 == r1) | (r3 == r2)
            r3[bad] = rng.integers(population_size, size=bad.sum())

        scale = 0.72 - 0.48 * generation / 3600.0
        donor = z[r1] + scale * (z[r2] - z[r3])
        crossover = rng.random(z.shape) < 0.72
        crossover[:, rng.integers(10), rng.integers(2)] = True
        trial = project_simplex(np.where(crossover, donor, z).copy())

        tmin, tarea = values(trial)
        tlow = np.partition(tarea / h, 7, axis=1)[:, :8].mean(axis=1)
        tfitness = tmin + weight * tlow
        z[tfitness > fitness] = trial[tfitness > fitness]

    # Deterministic fine polish of the best population member.
    current = best_z.copy()
    current_value = float(values(current[None, ...])[0][0])
    for step in range(18000):
        candidate = current.copy()
        point = int(rng.integers(10))
        sigma = 0.025 * (1.0 - step / 18000.0) + 0.00015
        candidate[point] += rng.normal(scale=sigma, size=2)
        candidate = project_simplex(candidate[None, ...])[0]
        candidate_value = float(values(candidate[None, ...])[0][0])
        if candidate_value > current_value:
            current, current_value = candidate, candidate_value

    optimized = make_points(current[None, ...])[0]

    # The regular polygon is a quick, robust non-degenerate baseline.  Keeping
    # it as a fallback guarantees a large improvement even if a future platform
    # has unusually unfavorable floating-point optimization behavior.
    angles = 2.0 * np.pi * np.arange(13) / 13.0
    regular = np.column_stack((np.cos(angles), np.sin(angles)))
    ru = regular[triples[:, 1]] - regular[triples[:, 0]]
    rv = regular[triples[:, 2]] - regular[triples[:, 0]]
    regular_min = np.abs(ru[:, 0] * rv[:, 1] - ru[:, 1] * rv[:, 0]).min()
    regular_hull_twice_area = 13.0 * np.sin(2.0 * np.pi / 13.0)
    regular_value = regular_min / regular_hull_twice_area

    return optimized if current_value >= regular_value else regular


# EVOLVE-BLOCK-END
