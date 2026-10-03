# EVOLVE-BLOCK-START
import itertools
import numpy as np


def heilbronn_convex13() -> np.ndarray:
    """
    Return a unit-area regular-11-gon hull with two deterministically optimized
    interior points, using exhaustive polar sampling and maximin polishing.
    """
    # An 11-vertex hull leaves two degrees of interior freedom while avoiding
    # the small consecutive triples forced by a regular 13-gon.  Every free
    # point is constrained to the 11-gon's incircle, hence containment is
    # guaranteed without needing a general convex-hull projection.
    triples11 = np.asarray(list(itertools.combinations(range(13), 3)), dtype=int)
    angles11 = 2.0 * np.pi * np.arange(11, dtype=float) / 11.0
    outer11 = np.column_stack((np.cos(angles11), np.sin(angles11)))
    hull_twice = 11.0 * np.sin(2.0 * np.pi / 11.0)
    apothem = np.cos(np.pi / 11.0)

    def score11(points):
        u = points[triples11[:, 1]] - points[triples11[:, 0]]
        v = points[triples11[:, 2]] - points[triples11[:, 0]]
        return float(np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]).min() / hull_twice)

    def make11(q):
        return np.vstack((
            outer11,
            [q[0] * np.cos(q[2]), q[0] * np.sin(q[2])],
            [q[1] * np.cos(q[3]), q[1] * np.sin(q[3])],
        ))

    # Fixing the first interior angle to one hull sector removes rotational
    # redundancy.  Batched scoring evaluates all 286 triangle constraints
    # cheaply and reproducibly.
    radii = np.linspace(0.35, 0.78, 12)
    first_angles = np.linspace(0.0, 2.0 * np.pi / 11.0, 9, endpoint=False)
    second_angles = np.linspace(0.0, 2.0 * np.pi, 22, endpoint=False)
    r1, r2, t1, t2 = np.meshgrid(
        radii, radii, first_angles, second_angles, indexing="ij"
    )
    parameters = np.column_stack((r1.ravel(), r2.ravel(), t1.ravel(), t2.ravel()))

    best_value = -np.inf
    best_q = None
    for start in range(0, len(parameters), 512):
        q = parameters[start:start + 512]
        candidates = np.empty((len(q), 13, 2), dtype=float)
        candidates[:, :11] = outer11
        candidates[:, 11] = np.column_stack((
            q[:, 0] * np.cos(q[:, 2]), q[:, 0] * np.sin(q[:, 2])
        ))
        candidates[:, 12] = np.column_stack((
            q[:, 1] * np.cos(q[:, 3]), q[:, 1] * np.sin(q[:, 3])
        ))
        u = candidates[:, triples11[:, 1]] - candidates[:, triples11[:, 0]]
        v = candidates[:, triples11[:, 2]] - candidates[:, triples11[:, 0]]
        values = np.abs(u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]).min(axis=1)
        winner = int(np.argmax(values))
        value = float(values[winner] / hull_twice)
        if value > best_value:
            best_value = value
            best_q = q[winner].copy()

    # Coordinate pattern search supplies continuous refinement after the
    # coarse angular grid, accepting only strict genuine maximin improvements.
    q = best_q.copy()
    steps = np.array([0.030, 0.030, 0.070, 0.180])
    for _ in range(28):
        changed = False
        for coordinate in range(4):
            local_q = q
            local_value = best_value
            for direction in (-1.0, 1.0):
                trial = q.copy()
                trial[coordinate] += direction * steps[coordinate]
                if coordinate < 2:
                    trial[coordinate] = np.clip(
                        trial[coordinate], 0.05, apothem - 1.0e-10
                    )
                else:
                    trial[coordinate] %= 2.0 * np.pi
                value = score11(make11(trial))
                if value > local_value:
                    local_q, local_value = trial, value
            if local_value > best_value:
                q, best_value, changed = local_q, local_value, True
        steps *= 0.82
        if not changed and steps.max() < 2.0e-5:
            break

    # The regular hull is a good starting point but need not be maximin once
    # its two interior points have been chosen.  Allow small independent
    # movements of ten cyclic hull angles.  The first angle fixes rotation.
    # Offsets are bounded by 0.025 radians, so adjacent vertices remain in
    # cyclic order and every gap is below 0.622 radians.  Consequently the
    # hull contains the disk of radius cos(0.311) > 0.95; the free-point cap
    # below therefore certifies containment without a costly hull projection.
    base_points = make11(q)
    winning_points = base_points
    winning_twice = hull_twice
    winning_value = best_value

    def variable_candidate(x):
        """Build a cyclic 11-gon and two certified interior polar points."""
        offsets = np.empty(11, dtype=float)
        offsets[0] = 0.0
        offsets[1:] = x[4:]
        angles = angles11 + offsets
        outer = np.column_stack((np.cos(angles), np.sin(angles)))
        points = np.vstack((
            outer,
            [x[0] * np.cos(x[2]), x[0] * np.sin(x[2])],
            [x[1] * np.cos(x[3]), x[1] * np.sin(x[3])],
        ))
        twice = np.sin(np.roll(angles, -1) - angles).sum()
        return points, twice

    # If the preceding regular-polygon polish used a radius very near its
    # apothem, reduce only the starting point for the variable-hull search.
    # The unmodified regular candidate remains the guaranteed fallback.
    x = np.empty(14, dtype=float)
    x[:4] = q
    x[:2] = np.minimum(x[:2], 0.94)
    x[4:] = 0.0
    variable_steps = np.array(
        [0.018, 0.018, 0.045, 0.090] + [0.014] * 10, dtype=float
    )

    for _ in range(34):
        changed = False
        for coordinate in range(14):
            local_x = x
            local_value = winning_value
            for direction in (-1.0, 1.0):
                trial = x.copy()
                trial[coordinate] += direction * variable_steps[coordinate]
                if coordinate < 2:
                    trial[coordinate] = np.clip(trial[coordinate], 0.05, 0.94)
                elif coordinate < 4:
                    trial[coordinate] %= 2.0 * np.pi
                else:
                    trial[coordinate] = np.clip(trial[coordinate], -0.025, 0.025)

                points, twice = variable_candidate(trial)
                value = score11(points) * hull_twice / twice
                if value > local_value:
                    local_x, local_value = trial, value

            if local_value > winning_value:
                x, winning_value, changed = local_x, local_value, True
                winning_points, winning_twice = variable_candidate(x)

        variable_steps *= 0.82
        if not changed and variable_steps.max() < 2.0e-5:
            break

    # Scale by the actual cyclic hull area, including any accepted angular
    # refinement.  This makes the returned convex hull have unit area.
    return winning_points * np.sqrt(2.0 / winning_twice)

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
