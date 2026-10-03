# EVOLVE-BLOCK-START
import itertools
import numpy as np


def heilbronn_convex13() -> np.ndarray:
    """
    Deterministically optimize ten points inside a fixed equilateral hull using
    differential evolution followed by coordinate hill-climbing; retain a
    regular 13-gon fallback if the search does not beat it.
    """
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
