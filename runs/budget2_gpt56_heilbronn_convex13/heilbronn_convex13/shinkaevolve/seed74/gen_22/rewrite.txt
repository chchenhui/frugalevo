# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_convex13() -> np.ndarray:
    """
    Construct a deterministic 13-point maximin configuration in a unit-area
    equilateral triangle.

    A threefold-symmetric differential-evolution search first optimizes three
    radial point orbits plus a center point.  A second sparse-coordinate phase
    then breaks symmetry when beneficial and directly polishes the exact
    minimum of all 286 triangle areas.
    """
    rng = np.random.default_rng(1301957)

    # Work initially in an equilateral triangle of side one.  The final affine
    # scale makes its area exactly one without changing normalized quality.
    root3 = np.sqrt(3.0)
    vertices = np.array(
        [
            [0.0, root3 / 3.0],
            [-0.5, -root3 / 6.0],
            [0.5, -root3 / 6.0],
        ],
        dtype=float,
    )
    center = np.zeros(2, dtype=float)
    circumradius = root3 / 3.0
    inradius = root3 / 6.0
    orbit_angles = 2.0 * np.pi * np.arange(3, dtype=float) / 3.0

    triples = np.array(
        [(i, j, k) for i in range(13) for j in range(i + 1, 13)
         for k in range(j + 1, 13)],
        dtype=np.intp,
    )

    def radial_limit(theta: np.ndarray) -> np.ndarray:
        """
        Distance from the center to the triangular boundary in directions
        theta.  The three edge inequalities are evaluated simultaneously.
        """
        limits = []
        for alpha in (np.pi / 2.0, 7.0 * np.pi / 6.0, 11.0 * np.pi / 6.0):
            cosine = np.cos(theta - alpha)
            limits.append(
                np.where(cosine < -1.0e-12, circumradius / (-2.0 * cosine), 1e9)
            )
        return np.minimum(np.minimum(limits[0], limits[1]), limits[2])

    def symmetric_layouts(parameters: np.ndarray) -> np.ndarray:
        """
        Parameters contain three (radius-fraction, phase) pairs.  Each pair
        produces a 120-degree orbit; together with center and hull vertices
        this gives exactly thirteen points.
        """
        count = parameters.shape[0]
        layouts = np.empty((count, 13, 2), dtype=float)
        layouts[:, :3] = vertices
        layouts[:, 3] = center
        for ring in range(3):
            fraction = np.clip(parameters[:, 2 * ring], 0.025, 0.985)
            phase = np.mod(parameters[:, 2 * ring + 1], 2.0 * np.pi / 3.0)
            directions = phase[:, None] + orbit_angles[None, :]
            radius = fraction[:, None] * radial_limit(directions)
            layouts[:, 4 + 3 * ring:7 + 3 * ring, 0] = radius * np.cos(directions)
            layouts[:, 4 + 3 * ring:7 + 3 * ring, 1] = radius * np.sin(directions)
        return layouts

    def minimum_areas(layouts: np.ndarray) -> np.ndarray:
        a = layouts[:, triples[:, 0]]
        b = layouts[:, triples[:, 1]]
        c = layouts[:, triples[:, 2]]
        values = 0.5 * np.abs(
            (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1])
            - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])
        )
        return values.min(axis=1)

    # Differential evolution over three radii and three angular offsets.
    population_size = 96
    dimensions = 6
    population = np.empty((population_size, dimensions), dtype=float)
    population[:, 0::2] = rng.uniform(0.08, 0.94, size=(population_size, 3))
    population[:, 1::2] = rng.uniform(
        0.0, 2.0 * np.pi / 3.0, size=(population_size, 3)
    )

    # Several deliberately separated orbit patterns improve initial coverage.
    population[0] = [0.23, 0.08, 0.53, 0.42, 0.82, 1.04]
    population[1] = [0.31, 0.59, 0.61, 0.11, 0.90, 0.89]
    population[2] = [0.18, 0.27, 0.48, 0.94, 0.78, 0.53]
    scores = minimum_areas(symmetric_layouts(population))

    for generation in range(720):
        order = np.argsort(scores)[::-1]
        best = population[order[0]]
        scale = 0.82 - 0.25 * generation / 719.0
        crossover_rate = 0.82

        a = rng.integers(0, population_size, size=population_size)
        b = rng.integers(0, population_size, size=population_size)
        c = rng.integers(0, population_size, size=population_size)

        donor = population[a] + scale * (population[b] - population[c])
        donor += 0.16 * (1.0 - generation / 719.0) * (best - population[a])

        mask = rng.random((population_size, dimensions)) < crossover_rate
        mask[np.arange(population_size), rng.integers(0, dimensions, population_size)] = True
        trial = np.where(mask, donor, population)
        trial[:, 0::2] = np.clip(trial[:, 0::2], 0.018, 0.992)
        trial[:, 1::2] = np.mod(trial[:, 1::2], 2.0 * np.pi / 3.0)

        trial_scores = minimum_areas(symmetric_layouts(trial))
        accepted = trial_scores >= scores
        population[accepted] = trial[accepted]
        scores[accepted] = trial_scores[accepted]

        # Diversity-aware survivor injection: retain elites but periodically
        # refill a small low-score portion from distant random orbit patterns.
        if generation in (180, 360, 540):
            worst = np.argsort(scores)[:18]
            injected = np.empty((len(worst), dimensions), dtype=float)
            injected[:, 0::2] = rng.uniform(0.04, 0.97, size=(len(worst), 3))
            injected[:, 1::2] = rng.uniform(
                0.0, 2.0 * np.pi / 3.0, size=(len(worst), 3)
            )
            population[worst] = injected
            scores[worst] = minimum_areas(symmetric_layouts(injected))

    best_parameters = population[int(np.argmax(scores))][None, :]
    current = symmetric_layouts(best_parameters)[0]
    current_score = float(minimum_areas(current[None, :])[0])

    # Barycentric projection keeps arbitrary polishing moves inside the hull.
    # For this centered equilateral triangle:
    # p = w0*v0 + w1*v1 + w2*v2, w0+w1+w2=1.
    inverse_basis = np.linalg.inv(
        np.column_stack((vertices[0] - vertices[2], vertices[1] - vertices[2]))
    )

    def project_inside(points: np.ndarray) -> np.ndarray:
        flat = points.reshape(-1, 2)
        uv = (flat - vertices[2]) @ inverse_basis.T
        weights = np.column_stack((uv[:, 0], uv[:, 1], 1.0 - uv[:, 0] - uv[:, 1]))
        weights = np.maximum(weights, 1.0e-7)
        weights /= weights.sum(axis=1, keepdims=True)
        projected = weights @ vertices
        return projected.reshape(points.shape)

    # Sparse local moves preserve unrelated active triangles much more often
    # than whole-layout perturbations.  The hull vertices remain fixed.
    for iteration in range(620):
        fraction = iteration / 619.0
        step = 0.050 * (0.0012 / 0.050) ** fraction
        trial_count = 56
        trials = np.repeat(current[None, :, :], trial_count, axis=0)

        probability = 0.30 - 0.18 * fraction
        moving = rng.random((trial_count, 10, 1)) < probability
        moving[0, rng.integers(0, 10), 0] = True
        trials[:, 3:] += moving * rng.normal(0.0, step, size=(trial_count, 10, 2))
        trials[:, 3:] = project_inside(trials[:, 3:])

        trial_scores = minimum_areas(trials)
        winner = int(np.argmax(trial_scores))
        if trial_scores[winner] > current_score:
            current = trials[winner]
            current_score = float(trial_scores[winner])

    # Equilateral side one has area sqrt(3)/4.  Scaling coordinates produces
    # a convex hull of unit area, hence raw and normalized triangle areas match.
    scale_to_unit_area = np.sqrt(4.0 / root3)
    return current * scale_to_unit_area


# EVOLVE-BLOCK-END