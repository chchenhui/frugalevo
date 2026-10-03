# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct eleven points inside or on the boundary of the reference
    equilateral triangle with vertices (0,0), (1,0), and
    (0.5, sqrt(3)/2).

    Internally, points use simplex coordinates (u, v):
        (x, y) = (u + v / 2, sqrt(3) * v / 2).
    In these coordinates, the absolute determinant of a point triple is
    its area normalized by the containing triangle's area.
    """
    rng = np.random.default_rng(11031987)
    n = 11
    fixed_vertices = np.array(
        ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
        dtype=float,
    )

    triples = np.array(
        [(i, j, k) for i in range(n) for j in range(i + 1, n)
         for k in range(j + 1, n)],
        dtype=np.intp,
    )

    def score(configurations: np.ndarray) -> np.ndarray:
        """Return the minimum normalized triangle determinant per layout."""
        a = configurations[:, triples[:, 0]]
        b = configurations[:, triples[:, 1]]
        c = configurations[:, triples[:, 2]]
        det = (
            (b[:, :, 0] - a[:, :, 0]) * (c[:, :, 1] - a[:, :, 1])
            - (b[:, :, 1] - a[:, :, 1]) * (c[:, :, 0] - a[:, :, 0])
        )
        return np.min(np.abs(det), axis=1)

    def soft_score(configurations: np.ndarray, temperature: float) -> np.ndarray:
        """
        Stable soft minimum of normalized determinants.

        This is used only while exploring: it distinguishes layouts having the
        same active minimum by rewarding improvement of nearby constraints.
        """
        a = configurations[:, triples[:, 0]]
        b = configurations[:, triples[:, 1]]
        c = configurations[:, triples[:, 2]]
        det = np.abs(
            (b[:, :, 0] - a[:, :, 0]) * (c[:, :, 1] - a[:, :, 1])
            - (b[:, :, 1] - a[:, :, 1]) * (c[:, :, 0] - a[:, :, 0])
        )
        lo = det.min(axis=1)
        return lo - temperature * np.log(
            np.exp(-(det - lo[:, None]) / temperature).sum(axis=1)
        )

    def fold_into_simplex(points_uv: np.ndarray) -> np.ndarray:
        """Reflect and normalize arbitrary proposed coordinates into u+v<=1."""
        q = np.abs(points_uv)
        total = q[..., 0] + q[..., 1]
        reflected = total > 1.0
        q[reflected] = 1.0 - q[reflected]
        q = np.maximum(q, 0.0)

        total = q[..., 0] + q[..., 1]
        over = total > 1.0
        if np.any(over):
            q[over] /= total[over, None]
        return q

    population_size = 240
    layouts = np.empty((population_size, n, 2), dtype=float)
    layouts[:, :3] = fixed_vertices

    raw = rng.exponential(1.0, size=(population_size, 8, 3))
    raw /= raw.sum(axis=2, keepdims=True)
    layouts[:, 3:, 0] = raw[:, :, 1]
    layouts[:, 3:, 1] = raw[:, :, 2]

    # Add a set of deliberately spread-out initial layouts.  They improve
    # early coverage while small deterministic noise avoids collinear grids.
    stencil = np.array(
        (
            (0.18, 0.10), (0.48, 0.08), (0.76, 0.10),
            (0.10, 0.43), (0.39, 0.34), (0.65, 0.27),
            (0.12, 0.72), (0.38, 0.52),
        ),
        dtype=float,
    )
    seed_count = 24
    layouts[:seed_count, 3:] = fold_into_simplex(
        stencil[None, :, :] + rng.normal(0.0, 0.045, size=(seed_count, 8, 2))
    )

    values = score(layouts)
    best_index = int(np.argmax(values))
    best_layout = layouts[best_index].copy()
    best_value = float(values[best_index])

    # Vectorized current-to-pbest differential evolution.  This generates
    # coordinated changes in several points, unlike isolated annealing moves.
    generations = 3600
    member_indices = np.arange(population_size)

    for generation in range(generations):
        # In the first stage, optimize a soft minimum so that triangles close
        # to the active constraint influence selection.  The final stage uses
        # the exact objective exclusively.
        exploratory = generation < 2200
        if exploratory:
            temperature = 0.0055 - 0.0035 * (generation / 2200.0)
            ranking = soft_score(layouts, temperature)
            elite_count = population_size // 3
        else:
            temperature = 0.0
            ranking = values
            elite_count = max(10, population_size // 12)

        order = np.argsort(ranking)
        pbest = order[rng.integers(0, elite_count, size=population_size)]

        r1 = rng.integers(0, population_size, size=population_size)
        r2 = rng.integers(0, population_size, size=population_size)
        same = (r1 == member_indices) | (r2 == member_indices) | (r1 == r2)
        while np.any(same):
            r1[same] = rng.integers(0, population_size, size=np.count_nonzero(same))
            r2[same] = rng.integers(0, population_size, size=np.count_nonzero(same))
            same = (r1 == member_indices) | (r2 == member_indices) | (r1 == r2)

        progress = generation / (generations - 1)
        if exploratory:
            scale_center = 0.82 - 0.16 * (generation / 2200.0)
        else:
            scale_center = 0.52 - 0.12 * (
                (generation - 2200) / (generations - 2200)
            )
        scale = scale_center + rng.uniform(-0.07, 0.07, population_size)
        scale = scale[:, None, None]

        donor = (
            layouts[:, 3:]
            + scale * (layouts[pbest, 3:] - layouts[:, 3:])
            + scale * (layouts[r1, 3:] - layouts[r2, 3:])
        )

        crossover = 0.94 - (0.12 if exploratory else 0.26) * progress
        take_donor = rng.random((population_size, 8, 2)) < crossover
        forced = rng.integers(0, 16, size=population_size)
        take_donor.reshape(population_size, 16)[member_indices, forced] = True

        proposal = layouts.copy()
        proposal[:, 3:] = np.where(take_donor, donor, layouts[:, 3:])
        proposal[:, 3:] = fold_into_simplex(proposal[:, 3:])

        proposal_values = score(proposal)
        if exploratory:
            proposal_ranking = soft_score(proposal, temperature)
        else:
            proposal_ranking = proposal_values

        accepted = proposal_ranking >= ranking
        layouts[accepted] = proposal[accepted]
        values[accepted] = proposal_values[accepted]

        current_best = int(np.argmax(values))
        if values[current_best] > best_value:
            best_value = float(values[current_best])
            best_layout = layouts[current_best].copy()

        # Reintroduce diverse candidates near the incumbent periodically.
        # The best layout itself is retained separately and can never be lost.
        if generation > 0 and generation % 300 == 0:
            worst = np.argsort(values)[:population_size // 5]
            refreshed = np.repeat(best_layout[None, :, :], len(worst), axis=0)
            noise_scale = (
                0.095 * (1.0 - generation / 2200.0) + 0.030
                if exploratory else 0.018
            )
            refreshed[:, 3:] += rng.normal(
                0.0, noise_scale, size=(len(worst), 8, 2)
            )
            refreshed[:, 3:] = fold_into_simplex(refreshed[:, 3:])
            refreshed_values = score(refreshed)
            layouts[worst] = refreshed
            values[worst] = refreshed_values

    # Deterministic maximin coordinate polish around the strongest layout.
    directions = np.array(
        (
            (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
            (0.7071067812, 0.7071067812),
            (-0.7071067812, -0.7071067812),
            (0.7071067812, -0.7071067812),
            (-0.7071067812, 0.7071067812),
        ),
        dtype=float,
    )

    for step in (0.015, 0.008, 0.004, 0.002, 0.001):
        improved = True
        while improved:
            improved = False
            for point_index in range(3, n):
                candidates = np.repeat(best_layout[None, :, :], len(directions), axis=0)
                candidates[:, point_index] += step * directions
                candidates[:, 3:] = fold_into_simplex(candidates[:, 3:])
                candidate_values = score(candidates)
                choice = int(np.argmax(candidate_values))
                if candidate_values[choice] > best_value:
                    best_layout = candidates[choice]
                    best_value = float(candidate_values[choice])
                    improved = True

    points = np.empty((n, 2), dtype=float)
    points[:, 0] = best_layout[:, 0] + 0.5 * best_layout[:, 1]
    points[:, 1] = (np.sqrt(3.0) / 2.0) * best_layout[:, 1]
    return points


# EVOLVE-BLOCK-END