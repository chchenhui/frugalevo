# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct eleven points in the reference equilateral triangle.

    Internal coordinates are simplex coordinates (u, v), where
        x = u + v / 2
        y = sqrt(3) * v / 2

    In these coordinates, absolute 2D determinants equal normalized
    triangle areas relative to the containing equilateral triangle.
    """
    rng = np.random.default_rng(11031987)

    n = 11
    movable_start = 3
    movable_count = 8
    vertices = np.array(
        ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
        dtype=float,
    )

    triples = np.array(
        [(i, j, k)
         for i in range(n)
         for j in range(i + 1, n)
         for k in range(j + 1, n)],
        dtype=np.intp,
    )
    triple_a = triples[:, 0]
    triple_b = triples[:, 1]
    triple_c = triples[:, 2]

    def project_simplex(q: np.ndarray) -> np.ndarray:
        """Euclidean projection of uv coordinates onto u>=0, v>=0, u+v<=1."""
        p = np.maximum(q, 0.0).copy()
        total = p[..., 0] + p[..., 1]
        outside = total > 1.0
        if np.any(outside):
            # Projection onto u + v = 1, followed by nonnegative clipping.
            d = (total[outside] - 1.0) * 0.5
            p[outside, 0] -= d
            p[outside, 1] -= d
            p[outside] = np.maximum(p[outside], 0.0)
            total2 = p[outside, 0] + p[outside, 1]
            bad = total2 > 1.0
            if np.any(bad):
                p_out = p[outside]
                p_out[bad] /= total2[bad, None]
                p[outside] = p_out
        return p

    def determinants(layouts: np.ndarray) -> np.ndarray:
        a = layouts[:, triple_a]
        b = layouts[:, triple_b]
        c = layouts[:, triple_c]
        return np.abs(
            (b[:, :, 0] - a[:, :, 0]) * (c[:, :, 1] - a[:, :, 1])
            - (b[:, :, 1] - a[:, :, 1]) * (c[:, :, 0] - a[:, :, 0])
        )

    def score(layouts: np.ndarray) -> np.ndarray:
        return np.min(determinants(layouts), axis=1)

    population = 192
    layouts = np.empty((population, n, 2), dtype=float)
    layouts[:, :3] = vertices

    # Dirichlet samples cover the simplex without the corner bias caused by
    # independent uniform coordinates.
    bary = rng.gamma(1.25, 1.0, size=(population, movable_count, 3))
    bary /= bary.sum(axis=2, keepdims=True)
    layouts[:, 3:, 0] = bary[:, :, 1]
    layouts[:, 3:, 1] = bary[:, :, 2]

    # Several staggered near-lattice seeds give the maximin search useful
    # large-scale structure from its first generation.
    stencil = np.array(
        (
            (0.10, 0.13), (0.31, 0.09), (0.56, 0.09), (0.78, 0.13),
            (0.16, 0.38), (0.43, 0.31), (0.67, 0.29), (0.31, 0.60),
        ),
        dtype=float,
    )
    seed_count = 40
    for s in range(seed_count):
        shift = rng.normal(0.0, 0.035, size=(movable_count, 2))
        # Alternate a reflected stencil, giving both simplex orientations.
        base = stencil if (s & 1) == 0 else stencil[:, ::-1]
        layouts[s, 3:] = project_simplex(base + shift)

    values = score(layouts)
    best_slot = int(np.argmax(values))
    best_layout = layouts[best_slot].copy()
    best_value = float(values[best_slot])

    indices = np.arange(population)
    generations = 3100

    try:
        for generation in range(generations):
            fraction = generation / float(generations - 1)
            ranking = np.argsort(values)
            elite_count = max(14, population // 6)
            elites = ranking[-elite_count:]

            # Parent selection is rank-biased but deliberately not restricted
            # to the top few layouts, which prevents early geometric collapse.
            rank_choice = rng.random(population) ** 2
            parent_pos = np.minimum(
                (rank_choice * elite_count).astype(np.intp), elite_count - 1
            )
            pbest = elites[-1 - parent_pos]

            r1 = rng.integers(0, population, size=population)
            r2 = rng.integers(0, population, size=population)
            invalid = (r1 == indices) | (r2 == indices) | (r1 == r2)
            while np.any(invalid):
                count = int(np.count_nonzero(invalid))
                r1[invalid] = rng.integers(0, population, size=count)
                r2[invalid] = rng.integers(0, population, size=count)
                invalid = (r1 == indices) | (r2 == indices) | (r1 == r2)

            proposal = layouts.copy()
            mode = rng.random(population)

            # 1) Coordinated differential proposals.
            de_rows = mode < 0.52
            if np.any(de_rows):
                scale = (
                    0.72 - 0.34 * fraction
                    + rng.uniform(-0.10, 0.10, size=population)
                )
                donor = (
                    layouts[:, 3:]
                    + scale[:, None, None] * (layouts[pbest, 3:] - layouts[:, 3:])
                    + scale[:, None, None] * (layouts[r1, 3:] - layouts[r2, 3:])
                )
                mask = rng.random((population, movable_count, 2)) < (
                    0.88 - 0.20 * fraction
                )
                forced = rng.integers(0, movable_count * 2, size=population)
                mask.reshape(population, -1)[indices, forced] = True
                proposal[de_rows, 3:] = np.where(
                    mask[de_rows], donor[de_rows], layouts[de_rows, 3:]
                )

            # 2) Sparse local mutations retain useful arrangements of the
            # other seven points while moving only one to three coordinates.
            sparse_rows = (mode >= 0.52) & (mode < 0.76)
            if np.any(sparse_rows):
                count = int(np.count_nonzero(sparse_rows))
                bases = elites[rng.integers(0, elite_count, size=count)]
                proposal[sparse_rows] = layouts[bases]
                sigma = 0.085 * (1.0 - fraction) + 0.008
                changed = rng.random((count, movable_count)) < 0.24
                changed[np.arange(count), rng.integers(0, movable_count, count)] = True
                noise = rng.normal(0.0, sigma, size=(count, movable_count, 2))
                proposal[sparse_rows, 3:] += noise * changed[:, :, None]

            # 3) Constraint-directed offspring: inspect the small triples of
            # each parent and mutate points occurring in those active triples.
            active_rows = mode >= 0.76
            if np.any(active_rows):
                row_ids = np.flatnonzero(active_rows)
                local_det = determinants(layouts[row_ids])
                worst_count = 8 if fraction < 0.65 else 12
                active = np.argpartition(local_det, worst_count, axis=1)[:, :worst_count]

                for local_row, row in enumerate(row_ids):
                    incidence = np.bincount(
                        triples[active[local_row]].ravel(),
                        minlength=n,
                    ).astype(float)
                    incidence[:3] = 0.0
                    if incidence.sum() <= 0.0:
                        point = int(rng.integers(3, n))
                    else:
                        weights = incidence[3:] + 0.15
                        point = 3 + int(rng.choice(movable_count, p=weights / weights.sum()))

                    proposal[row] = layouts[row]
                    sigma = 0.045 * (1.0 - fraction) + 0.0035
                    proposal[row, point] += rng.normal(0.0, sigma, size=2)

                    # Occasionally move a second active point in a correlated
                    # direction to escape a rigid limiting-triple cluster.
                    if rng.random() < 0.32:
                        weights = incidence[3:] + 0.15
                        second = 3 + int(
                            rng.choice(movable_count, p=weights / weights.sum())
                        )
                        proposal[row, second] += rng.normal(
                            0.0, sigma * 0.65, size=2
                        )

            proposal[:, 3:] = project_simplex(proposal[:, 3:])
            proposed_values = score(proposal)

            accept = proposed_values >= values
            layouts[accept] = proposal[accept]
            values[accept] = proposed_values[accept]

            current = int(np.argmax(values))
            if values[current] > best_value:
                best_value = float(values[current])
                best_layout = layouts[current].copy()

            # Island-style renewal around the incumbent.  Half are local
            # descendants and half are fresh global simplex samples.
            if generation > 0 and generation % 350 == 0:
                worst = np.argsort(values)[:population // 5]
                split = len(worst) // 2

                local = np.repeat(best_layout[None], split, axis=0)
                local[:, 3:] += rng.normal(
                    0.0,
                    0.075 * (1.0 - fraction) + 0.018,
                    size=(split, movable_count, 2),
                )
                local[:, 3:] = project_simplex(local[:, 3:])

                fresh_count = len(worst) - split
                fresh = np.empty((fresh_count, n, 2), dtype=float)
                fresh[:, :3] = vertices
                raw = rng.gamma(1.1, 1.0, size=(fresh_count, movable_count, 3))
                raw /= raw.sum(axis=2, keepdims=True)
                fresh[:, 3:, 0] = raw[:, :, 1]
                fresh[:, 3:, 1] = raw[:, :, 2]

                renewed = np.concatenate((local, fresh), axis=0)
                layouts[worst] = renewed
                values[worst] = score(renewed)

    except (FloatingPointError, ValueError):
        # The best valid incumbent remains a safe deterministic result.
        pass

    # Final active-set pattern search.  It only tests displacements of points
    # participating in current limiting triples, making it substantially more
    # effective than indiscriminate coordinate sweeps.
    angles = np.linspace(0.0, 2.0 * np.pi, 20, endpoint=False)
    directions = np.column_stack((np.cos(angles), np.sin(angles)))

    for step in (0.018, 0.010, 0.005, 0.0025, 0.001):
        for _ in range(10):
            det = determinants(best_layout[None])[0]
            active_ids = np.argpartition(det, 12)[:12]
            incidence = np.bincount(triples[active_ids].ravel(), minlength=n)
            candidates_points = np.argsort(incidence[3:])[::-1][:5] + 3

            candidate_layouts = []
            for point in candidates_points:
                for direction in directions:
                    candidate = best_layout.copy()
                    candidate[point] += step * direction
                    candidate[3:] = project_simplex(candidate[3:])
                    candidate_layouts.append(candidate)

            batch = np.asarray(candidate_layouts)
            batch_values = score(batch)
            choice = int(np.argmax(batch_values))
            if batch_values[choice] > best_value + 1e-14:
                best_layout = batch[choice]
                best_value = float(batch_values[choice])
            else:
                break

    result = np.empty((n, 2), dtype=float)
    result[:, 0] = best_layout[:, 0] + 0.5 * best_layout[:, 1]
    result[:, 1] = (np.sqrt(3.0) * 0.5) * best_layout[:, 1]
    return result


# EVOLVE-BLOCK-END