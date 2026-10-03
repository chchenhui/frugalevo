# EVOLVE-BLOCK-START
import itertools

import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """
    Deterministically construct an 11-point configuration in the reference
    equilateral triangle.

    Internally points use coordinates (u, v), with u >= 0, v >= 0, u + v <= 1.
    Cartesian coordinates are recovered by

        x = u + v / 2
        y = sqrt(3) * v / 2

    and determinants in (u, v) coordinates equal normalized triangle areas.
    """
    rng = np.random.default_rng(11031987)

    n_points = 11
    fixed_count = 3
    triples = np.asarray(
        list(itertools.combinations(range(n_points), 3)), dtype=np.intp
    )
    triple_count = len(triples)

    corner_layout = np.array(
        ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
        dtype=float,
    )

    def project_simplex(points: np.ndarray) -> np.ndarray:
        """Project coordinate pairs safely into u>=0, v>=0, u+v<=1."""
        weights = np.empty(points.shape[:-1] + (3,), dtype=float)
        weights[..., 0] = 1.0 - points[..., 0] - points[..., 1]
        weights[..., 1:] = points
        np.maximum(weights, 1.0e-11, out=weights)
        weights /= weights.sum(axis=-1, keepdims=True)
        return weights[..., 1:]

    def determinants(layouts: np.ndarray) -> np.ndarray:
        """Return all normalized signed triangle determinants."""
        chosen = layouts[:, triples]
        first = chosen[:, :, 1] - chosen[:, :, 0]
        second = chosen[:, :, 2] - chosen[:, :, 0]
        return first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]

    def exact_score(layouts: np.ndarray) -> np.ndarray:
        """The reported maximin objective."""
        return np.abs(determinants(layouts)).min(axis=1)

    def evolutionary_score(layouts: np.ndarray) -> np.ndarray:
        """
        A conservative lexicographic-like surrogate.

        The true minimum remains dominant; a small lower-tail bonus helps
        distinguish layouts which have equal active constraints but different
        room for subsequent mutation and coordinate polishing.
        """
        areas = np.abs(determinants(layouts))
        lower_tail = np.partition(areas, 9, axis=1)[:, :10]
        return lower_tail[:, 0] + 0.0035 * lower_tail.mean(axis=1)

    def random_layouts(count: int) -> np.ndarray:
        """Generate feasible layouts with the enclosing corners retained."""
        layouts = np.empty((count, n_points, 2), dtype=float)
        layouts[:, :fixed_count] = corner_layout
        weights = rng.exponential(
            1.0, size=(count, n_points - fixed_count, 3)
        )
        weights /= weights.sum(axis=-1, keepdims=True)
        layouts[:, fixed_count:, 0] = weights[:, :, 1]
        layouts[:, fixed_count:, 1] = weights[:, :, 2]
        return layouts

    # ------------------------------------------------------------------
    # Global phase: an elite island-style population with sparse crossover.
    # ------------------------------------------------------------------
    population_size = 192
    elite_count = 24
    generations = 3600

    population = random_layouts(population_size)

    # A small structured component gives the evolutionary population layouts
    # with broader coverage than independent Dirichlet samples alone.
    for row in range(0, 36):
        free = population[row, fixed_count:]
        base = rng.uniform(0.08, 0.92, size=(n_points - fixed_count, 2))
        base[:, 1] *= 0.72
        population[row, fixed_count:] = project_simplex(base)
        population[row, fixed_count:] = project_simplex(
            population[row, fixed_count:]
            + rng.normal(0.0, 0.025, size=(n_points - fixed_count, 2))
        )
        population[row, :fixed_count] = corner_layout

    fitness = evolutionary_score(population)
    actual_values = exact_score(population)
    best_index = int(np.argmax(actual_values))
    best_layout = population[best_index].copy()
    best_value = float(actual_values[best_index])

    for generation in range(generations):
        order = np.argsort(fitness)[::-1]
        elites = population[order[:elite_count]]

        progress = generation / (generations - 1.0)
        offspring = elites[
            rng.integers(0, elite_count, size=population_size)
        ].copy()

        # Conservative pointwise elite recombination.  Complete coordinates are
        # inherited rather than averaged, which prevents crossover from making
        # already well-separated points collapse into common central locations.
        mate = elites[rng.integers(0, elite_count, size=population_size)]
        recombine_rows = rng.random(population_size) < 0.34
        take_mate = rng.random(
            (population_size, n_points - fixed_count, 1)
        ) < 0.50
        mixed = np.where(take_mate, mate[:, fixed_count:], offspring[:, fixed_count:])
        offspring[recombine_rows, fixed_count:] = mixed[recombine_rows]

        # Some offspring instead use convex elite recombination.  This is kept
        # relatively rare because coordinate inheritance is generally safer.
        blend_rows = (
            (~recombine_rows)
            & (rng.random(population_size) < 0.16)
        )
        blend_mate = elites[rng.integers(0, elite_count, size=population_size)]
        blend_weight = rng.uniform(
            0.20, 0.80, size=(population_size, 1, 1)
        )
        blended = (
            blend_weight * offspring[:, fixed_count:]
            + (1.0 - blend_weight) * blend_mate[:, fixed_count:]
        )
        offspring[blend_rows, fixed_count:] = blended[blend_rows]

        sigma = 0.105 * (1.0 - progress) ** 1.55 + 0.0010
        mutation_rate = 0.34 - 0.19 * progress
        change_mask = rng.random(
            (population_size, n_points - fixed_count, 1)
        ) < mutation_rate
        noise = rng.normal(
            0.0, sigma, size=(population_size, n_points - fixed_count, 2)
        )
        offspring[:, fixed_count:] += noise * change_mask
        offspring[:, fixed_count:] = project_simplex(offspring[:, fixed_count:])

        # Near a maximin layout only a few limiting triples determine the
        # objective.  Late offspring therefore include a sparse active-set
        # class: move the free point occurring most often in their twelve
        # smallest triangles.  This complements, rather than replaces, the
        # unbiased mutations above.
        if progress >= 0.68:
            targeted_rows = rng.random(population_size) < 0.28
            targeted_rows[:elite_count] = False
            if np.any(targeted_rows):
                trial_determinants = np.abs(determinants(offspring))
                limiting = np.argpartition(
                    trial_determinants, 11, axis=1
                )[:, :12]
                occurrences = np.zeros(
                    (population_size, n_points), dtype=np.intp
                )
                row_numbers = np.arange(population_size)[:, None, None]
                np.add.at(
                    occurrences,
                    (np.broadcast_to(row_numbers, limiting.shape + (3,)),
                     triples[limiting]),
                    1,
                )
                occurrences[:, :fixed_count] = -1
                targeted_point = np.argmax(occurrences, axis=1)
                selected = np.flatnonzero(targeted_rows)
                offspring[selected, targeted_point[selected]] += rng.normal(
                    0.0, 0.38 * sigma, size=(len(selected), 2)
                )
                offspring[:, fixed_count:] = project_simplex(
                    offspring[:, fixed_count:]
                )

        # Exact elite retention makes objective regressions impossible.
        offspring[:elite_count] = elites
        offspring[:, :fixed_count] = corner_layout

        population = offspring
        fitness = evolutionary_score(population)
        actual_values = exact_score(population)

        generation_best = int(np.argmax(actual_values))
        if actual_values[generation_best] > best_value:
            best_layout = population[generation_best].copy()
            best_value = float(actual_values[generation_best])

    # ------------------------------------------------------------------
    # Local phase: exact coordinate-wise maximin improvement.
    #
    # With all other points fixed and determinant signs retained, every area
    # involving one selected point is affine in that point.  The local problem
    # is therefore a small linear program in (u, v, t).  Rather than requiring
    # scipy, all intersections of three constraint planes are enumerated.
    # ------------------------------------------------------------------
    all_constraint_combinations = None

    def improve_one_coordinate(layout: np.ndarray, point: int) -> np.ndarray:
        nonlocal all_constraint_combinations

        current_dets = determinants(layout[None, ...])[0]
        current_abs = np.abs(current_dets)
        involved = np.any(triples == point, axis=1)
        active_indices = np.flatnonzero(involved)

        rows = []
        rhs = []

        # Each signed determinant has affine form a*u+b*v+c and must satisfy
        # sign(det)*(a*u+b*v+c) >= t.
        for tri_index in active_indices:
            i, j, k = triples[tri_index]
            other = layout.copy()

            def det_at(u: float, v: float) -> float:
                other[point, 0] = u
                other[point, 1] = v
                a = other[i]
                b = other[j]
                c = other[k]
                return float(
                    (b[0] - a[0]) * (c[1] - a[1])
                    - (b[1] - a[1]) * (c[0] - a[0])
                )

            constant = det_at(0.0, 0.0)
            coeff_u = det_at(1.0, 0.0) - constant
            coeff_v = det_at(0.0, 1.0) - constant
            sign = 1.0 if current_dets[tri_index] >= 0.0 else -1.0

            rows.append((sign * coeff_u, sign * coeff_v, -1.0))
            rhs.append(-sign * constant)

        # Triangle domain: u >= 0, v >= 0, u + v <= 1.
        rows.extend(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (-1.0, -1.0, 0.0)))
        rhs.extend((0.0, 0.0, -1.0))

        # Areas of triangles not containing this coordinate remain fixed and
        # give an absolute upper cap on the local variable t.
        unaffected = current_abs[~involved]
        if unaffected.size:
            rows.append((0.0, 0.0, -1.0))
            rhs.append(-float(unaffected.min()))

        matrix = np.asarray(rows, dtype=float)
        lower = np.asarray(rhs, dtype=float)
        constraint_count = len(matrix)

        if all_constraint_combinations is None or (
            all_constraint_combinations.size
            and all_constraint_combinations.max() >= constraint_count
        ):
            all_constraint_combinations = np.asarray(
                list(itertools.combinations(range(constraint_count), 3)),
                dtype=np.intp,
            )

        combinations = all_constraint_combinations
        systems = matrix[combinations]
        determinants_3x3 = np.linalg.det(systems)
        nonsingular = np.abs(determinants_3x3) > 1.0e-10
        systems = systems[nonsingular]
        selected_rhs = lower[combinations[nonsingular]]

        if len(systems) == 0:
            return layout

        vertices = np.linalg.solve(systems, selected_rhs[..., None])[..., 0]
        feasible = np.all(
            vertices @ matrix.T >= lower[None, :] - 2.0e-9,
            axis=1,
        )
        if not np.any(feasible):
            return layout

        candidates = vertices[feasible]
        candidate_index = int(np.argmax(candidates[:, 2]))
        uv = candidates[candidate_index, :2]

        proposal = layout.copy()
        proposal[point] = uv
        return proposal

    # Several passes are enough because each accepted coordinate move improves
    # the global objective exactly, and the last pass generally stabilizes.
    for _ in range(5):
        changed = False
        order = rng.permutation(np.arange(fixed_count, n_points))
        for point in order:
            candidate = improve_one_coordinate(best_layout, int(point))
            candidate_value = float(exact_score(candidate[None, ...])[0])
            if candidate_value > best_value + 1.0e-11:
                best_layout = candidate
                best_value = candidate_value
                changed = True
        if not changed:
            break

    height = np.sqrt(3.0) / 2.0
    result = np.empty((n_points, 2), dtype=float)
    result[:, 0] = best_layout[:, 0] + 0.5 * best_layout[:, 1]
    result[:, 1] = height * best_layout[:, 1]
    return result


# EVOLVE-BLOCK-END