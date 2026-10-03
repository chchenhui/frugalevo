# EVOLVE-BLOCK-START
import numpy as np


def _incumbent_heilbronn_convex13() -> np.ndarray:
    """Deterministic square construction with exact local bottleneck polling."""
    n = 13
    rng = np.random.default_rng(42)

    corners = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=float,
    )
    triples = np.array(
        [(i, j, k) for i in range(n) for j in range(i + 1, n)
         for k in range(j + 1, n)],
        dtype=np.int16,
    )

    def areas(points: np.ndarray) -> np.ndarray:
        a = points[triples[:, 0]]
        b = points[triples[:, 1]]
        c = points[triples[:, 2]]
        cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
        cross -= (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        return 0.5 * np.abs(cross)

    def minimum_area(points: np.ndarray) -> float:
        return float(np.min(areas(points)))

    base = np.array(
        [
            [0.18, 0.18], [0.42, 0.16], [0.68, 0.18],
            [0.82, 0.40], [0.58, 0.40], [0.30, 0.40],
            [0.16, 0.64], [0.42, 0.64], [0.70, 0.64],
        ],
        dtype=float,
    )

    best_points = np.vstack((corners, base))
    best_value = minimum_area(best_points)

    for restart in range(32):
        interior = base.copy()
        if restart:
            noise = 0.035 if restart % 2 else 0.075
            interior += rng.normal(0.0, noise, interior.shape)
            interior = np.clip(interior, 0.045, 0.955)

        points = np.vstack((corners, interior))
        value = minimum_area(points)
        restart_best_value = value
        restart_best_points = points.copy()

        for iteration in range(5000):
            fraction = iteration / 5000.0
            scale = 0.085 * (1.0 - fraction) + 0.0012
            index = 4 + int(rng.integers(9))
            old = points[index].copy()
            proposal = old + rng.normal(0.0, scale, 2)
            proposal = np.clip(proposal, 0.02, 0.98)

            points[index] = proposal
            candidate = minimum_area(points)

            temperature = 0.00035 * (1.0 - fraction)
            accept = candidate >= value
            if not accept and temperature > 0.0:
                accept = rng.random() < np.exp((candidate - value) / temperature)

            if accept:
                value = candidate
                if value > restart_best_value:
                    restart_best_value = value
                    restart_best_points = points.copy()
            else:
                points[index] = old

        if restart_best_value > best_value:
            best_value = restart_best_value
            best_points = restart_best_points.copy()

    directions = np.array(
        [
            [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0],
            [0.7071067811865476, 0.7071067811865476],
            [0.7071067811865476, -0.7071067811865476],
            [-0.7071067811865476, 0.7071067811865476],
            [-0.7071067811865476, -0.7071067811865476],
        ],
        dtype=float,
    )

    step = 0.012
    for _ in range(28):
        tri_areas = areas(best_points)
        active = np.argsort(tri_areas)[:24]
        incidence = np.zeros(n, dtype=np.int16)
        for tri_index in active:
            incidence[triples[tri_index]] += 1
        order = 4 + np.argsort(-incidence[4:])

        improved = False
        for index in order:
            old = best_points[index].copy()
            local_best = best_value
            local_point = old
            for direction in directions:
                candidate_point = np.clip(old + step * direction, 0.02, 0.98)
                best_points[index] = candidate_point
                candidate = minimum_area(best_points)
                if candidate > local_best:
                    local_best = candidate
                    local_point = candidate_point.copy()
            best_points[index] = local_point
            if local_best > best_value:
                best_value = local_best
                improved = True

        if not improved:
            step *= 0.5
            if step < 2.0e-5:
                break

    return best_points


def heilbronn_convex13() -> np.ndarray:
    """Optimize a center plus four threefold rotational point orbits.

    The eight orbit parameters (four radii and four angles) are searched with
    deterministic simulated annealing, using the exact minimum triangle area
    divided by convex-hull area as the objective.  The best configuration is
    finally scaled to unit convex-hull area.
    """
    n = 13
    rng = np.random.default_rng(42)
    twopi_over_three = 2.0 * np.pi / 3.0
    triples = np.array(
        [(i, j, k) for i in range(n) for j in range(i + 1, n)
         for k in range(j + 1, n)],
        dtype=np.int16,
    )

    def orbit_points(params: np.ndarray) -> np.ndarray:
        """Materialize the center and four three-point rotational orbits."""
        radii = params[:4]
        angles = params[4:]
        points = np.zeros((n, 2), dtype=float)
        cursor = 1
        for radius, angle in zip(radii, angles):
            for k in range(3):
                a = angle + twopi_over_three * k
                points[cursor] = radius * np.array(
                    [np.cos(a), np.sin(a)], dtype=float
                )
                cursor += 1
        return points

    def hull_area(points: np.ndarray) -> float:
        """Return the area of the convex hull by a monotone-chain scan."""
        ordered = sorted((float(x), float(y)) for x, y in points)
        if len(ordered) < 3:
            return 0.0

        def cross(o, a, b):
            return ((a[0] - o[0]) * (b[1] - o[1])
                    - (a[1] - o[1]) * (b[0] - o[0]))

        lower = []
        for point in ordered:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
                lower.pop()
            lower.append(point)

        upper = []
        for point in reversed(ordered):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
                upper.pop()
            upper.append(point)

        polygon = lower[:-1] + upper[:-1]
        area = 0.0
        for i, point in enumerate(polygon):
            other = polygon[(i + 1) % len(polygon)]
            area += point[0] * other[1] - point[1] * other[0]
        return 0.5 * abs(area)

    def objective(params: np.ndarray) -> float:
        """Evaluate the exact normalized minimum triangle area."""
        points = orbit_points(params)
        a = points[triples[:, 0]]
        b = points[triples[:, 1]]
        c = points[triples[:, 2]]
        cross = ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                 - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
        minimum = float(np.min(0.5 * np.abs(cross)))
        area = hull_area(points)
        return minimum / area if area > 1.0e-14 else 0.0

    # Cross four radial-layer orderings with six distinct relative angular
    # arrangements.  All arrays are writable float copies so the annealing
    # loop can safely perturb and wrap the eight parameters in place.
    radial_seed = np.array((0.16, 0.27, 0.39, 0.52), dtype=float)
    radii_patterns = (
        radial_seed,
        radial_seed[[1, 3, 0, 2]],
        radial_seed[[2, 0, 3, 1]],
        radial_seed[[3, 2, 1, 0]],
    )
    angle_patterns = (
        np.array((0.00, 0.07, 0.14, 0.21), dtype=float),
        np.array((0.00, 0.21, 0.07, 0.14), dtype=float),
        np.array((0.00, 0.14, 0.21, 0.07), dtype=float),
        np.array((0.00, 0.11, 0.23, 0.04), dtype=float),
        np.array((0.00, 0.19, 0.03, 0.16), dtype=float),
        np.array((0.00, 0.25, 0.12, 0.05), dtype=float),
    )

    best_params = None
    best_value = -1.0

    for restart in range(24):
        radial_pattern = radii_patterns[restart // 6]
        angular_pattern = angle_patterns[restart % 6]
        radii = np.asarray(radial_pattern, dtype=float).copy()
        offset = 0.04 * (restart % 6)
        angles = np.asarray(angular_pattern + offset, dtype=float).copy()
        params = np.concatenate((radii, angles)).astype(float, copy=True)
        params[4:] %= twopi_over_three
        value = objective(params)
        local_best = value
        local_params = params.copy()

        for iteration in range(7000):
            fraction = iteration / 7000.0
            proposal = params.copy()
            if iteration < 4500:
                radius_step = 0.055 * (1.0 - fraction) + 0.002
                angle_step = 0.16 * (1.0 - fraction) + 0.006
            else:
                radius_step = 0.012 * (1.0 - fraction) + 0.0004
                angle_step = 0.035 * (1.0 - fraction) + 0.001
            # Mostly use independent coordinate moves, but periodically move
            # one complete orbit coherently.  Coupled orbit moves are useful
            # near local optima where changing a radius and its phase together
            # can remove several simultaneous bottleneck triangles.
            if rng.random() < 0.28:
                orbit = int(rng.integers(4))
                proposal[orbit] += rng.normal(0.0, radius_step)
                proposal[4 + orbit] += rng.normal(0.0, angle_step)
                # Apply a small compensating displacement to a second layer;
                # this changes hull topology without destabilizing all layers.
                partner = (orbit + 1 + int(rng.integers(3))) % 4
                proposal[partner] += rng.normal(0.0, 0.45 * radius_step)
                proposal[4 + partner] += rng.normal(
                    0.0, 0.45 * angle_step
                )
            else:
                proposal[:4] += rng.normal(0.0, radius_step, 4)
                proposal[4:] += rng.normal(0.0, angle_step, 4)

            proposal[:4] = np.clip(proposal[:4], 0.12, 0.55)
            proposal[4:] %= twopi_over_three

            candidate = objective(proposal)
            temperature = 0.00022 * (1.0 - fraction)
            accept = candidate >= value
            if not accept and temperature > 0.0:
                accept = rng.random() < np.exp(
                    (candidate - value) / temperature
                )
            if accept:
                params = proposal
                value = candidate
                if value > local_best:
                    local_best = value
                    local_params = params.copy()

        # Deterministic coordinate polishing of the annealed candidate.
        # Each pass tests both signs of every radius and angle coordinate,
        # retaining only strict improvements of the exact normalized minimum.
        params = local_params.copy()
        value = local_best
        # Continue with two fine scales to resolve nearly active triangle
        # constraints left by the stochastic phase.
        for polish_step in (
            0.010, 0.004, 0.0015, 0.0005, 0.00015, 0.00005, 0.00002
        ):
            improved = True
            while improved:
                """Polish single coordinates and coupled orbit radius/phase pairs."""
                improved = False
                for coordinate in range(8):
                    for sign in (-1.0, 1.0):
                        candidate_params = params.copy()
                        candidate_params[coordinate] += sign * polish_step
                        candidate_params[:4] = np.clip(
                            candidate_params[:4], 0.12, 0.55
                        )
                        candidate_params[4:] %= twopi_over_three
                        candidate = objective(candidate_params)
                        if candidate > value:
                            params = candidate_params
                            value = candidate
                            improved = True
                            if value > local_best:
                                local_best = value
                                local_params = params.copy()

                # The exact bottleneck often depends on a layer's radius and
                # phase together; poll these coupled directions after the
                # coordinate sweep without adding another global search.
                for orbit in range(4):
                    for radius_sign in (-1.0, 1.0):
                        for angle_sign in (-1.0, 1.0):
                            candidate_params = params.copy()
                            candidate_params[orbit] += (
                                radius_sign * polish_step
                            )
                            candidate_params[4 + orbit] += (
                                angle_sign * 0.8 * polish_step
                            )
                            candidate_params[:4] = np.clip(
                                candidate_params[:4], 0.12, 0.55
                            )
                            candidate_params[4:] %= twopi_over_three
                            candidate = objective(candidate_params)
                            if candidate > value:
                                params = candidate_params
                                value = candidate
                                improved = True
                                if value > local_best:
                                    local_best = value
                                    local_params = params.copy()

        if local_best > best_value:
            best_value = local_best
            best_params = local_params.copy()

    if best_params is None:
        best_params = np.array(
            [0.16, 0.27, 0.39, 0.52, 0.0, 0.07, 0.14, 0.21],
            dtype=float,
        )

    result = orbit_points(best_params).astype(float, copy=True)

    # Jointly refine all non-gauge Cartesian coordinates with bounded,
    # deterministic softmin continuation over all 286 triangle constraints.
    try:
        from scipy.optimize import minimize

        def gauge_points(points: np.ndarray) -> np.ndarray:
            """Fix translation, rotation, and scale using the first two points."""
            transformed = np.asarray(points, dtype=float).copy()
            transformed -= transformed[0]
            baseline = transformed[1].copy()
            length = float(np.linalg.norm(baseline))
            if length <= 1.0e-12 or not np.isfinite(length):
                raise ValueError("degenerate gauge")
            rotation = np.array(
                [[baseline[0], baseline[1]],
                 [-baseline[1], baseline[0]]],
                dtype=float,
            ) / length
            transformed = transformed @ rotation.T
            transformed /= float(transformed[1, 0])
            transformed[0] = (0.0, 0.0)
            transformed[1] = (1.0, 0.0)
            return transformed

        initial = gauge_points(result)
        variable_indices = np.arange(2, n, dtype=np.intp)
        initial_vector = initial[variable_indices].reshape(-1).copy()

        def unpack(vector: np.ndarray) -> np.ndarray:
            """Restore thirteen points from the gauge-fixed free coordinates."""
            points = np.empty((n, 2), dtype=float)
            points[0] = (0.0, 0.0)
            points[1] = (1.0, 0.0)
            points[variable_indices] = np.asarray(vector).reshape(-1, 2)
            return points

        def exact_free_value(vector: np.ndarray) -> float:
            """Return the exact normalized minimum triangle area."""
            points = unpack(vector)
            if not np.all(np.isfinite(points)):
                return -1.0
            area = hull_area(points)
            if area <= 1.0e-14:
                return -1.0
            a = points[triples[:, 0]]
            b = points[triples[:, 1]]
            c = points[triples[:, 2]]
            cross = (
                (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
            )
            return float(np.min(0.5 * np.abs(cross)) / area)

        def smooth_free_value(
            vector: np.ndarray, temperature: float
        ) -> float:
            """Return a stable smooth lower-tail objective for minimization."""
            points = unpack(vector)
            if not np.all(np.isfinite(points)):
                return 1.0e6
            area = hull_area(points)
            if area <= 1.0e-14:
                return 1.0e6
            a = points[triples[:, 0]]
            b = points[triples[:, 1]]
            c = points[triples[:, 2]]
            cross = (
                (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
            )
            magnitudes = np.sqrt(0.25 * cross * cross + 1.0e-20)
            scaled = -magnitudes / temperature
            shift = float(np.max(scaled))
            tail = shift + np.log(np.sum(np.exp(scaled - shift)))
            return float(temperature * tail / area)

        original_value = exact_free_value(initial_vector)
        vector = initial_vector.copy()
        incumbent = original_value
        temperatures = (0.0015, 0.0007, 0.0003, 0.00012)
        bounds = [(-1.5, 1.5)] * vector.size

        for stage, temperature in enumerate(temperatures):
            """Run one bounded continuation stage with at most three starts."""
            """Run bounded L-BFGS-B from three deterministic continuation starts."""
            perturbation = 0.0015 * rng.normal(size=vector.size)
            starts = [
                vector.copy(),
                vector + perturbation,
                vector - perturbation,
            ]

            stage_vector = vector.copy()
            stage_value = incumbent
            for start in starts[:3]:
                try:
                    outcome = minimize(
                        lambda x: smooth_free_value(x, temperature),
                        np.clip(start, -1.5, 1.5),
                        method="L-BFGS-B",
                        bounds=bounds,
                        options={
                            "maxiter": 180,
                            "ftol": 1.0e-12,
                            "gtol": 1.0e-7,
                            "maxls": 40,
                        },
                    )
                except (FloatingPointError, ValueError):
                    continue
                candidate = np.asarray(outcome.x, dtype=float)
                if candidate.shape != vector.shape:
                    continue
                candidate_value = exact_free_value(candidate)
                if (
                    np.isfinite(candidate_value)
                    and candidate_value >= stage_value
                    and np.all(np.isfinite(candidate))
                ):
                    stage_vector = candidate.copy()
                    stage_value = candidate_value

            vector = stage_vector
            incumbent = stage_value

        # Exact deterministic coordinate and pair polling.  Single-coordinate
        # moves resolve isolated constraints; the additional coupled polls
        # allow two Cartesian coordinates involved in the same bottleneck
        # triangles to move together without requiring another global solve.
        for step in (0.0015, 0.0005, 0.00015, 0.00005):
            for _ in range(3):
                improved = False
                for coordinate in range(vector.size):
                    for sign in (-1.0, 1.0):
                        candidate = vector.copy()
                        candidate[coordinate] += sign * step
                        candidate[coordinate] = np.clip(
                            candidate[coordinate], -1.5, 1.5
                        )
                        candidate_value = exact_free_value(candidate)
                        if candidate_value > incumbent:
                            vector = candidate
                            incumbent = candidate_value
                            improved = True

                # Select coordinates participating most often in the current
                # lower-tail triangles, then test a bounded deterministic set
                # of paired sign combinations.
                points = unpack(vector)
                a = points[triples[:, 0]]
                b = points[triples[:, 1]]
                c = points[triples[:, 2]]
                cross = (
                    (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                    - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
                )
                active = np.argsort(0.5 * np.abs(cross))[:16]
                participation = np.zeros(vector.size, dtype=np.int32)
                for tri_index in active:
                    for point_index in triples[tri_index]:
                        if int(point_index) >= 2:
                            offset = 2 * (int(point_index) - 2)
                            participation[offset:offset + 2] += 1

                ranked = np.argsort(-participation)
                pair_coordinates = []
                for first_position in range(min(8, ranked.size)):
                    first = int(ranked[first_position])
                    for second_position in range(first_position + 1,
                                                 min(8, ranked.size)):
                        second = int(ranked[second_position])
                        if participation[first] > 0 and participation[second] > 0:
                            pair_coordinates.append((first, second))
                        if len(pair_coordinates) >= 12:
                            break
                    if len(pair_coordinates) >= 12:
                        break

                for first, second in pair_coordinates:
                    for first_sign, second_sign in (
                        (-1.0, -1.0), (-1.0, 1.0),
                        (1.0, -1.0), (1.0, 1.0),
                    ):
                        candidate = vector.copy()
                        candidate[first] += first_sign * step
                        candidate[second] += second_sign * step
                        candidate[first] = np.clip(
                            candidate[first], -1.5, 1.5
                        )
                        candidate[second] = np.clip(
                            candidate[second], -1.5, 1.5
                        )
                        candidate_value = exact_free_value(candidate)
                        if candidate_value > incumbent:
                            vector = candidate
                            incumbent = candidate_value
                            improved = True

                if not improved:
                    break

        refined = unpack(vector)
        if (
            np.all(np.isfinite(refined))
            and hull_area(refined) > 1.0e-14
            and incumbent > original_value
        ):
            result = refined
    except (ImportError, ValueError, FloatingPointError):
        pass

    def cartesian_objective(points: np.ndarray) -> float:
        """Evaluate the exact normalized minimum area for Cartesian points."""
        a = points[triples[:, 0]]
        b = points[triples[:, 1]]
        c = points[triples[:, 2]]
        cross = ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                 - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
        minimum = float(np.min(0.5 * np.abs(cross)))
        area = hull_area(points)
        return minimum / area if area > 1.0e-14 else 0.0

    """Refine all points with bounded active-triangle linearized feasibility moves."""
    """Refine bottleneck-sharing point pairs using active-gradient nullspaces."""
    cart_value = cartesian_objective(result)
    cart_best = result.copy()

    def optimize_orbit_block(points: np.ndarray, block: int) -> tuple:
        """Optimize one frozen three-point orbit with bounded Nelder–Mead."""
        indices = np.arange(1 + 3 * block, 4 + 3 * block, dtype=np.intp)
        original = points[indices].copy()
        center = np.mean(original, axis=0)
        radial = original - center
        tangent = np.column_stack((-radial[:, 1], radial[:, 0]))

        starts = [
            original.copy(),
            center + 1.035 * radial,
            center + 0.965 * radial,
            center + 0.018 * tangent,
            center - 0.018 * tangent,
            original + 0.012 * np.array(
                [[1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]], dtype=float
            ),
        ]

        def block_value(flat: np.ndarray) -> float:
            """Evaluate a candidate orbit using the exact global objective."""
            candidate = points.copy()
            candidate[indices] = flat.reshape(3, 2)
            if not np.all(np.isfinite(candidate)):
                return -1.0
            area = hull_area(candidate)
            if area <= 1.0e-14:
                return -1.0
            return cartesian_objective(candidate)

        best_local = cart_value
        best_orbit = original.copy()

        for start in starts:
            simplex = np.empty((7, 6), dtype=float)
            simplex[0] = start.reshape(-1)
            # Use a finer initial simplex so the bounded six-coordinate search
            # resolves active triangle constraints instead of contracting from
            # an unnecessarily coarse Cartesian perturbation.
            for coordinate in range(6):
                simplex[coordinate + 1] = simplex[0]
                simplex[coordinate + 1, coordinate] += 0.010

            values = np.array(
                [block_value(vertex) for vertex in simplex], dtype=float
            )

            # Allow the six-dimensional orbit block to converge more fully
            # against the frozen ten-point scaffold while retaining a hard
            # deterministic per-start evaluation limit.
            for _ in range(320):
                order = np.argsort(-values)
                simplex = simplex[order]
                values = values[order]

                if values[0] > best_local:
                    best_local = float(values[0])
                    best_orbit = simplex[0].reshape(3, 2).copy()

                centroid = np.mean(simplex[:6], axis=0)
                reflected = centroid + (centroid - simplex[6])
                reflected_value = block_value(reflected)

                if reflected_value > values[0]:
                    expanded = centroid + 2.0 * (reflected - centroid)
                    expanded_value = block_value(expanded)
                    if expanded_value > reflected_value:
                        simplex[6] = expanded
                        values[6] = expanded_value
                    else:
                        simplex[6] = reflected
                        values[6] = reflected_value
                elif reflected_value > values[5]:
                    simplex[6] = reflected
                    values[6] = reflected_value
                else:
                    if reflected_value > values[6]:
                        contracted = centroid + 0.5 * (reflected - centroid)
                    else:
                        contracted = centroid + 0.5 * (simplex[6] - centroid)
                    contracted_value = block_value(contracted)
                    if contracted_value > values[6]:
                        simplex[6] = contracted
                        values[6] = contracted_value
                    else:
                        for row in range(1, 7):
                            simplex[row] = (
                                simplex[0] + 0.5 * (simplex[row] - simplex[0])
                            )
                            values[row] = block_value(simplex[row])

                if float(np.max(np.ptp(simplex, axis=0))) < 1.0e-7:
                    break

        return best_local, best_orbit

    for _ in range(0):
        tri_a = result[triples[:, 0]]
        tri_b = result[triples[:, 1]]
        tri_c = result[triples[:, 2]]
        tri_cross = (
            (tri_b[:, 0] - tri_a[:, 0]) * (tri_c[:, 1] - tri_a[:, 1])
            - (tri_b[:, 1] - tri_a[:, 1]) * (tri_c[:, 0] - tri_a[:, 0])
        )
        participation = np.zeros(4, dtype=np.int32)
        for tri_index in np.argsort(0.5 * np.abs(tri_cross))[:36]:
            for block in range(4):
                block_indices = np.arange(
                    1 + 3 * block, 4 + 3 * block, dtype=np.intp
                )
                participation[block] += int(
                    np.any(np.isin(triples[tri_index], block_indices))
                )

        changed = False
        for block in np.argsort(-participation):
            candidate_value, candidate_orbit = optimize_orbit_block(
                result, int(block)
            )
            if candidate_value > cart_value:
                result[1 + 3 * block:4 + 3 * block] = candidate_orbit
                cart_value = candidate_value
                cart_best = result.copy()
                changed = True

        if not changed:
            break

    def optimize_active_star(
        points: np.ndarray, indices: np.ndarray
    ) -> tuple:
        """Optimize a frozen five-point active-star block with Nelder–Mead."""
        original = points[indices].copy()
        center = np.mean(original, axis=0)
        radial = original - center

        # Build a deterministic first-order direction from the eight smallest
        # triangles.  The direction is only an initialization; every accepted
        # move is checked by the exact global objective.
        local_gradient = np.zeros((5, 2), dtype=float)
        all_areas = 0.5 * np.abs(
            ((points[triples[:, 1], 0] - points[triples[:, 0], 0]) *
             (points[triples[:, 2], 1] - points[triples[:, 0], 1]))
            - ((points[triples[:, 1], 1] - points[triples[:, 0], 1]) *
               (points[triples[:, 2], 0] - points[triples[:, 0], 0]))
        )
        for tri_index in np.argsort(all_areas)[:8]:
            i, j, k = (int(x) for x in triples[tri_index])
            signed = 1.0 if (
                (points[j, 0] - points[i, 0]) *
                (points[k, 1] - points[i, 1]) -
                (points[j, 1] - points[i, 1]) *
                (points[k, 0] - points[i, 0])
            ) >= 0.0 else -1.0
            for vertex, u, v in ((i, j, k), (j, k, i), (k, i, j)):
                local = np.flatnonzero(indices == vertex)
                if local.size:
                    local_gradient[local[0]] += 0.5 * signed * np.array(
                        [points[u, 1] - points[v, 1],
                         points[v, 0] - points[u, 0]], dtype=float
                    )

        gradient_norm = float(np.linalg.norm(local_gradient))
        gradient_start = original.copy()
        if gradient_norm > 1.0e-14:
            gradient_start += 0.004 * local_gradient / gradient_norm

        starts = (
            original,
            center + 1.018 * radial,
            gradient_start,
        )

        def block_value(flat: np.ndarray) -> float:
            """Evaluate one finite star against the exact global objective."""
            candidate = points.copy()
            candidate[indices] = flat.reshape(5, 2)
            if not np.all(np.isfinite(candidate)):
                return -1.0
            return cartesian_objective(candidate)

        best_value = cart_value
        best_block = original.copy()

        for start in starts:
            simplex = np.empty((11, 10), dtype=float)
            simplex[0] = start.reshape(-1)
            for coordinate in range(10):
                simplex[coordinate + 1] = simplex[0]
                simplex[coordinate + 1, coordinate] += 0.006

            values = np.array(
                [block_value(vertex) for vertex in simplex], dtype=float
            )

            for _ in range(220):
                order = np.argsort(-values)
                simplex = simplex[order]
                values = values[order]

                if values[0] > best_value:
                    best_value = float(values[0])
                    best_block = simplex[0].reshape(5, 2).copy()

                centroid = np.mean(simplex[:10], axis=0)
                reflected = centroid + (centroid - simplex[10])
                reflected_value = block_value(reflected)

                if reflected_value > values[0]:
                    expanded = centroid + 2.0 * (reflected - centroid)
                    expanded_value = block_value(expanded)
                    if expanded_value > reflected_value:
                        simplex[10] = expanded
                        values[10] = expanded_value
                    else:
                        simplex[10] = reflected
                        values[10] = reflected_value
                elif reflected_value > values[9]:
                    simplex[10] = reflected
                    values[10] = reflected_value
                else:
                    if reflected_value > values[10]:
                        contracted = centroid + 0.5 * (
                            reflected - centroid
                        )
                    else:
                        contracted = centroid + 0.5 * (
                            simplex[10] - centroid
                        )
                    contracted_value = block_value(contracted)
                    if contracted_value > values[10]:
                        simplex[10] = contracted
                        values[10] = contracted_value
                    else:
                        for row in range(1, 11):
                            simplex[row] = simplex[0] + 0.5 * (
                                simplex[row] - simplex[0]
                            )
                            values[row] = block_value(simplex[row])

                if float(np.max(np.ptp(simplex, axis=0))) < 1.0e-7:
                    break

        return best_value, best_block

    # Repeatedly optimize a connected five-point star selected from the
    # lowest 28 triangles.  Recomputing the active set after each accepted
    # replacement lets successive stars follow the moving bottleneck.
    for _ in range(0):
        areas = 0.5 * np.abs(
            ((result[triples[:, 1], 0] - result[triples[:, 0], 0]) *
             (result[triples[:, 2], 1] - result[triples[:, 0], 1]))
            - ((result[triples[:, 1], 1] - result[triples[:, 0], 1]) *
               (result[triples[:, 2], 0] - result[triples[:, 0], 0]))
        )
        active = np.argsort(areas)[:28]

        incidence = np.zeros(n, dtype=np.int32)
        graph = np.zeros((n, n), dtype=bool)
        for tri_index in active:
            tri = triples[tri_index]
            incidence[tri] += 1
            for p in range(3):
                for q in range(p + 1, 3):
                    graph[tri[p], tri[q]] = True
                    graph[tri[q], tri[p]] = True

        ranked = [int(x) for x in np.argsort(-incidence)]
        selected = [ranked[0]]
        while len(selected) < 5:
            choices = [
                vertex for vertex in ranked
                if vertex not in selected
                and any(graph[vertex, old] for old in selected)
            ]
            if not choices:
                choices = [vertex for vertex in ranked
                           if vertex not in selected]
            selected.append(choices[0])

        selected = np.asarray(selected, dtype=np.intp)

        # The construction above is connected by induction; retain an
        # explicit verification for numerical or graph-building edge cases.
        seen = {0}
        changed = True
        while changed:
            changed = False
            for p in range(5):
                for q in range(5):
                    if p in seen and q not in seen and graph[
                        selected[p], selected[q]
                    ]:
                        seen.add(q)
                        changed = True
        if len(seen) < 5:
            continue

        candidate_value, candidate_block = optimize_active_star(
            result, selected
        )
        if candidate_value <= cart_value:
            break

        result[selected] = candidate_block
        cart_value = candidate_value
        cart_best = result.copy()

    result = cart_best
    area = hull_area(result)
    if area > 1.0e-14:
        result = result / np.sqrt(area)
    return np.asarray(result, dtype=float).copy()


# EVOLVE-BLOCK-END