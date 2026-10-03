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

    for round_index in range(16):
        a = result[triples[:, 0]]
        b = result[triples[:, 1]]
        c = result[triples[:, 2]]
        cross = ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                 - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
        tri_areas = 0.5 * np.abs(cross)
        active = np.argsort(tri_areas)[:32]

        gradients = np.zeros((len(active), n, 2), dtype=float)
        for row, tri_index in enumerate(active):
            i, j, k = triples[tri_index]
            sign = 1.0 if cross[tri_index] >= 0.0 else -1.0
            ab = result[j] - result[i]
            ac = result[k] - result[i]
            gradients[row, i] = sign * np.array(
                [ab[1] - ac[1], ac[0] - ab[0]], dtype=float
            ) * 0.5
            gradients[row, j] = sign * np.array(
                [-ac[1], ac[0]], dtype=float
            ) * 0.5
            gradients[row, k] = sign * np.array(
                [ab[1], -ab[0]], dtype=float
            ) * 0.5

        incidence = np.sum(np.abs(gradients), axis=(0, 2))
        pairs = []
        ranked = np.argsort(-incidence)
        for first_pos, first in enumerate(ranked):
            for second in ranked[first_pos + 1:]:
                weight = incidence[first] + incidence[second]
                pairs.append((float(weight), int(first), int(second)))
        pairs.sort(reverse=True)
        bottleneck_rows = min(8, len(active))

        for _, first, second in pairs[:18]:
            pair_gradient = np.concatenate(
                (gradients[:, first, :], gradients[:, second, :]), axis=1
            )
            if not np.all(np.isfinite(pair_gradient)):
                continue

            # Null directions of the less-critical active constraints preserve
            # tight triangles to first order while permitting a pair hinge.
            keep = max(1, len(active) - bottleneck_rows)
            try:
                _, _, vh = np.linalg.svd(pair_gradient[:keep], full_matrices=True)
            except np.linalg.LinAlgError:
                continue

            directions = []
            for row in vh[-4:]:
                direction = np.asarray(row, dtype=float).copy()
                norm = float(np.linalg.norm(direction))
                if norm > 1.0e-14:
                    direction /= norm
                    guide = np.sum(pair_gradient[-bottleneck_rows:], axis=0)
                    if float(np.dot(direction, guide)) < 0.0:
                        direction *= -1.0
                    directions.append(direction)
                    directions.append(-direction)

            for direction in directions[:12]:
                displacement = np.zeros((n, 2), dtype=float)
                displacement[first] = direction[:2]
                displacement[second] = direction[2:]
                displacement *= 0.006 * (0.86 ** round_index)

                for backtrack in range(7):
                    candidate = result + (0.5 ** backtrack) * displacement
                    if not np.all(np.isfinite(candidate)):
                        continue
                    candidate_value = cartesian_objective(candidate)
                    if candidate_value > cart_value:
                        result = candidate
                        cart_value = candidate_value
                        cart_best = result.copy()
                        break

    result = cart_best
    area = hull_area(result)
    if area > 1.0e-14:
        result = result / np.sqrt(area)
    return np.asarray(result, dtype=float).copy()


# EVOLVE-BLOCK-END