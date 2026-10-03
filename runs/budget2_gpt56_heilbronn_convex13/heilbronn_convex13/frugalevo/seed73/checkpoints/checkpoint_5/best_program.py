# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_convex13() -> np.ndarray:
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


# EVOLVE-BLOCK-END