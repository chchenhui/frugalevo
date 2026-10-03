# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """Use seeded bottleneck annealing followed by deterministic greedy polishing."""
    n = 11
    rng = np.random.default_rng(11011)
    height = np.sqrt(3.0) / 2.0

    triples = np.array(
        [(i, j, k) for i in range(n) for j in range(i + 1, n) for k in range(j + 1, n)],
        dtype=np.int32,
    )

    def sample_points() -> np.ndarray:
        """Generate uniformly distributed points in the equilateral triangle."""
        u = rng.random((n, 2))
        mask = (u[:, 0] + u[:, 1]) > 1.0
        u[mask] = 1.0 - u[mask]
        # Barycentric coordinates relative to (0,0), (1,0), and (1/2,height).
        return (
            u[:, 0:1] * np.array([1.0, 0.0])
            + u[:, 1:2] * np.array([0.5, height])
        )

    def minimum_area(points: np.ndarray) -> float:
        """Return the smallest absolute area over all point triples."""
        a = points[triples[:, 0]]
        b = points[triples[:, 1]]
        c = points[triples[:, 2]]
        cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (
            b[:, 1] - a[:, 1]
        ) * (c[:, 0] - a[:, 0])
        return 0.5 * np.min(np.abs(cross))

    best_points = None
    best_value = -1.0

    # Use bottleneck-directed annealing: most moves affect a point in one
    # of the currently smallest triangles, while random moves preserve escape
    # from local basins.
    # Use fewer, longer trajectories: this preserves a comparable evaluation
    # budget while allowing each annealing run to perform deeper refinement.
    for restart in range(30):
        points = sample_points()
        value = minimum_area(points)

        for iteration in range(20000):
            fraction = iteration / 20000.0
            temperature = 0.0022 * (1.0 - fraction) ** 2 + 2.0e-9
            step = 0.13 * (1.0 - fraction) + 0.0010

            a = points[triples[:, 0]]
            b = points[triples[:, 1]]
            c = points[triples[:, 2]]
            cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (
                b[:, 1] - a[:, 1]
            ) * (c[:, 0] - a[:, 0])
            areas = 0.5 * np.abs(cross)

            # Select from several near-bottleneck triples rather than always
            # following one possibly noisy minimum.
            if rng.random() < 0.86:
                # Include a wider bottleneck set so that moves address several
                # competing constraints instead of repeatedly perturbing one.
                bottleneck_rank = min(15, len(areas) - 1)
                cutoff = np.partition(areas, bottleneck_rank)[bottleneck_rank]
                candidates = np.flatnonzero(areas <= cutoff + 1.0e-12)
                triple = triples[candidates[rng.integers(len(candidates))]]
                index = int(triple[rng.integers(3)])
            else:
                index = int(rng.integers(n))

            candidate = points.copy()
            candidate[index] += rng.normal(0.0, step, 2)

            x, y = candidate[index]
            if (
                x < 0.0
                or y < 0.0
                or y > height
                or y > np.sqrt(3.0) * min(x, 1.0 - x)
            ):
                continue

            candidate_value = minimum_area(candidate)
            delta = candidate_value - value

            if delta >= 0.0 or rng.random() < np.exp(delta / temperature):
                points = candidate
                value = candidate_value

                if value > best_value:
                    best_value = value
                    best_points = points.copy()

    # Greedy local polishing removes residual slack after the annealing phase.
    # Only improving moves are accepted, so this phase cannot degrade the best
    # configuration already discovered.
    if best_points is not None:
        points = best_points.copy()
        value = best_value

        for polish_iteration in range(12000):
            fraction = polish_iteration / 12000.0
            step = 0.010 * (1.0 - fraction) + 2.0e-6

            # Prefer points participating in the current bottleneck, while
            # retaining occasional fully random coordinate updates.
            a = points[triples[:, 0]]
            b = points[triples[:, 1]]
            c = points[triples[:, 2]]
            cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (
                b[:, 1] - a[:, 1]
            ) * (c[:, 0] - a[:, 0])
            areas = 0.5 * np.abs(cross)

            if rng.random() < 0.90:
                rank = min(10, len(areas) - 1)
                cutoff = np.partition(areas, rank)[rank]
                candidates = np.flatnonzero(areas <= cutoff + 1.0e-12)
                triple = triples[candidates[rng.integers(len(candidates))]]
                index = int(triple[rng.integers(3)])
            else:
                index = int(rng.integers(n))

            candidate = points.copy()
            candidate[index] += rng.normal(0.0, step, 2)
            x, y = candidate[index]

            if (
                x < 0.0
                or y < 0.0
                or x > 1.0
                or y > height
                or y > np.sqrt(3.0) * min(x, 1.0 - x)
            ):
                continue

            candidate_value = minimum_area(candidate)
            if candidate_value > value:
                points = candidate
                value = candidate_value

                if value > best_value:
                    best_value = value
                    best_points = points.copy()

    return best_points


# EVOLVE-BLOCK-END
