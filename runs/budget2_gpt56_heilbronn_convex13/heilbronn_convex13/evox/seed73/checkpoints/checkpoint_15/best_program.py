# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_convex13() -> np.ndarray:
    """Deterministically optimize nine interior points of a unit square using
    multi-start simulated annealing on a smooth approximation of minimum area.

    The four square corners are retained permanently, so the convex hull has
    area exactly one and every returned point is inside the required convex
    region.  Candidate moves update only the 66 triangles incident to the
    moved point, while all 286 triangle areas are used by the objective.
    """
    rng = np.random.default_rng(13071957)

    # Keeping these four points fixes the enclosing convex region to the unit
    # square.  Consequently raw triangle area equals normalized triangle area.
    corners = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=np.float64,
    )

    triangles = np.array(
        [(i, j, k) for i in range(13) for j in range(i + 1, 13)
         for k in range(j + 1, 13)],
        dtype=np.intp,
    )
    incident = [
        np.flatnonzero(np.any(triangles == point_index, axis=1))
        for point_index in range(13)
    ]

    def areas(points: np.ndarray, ids: np.ndarray = None) -> np.ndarray:
        """Return unsigned areas for all, or a selected set of, triangles."""
        t = triangles if ids is None else triangles[ids]
        a = points[t[:, 0]]
        b = points[t[:, 1]]
        c = points[t[:, 2]]
        return 0.5 * np.abs(
            (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
            - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        )

    def soft_min(values: np.ndarray, temperature: float) -> float:
        """Stable log-sum-exp approximation to the minimum triangle area."""
        m = float(values.min())
        return m - temperature * np.log(
            np.exp(-(values - m) / temperature).sum()
        )

    best_points = None
    best_minimum = -1.0

    # A jittered 3 by 3 interior lattice gives a much stronger initial state
    # than independent random points, while independent restarts avoid locking
    # into its symmetry-related local optima.
    lattice = np.array(
        [[x, y] for y in (0.20, 0.50, 0.80) for x in (0.20, 0.50, 0.80)],
        dtype=np.float64,
    )

    restarts = 10
    iterations = 30000
    for restart in range(restarts):
        points = np.empty((13, 2), dtype=np.float64)
        points[:4] = corners
        points[4:] = np.clip(
            lattice + rng.uniform(-0.115, 0.115, size=(9, 2)),
            0.015,
            0.985,
        )

        current_areas = areas(points)
        current_minimum = float(current_areas.min())
        if current_minimum > best_minimum:
            best_minimum = current_minimum
            best_points = points.copy()

        for step in range(iterations):
            progress = step / (iterations - 1)

            # Broad early moves locate a basin; small late moves refine the
            # nearly active (small-area) triangles.
            move_scale = 0.085 * (1.0 - progress) ** 1.7 + 0.0012
            smooth_temperature = 0.010 * (1.0 - progress) + 0.00055
            accept_temperature = 0.0030 * (1.0 - progress) ** 2 + 0.000015

            point_index = int(rng.integers(4, 13))
            old_point = points[point_index].copy()
            proposal = np.clip(
                old_point + rng.normal(0.0, move_scale, size=2),
                0.001,
                0.999,
            )
            points[point_index] = proposal

            affected = incident[point_index]
            proposed_areas = current_areas.copy()
            proposed_areas[affected] = areas(points, affected)

            old_value = soft_min(current_areas, smooth_temperature)
            new_value = soft_min(proposed_areas, smooth_temperature)
            delta = new_value - old_value

            # Always retain an improvement; controlled downhill moves permit
            # rearrangements when a different triangle becomes limiting.
            if delta >= 0.0 or rng.random() < np.exp(delta / accept_temperature):
                current_areas = proposed_areas
                current_minimum = float(current_areas.min())

                if current_minimum > best_minimum:
                    best_minimum = current_minimum
                    best_points = points.copy()
            else:
                points[point_index] = old_point

    # Strict coordinate-poll refinement of the true nonsmooth max-min
    # objective.  Unlike annealing's soft minimum, every accepted move raises
    # the exact smallest triangle area, so this stage cannot degrade quality.
    angles = np.arange(16, dtype=np.float64) * (np.pi / 8.0)
    directions = np.column_stack((np.cos(angles), np.sin(angles)))

    best_areas = areas(best_points)
    best_minimum = float(best_areas.min())
    for radius in (0.012, 0.006, 0.003, 0.0015, 0.0007, 0.0003, 0.00012):
        # A cap avoids spending excessive time cycling through nearly flat
        # nonsmooth plateaus while still allowing interactions among points.
        for _ in range(20):
            changed = False
            for point_index in range(4, 13):
                original = best_points[point_index].copy()
                local_point = original
                local_minimum = best_minimum

                for direction in directions:
                    candidate = np.clip(
                        original + radius * direction, 0.001, 0.999
                    )
                    best_points[point_index] = candidate
                    candidate_areas = areas(best_points)
                    candidate_minimum = float(candidate_areas.min())

                    if candidate_minimum > local_minimum + 1e-14:
                        local_minimum = candidate_minimum
                        local_point = candidate.copy()
                        best_areas = candidate_areas

                best_points[point_index] = local_point
                if local_minimum > best_minimum + 1e-14:
                    best_minimum = local_minimum
                    changed = True
                else:
                    best_points[point_index] = original

            if not changed:
                break

    # Corners make the hull exactly the unit square, and copying prevents
    # callers from observing mutable optimization workspace.
    return best_points.copy()


# EVOLVE-BLOCK-END
