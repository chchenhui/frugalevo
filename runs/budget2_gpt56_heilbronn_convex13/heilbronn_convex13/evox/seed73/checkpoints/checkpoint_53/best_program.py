# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def heilbronn_convex13() -> np.ndarray:
    """Optimize ten points inside a fixed unit-area triangle deterministically.

    Three hull vertices are fixed, while seeded soft-min annealing, exact
    directional max-min polling, and a final signed-area SLSQP formulation
    optimize the remaining ten points.
    """
    rng = np.random.default_rng(13071957)

    # This right triangle has area one, so raw areas equal normalized areas.
    # Compared with a fixed square it leaves one additional point movable.
    corners = np.array(
        [[0.0, 0.0], [2.0, 0.0], [0.0, 1.0]],
        dtype=np.float64,
    )

    def project_triangle(p: np.ndarray) -> np.ndarray:
        """Project p into x >= 0, y >= 0, and x / 2 + y <= 1."""
        uv = np.maximum(np.array([0.5 * p[0], p[1]]), 0.001)
        total = float(uv.sum())
        if total > 0.999:
            uv *= 0.999 / total
        return np.array([2.0 * uv[0], uv[1]])

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

    # Staggered barycentric sites avoid the severe exact-collinearity defects
    # of an unperturbed triangular lattice.
    lattice = np.array(
        [[0.14, 0.14], [0.36, 0.14], [0.58, 0.14], [0.80, 0.14],
         [0.14, 0.36], [0.36, 0.36], [0.58, 0.36],
         [0.14, 0.58], [0.36, 0.58], [0.14, 0.80]],
        dtype=np.float64,
    )

    # Retain the original 32 deterministic trajectories unchanged, then
    # explore additional independent orientation/sign-pattern basins.  Since
    # best_points is a global incumbent, this can only improve quality.
    restarts = 80
    iterations = 45000
    for restart in range(restarts):
        points = np.empty((13, 2), dtype=np.float64)
        points[:3] = corners
        uv = lattice + rng.uniform(-0.105, 0.105, size=(10, 2))
        points[3:] = np.array(
            [project_triangle([2.0 * q[0], q[1]]) for q in uv]
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

            point_index = int(rng.integers(3, 13))
            old_point = points[point_index].copy()
            proposal = project_triangle(
                old_point + rng.normal(0.0, move_scale, size=2)
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
            for point_index in range(3, 13):
                original = best_points[point_index].copy()
                local_point = original
                local_minimum = best_minimum

                for direction in directions:
                    candidate = project_triangle(original + radius * direction)
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

    # Simultaneously polish all movable coordinates.  Around this nondegenerate
    # incumbent, determinant signs are locally fixed; maximizing t subject to
    # signed_area >= t is therefore a smooth local max-min formulation.
    signed_determinants = (
        (best_points[triangles[:, 1], 0] - best_points[triangles[:, 0], 0])
        * (best_points[triangles[:, 2], 1] - best_points[triangles[:, 0], 1])
        - (best_points[triangles[:, 1], 1] - best_points[triangles[:, 0], 1])
        * (best_points[triangles[:, 2], 0] - best_points[triangles[:, 0], 0])
    )
    orientation = np.sign(signed_determinants)
    orientation[orientation == 0.0] = 1.0

    def unpack_polish(z: np.ndarray) -> np.ndarray:
        """Rebuild fixed-triangle points from ten coordinates and t."""
        polished = np.empty((13, 2), dtype=np.float64)
        polished[:3] = corners
        polished[3:] = z[:-1].reshape(10, 2)
        return polished

    def polish_constraints(z: np.ndarray) -> np.ndarray:
        """Return signed-area and triangular-domain feasibility margins."""
        polished = unpack_polish(z)
        a = polished[triangles[:, 0]]
        b = polished[triangles[:, 1]]
        c = polished[triangles[:, 2]]
        signed_area = 0.5 * orientation * (
            (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
            - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        )
        movable = polished[3:]
        domain = np.column_stack((
            0.5 * movable[:, 0] - 0.001,
            movable[:, 1] - 0.001,
            0.999 - 0.5 * movable[:, 0] - movable[:, 1],
        )).ravel()
        return np.concatenate((signed_area - z[-1], domain))

    polish_start = np.concatenate((
        best_points[3:].ravel(),
        [best_minimum - 1e-10],
    ))
    try:
        polished_result = minimize(
            lambda z: -z[-1],
            polish_start,
            method="SLSQP",
            constraints={"type": "ineq", "fun": polish_constraints},
            options={"maxiter": 700, "ftol": 1e-12, "disp": False},
        )
        if np.all(np.isfinite(polished_result.x)):
            polished_points = unpack_polish(polished_result.x)
            polished_minimum = float(areas(polished_points).min())
            # Verify using unsigned areas, rather than trusting solver status.
            if polished_minimum > best_minimum + 1e-12:
                best_points = polished_points
                best_minimum = polished_minimum
    except (FloatingPointError, ValueError):
        # The already validated annealing/polling incumbent is always safe.
        pass

    # The three fixed vertices retain a unit-area triangular hull exactly.
    return best_points.copy()


# EVOLVE-BLOCK-END
