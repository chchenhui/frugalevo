# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_convex13() -> np.ndarray:
    """
    Deterministically construct 13 points in the unit square.

    The four square corners are retained throughout the search, so the convex
    hull has area exactly one.  Consequently, maximizing the raw smallest
    triangle area is also maximizing the normalized score.
    """
    rng = np.random.default_rng(13031957)

    n = 13
    corners = np.array(
        [[0.0, 0.0],
         [1.0, 0.0],
         [1.0, 1.0],
         [0.0, 1.0]],
        dtype=float,
    )

    tri = np.array(
        [(i, j, k)
         for i in range(n - 2)
         for j in range(i + 1, n - 1)
         for k in range(j + 1, n)],
        dtype=np.intp,
    )

    incident = []
    for p in range(n):
        incident.append(np.flatnonzero(np.any(tri == p, axis=1)))

    def triangle_areas(pts: np.ndarray) -> np.ndarray:
        a = pts[tri[:, 0]]
        b = pts[tri[:, 1]]
        c = pts[tri[:, 2]]
        return 0.5 * np.abs(
            (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
            - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        )

    def incident_areas(candidates: np.ndarray, point_index: int) -> np.ndarray:
        """Areas of all triangles containing point_index for many candidates."""
        local_tri = tri[incident[point_index]]
        m = candidates.shape[0]
        work = np.broadcast_to(points, (m, n, 2)).copy()
        work[:, point_index, :] = candidates

        a = work[:, local_tri[:, 0], :]
        b = work[:, local_tri[:, 1], :]
        c = work[:, local_tri[:, 2], :]
        return 0.5 * np.abs(
            (b[:, :, 0] - a[:, :, 0]) * (c[:, :, 1] - a[:, :, 1])
            - (b[:, :, 1] - a[:, :, 1]) * (c[:, :, 0] - a[:, :, 0])
        )

    def soft_bottleneck(values: np.ndarray, temperature: float) -> np.ndarray:
        """
        Stable soft-min score.  The lower tail of the triangle-area spectrum
        controls the score, while temperature permits early global reshaping.
        """
        minimum = np.min(values, axis=1)
        z = np.exp(-(values - minimum[:, None]) / temperature).sum(axis=1)
        return minimum - temperature * np.log(z)

    best_points = None
    best_value = -np.inf

    # Several deterministic initializations are preferable to a single
    # expensive stochastic trajectory for this highly nonsmooth objective.
    # Retain the successful lattice starts, but add structured non-lattice
    # basins.  The objective has many local max-min arrangements and these
    # staggered/rotated patterns explore qualitatively different adjacency
    # graphs before the common local optimizer takes over.
    restarts = 16
    sweeps = 520

    for restart in range(restarts):
        # The first ten starts preserve the established perturbed square-grid
        # family.  Later starts use staggered and rotated grids, whose
        # different near-neighbor graphs are useful for escaping grid-derived
        # local optima.  Jitter deliberately removes all exact collinearities.
        square_base = np.array(
            [[0.22, 0.22], [0.50, 0.22], [0.78, 0.22],
             [0.22, 0.50], [0.50, 0.50], [0.78, 0.50],
             [0.22, 0.78], [0.50, 0.78], [0.78, 0.78]],
            dtype=float,
        )
        hex_base = np.array(
            [[0.25, 0.16], [0.50, 0.16], [0.75, 0.16],
             [0.15, 0.43], [0.50, 0.43], [0.85, 0.43],
             [0.25, 0.70], [0.50, 0.70], [0.75, 0.70]],
            dtype=float,
        )

        if restart < 10:
            base = square_base
            jitter_scale = 0.125
        elif restart < 13:
            base = hex_base
            jitter_scale = 0.090
        else:
            # Rotating the square grid about its center gives a further
            # structured family while remaining comfortably inside the box.
            angle = (restart - 12) * np.pi / 18.0
            rotation = np.array(
                [[np.cos(angle), -np.sin(angle)],
                 [np.sin(angle), np.cos(angle)]],
                dtype=float,
            )
            base = (square_base - 0.5) @ rotation.T + 0.5
            jitter_scale = 0.080

        jitter = rng.uniform(-jitter_scale, jitter_scale, size=(9, 2))
        points = np.vstack((corners, np.clip(base + jitter, 0.045, 0.955)))
        areas = triangle_areas(points)

        for sweep in range(sweeps):
            progress = sweep / max(1, sweeps - 1)
            # Large early moves escape lattice-like arrangements; late moves
            # resolve the bottleneck triangles precisely.
            step = 0.105 * (1.0 - progress) ** 1.65 + 0.0015
            temperature = 0.010 * (1.0 - progress) ** 2.0 + 0.00022

            # Cycle through a randomly permuted set of movable coordinates.
            for p in rng.permutation(np.arange(4, n)):
                current = points[p].copy()

                # Include the present location so every coordinate update is
                # non-destructive with respect to the current smooth score.
                proposals = np.empty((13, 2), dtype=float)
                proposals[0] = current

                directions = rng.normal(size=(12, 2))
                directions /= np.maximum(
                    np.linalg.norm(directions, axis=1, keepdims=True), 1e-12
                )
                radii = step * (0.25 + 1.15 * rng.random(12))
                proposals[1:] = current + directions * radii[:, None]

                # Occasional broad deterministic proposals prevent individual
                # coordinates from becoming permanently trapped.
                if sweep < sweeps // 3:
                    proposals[-2:] = rng.uniform(0.075, 0.925, size=(2, 2))

                proposals = np.clip(proposals, 0.035, 0.965)

                local = incident_areas(proposals, p)
                mask = np.ones(len(tri), dtype=bool)
                mask[incident[p]] = False
                fixed = areas[mask]

                all_values = np.empty((len(proposals), len(tri)), dtype=float)
                all_values[:, mask] = fixed
                all_values[:, incident[p]] = local

                # The evaluated objective is the strict smallest area.  Use it
                # as the dominant coordinate-selection criterion; otherwise a
                # soft-min improvement can silently reduce the true bottleneck.
                candidate_min = np.minimum(
                    np.min(local, axis=1),
                    np.min(fixed),
                )
                maximum_min = float(np.max(candidate_min))

                # Among essentially equal max-min alternatives, the smooth
                # lower-tail score encourages future improvements by enlarging
                # several currently tight triangles rather than just one.
                tied = candidate_min >= maximum_min - 2.0e-7
                scores = soft_bottleneck(all_values, temperature)
                scores[~tied] = -np.inf
                choice = int(np.argmax(scores))

                points[p] = proposals[choice]
                areas[incident[p]] = local[choice]

        # Final strict max-min coordinate polishing.  At this point the
        # objective is the actual evaluation metric rather than its surrogate.
        for polish_step in (0.008, 0.004, 0.002, 0.0008):
            for _ in range(4):
                for p in range(4, n):
                    current = points[p].copy()
                    proposals = np.empty((49, 2), dtype=float)
                    proposals[0] = current

                    angles = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
                    proposals[1:] = current + polish_step * np.column_stack(
                        (np.cos(angles), np.sin(angles))
                    )
                    proposals = np.clip(proposals, 0.035, 0.965)

                    local = incident_areas(proposals, p)
                    mask = np.ones(len(tri), dtype=bool)
                    mask[incident[p]] = False

                    candidate_min = np.minimum(
                        np.min(local, axis=1),
                        np.min(areas[mask]),
                    )
                    choice = int(np.argmax(candidate_min))

                    if candidate_min[choice] >= np.min(areas) - 1e-15:
                        points[p] = proposals[choice]
                        areas[incident[p]] = local[choice]

        value = float(np.min(areas))
        if value > best_value:
            best_value = value
            best_points = points.copy()

    # Defensive fallback is never normally reached, but preserves the required
    # output shape if an unexpected numerical failure occurs.
    if best_points is None or not np.all(np.isfinite(best_points)):
        return np.vstack((corners, np.full((9, 2), 0.5, dtype=float)))

    return best_points


# EVOLVE-BLOCK-END