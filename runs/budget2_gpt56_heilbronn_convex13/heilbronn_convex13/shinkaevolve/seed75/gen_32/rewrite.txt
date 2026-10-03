# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_convex13() -> np.ndarray:
    """
    Deterministically construct 13 points in the unit square.

    The four fixed corners make the convex hull the unit square, hence raw
    minimum triangle area equals the normalized evaluation objective.
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
    tri_count = len(tri)

    incident = [np.flatnonzero(np.any(tri == p, axis=1)) for p in range(n)]
    fixed_indices = []
    local_other = []

    for p in range(n):
        ids = incident[p]
        local_tri = tri[ids]
        others = np.empty((len(ids), 2), dtype=np.intp)
        for r, t in enumerate(local_tri):
            others[r] = t[t != p]
        local_other.append(others)

        mask = np.ones(tri_count, dtype=bool)
        mask[ids] = False
        fixed_indices.append(np.flatnonzero(mask))

    def triangle_areas(pts: np.ndarray) -> np.ndarray:
        a = pts[tri[:, 0]]
        b = pts[tri[:, 1]]
        c = pts[tri[:, 2]]
        return 0.5 * np.abs(
            (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
            - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        )

    def incident_areas(candidates: np.ndarray, point_index: int) -> np.ndarray:
        """
        Evaluate triangles containing point_index without materializing a
        candidate-by-full-point-array tensor.  Each relevant area is simply
        the altitude of the candidate above the edge between the other points.
        """
        other = local_other[point_index]
        u = points[other[:, 0]]
        v = points[other[:, 1]]
        edge = v - u
        displacement = candidates[:, None, :] - u[None, :, :]
        return 0.5 * np.abs(
            edge[None, :, 0] * displacement[:, :, 1]
            - edge[None, :, 1] * displacement[:, :, 0]
        )

    def lower_tail_score(
        local: np.ndarray,
        fixed: np.ndarray,
        temperature: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Strict bottleneck plus stable soft-min tie breaking."""
        local_min = np.min(local, axis=1)
        fixed_min = float(np.min(fixed))
        minimum = np.minimum(local_min, fixed_min)

        z_local = np.exp(
            -(local - minimum[:, None]) / temperature
        ).sum(axis=1)
        z_fixed = np.exp(
            -(fixed[None, :] - minimum[:, None]) / temperature
        ).sum(axis=1)
        return minimum, minimum - temperature * np.log(z_local + z_fixed)

    base = np.array(
        [[0.22, 0.22], [0.50, 0.22], [0.78, 0.22],
         [0.22, 0.50], [0.50, 0.50], [0.78, 0.50],
         [0.22, 0.78], [0.50, 0.78], [0.78, 0.78]],
        dtype=float,
    )

    best_points = None
    best_value = -np.inf
    restarts = 12
    sweeps = 500

    for _ in range(restarts):
        jitter = rng.uniform(-0.125, 0.125, size=(9, 2))
        points = np.vstack((corners, np.clip(base + jitter, 0.035, 0.965)))
        areas = triangle_areas(points)

        for sweep in range(sweeps):
            progress = sweep / max(1, sweeps - 1)
            step = 0.105 * (1.0 - progress) ** 1.65 + 0.0015
            temperature = 0.010 * (1.0 - progress) ** 2.0 + 0.00022

            for p in rng.permutation(np.arange(4, n)):
                current = points[p].copy()
                proposals = np.empty((19, 2), dtype=float)
                proposals[0] = current

                directions = np.empty((14, 2), dtype=float)
                order = np.argsort(areas[incident[p]])

                for slot, local_index in enumerate(order[:4]):
                    t = tri[incident[p][local_index]]
                    other = t[t != p]
                    edge = points[other[1]] - points[other[0]]
                    normal = np.array([-edge[1], edge[0]], dtype=float)
                    if np.cross(edge, current - points[other[0]]) < 0.0:
                        normal = -normal
                    directions[slot] = normal

                delta = points - current
                distances = np.linalg.norm(delta, axis=1)
                distances[p] = np.inf
                for slot, neighbor in enumerate(np.argsort(distances)[:3], start=4):
                    directions[slot] = current - points[neighbor]

                directions[7:] = rng.normal(size=(7, 2))
                directions /= np.maximum(
                    np.linalg.norm(directions, axis=1, keepdims=True), 1e-12
                )
                radii = step * (0.22 + 1.18 * rng.random(14))
                proposals[1:15] = current + directions * radii[:, None]

                proposals[15:] = np.array(
                    [[0.0, current[1]], [1.0, current[1]],
                     [current[0], 0.0], [current[0], 1.0]],
                    dtype=float,
                )

                if sweep < sweeps // 3:
                    proposals[13:15] = rng.uniform(0.0, 1.0, size=(2, 2))

                proposals = np.clip(proposals, 0.0, 1.0)

                local = incident_areas(proposals, p)
                fixed = areas[fixed_indices[p]]
                candidate_min, scores = lower_tail_score(
                    local, fixed, temperature
                )

                maximum = float(np.max(candidate_min))
                scores[candidate_min < maximum - 2.0e-7] = -np.inf
                choice = int(np.argmax(scores))

                points[p] = proposals[choice]
                areas[incident[p]] = local[choice]

        for _ in range(42):
            critical = tri[int(np.argmin(areas))]
            movable = critical[critical >= 4]

            if len(movable) < 2:
                tail = tri[np.argsort(areas)[:14]].ravel()
                movable = np.unique(tail[tail >= 4])
            if len(movable) < 2:
                continue

            p, q = movable[:2]
            current_value = float(np.min(areas))
            proposals = np.empty((29, 2, 2), dtype=float)
            proposals[0] = np.array([points[p], points[q]])

            directions = rng.normal(size=(28, 2, 2))
            directions /= np.maximum(
                np.linalg.norm(directions, axis=2, keepdims=True), 1e-12
            )
            radii = rng.uniform(0.0012, 0.019, size=(28, 2, 1))
            proposals[1:] = np.array([points[p], points[q]]) + directions * radii

            shared = rng.normal(size=(14, 2))
            shared /= np.maximum(
                np.linalg.norm(shared, axis=1, keepdims=True), 1e-12
            )
            shared *= rng.uniform(0.0012, 0.017, size=(14, 1))
            proposals[1:15, 0] = points[p] + shared
            proposals[1:15, 1] = points[q] + shared
            proposals = np.clip(proposals, 0.0, 1.0)

            work = np.broadcast_to(points, (len(proposals), n, 2)).copy()
            work[:, p] = proposals[:, 0]
            work[:, q] = proposals[:, 1]

            a = work[:, tri[:, 0]]
            b = work[:, tri[:, 1]]
            c = work[:, tri[:, 2]]
            trial_areas = 0.5 * np.abs(
                (b[:, :, 0] - a[:, :, 0]) * (c[:, :, 1] - a[:, :, 1])
                - (b[:, :, 1] - a[:, :, 1]) * (c[:, :, 0] - a[:, :, 0])
            )
            trial_min = np.min(trial_areas, axis=1)
            choice = int(np.argmax(trial_min))

            if trial_min[choice] >= current_value - 1e-15:
                points[p] = proposals[choice, 0]
                points[q] = proposals[choice, 1]
                areas = trial_areas[choice]

        angles = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
        unit_circle = np.column_stack((np.cos(angles), np.sin(angles)))

        for polish_step in (0.009, 0.0045, 0.002, 0.0008):
            for _ in range(4):
                changed = False
                for p in range(4, n):
                    current = points[p].copy()
                    proposals = np.empty((49, 2), dtype=float)
                    proposals[0] = current
                    proposals[1:] = current + polish_step * unit_circle
                    proposals = np.clip(proposals, 0.0, 1.0)

                    local = incident_areas(proposals, p)
                    fixed = areas[fixed_indices[p]]
                    candidate_min = np.minimum(
                        np.min(local, axis=1), np.min(fixed)
                    )
                    choice = int(np.argmax(candidate_min))

                    if candidate_min[choice] >= np.min(areas) - 1e-15:
                        if choice != 0:
                            changed = True
                        points[p] = proposals[choice]
                        areas[incident[p]] = local[choice]
                if not changed:
                    break

        value = float(np.min(areas))
        if value > best_value:
            best_value = value
            best_points = points.copy()

    if best_points is None or not np.all(np.isfinite(best_points)):
        return np.vstack((corners, np.full((9, 2), 0.5, dtype=float)))

    return best_points


# EVOLVE-BLOCK-END