import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """Seeded bottleneck annealing with a deterministic epigraph final polish."""
    n = 11
    rng = np.random.default_rng(11011)
    height = np.sqrt(3.0) / 2.0
    root3 = np.sqrt(3.0)

    triples = np.array(
        [(i, j, k) for i in range(n) for j in range(i + 1, n)
         for k in range(j + 1, n)],
        dtype=np.int32,
    )

    def sample_points() -> np.ndarray:
        u = rng.random((n, 2))
        u[u[:, 0] + u[:, 1] > 1.0] = 1.0 - u[u[:, 0] + u[:, 1] > 1.0]
        return u[:, 0:1] * np.array([1.0, 0.0]) + u[:, 1:2] * np.array([0.5, height])

    def signed_double_areas(points: np.ndarray) -> np.ndarray:
        a = points[triples[:, 0]]
        b = points[triples[:, 1]]
        c = points[triples[:, 2]]
        return ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))

    def minimum_area(points: np.ndarray) -> float:
        return 0.5 * float(np.min(np.abs(signed_double_areas(points))))

    def inside(points: np.ndarray) -> bool:
        x = points[:, 0]
        y = points[:, 1]
        return bool(
            np.all(x >= -1e-10)
            and np.all(x <= 1.0 + 1e-10)
            and np.all(y >= -1e-10)
            and np.all(y <= height + 1e-10)
            and np.all(y <= root3 * x + 1e-10)
            and np.all(y <= root3 * (1.0 - x) + 1e-10)
        )

    best_points = None
    best_value = -1.0

    for _restart in range(45):
        points = sample_points()
        value = minimum_area(points)

        for iteration in range(24000):
            fraction = iteration / 24000.0
            temperature = 0.0022 * (1.0 - fraction) ** 2 + 2.0e-9
            step = 0.13 * (1.0 - fraction) + 0.0010

            areas = 0.5 * np.abs(signed_double_areas(points))

            # Fifteen percent of moves are collective affine proposals on a
            # connected active cloud; the remaining 85 percent retain the
            # original bottleneck-directed one-point proposal.
            if rng.random() < 0.15:
                active_count = min(24, len(areas))
                active = np.argpartition(
                    areas, active_count - 1
                )[:active_count]

                seed = int(active[rng.integers(active_count)])
                seed_triangle = triples[seed]
                cloud = [int(v) for v in seed_triangle]
                cloud_set = set(cloud)

                # Add vertices only from active triangles incident to the
                # current cloud.  Thus every added point belongs to the same
                # active hypergraph component, with no unrelated filler.
                remaining = list(
                    int(i) for i in rng.permutation(active_count)
                )
                changed = True
                while len(cloud) < 6 and changed:
                    changed = False
                    for position in remaining:
                        triangle = triples[int(active[position])]
                        if not any(int(v) in cloud_set for v in triangle):
                            continue
                        for vertex in triangle:
                            vertex = int(vertex)
                            if vertex not in cloud_set:
                                cloud_set.add(vertex)
                                cloud.append(vertex)
                                changed = True
                                if len(cloud) == 6:
                                    break
                        if len(cloud) == 6:
                            break

                cloud_indices = np.asarray(cloud, dtype=np.int32)
                center = np.mean(points[cloud_indices], axis=0)
                relative = points[cloud_indices] - center

                # Trace-zero M gives a first-order shear/anisotropic change;
                # the common rotation and translation remain deliberately
                # small so active constraints are not destroyed wholesale.
                affine_scale = 0.16 * step
                m00 = float(rng.normal(0.0, affine_scale))
                m01 = float(rng.normal(0.0, affine_scale))
                m10 = float(rng.normal(0.0, affine_scale))
                matrix = np.array(
                    [[m00, m01], [m10, -m00]], dtype=float
                )

                angle = float(rng.normal(0.0, 0.12 * step))
                cs = np.cos(angle)
                sn = np.sin(angle)
                rotation = np.array(
                    [[cs, -sn], [sn, cs]], dtype=float
                )
                translation = rng.normal(0.0, 0.18 * step, 2)

                # p' = c + R (I + M)(p-c) + d.
                transformed = (
                    center
                    + relative @ (np.eye(2) + matrix).T @ rotation.T
                    + translation
                )

                candidate = points.copy()
                candidate[cloud_indices] = transformed

                moved = candidate[cloud_indices]
                x = moved[:, 0]
                y = moved[:, 1]
                if not bool(
                    np.all(x >= -1e-10)
                    and np.all(x <= 1.0 + 1e-10)
                    and np.all(y >= -1e-10)
                    and np.all(y <= height + 1e-10)
                    and np.all(y <= root3 * x + 1e-10)
                    and np.all(y <= root3 * (1.0 - x) + 1e-10)
                ):
                    continue
            else:
                rank = min(15, len(areas) - 1)
                cutoff = np.partition(areas, rank)[rank]
                choices = np.flatnonzero(areas <= cutoff + 1e-12)
                tri = triples[choices[rng.integers(len(choices))]]
                index = int(tri[rng.integers(3)])

                candidate = points.copy()
                candidate[index] += rng.normal(0.0, step, 2)
                if not inside(candidate):
                    continue

            candidate_value = minimum_area(candidate)
            delta = candidate_value - value
            if delta >= 0.0 or rng.random() < np.exp(delta / temperature):
                points = candidate
                value = candidate_value
                if value > best_value:
                    best_value = value
                    best_points = points.copy()

    # Deterministic smooth active-sign epigraph continuation.  It is optional:
    # a missing or failed scipy installation leaves the annealing incumbent valid.
    if best_points is not None:
        try:
            from scipy.optimize import minimize

            current = best_points.copy()
            current_value = best_value

            for _pass in range(3):
                signs = np.sign(signed_double_areas(current))
                signs[signs == 0.0] = 1.0
                x0 = np.empty(23, dtype=float)
                x0[:22] = current.ravel()
                # Start infinitesimally inside the epigraph feasible region.
                # This avoids numerical issues from beginning with many
                # determinant constraints exactly active.
                x0[22] = max(0.0, current_value - 1.0e-9)

                def constraints(z):
                    p = z[:22].reshape(n, 2)
                    x = p[:, 0]
                    y = p[:, 1]
                    oriented = 0.5 * signs * signed_double_areas(p) - z[22]
                    boundary = np.concatenate((
                        x, 1.0 - x, y,
                        root3 * x - y,
                        root3 * (1.0 - x) - y,
                    ))
                    return np.concatenate((oriented, boundary))

                result = minimize(
                    lambda z: -z[22],
                    x0,
                    method="SLSQP",
                    constraints={"type": "ineq", "fun": constraints},
                    options={"maxiter": 350, "ftol": 1e-11, "disp": False},
                )
                # SLSQP can return a useful feasible point while reporting
                # iteration-limit or precision-loss status.  Validate the
                # complete candidate independently using all 165 triangles,
                # rather than discarding such an improving point.
                if np.all(np.isfinite(result.x)) and result.x.size == 23:
                    candidate = np.array(
                        result.x[:22].reshape(n, 2), dtype=float, copy=True
                    )
                    candidate_value = minimum_area(candidate)
                    if inside(candidate) and candidate_value > current_value:
                        current = candidate
                        current_value = candidate_value
                        if candidate_value > best_value:
                            best_points = candidate.copy()
                            best_value = candidate_value
        except Exception:
            pass

    return best_points