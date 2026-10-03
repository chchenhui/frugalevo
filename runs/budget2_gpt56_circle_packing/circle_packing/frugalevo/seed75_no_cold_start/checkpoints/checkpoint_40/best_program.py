"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """Optimize eight deterministic six-row, variable-radius packing seeds."""
    # Keep the original construction as a guaranteed-valid fallback.
    fallback_r = 0.1
    fallback = np.array(
        [[x, y] for y in (0.1, 0.3, 0.5, 0.7, 0.9)
                  for x in (0.1, 0.3, 0.5, 0.7, 0.9)],
        dtype=float,
    )
    small = 0.1 * (3.0 - 2.0 * np.sqrt(2.0)) - 1.0e-8
    fallback = np.vstack((fallback, [[small, small]]))
    fallback_radii = np.r_[np.full(25, fallback_r, dtype=float), small]
    best = fallback.copy()
    best_radii = fallback_radii.copy()
    best_sum = float(best_radii.sum())

    # The six rows have 4,5,4,5,4,4 circles, respectively.  Alternating
    # offsets expose diagonal contacts instead of restricting the search to
    # the incumbent's square contact graph.
    try:
        from scipy.optimize import minimize

        # Generate six boundary-aware staggered triangular starts.  Each
        # layout is relaxed by fixed inverse-cubic repulsion from both the
        # other centers and their four mirrored wall images.  The final
        # displaced site creates a controlled odd-count dislocation.
        seeds = []
        base = []
        for row in range(5):
            y = 0.105 + 0.198 * row
            offset = 0.0 if row % 2 == 0 else 0.099
            count = 5 if row % 2 == 0 else 5
            for col in range(count):
                base.append((0.105 + 0.198 * col + offset, y))
        base.append((0.515, 0.515))
        base = np.asarray(base, dtype=float)
        base[-1] = base[12] + np.array([0.061, -0.047], dtype=float)

        transforms = (
            (0.000, 1.00, 1.00),
            (0.035, 0.98, 1.02),
            (-0.035, 1.02, 0.98),
            (0.070, 0.96, 1.04),
            (-0.070, 1.04, 0.96),
            (0.120, 1.01, 0.99),
        )
        for angle, sx, sy in transforms:
            ca, sa = np.cos(angle), np.sin(angle)
            linear = np.array(
                [[ca * sx, -sa * sy], [sa * sx, ca * sy]],
                dtype=float,
            )
            points = 0.5 + (base - 0.5) @ linear.T
            points = np.clip(points, 0.035, 0.965)

            for _ in range(90):
                delta = points[:, None, :] - points[None, :, :]
                dist2 = np.sum(delta * delta, axis=2) + np.eye(26)
                inv3 = 1.0 / np.maximum(dist2, 1.0e-5) ** 1.5
                # Include the second neighbor shell so the relaxed sites
                # develop triangular cells rather than only eliminating
                # the most obvious local crowding.
                pair_force = np.sum(
                    delta * inv3[:, :, None] * (dist2 < 0.19)[:, :, None],
                    axis=1,
                )

                image_force = np.zeros_like(points)
                for axis in (0, 1):
                    for side in (0.0, 1.0):
                        image = points.copy()
                        image[:, axis] = 2.0 * side - image[:, axis]
                        d = points - image
                        d2 = np.sum(d * d, axis=1) + 1.0e-5
                        # A slightly stronger image force forms a useful
                        # boundary belt without pinning sites to the walls.
                        image_force += 0.022 * d / d2[:, None] ** 1.5

                step = pair_force + image_force
                scale = np.maximum(
                    1.0, np.max(np.linalg.norm(step, axis=1))
                )
                points = np.clip(
                    points + 0.0022 * step / scale,
                    0.035,
                    0.965,
                )

            clearance = np.min(
                np.column_stack(
                    (points[:, 0], points[:, 1],
                     1.0 - points[:, 0], 1.0 - points[:, 1])
                ),
                axis=1,
            )
            d = points[:, None, :] - points[None, :, :]
            d2 = np.sum(d * d, axis=2) + np.eye(26)
            nearest = np.sqrt(np.min(d2, axis=1))
            rr = 0.45 * np.minimum(clearance, 0.5 * nearest)
            rr = np.maximum(rr, 1.0e-5)
            seeds.append(np.r_[points.ravel(), rr])

        def constraints(z):
            """Return wall and pairwise squared-distance inequalities."""
            p = z[:52].reshape(26, 2)
            r = z[52:]
            out = [p[:, 0] - r, p[:, 1] - r,
                   1.0 - p[:, 0] - r, 1.0 - p[:, 1] - r]
            for i in range(26):
                d = p[i + 1:] - p[i]
                out.append(np.einsum("ij,ij->i", d, d) -
                           (r[i] + r[i + 1:]) ** 2 - 2.0e-9)
            return np.concatenate(out)

        for seed in seeds:
            result = minimize(
                lambda z: -float(np.sum(z[52:])),
                seed,
                method="SLSQP",
                bounds=[(0.0, 1.0)] * 52 + [(1.0e-5, 0.25)] * 26,
                constraints={"type": "ineq", "fun": constraints},
                options={"maxiter": 450, "ftol": 2.0e-9, "disp": False},
            )
            if not result.success:
                continue
            z = np.asarray(result.x, dtype=float)
            p, r = z[:52].reshape(26, 2), z[52:].copy()
            slack = np.min(constraints(z))
            if slack < 0.0:
                r *= max(0.0, 1.0 + slack / max(1.0e-12, r.max()))
                z[52:] = r
            if float(r.sum()) > best_sum and np.min(constraints(z)) >= -2.0e-8:
                best, best_radii, best_sum = p.copy(), r.copy(), float(r.sum())

        # Replace regular-hexagon restarts with three weighted seven-circle
        # void patches.  Each patch freezes the other 19 circles, explores
        # wall-offset and weighted-bisector candidates, then receives one
        # bounded local solve followed by at most one full polishing solve.
        patch_centers = [
            int(np.argmin(best_radii)),
            int(np.argmin(np.sum((best - np.array([0.5, 0.5])) ** 2, axis=1))),
            int(np.argmax(np.min(
                np.column_stack((best[:, 0], best[:, 1],
                                 1.0 - best[:, 0], 1.0 - best[:, 1])),
                axis=1))),
        ]
        tried = set()

        for cavity in patch_centers:
            if cavity in tried:
                continue
            tried.add(cavity)

            distance = np.sqrt(np.maximum(
                np.sum((best - best[cavity]) ** 2, axis=1), 1.0e-24))
            selected = np.argsort(distance)[:7]
            selected_set = set(int(i) for i in selected)
            exterior = np.asarray(
                [i for i in range(26) if i not in selected_set], dtype=int
            )

            # Candidate locations are intersections of offset wall lines and
            # weighted bisector samples.  The latter use the exact radical
            # axis direction and radius-weighted interpolation between two
            # exterior circles, rather than an unweighted Cartesian grid.
            candidates = []
            wall_levels = (0.075, 0.14, 0.23, 0.34, 0.50)
            for a in wall_levels:
                for b in wall_levels:
                    candidates.extend((
                        np.array([a, b], dtype=float),
                        np.array([1.0 - a, b], dtype=float),
                        np.array([a, 1.0 - b], dtype=float),
                        np.array([1.0 - a, 1.0 - b], dtype=float),
                    ))
            for u in range(min(10, exterior.size)):
                i = int(exterior[u])
                for v in range(u + 1, min(10, exterior.size)):
                    j = int(exterior[v])
                    pi, pj = best[i], best[j]
                    wi, wj = best_radii[i], best_radii[j]
                    dvec = pj - pi
                    d2 = float(np.dot(dvec, dvec))
                    if d2 < 1.0e-10:
                        continue
                    # Point on the weighted radical axis, plus two normal
                    # displacements to expose both sides of the void.
                    t = 0.5 + (wi * wi - wj * wj) / (2.0 * d2)
                    mid = pi + t * dvec
                    normal = np.array([-dvec[1], dvec[0]]) / np.sqrt(d2)
                    for off in (-0.18, -0.08, 0.08, 0.18):
                        candidates.append(np.clip(mid + off * normal,
                                                   0.025, 0.975))

            scored = []
            for q in candidates:
                wall = min(q[0], q[1], 1.0 - q[0], 1.0 - q[1])
                if exterior.size:
                    gap = np.sqrt(np.sum((best[exterior] - q) ** 2,
                                         axis=1)) - best_radii[exterior]
                    wall = min(wall, float(np.min(gap)))
                scored.append((wall, q))
            scored.sort(key=lambda item: item[0], reverse=True)

            chosen = []
            for _, q in scored:
                if all(np.linalg.norm(q - old) > 0.055 for old in chosen):
                    chosen.append(q.copy())
                if len(chosen) == 7:
                    break
            if len(chosen) < 7:
                chosen = [best[i].copy() for i in selected]
            chosen = np.asarray(chosen[:7], dtype=float)

            # Start each weighted-Voronoi site near its largest locally
            # feasible radius.  The 0.82 factor leaves numerical slack for
            # the first SLSQP step while avoiding the overly small circles
            # produced by the former 0.30 factor.
            initial_r = np.empty(7, dtype=float)
            for k, q in enumerate(chosen):
                limit = min(q[0], q[1], 1.0 - q[0], 1.0 - q[1])
                if exterior.size:
                    gap = np.sqrt(np.sum((best[exterior] - q) ** 2,
                                         axis=1)) - best_radii[exterior]
                    limit = min(limit, float(np.min(gap)))
                # Permit the weighted patch to seed one enlarged void-filling
                # circle; the SLSQP constraints still enforce every wall and
                # exterior-circle clearance exactly.
                initial_r[k] = max(1.0e-5, min(0.16, 0.82 * limit))

            def patch_constraints(v):
                """Return wall, frozen-exterior, and internal patch clearances."""
                q = v[:14].reshape(7, 2)
                rr = v[14:]
                out = [q[:, 0] - rr, q[:, 1] - rr,
                       1.0 - q[:, 0] - rr, 1.0 - q[:, 1] - rr]
                for k in range(7):
                    if exterior.size:
                        d = best[exterior] - q[k]
                        out.append(np.sum(d * d, axis=1) -
                                   (rr[k] + best_radii[exterior]) ** 2 -
                                   2.0e-9)
                    if k:
                        d = q[:k] - q[k]
                        out.append(np.sum(d * d, axis=1) -
                                   (rr[k] + rr[:k]) ** 2 - 2.0e-9)
                return np.concatenate(out)

            local = minimize(
                lambda v: -float(np.sum(v[14:])),
                np.r_[chosen.ravel(), initial_r],
                method="SLSQP",
                bounds=[(1.0e-5, 1.0 - 1.0e-5)] * 14 +
                       [(1.0e-5, 0.25)] * 7,
                constraints={"type": "ineq", "fun": patch_constraints},
                options={"maxiter": 180, "ftol": 2.0e-9, "disp": False},
            )
            if not local.success or np.min(patch_constraints(local.x)) < -2.0e-8:
                continue

            candidate_p = best.copy()
            candidate_r = best_radii.copy()
            candidate_p[selected] = local.x[:14].reshape(7, 2)
            candidate_r[selected] = local.x[14:]
            polish = minimize(
                lambda z: -float(np.sum(z[52:])),
                np.r_[candidate_p.ravel(), candidate_r],
                method="SLSQP",
                bounds=[(0.0, 1.0)] * 52 + [(1.0e-5, 0.25)] * 26,
                constraints={"type": "ineq", "fun": constraints},
                options={"maxiter": 350, "ftol": 2.0e-9, "disp": False},
            )
            if polish.success:
                z = np.asarray(polish.x, dtype=float)
                if np.min(constraints(z)) >= -2.0e-8:
                    value = float(np.sum(z[52:]))
                    if value > best_sum:
                        best = z[:52].reshape(26, 2).copy()
                        best_radii = z[52:].copy()
                        best_sum = value

        return best, best_radii, best_sum

        # Local six-circle Delaunay cavity retessellation.  Each cavity
        # deletes one stressed circle and its five nearest neighbors,
        # reseeds those six sites as a rotated hexagon, and then releases
        # all center/radius variables in one bounded constrained polish.
        # The old incumbent is retained unless the complete result is valid
        # and strictly improves its radius sum.
        cavity_scores = [
            int(np.argmin(best_radii)),
            int(np.argmin(np.sum(
                np.maximum(
                    np.sqrt(np.maximum(
                        np.sum((best[:, None, :] - best[None, :, :]) ** 2,
                               axis=2), 1.0e-24))
                        - best_radii[:, None] - best_radii[None, :],
                    0.0), axis=1))),
            int(np.argmin(np.sum((best - np.array([0.5, 0.5])) ** 2, axis=1))),
        ]
        seen = set()
        for cavity, angle in zip(cavity_scores, (0.0, np.pi / 12.0,
                                                  np.pi / 6.0)):
            if cavity in seen:
                continue
            seen.add(cavity)

            delta = best - best[cavity]
            distances = np.sqrt(np.maximum(np.sum(delta * delta, axis=1),
                                            1.0e-24))
            selected = np.argsort(distances)[:6]
            centroid = np.mean(best[selected], axis=0)
            neighbor_distances = distances[selected[1:]]
            hex_radius = float(np.median(neighbor_distances))
            hex_radius = max(0.055, min(0.145, 0.72 * hex_radius))

            start_p = best.copy()
            for k, index in enumerate(selected):
                theta = angle + 2.0 * np.pi * k / 6.0
                start_p[index] = centroid + hex_radius * np.array(
                    [np.cos(theta), np.sin(theta)], dtype=float)

            # Give the cavity a conservative feasible scale before polishing.
            # For each selected site, both walls and the closest exterior
            # center provide a local upper bound on its initial radius.
            start_r = best_radii.copy()
            for index in selected:
                wall = min(start_p[index, 0], start_p[index, 1],
                           1.0 - start_p[index, 0], 1.0 - start_p[index, 1])
                exterior = np.delete(np.arange(26), selected)
                if exterior.size:
                    gap = np.sqrt(np.sum(
                        (best[exterior] - start_p[index]) ** 2, axis=1))
                    wall = min(wall, 0.5 * float(np.min(gap)))
                start_r[index] = max(1.0e-5, 0.90 * wall)

            start_p = np.clip(start_p, 1.0e-5, 1.0 - 1.0e-5)
            start = np.r_[start_p.ravel(), start_r]
            result = minimize(
                lambda z: -float(np.sum(z[52:])),
                start,
                method="SLSQP",
                bounds=[(0.0, 1.0)] * 52 + [(1.0e-5, 0.25)] * 26,
                constraints={"type": "ineq", "fun": constraints},
                options={"maxiter": 320, "ftol": 2.0e-9, "disp": False},
            )
            if not result.success:
                continue

            z = np.asarray(result.x, dtype=float).copy()
            p = z[:52].reshape(26, 2).copy()
            r = z[52:].copy()
            checked = np.r_[p.ravel(), r]
            if np.min(constraints(checked)) >= -2.0e-8:
                value = float(np.sum(r))
                if value > best_sum:
                    best, best_radii, best_sum = p, r, value
    except Exception:
        pass

    return best, best_radii, best_sum


def compute_max_radii(centers):
    """Compute feasible radii by boundary initialization and pairwise projection."""
    n = centers.shape[0]
    radii = np.min(
        np.column_stack((centers[:, 0], centers[:, 1],
                         1.0 - centers[:, 0], 1.0 - centers[:, 1])),
        axis=1,
    )

    for _ in range(12):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.linalg.norm(centers[i] - centers[j]))
                excess = radii[i] + radii[j] - d
                if excess > 1e-12:
                    total = radii[i] + radii[j]
                    if total > 0.0:
                        radii[i] -= excess * radii[i] / total
                        radii[j] -= excess * radii[j] / total
                        changed = True
        if not changed:
            break
    return np.maximum(radii, 0.0)


def run_packing():
    """Run the circle packing constructor for n=26."""
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii


def visualize(centers, radii):
    """Visualize the circle packing."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True)

    for i, (center, radius) in enumerate(zip(centers, radii)):
        ax.add_patch(Circle(center, radius, alpha=0.5))
        ax.text(center[0], center[1], str(i), ha="center", va="center")

    plt.title(f"Circle Packing (n={len(centers)}, sum={sum(radii):.6f})")
    plt.show()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii}")
    visualize(centers, radii)