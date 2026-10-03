# EVOLVE-BLOCK-START
"""Deterministic multistart constructor for 26 circles in a unit square."""
import numpy as np


def compute_max_radii(centers):
    """Maximum sum radii for fixed centers, subject to wall and pair constraints."""
    n = len(centers)
    try:
        from scipy.optimize import linprog

        A = []
        b = []
        clearance = np.min(np.column_stack((
            centers[:, 0], centers[:, 1],
            1.0 - centers[:, 0], 1.0 - centers[:, 1]
        )), axis=1)

        for i in range(n):
            row = np.zeros(n)
            row[i] = 1.0
            A.append(row)
            b.append(clearance[i])

        for i in range(n):
            for j in range(i + 1, n):
                row = np.zeros(n)
                row[i] = row[j] = 1.0
                A.append(row)
                b.append(float(np.linalg.norm(centers[i] - centers[j])))

        ans = linprog(
            -np.ones(n), A_ub=np.asarray(A), b_ub=np.asarray(b),
            bounds=[(0.0, None)] * n, method="highs"
        )
        if ans.success:
            return np.maximum(ans.x, 0.0)
    except Exception:
        pass

    r = np.min(np.column_stack((
        centers[:, 0], centers[:, 1],
        1.0 - centers[:, 0], 1.0 - centers[:, 1]
    )), axis=1)
    for _ in range(60):
        changed = False
        for i in range(n):
            for j in range(i):
                d = float(np.linalg.norm(centers[i] - centers[j]))
                excess = r[i] + r[j] - d
                if excess > 0.0:
                    total = r[i] + r[j]
                    if total > 0.0:
                        r[i] -= excess * r[i] / total
                        r[j] -= excess * r[j] / total
                        changed = True
        if not changed:
            break
    return np.maximum(r, 0.0)


def _sanitize(centers, radii):
    """Make a candidate safely evaluator-feasible using reductions only."""
    centers = np.asarray(centers, dtype=float).copy()
    centers = np.clip(centers, 1e-7, 1.0 - 1e-7)
    r = np.maximum(np.asarray(radii, dtype=float).copy(), 0.0)
    wall = np.min(np.column_stack((
        centers[:, 0], centers[:, 1],
        1.0 - centers[:, 0], 1.0 - centers[:, 1]
    )), axis=1)
    r = np.minimum(r, np.maximum(0.0, wall - 5e-8))

    for _ in range(3):
        for i in range(len(r)):
            for j in range(i):
                d = float(np.linalg.norm(centers[i] - centers[j]))
                allowed = max(0.0, d - 5e-8)
                if r[i] + r[j] > allowed:
                    # Reducing one radius cannot create a new overlap.
                    r[i] = max(0.0, allowed - r[j])
    return centers, r


def _row_seed(shear=0.0, phase=0.0, tilt=0.0):
    """A centered six-band seed, unlike the old strongly one-sided 5-circle rows."""
    counts = (4, 5, 4, 5, 4, 4)
    ys = np.array((0.085, 0.250, 0.415, 0.580, 0.745, 0.910))
    points = []
    for k, count in enumerate(counts):
        if count == 4:
            xs = np.array((0.185, 0.395, 0.605, 0.815))
        else:
            xs = np.array((0.140, 0.320, 0.500, 0.680, 0.860))
        # Opposite phases make the five-circle interfaces genuinely staggered.
        xs = xs + (phase if k % 2 else -phase)
        for x in xs:
            xx = x + shear * (ys[k] - 0.5)
            yy = ys[k] + tilt * (x - 0.5)
            points.append((xx, yy))
    return np.clip(np.asarray(points), 0.015, 0.985)


def _belt_seed(phase=0.0):
    """A distinct boundary-belt / interior-core topology."""
    p = []
    for x in (0.12, 0.37, 0.63, 0.88):
        p.append((x + phase, 0.075))
        p.append((x - phase, 0.925))
    for y in (0.23, 0.42, 0.61, 0.80):
        p.append((0.075, y))
        p.append((0.925, y))
    for x in (0.28, 0.50, 0.72):
        p.append((x - phase, 0.32))
    for x in (0.20, 0.40, 0.60, 0.80):
        p.append((x + phase, 0.50))
    for x in (0.28, 0.50, 0.72):
        p.append((x - phase, 0.68))
    return np.clip(np.asarray(p), 0.015, 0.985)


def construct_packing():
    """Generate six deterministic jammed seeds, then polish the best three with SLSQP."""
    n = 26
    rng_seeds = (11, 29, 47, 71, 101, 149)

    def annealed_seed(seed):
        """Anneal a staggered lattice using LP radius value and topology penalties."""
        rng = np.random.default_rng(seed)

        # Start from a staggered 5-by-5 interior lattice, remove its
        # least useful central point, then add two boundary-void points.
        p = []
        for iy in range(5):
            for ix in range(5):
                x = 0.14 + 0.18 * ix + (0.045 if iy % 2 else 0.0)
                y = 0.14 + 0.18 * iy
                p.append((x, y))
        p = np.asarray(p, dtype=float)
        remove = int(np.argmin(np.sum((p - 0.5) ** 2, axis=1)))
        p = np.delete(p, remove, axis=0)
        p = np.vstack((p, [[0.08, 0.50], [0.92, 0.50]]))

        def value(q):
            """Return LP radius sum with a penalty for dangerously short edges."""
            d = q[:, None, :] - q[None, :, :]
            d = np.sqrt(np.sum(d * d, axis=2) + np.eye(n))
            short = d[np.triu_indices(n, 1)]
            penalty = np.sum(np.maximum(0.025 - short, 0.0) ** 2) * 18.0
            return float(np.sum(compute_max_radii(q)) - penalty)

        p = np.clip(p, 0.035, 0.965)
        current = value(p)
        best = p.copy()
        best_value = current

        for step in range(420):
            temperature = 0.012 * (0.0004 / 0.012) ** (step / 419.0)
            trial = p.copy()

            # Bias single-circle moves toward the currently smallest LP radius.
            radii = compute_max_radii(p)
            order = np.argsort(radii)
            i = int(order[(step + seed) % min(8, n)])
            angle = rng.uniform(0.0, 2.0 * np.pi)
            scale = 0.040 * (1.0 - step / 500.0)
            trial[i] += scale * np.array((np.cos(angle), np.sin(angle)))

            # Periodically move both ends of the shortest non-contact edge.
            if step % 35 == 0:
                d = trial[:, None, :] - trial[None, :, :]
                dist = np.sqrt(np.sum(d * d, axis=2) + np.eye(n))
                radii_now = compute_max_radii(trial)
                gap = dist - radii_now[:, None] - radii_now[None, :]
                gap += np.eye(n) * 10.0
                a, b = np.unravel_index(np.argmin(np.where(
                    gap > 0.025, gap, 10.0)), gap.shape)
                if a != b and gap[a, b] < 10.0:
                    direction = trial[b] - trial[a]
                    length = max(float(np.linalg.norm(direction)), 1e-9)
                    direction /= length
                    trial[a] -= 0.012 * direction
                    trial[b] += 0.012 * direction

            trial = np.clip(trial, 0.025, 0.975)
            trial_value = value(trial)
            delta = trial_value - current
            if delta >= 0.0 or rng.random() < np.exp(delta / max(temperature, 1e-9)):
                p = trial
                current = trial_value
                if current > best_value:
                    best = p.copy()
                    best_value = current

        return best

    seeds = [annealed_seed(s) for s in rng_seeds]
    fallback_centers = seeds[0]
    fallback_radii = compute_max_radii(fallback_centers)
    best_c, best_r = _sanitize(fallback_centers, fallback_radii)
    best_value = float(np.sum(best_r))

    try:
        from scipy.optimize import minimize

        def objective(z):
            return -float(np.sum(z[2 * n:]))

        def inequalities(z):
            p = z[:2 * n].reshape(n, 2)
            r = z[2 * n:]
            parts = [p[:, 0] - r, p[:, 1] - r,
                     1.0 - p[:, 0] - r, 1.0 - p[:, 1] - r]
            for i in range(n - 1):
                d = p[i + 1:] - p[i]
                parts.append(np.sum(d * d, axis=1) -
                             (r[i] + r[i + 1:]) ** 2)
            return np.concatenate(parts)

        ranked = []
        for p in seeds:
            rr = compute_max_radii(p)
            ranked.append((float(np.sum(rr)), np.asarray(p, dtype=float).copy(), rr))
        ranked.sort(key=lambda x: x[0], reverse=True)

        # Alternate exact fixed-center radius optimization with center-only
        # clearance ascent, then perform one final joint polish.
        for _, p, rr in ranked[:4]:
            c = np.asarray(p, dtype=float).copy()
            r = np.asarray(rr, dtype=float).copy()

            def center_slack(z):
                """Maximize the common additive clearance while radii stay fixed."""
                q = z[:2 * n].reshape(n, 2)
                t = z[-1]
                # Use clearance in radius units rather than squared-distance
                # units; this gives wall and pair constraints comparable
                # derivatives and makes the auxiliary slack meaningful.
                parts = [
                    q[:, 0] - r - t,
                    q[:, 1] - r - t,
                    1.0 - q[:, 0] - r - t,
                    1.0 - q[:, 1] - r - t,
                ]
                for i in range(n - 1):
                    d = q[i + 1:] - q[i]
                    dist = np.sqrt(np.sum(d * d, axis=1) + 1e-14)
                    parts.append(
                        dist - (r[i] + r[i + 1:]) - t
                    )
                return np.concatenate(parts)

            for _ in range(3):
                # Recompute the globally optimal radii before moving centers.
                r = compute_max_radii(c)
                zc = np.concatenate((c.ravel(), np.array([0.0])))
                center_result = minimize(
                    lambda z: -z[-1],
                    zc,
                    method="SLSQP",
                    bounds=[(1e-6, 1.0 - 1e-6)] * (2 * n) +
                           [(-0.05, 0.05)],
                    constraints={"type": "ineq", "fun": center_slack},
                    options={"maxiter": 180, "ftol": 1e-9, "disp": False},
                )
                if center_result.success or center_result.x is not None:
                    c = center_result.x[:2 * n].reshape(n, 2).copy()

            r = compute_max_radii(c)
            z0 = np.concatenate((c.ravel(), np.maximum(r, 1e-7)))
            result = minimize(
                objective, z0, method="SLSQP",
                bounds=[(1e-6, 1.0 - 1e-6)] * (2 * n) +
                       [(0.0, 0.5)] * n,
                constraints={"type": "ineq", "fun": inequalities},
                options={"maxiter": 420, "ftol": 2e-10, "disp": False},
            )
            if result.x is None:
                continue

            def polish_release(q, limit):
                """Polish a released contact configuration with bounded joint SLSQP."""
                q = np.asarray(q, dtype=float).reshape(n, 2).copy()
                qr = compute_max_radii(q)
                start = np.concatenate((q.ravel(), np.maximum(qr, 1e-7)))
                return minimize(
                    objective, start, method="SLSQP",
                    bounds=[(1e-6, 1.0 - 1e-6)] * (2 * n) +
                           [(0.0, 0.5)] * n,
                    constraints={"type": "ineq", "fun": inequalities},
                    options={"maxiter": limit, "ftol": 2e-10, "disp": False},
                )

            # Retain the ordinary incumbent, then perform deterministic
            # Delaunay-diagonal cavity reconstruction.  The cavity solve has
            # frozen exterior centers and radii; the subsequent full polish
            # restores global feasibility and permits the new local topology.
            candidates = [result]
            base = np.asarray(result.x, dtype=float).copy()
            base_centers = base[:2 * n].reshape(n, 2).copy()
            base_radii = np.maximum(base[2 * n:], 1e-9)

            # Replace isolated Delaunay cavities by three horizontal
            # separator patches.  Each patch contains both sides of a
            # high-clearance interface, allowing an entire row boundary to
            # exchange neighbors simultaneously while the exterior is frozen.
            separator_order = np.argsort(
                np.diff(np.sort(base_centers[:, 1]))
            )[::-1][:3]

            for sep_index in separator_order:
                sorted_y = np.sort(base_centers[:, 1])
                separator = 0.5 * (
                    sorted_y[int(sep_index)] +
                    sorted_y[int(sep_index) + 1]
                )

                distance = np.abs(base_centers[:, 1] - separator)
                ids = list(np.argsort(distance)[:10])

                # Include every circle whose current disk reaches the
                # separator, then cap the patch at eleven circles.
                crossing = np.where(
                    np.abs(base_centers[:, 1] - separator) <=
                    base_radii + 0.035
                )[0]
                for k in crossing:
                    if int(k) not in ids:
                        ids.append(int(k))
                ids = np.asarray(ids[:11], dtype=int)

                if len(ids) < 8:
                    continue

                outside = np.asarray(
                    [k for k in range(n) if k not in set(ids)],
                    dtype=int
                )
                local = base_centers.copy()

                # Alternating shifts open a new row-interface topology.
                above = base_centers[ids, 1] >= separator
                local[ids[above], 0] += 0.006
                local[ids[~above], 0] -= 0.006
                local[ids] = np.clip(local[ids], 1e-6, 1.0 - 1e-6)

                local_r = compute_max_radii(local[ids])
                zc = base.copy()
                zc[2 * ids] = local[ids, 0]
                zc[2 * ids + 1] = local[ids, 1]
                zc[2 * n + ids] = np.maximum(local_r, 1e-7)

                def separator_objective(z):
                    """Maximize the radii sum of the frozen-exterior separator patch."""
                    return -float(np.sum(z[2 * n + ids]))

                def separator_constraints(z):
                    """Enforce walls, patch contacts, and all frozen-exterior contacts."""
                    p = z[:2 * n].reshape(n, 2)
                    r = z[2 * n:]
                    parts = [
                        p[ids, 0] - r[ids],
                        p[ids, 1] - r[ids],
                        1.0 - p[ids, 0] - r[ids],
                        1.0 - p[ids, 1] - r[ids],
                    ]

                    for aa in range(len(ids) - 1):
                        i = int(ids[aa])
                        jj = ids[aa + 1:]
                        d = p[jj] - p[i]
                        parts.append(
                            np.sum(d * d, axis=1) -
                            (r[i] + r[jj]) ** 2
                        )

                    if len(outside):
                        for i in ids:
                            d = p[outside] - p[i]
                            parts.append(
                                np.sum(d * d, axis=1) -
                                (r[i] + r[outside]) ** 2
                            )
                    return np.concatenate(parts)

                patch_bounds = [(1e-6, 1.0 - 1e-6)] * (2 * n)
                patch_bounds += [(0.0, 0.5)] * n
                for k in outside:
                    patch_bounds[2 * int(k)] = (
                        float(base_centers[k, 0]),
                        float(base_centers[k, 0])
                    )
                    patch_bounds[2 * int(k) + 1] = (
                        float(base_centers[k, 1]),
                        float(base_centers[k, 1])
                    )
                    patch_bounds[2 * n + int(k)] = (
                        float(base_radii[k]),
                        float(base_radii[k])
                    )

                patch = minimize(
                    separator_objective, zc, method="SLSQP",
                    bounds=patch_bounds,
                    constraints={"type": "ineq",
                                 "fun": separator_constraints},
                    options={"maxiter": 280, "ftol": 2e-10, "disp": False},
                )
                if patch.x is None:
                    continue

                # Release the frozen exterior with one bounded global polish.
                release_start = np.asarray(patch.x, dtype=float).copy()
                released = minimize(
                    objective, release_start, method="SLSQP",
                    bounds=[(1e-6, 1.0 - 1e-6)] * (2 * n) +
                           [(0.0, 0.5)] * n,
                    constraints={"type": "ineq", "fun": inequalities},
                    options={"maxiter": 220, "ftol": 2e-10, "disp": False},
                )
                if released.x is not None:
                    candidates.append(released)

            # The separator patches supersede the former Delaunay-cavity
            # candidates.  Keep the old loop structurally harmless and retain
            # the ordinary incumbent in candidates as a fallback.
            diagonals = []

            try:
                from scipy.spatial import Delaunay

                triangulation = Delaunay(base_centers)
                diagonal_set = set()
                for simplex in triangulation.simplices:
                    for a in range(3):
                        for b in range(a + 1, 3):
                            i, j = sorted((int(simplex[a]), int(simplex[b])))
                            diagonal_set.add((i, j))

                # Prefer short non-contact edges: these are the most likely
                # locations where a local Delaunay flip can improve packing.
                diagonal_order = []
                for i, j in diagonal_set:
                    d = float(np.linalg.norm(
                        base_centers[i] - base_centers[j]
                    ))
                    gap = d - base_radii[i] - base_radii[j]
                    diagonal_order.append((gap, d, i, j))
                diagonal_order.sort()
                diagonals = diagonal_order[:6]
            except Exception:
                # The fallback preserves the same cavity mechanism without
                # requiring scipy.spatial.
                pairs = []
                for i in range(n - 1):
                    for j in range(i + 1, n):
                        d = float(np.linalg.norm(
                            base_centers[i] - base_centers[j]
                        ))
                        gap = d - base_radii[i] - base_radii[j]
                        pairs.append((gap, d, i, j))
                pairs.sort()
                diagonals = pairs[:6]

            for _, _, u, v in diagonals:
                midpoint = 0.5 * (base_centers[u] + base_centers[v])
                cavity = {int(u), int(v)}

                # Include nearby circles around the diagonal midpoint.
                distances = np.linalg.norm(base_centers - midpoint, axis=1)
                for k in np.argsort(distances):
                    if distances[k] <= 0.22 or len(cavity) < 5:
                        cavity.add(int(k))
                    if len(cavity) >= 8:
                        break

                ids = np.asarray(sorted(cavity), dtype=int)
                if len(ids) < 5:
                    continue
                outside = np.asarray(
                    [k for k in range(n) if k not in cavity], dtype=int
                )

                local = base_centers[ids].copy()
                direction = base_centers[v] - base_centers[u]
                length = max(float(np.linalg.norm(direction)), 1e-9)
                normal = np.asarray(
                    (-direction[1], direction[0]), dtype=float
                ) / length
                signed = 0.004 if (int(u) + int(v)) % 2 else -0.004
                local += signed * normal[None, :]
                local = np.clip(local, 1e-6, 1.0 - 1e-6)

                local_r = compute_max_radii(local)
                zc = np.asarray(base, dtype=float).copy()
                zc[2 * ids] = local[:, 0]
                zc[2 * ids + 1] = local[:, 1]
                zc[2 * n + ids] = np.maximum(local_r, 1e-7)

                def cavity_objective(z):
                    """Maximize the sum of radii belonging to one frozen cavity."""
                    return -float(np.sum(z[2 * n + ids]))

                def cavity_constraints(z):
                    """Enforce walls, cavity contacts, and frozen-exterior contacts."""
                    p = z[:2 * n].reshape(n, 2)
                    r = z[2 * n:]
                    parts = [
                        p[ids, 0] - r[ids],
                        p[ids, 1] - r[ids],
                        1.0 - p[ids, 0] - r[ids],
                        1.0 - p[ids, 1] - r[ids],
                    ]
                    for aa in range(len(ids) - 1):
                        i = int(ids[aa])
                        jj = ids[aa + 1:]
                        d = p[jj] - p[i]
                        parts.append(
                            np.sum(d * d, axis=1) -
                            (r[i] + r[jj]) ** 2
                        )
                    for i in ids:
                        jj = outside
                        if len(jj):
                            d = p[jj] - p[i]
                            parts.append(
                                np.sum(d * d, axis=1) -
                                (r[i] + r[jj]) ** 2
                            )
                    return np.concatenate(parts)

                bounds = [(1e-6, 1.0 - 1e-6)] * (2 * n)
                bounds += [(0.0, 0.5)] * n
                for k in outside:
                    bounds[2 * int(k)] = (
                        float(base_centers[k, 0]),
                        float(base_centers[k, 0])
                    )
                    bounds[2 * int(k) + 1] = (
                        float(base_centers[k, 1]),
                        float(base_centers[k, 1])
                    )
                    bounds[2 * n + int(k)] = (
                        float(base_radii[k]),
                        float(base_radii[k])
                    )

                cavity_result = minimize(
                    cavity_objective, zc, method="SLSQP",
                    bounds=bounds,
                    constraints={"type": "ineq", "fun": cavity_constraints},
                    options={"maxiter": 260, "ftol": 2e-10, "disp": False},
                )
                if cavity_result.x is None:
                    continue

                cavity_centers = cavity_result.x[:2 * n].reshape(n, 2)
                polished = polish_release(cavity_centers, 180)
                if polished.x is not None:
                    candidates.append(polished)

            for candidate in candidates:
                c = candidate.x[:2 * n].reshape(n, 2)
                r = candidate.x[2 * n:]
                c, r = _sanitize(c, r)
                value = float(np.sum(r))
                if value > best_value:
                    best_c, best_r, best_value = c, r, value
    except Exception:
        pass

    best_c, best_r = _sanitize(best_c, best_r)
    return best_c, best_r, float(np.sum(best_r))


# EVOLVE-BLOCK-END


def run_packing():
    """Run the circle packing constructor for n=26."""
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii


def visualize(centers, radii):
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