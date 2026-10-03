"""Deterministic constrained constructor for 26 circles in the unit square."""
import numpy as np


def compute_max_radii(centers):
    n = len(centers)
    r = np.minimum.reduce(
        (centers[:, 0], centers[:, 1], 1.0 - centers[:, 0], 1.0 - centers[:, 1])
    ).copy()
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(centers[i] - centers[j]))
            if r[i] + r[j] > d and r[i] + r[j] > 0.0:
                scale = d / (r[i] + r[j])
                r[i] *= scale
                r[j] *= scale
    return np.maximum(r, 0.0)


def _row_seed(counts, offsets, phase=0.0):
    rows = len(counts)
    ys = np.linspace(0.095, 0.905, rows)
    points = []
    for k, (m, off) in enumerate(zip(counts, offsets)):
        width = 0.79 + 0.018 * (m - 5)
        left = 0.5 - width / 2.0 + off
        right = 0.5 + width / 2.0 + off
        xs = np.linspace(left, right, m)
        stagger = 0.5 * (right - left) / max(1, m - 1) if k % 2 else 0.0
        for q, x in enumerate(xs):
            dx = 0.0055 * np.sin(1.71 * (k + 1) * (q + 1) + phase)
            dy = 0.0035 * np.cos(1.13 * (k + 1) * (q + 1) + phase)
            points.append((
                np.clip(x + stagger + dx, 0.035, 0.965),
                np.clip(ys[k] + dy, 0.035, 0.965),
            ))
    return np.asarray(points, dtype=float)


def _frame_seed(kind):
    if kind == 0:
        a, b, skew = 0.105, 0.895, 0.0
    elif kind == 1:
        a, b, skew = 0.085, 0.915, 0.012
    else:
        a, b, skew = 0.11, 0.89, -0.014

    points = []
    for t in np.linspace(0.18, 0.82, 4):
        points.extend(((t + skew, a), (t - skew, b),
                       (a, t - skew), (b, t + skew)))
    for y, shift in ((0.405, -0.035 + skew), (0.595, 0.035 - skew)):
        for x in np.linspace(0.19, 0.81, 5):
            points.append((x + shift, y))
    return np.clip(np.asarray(points, dtype=float), 0.035, 0.965)


def _cellular_seed(defect, power=0.0):
    """A deterministic power-CVT seed; power=0 is the ordinary CVT case."""
    grid = np.linspace(0.12, 0.88, 5)
    sites = []
    for row, y in enumerate(grid):
        shift = 0.038 if row % 2 else 0.0
        for x in grid:
            sites.append((np.clip(x + shift, 0.035, 0.965), y))
    sites.append(defect)
    sites = np.asarray(sites, dtype=float)

    samples = np.linspace(0.02, 0.98, 33)
    sample_grid = np.asarray([(x, y) for y in samples for x in samples],
                             dtype=float)

    # Positive power gives edge-band cells a modestly greater initial capacity.
    weights = np.zeros(26, dtype=float)
    if power != 0.0:
        for k, (x, y) in enumerate(sites):
            wall = min(x, y, 1.0 - x, 1.0 - y)
            if wall < 0.18:
                weights[k] = power
            elif wall > 0.30:
                weights[k] = -0.35 * power
        weights[25] = -0.15 * power

    corners = {0, 4, 20, 24}
    for _ in range(16):
        delta = sample_grid[:, None, :] - sites[None, :, :]
        d2 = np.einsum("gsi,gsi->gs", delta, delta) - weights[None, :]
        labels = np.argmin(d2, axis=1)
        updated = sites.copy()
        for k in range(26):
            assigned = sample_grid[labels == k]
            if len(assigned):
                rate = 0.10 if k in corners else 0.34
                updated[k] += rate * (np.mean(assigned, axis=0) - updated[k])
        sites = np.clip(updated, 0.035, 0.965)
    return sites


def _square_radial_seed(phases, reflected=False, skew=0.0):
    """Create 26 centers on three staggered square-radial annular rings."""
    center = np.asarray((0.5, 0.5), dtype=float)
    counts = (6, 10, 10)
    radial_fractions = (0.32, 0.62, 0.88)
    points = []

    for ring, (count, rho, phase) in enumerate(
            zip(counts, radial_fractions, phases)):
        angles = phase + 2.0 * np.pi * np.arange(count, dtype=float) / count
        for theta in angles:
            c = float(np.cos(theta))
            s = float(np.sin(theta))
            square_radius = 0.5 / max(abs(c), abs(s))
            local_rho = rho + skew * np.sin(2.0 * theta + 0.37 * ring)
            point = center + local_rho * square_radius * np.asarray(
                (c, s), dtype=float)
            if reflected:
                point[0] = 1.0 - point[0]
            points.append(point)

    return np.clip(np.asarray(points, dtype=float), 0.02, 0.98).copy()


def _radius_lp(centers, fallback):
    try:
        from scipy.optimize import linprog
        n = len(centers)
        A, b = [], []
        for i, (x, y) in enumerate(centers):
            row = np.zeros(n)
            row[i] = 1.0
            for limit in (x, y, 1.0 - x, 1.0 - y):
                A.append(row.copy())
                b.append(limit)
        for i in range(n):
            for j in range(i + 1, n):
                row = np.zeros(n)
                row[i] = row[j] = 1.0
                A.append(row)
                b.append(float(np.linalg.norm(centers[i] - centers[j])))
        ans = linprog(-np.ones(n), A_ub=np.asarray(A), b_ub=np.asarray(b),
                      bounds=[(0.0, 0.25)] * n, method="highs")
        if ans.success and np.all(np.isfinite(ans.x)):
            return ans.x
    except Exception:
        pass
    return fallback


def _feasible(centers, radii, tol=2e-7):
    if centers.shape != (26, 2) or radii.shape != (26,):
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(radii)):
        return False
    if np.min(radii) < -tol:
        return False
    if np.min(centers[:, 0] - radii) < -tol:
        return False
    if np.min(centers[:, 1] - radii) < -tol:
        return False
    if np.min(1.0 - centers[:, 0] - radii) < -tol:
        return False
    if np.min(1.0 - centers[:, 1] - radii) < -tol:
        return False
    for i in range(26):
        delta = centers[i + 1:] - centers[i]
        if len(delta):
            gap = np.sum(delta * delta, axis=1) - (radii[i] + radii[i + 1:]) ** 2
            if np.min(gap) < -tol:
                return False
    return True


def construct_packing():
    n = 26
    seeds = [
        _row_seed((5, 5, 6, 5, 5), (0.00, 0.018, 0.00, 0.018, 0.00), 0.0),
        _row_seed((5, 6, 5, 5, 5), (0.00, -0.018, 0.018, -0.010, 0.012), 0.7),
        _row_seed((5, 5, 5, 6, 5), (0.012, -0.010, 0.018, -0.018, 0.00), 1.4),
        _row_seed((4, 5, 4, 5, 4, 4),
                  (0.00, 0.018, -0.010, 0.018, 0.00, 0.012), 2.1),
        _row_seed((4, 5, 5, 4, 5, 3),
                  (0.008, -0.014, 0.012, -0.012, 0.014, 0.0), 2.8),
        _frame_seed(0), _frame_seed(1), _frame_seed(2),
        _cellular_seed((0.50, 0.43), 0.0),
        _cellular_seed((0.57, 0.52), 0.0),
        _cellular_seed((0.50, 0.43), 0.0018),
        _cellular_seed((0.57, 0.52), -0.0018),
        _square_radial_seed(
            (0.0, np.pi / 10.0, np.pi / 20.0),
            reflected=False, skew=0.0),
        _square_radial_seed(
            (0.018, np.pi / 10.0 + 0.012, np.pi / 20.0 - 0.009),
            reflected=False, skew=0.008),
        _square_radial_seed(
            (0.0, np.pi / 10.0, np.pi / 20.0),
            reflected=True, skew=0.0),
        _square_radial_seed(
            (-0.014, np.pi / 10.0 - 0.010, np.pi / 20.0 + 0.013),
            reflected=True, skew=-0.008),
    ]

    best_centers = seeds[0].copy()
    best_radii = compute_max_radii(best_centers)
    best_sum = float(np.sum(best_radii))

    try:
        from scipy.optimize import minimize

        ii, jj = np.triu_indices(n, 1)

        def objective(z):
            return -float(np.sum(z[2 * n:]))

        def constraints(z):
            centers = z[:2 * n].reshape(n, 2)
            radii = z[2 * n:]
            delta = centers[jj] - centers[ii]
            return np.concatenate((
                centers[:, 0] - radii,
                centers[:, 1] - radii,
                1.0 - centers[:, 0] - radii,
                1.0 - centers[:, 1] - radii,
                np.einsum("ij,ij->i", delta, delta) -
                (radii[ii] + radii[jj]) ** 2,
            ))

        candidates = []
        for seed in seeds:
            initial_radii = _radius_lp(
                seed, np.full(n, 0.018, dtype=float)).copy()
            z0 = np.concatenate((seed.reshape(-1), initial_radii))
            result = minimize(
                objective, z0, method="SLSQP",
                bounds=[(0.001, 0.999)] * (2 * n) + [(0.001, 0.22)] * n,
                constraints={"type": "ineq", "fun": constraints},
                options={"maxiter": 550, "ftol": 3e-10, "disp": False},
            )
            centers = result.x[:2 * n].reshape(n, 2)
            radii = result.x[2 * n:]
            if _feasible(centers, radii, tol=5e-7):
                candidates.append((float(np.sum(radii)), centers.copy(), radii.copy()))

        candidates.sort(key=lambda q: q[0], reverse=True)

        def _reinsert_eight(parent, parent_radii, weighted):
            """Replace the eight central circles using deterministic radius-aware maximin insertion."""
            center = np.asarray((0.5, 0.5), dtype=float)
            distances = np.linalg.norm(parent - center, axis=1)
            removed = np.argsort(distances, kind="stable")[:8]

            keep = np.ones(n, dtype=bool)
            keep[removed] = False
            retained = np.asarray(parent[keep], dtype=float).copy()
            retained_radii = np.asarray(parent_radii, dtype=float)[keep].copy()

            grid = np.linspace(0.16, 0.84, 37)
            available = np.asarray(
                [(x, y) for y in grid for x in grid], dtype=float)
            inserted = []

            for _ in range(8):
                trial = available
                clearance = np.minimum.reduce((
                    trial[:, 0], trial[:, 1],
                    1.0 - trial[:, 0], 1.0 - trial[:, 1]))

                # In the weighted variant, maximize the actual free distance
                # from each retained disk rather than distance from its center.
                # The ordinary variant preserves the unweighted maximin graph.
                for index, point in enumerate(retained):
                    gap = np.linalg.norm(trial - point, axis=1)
                    if weighted:
                        gap = gap - retained_radii[index]
                    clearance = np.minimum(clearance, gap)

                # Keep each new center at least one initial diameter from all
                # previously inserted centers, making the SLSQP start less
                # singular while still allowing the optimizer to move freely.
                for point in inserted:
                    gap = np.linalg.norm(trial - point, axis=1) - 0.024
                    clearance = np.minimum(clearance, gap)

                # Stable tie-breaking favors points nearer the square center;
                # this avoids accidental edge-biased choices on equal lattice
                # clearances without changing the maximin objective materially.
                tie = 1e-10 * np.linalg.norm(trial - center, axis=1)
                chosen = int(np.argmax(clearance - tie))
                inserted.append(trial[chosen].copy())
                available = np.delete(available, chosen, axis=0)

            result = np.empty((n, 2), dtype=float)
            result[keep] = retained
            result[removed] = np.asarray(inserted, dtype=float)
            return result

        def _tangent_restarts(parent, parent_radii):
            """Apply four alternating support-coordinate accordion relaxations."""
            restarts = []
            base = np.asarray(parent, dtype=float).copy()
            base_radii = np.asarray(parent_radii, dtype=float).copy()

            for sweep in range(4):
                xwall = np.minimum(base[:, 0], 1.0 - base[:, 0])
                ywall = np.minimum(base[:, 1], 1.0 - base[:, 1])
                x_supported = xwall <= ywall
                if sweep >= 2:
                    x_supported = ~x_supported

                axis = sweep % 2
                selected = x_supported if axis == 0 else ~x_supported
                indices = np.flatnonzero(selected)
                if len(indices) < 2:
                    continue

                # Use a centered quadratic-wave mode.  Unlike a monotone
                # displacement, this opens alternating gaps within a chain
                # while preserving its mean coordinate.
                transverse = base[indices, 1 - axis]
                order = indices[np.argsort(transverse, kind="stable")]
                phase = np.linspace(-1.0, 1.0, len(order), dtype=float)
                wave = np.sin(0.5 * np.pi * phase)
                wave -= np.mean(wave)
                profile = np.zeros(n, dtype=float)
                profile[order] = wave
                scale = 0.018 / max(1.0, float(np.max(np.abs(profile))))

                best_value = float(np.sum(base_radii))
                best_state = None
                for sign in (-1.0, 1.0):
                    moved = base.copy()
                    moved[:, axis] += sign * scale * profile
                    moved[:, axis] = np.clip(moved[:, axis], 0.012, 0.988)

                    radii0 = _radius_lp(moved, base_radii).copy()
                    z0 = np.concatenate((moved.reshape(-1), radii0))
                    result = minimize(
                        objective, z0, method="SLSQP",
                        bounds=[(0.001, 0.999)] * (2 * n) +
                               [(0.001, 0.22)] * n,
                        constraints={"type": "ineq", "fun": constraints},
                        options={"maxiter": 180, "ftol": 3e-10,
                                 "disp": False},
                    )
                    if not np.all(np.isfinite(result.x)):
                        continue
                    trial_centers = result.x[:2 * n].reshape(n, 2)
                    trial_radii = result.x[2 * n:].copy()
                    if not _feasible(trial_centers, trial_radii, tol=5e-7):
                        continue

                    value = float(np.sum(trial_radii))
                    restarts.append((value, trial_centers, trial_radii))
                    if value > best_value:
                        best_value = value
                        best_state = (trial_centers.copy(), trial_radii.copy())

                # Carry the best signed result into the next alternating
                # pass, so the relaxation is genuinely layered rather than
                # four independent perturbations of the same parent.
                if best_state is not None:
                    base, base_radii = best_state

            return restarts

        tangent_candidates = []
        for _, parent, parent_radii in candidates[:3]:
            tangent_candidates.extend(_tangent_restarts(parent, parent_radii))
        candidates.extend(tangent_candidates)
        candidates.sort(key=lambda q: q[0], reverse=True)
        for _, centers, radii in candidates[:8]:
            radii = _radius_lp(centers, radii)
            if _feasible(centers, radii, tol=5e-7):
                value = float(np.sum(radii))
                if value > best_sum:
                    best_sum = value
                    best_centers = centers.copy()
                    best_radii = radii.copy()

    except (ImportError, ValueError, RuntimeError, FloatingPointError):
        pass

    best_radii = np.maximum(0.0, best_radii * (1.0 - 3e-7))
    return best_centers, best_radii, float(np.sum(best_radii))


def run_packing():
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