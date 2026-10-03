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

    def jammed_seed(seed):
        rng = np.random.default_rng(seed)
        lo, hi = 0.055, 0.945
        points = []
        # Deterministic dart throwing gives a non-lattice starting topology.
        for _ in range(7000):
            q = rng.uniform(lo, hi, 2)
            if not points or min(np.linalg.norm(q - p) for p in points) > 0.115:
                points.append(q)
                if len(points) == n:
                    break
        while len(points) < n:
            points.append(rng.uniform(lo, hi, 2))
        p = np.asarray(points[:n], dtype=float)

        # Stronger bounded repulsion clears short-range defects while
        # preserving the deliberately non-row contact topology.
        for _ in range(250):
            force = np.zeros_like(p)
            for i in range(n):
                for j in range(i):
                    d = p[i] - p[j]
                    dist = max(float(np.linalg.norm(d)), 1e-8)
                    if dist < 0.17:
                        f = 0.006 * (0.17 - dist) / dist * d
                        force[i] += f
                        force[j] -= f
            wall = 0.085
            force[:, 0] += np.where(p[:, 0] < wall,
                                    0.0025 * (wall - p[:, 0]) / wall, 0.0)
            force[:, 0] -= np.where(p[:, 0] > 1.0 - wall,
                                    0.0025 * (p[:, 0] - 1.0 + wall) / wall, 0.0)
            force[:, 1] += np.where(p[:, 1] < wall,
                                    0.0025 * (wall - p[:, 1]) / wall, 0.0)
            force[:, 1] -= np.where(p[:, 1] > 1.0 - wall,
                                    0.0025 * (p[:, 1] - 1.0 + wall) / wall, 0.0)
            p = np.clip(p + force, lo, hi)
        return p

    seeds = [jammed_seed(s) for s in rng_seeds]
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
            ranked.append((float(np.sum(rr)), p, rr))
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
                options={"maxiter": 500, "ftol": 2e-10, "disp": False},
            )
            if result.x is None:
                continue
            c = result.x[:2 * n].reshape(n, 2)
            r = result.x[2 * n:]
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