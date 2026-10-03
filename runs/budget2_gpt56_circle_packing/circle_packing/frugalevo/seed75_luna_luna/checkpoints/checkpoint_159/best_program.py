"""Deterministic constrained constructor for 26 non-overlapping circles."""
import numpy as np


def construct_packing():
    """Construct four clipped triangular-lattice packings and optimize their radii."""
    n = 26

    # Four distinct five-layer populations.  The last two are asymmetric
    # enough to give SLSQP different contact graphs without using random data.
    patterns = [
        ((5, 5, 6, 5, 5), (0.095, 0.297, 0.500, 0.703, 0.905)),
        ((4, 6, 6, 6, 4), (0.085, 0.285, 0.500, 0.715, 0.915)),
        ((6, 4, 6, 4, 6), (0.090, 0.292, 0.500, 0.708, 0.910)),
        ((5, 6, 4, 6, 5), (0.090, 0.292, 0.500, 0.708, 0.910)),
    ]

    variations = (
        (1.00, 1.00, 0.000),
        (0.97, 1.00, -0.008),
        (1.03, 1.00, 0.008),
        (1.00, 0.975, 0.000),
    )

    seeds = []
    for row_sizes, ys in patterns:
        points = []
        rows = []
        for iy, (count, y) in enumerate(zip(row_sizes, ys)):
            if count == 6:
                pitch = 0.158
            elif count == 5:
                pitch = 0.175
            else:
                pitch = 0.205
            start = 0.5 - 0.5 * pitch * (count - 1)
            if iy & 1:
                start += 0.5 * pitch
            lo = 0.060
            hi = 0.940 - pitch * (count - 1)
            start = min(max(start, lo), hi)
            for k in range(count):
                points.append((start + k * pitch, y))
                rows.append(iy)

        seed = np.asarray(points, dtype=float)
        rows = np.asarray(rows, dtype=float)

        for ys_scale, xs_scale, shear in variations:
            q = seed.copy()
            q[:, 1] = 0.5 + ys_scale * (q[:, 1] - 0.5)
            q[:, 0] = 0.5 + xs_scale * (q[:, 0] - 0.5)
            q[:, 0] += shear * (rows - 2.0)
            seeds.append(q)

    # Add a deterministic clearance-insertion seed.  It is deliberately
    # generated from geometry rather than from another row population.
    for mode in range(2):
        pts = []
        rr = []
        candidates = [(0.08, 0.08), (0.92, 0.08),
                      (0.08, 0.92), (0.92, 0.92)]
        for p in candidates:
            pts.append(p)
            rr.append(0.045)

        while len(pts) < n:
            best_p = None
            best_r = -1.0
            for gx in range(1, 18):
                for gy in range(1, 18):
                    p = np.array((gx / 18.0, gy / 18.0))
                    wall = min(p[0], p[1], 1-p[0], 1-p[1])
                    if pts:
                        d = np.linalg.norm(np.asarray(pts) - p, axis=1)
                        avail = np.min(d - np.asarray(rr))
                    else:
                        avail = wall
                    rad = min(wall, 0.5 * avail)
                    if rad <= 0.010:
                        continue
                    tie = (p[0] + p[1]) if mode == 0 else (p[0] - p[1])
                    score = rad + 1e-5 * tie
                    if score > best_r:
                        best_r = score
                        best_p = p
            if best_p is None:
                break
            pts.append(tuple(best_p))
            rr.append(best_r)
        if len(pts) == n:
            seeds.append(np.asarray(pts, dtype=float))

    def cons(z):
        c = z[:2*n].reshape(n, 2)
        r = z[2*n:]
        out = [
            c[:, 0] - r,
            c[:, 1] - r,
            1.0 - c[:, 0] - r,
            1.0 - c[:, 1] - r
        ]
        for i in range(n - 1):
            d = c[i+1:] - c[i]
            out.append(np.einsum("ij,ij->i", d, d) -
                       (r[i] + r[i+1:])**2)
        return np.concatenate(out)

    try:
        from scipy.optimize import minimize

        def augmented_cons(w):
            """Return wall and pair clearances for radii r_i=t+u_i."""
            c = w[:2*n].reshape(n, 2)
            t = float(w[2*n])
            u = w[2*n+1:]
            return cons(np.r_[c.ravel(), t + u])

        def unpack(w):
            """Decode centers and the common-scale radius deviations."""
            return (w[:2*n].reshape(n, 2).copy(),
                    (w[2*n] + w[2*n+1:]).copy())

        # Add a vertical-bisection contact graph.  Each half has four
        # staggered three-circle bands; the two bridge circles are offset in
        # opposite directions so that the central interface is asymmetric.
        vertical = []
        for flip in (0, 1):
            points = []
            for side in (0, 1):
                left, right = ((0.075, 0.475) if side == 0
                               else (0.525, 0.925))
                for band, y in enumerate((0.16, 0.385, 0.615, 0.84)):
                    edge = 0.020
                    gap = (right - left - 2.0*edge) / 2.0
                    shift = 0.035 if ((band + side + flip) & 1) else 0.0
                    for k in range(3):
                        x = left + edge + k*gap + shift
                        if x > right - edge:
                            x -= shift
                        points.append((x, y))
            delta = 0.018 if flip else -0.018
            points.extend(((0.5 + delta, 0.50), (0.5 - delta, 0.50)))
            vertical.append(np.asarray(points, dtype=float))

        # Build four deterministic triangular-lattice skeletons.  Each rule
        # selects a different clipped 26-point contact graph, then normalizes
        # it into a conservative square and applies a small alternating radial
        # perturbation to avoid collapsing all starts onto one symmetry basin.
        starts = []
        lattice = []
        for j in range(10):
            for i in range(-2, 11):
                lattice.append((i + 0.5 * (j & 1), 0.866025403784 * j))
        lattice = np.asarray(lattice, dtype=float)

        center_l = np.mean(lattice, axis=0)
        rel = lattice - center_l
        angle = np.arctan2(rel[:, 1], rel[:, 0])
        radius_l = np.sqrt(np.sum(rel * rel, axis=1))

        clipping_rules = (
            # Compact central hexagonal crop.
            np.abs(rel[:, 0]) / 1.65 + np.abs(rel[:, 1]) / 1.55,
            # Boundary-heavy crop, favoring points near the four corners.
            -0.35 * np.abs(rel[:, 0] * rel[:, 1]) +
            0.08 * radius_l,
            # Diagonal crop with a mild alternating diagonal preference.
            np.abs(rel[:, 0] + rel[:, 1]) / 2.0 +
            0.12 * np.abs(rel[:, 0] - rel[:, 1]),
            # Rotated 30-degree crop.
            np.abs(rel[:, 0] * np.cos(np.pi / 6.0) +
                   rel[:, 1] * np.sin(np.pi / 6.0)) / 1.7 +
            0.10 * np.abs(angle - np.pi / 6.0),
        )

        for rule_id, key in enumerate(clipping_rules):
            if rule_id == 1:
                order = np.lexsort((angle, key))
            elif rule_id == 2:
                order = np.lexsort((radius_l, key))
            else:
                order = np.lexsort((angle, radius_l, key))

            chosen = order[:n].copy()

            # Give the alternate crops distinct but deterministic boundary
            # support by replacing one or two nearly central points.
            if rule_id == 1:
                chosen[-1] = order[n + 4]
            elif rule_id == 2:
                chosen[-1] = order[n + 3]
                chosen[-2] = order[n + 6]
            elif rule_id == 3:
                chosen[-1] = order[n + 5]

            q = lattice[chosen].copy()
            lo = np.min(q, axis=0)
            hi = np.max(q, axis=0)
            span = np.maximum(hi - lo, 1e-12)
            q = 0.055 + 0.890 * (q - lo) / span

            # Alternate points are pushed very slightly away from the square
            # center, changing local angles without threatening feasibility.
            delta = q - 0.5
            norm = np.sqrt(np.sum(delta * delta, axis=1))
            direction = np.divide(
                delta, norm[:, None],
                out=np.zeros_like(delta),
                where=norm[:, None] > 1e-12,
            )
            q += (0.006 * ((np.arange(n) + rule_id) & 1))[:, None] * direction
            q = np.clip(q, 0.055, 0.945)
            starts.append(np.asarray(q, dtype=float))
        best = None
        # Evaluate exactly the four clipped triangular skeletons.  Directly
        # optimizing the radius sum avoids the artificial common-scale
        # bottleneck of the t/u parameterization while retaining its bounded
        # deviations and hard nonlinear clearance constraints.
        for seed in starts[-4:]:
            d = seed[:, None, :] - seed[None, :, :]
            dist = np.sqrt(np.sum(d*d, axis=2) + np.eye(n))
            upper = np.full(n, 0.060, dtype=float)
            for axis in (0, 1):
                upper = np.minimum(upper, seed[:, axis])
                upper = np.minimum(upper, 1.0 - seed[:, axis])
            for i in range(n):
                for j in range(i):
                    v = 0.46 * dist[i, j]
                    upper[i] = min(upper[i], v)
                    upper[j] = min(upper[j], v)

            r0 = np.maximum(0.020, upper)
            t0 = float(np.clip(np.min(r0), 0.020, 0.060))
            # Preserve the feasible geometry-derived initialization, but give
            # boundary and interior circles more freedom to acquire distinct
            # optimal radii than the original narrow deviation band allowed.
            u0 = np.clip(r0 - t0, -0.018, 0.018)
            w0 = np.r_[seed.ravel(), t0, u0]
            bounds = ([(0.0, 1.0)]*(2*n) + [(0.014, 0.25)] +
                      [(-0.018, 0.018)]*n)

            structural = minimize(
                lambda w: -float(n*w[2*n] + np.sum(w[2*n+1:])),
                w0, method="SLSQP", bounds=bounds,
                constraints={"type": "ineq", "fun": augmented_cons},
                options={"maxiter": 260, "ftol": 2e-9, "disp": False},
            )
            if (structural.success and np.all(np.isfinite(structural.x)) and
                    np.min(augmented_cons(structural.x)) >= -2e-8):
                if best is None or structural.fun < best.fun:
                    best = structural

        if best is None:
            centers = seeds[0].copy()
            radii = np.full(n, 0.055, dtype=float)
        else:
            centers, radii = unpack(best.x)

        def active_contact_polish(c0, r0):
            """Open weighted near-contacts, then optimize three tangent-cone starts."""
            incumbent = np.r_[np.asarray(c0, dtype=float).reshape(n, 2).ravel(),
                              np.asarray(r0, dtype=float).copy()]
            base_c = incumbent[:2*n].reshape(n, 2).copy()
            base_r = incumbent[2*n:].copy()
            normals = np.zeros((n, 2), dtype=float)
            weights = np.zeros(n, dtype=float)
            threshold = 8e-4

            # Weight congested circles by radius and give wall contacts extra
            # influence; all directions point into the available square.
            for i in range(n):
                x, y = base_c[i]
                wall_contacts = (
                    (x - base_r[i], np.array((1.0, 0.0))),
                    (y - base_r[i], np.array((0.0, 1.0))),
                    (1.0 - x - base_r[i], np.array((-1.0, 0.0))),
                    (1.0 - y - base_r[i], np.array((0.0, -1.0))),
                )
                for gap, direction in wall_contacts:
                    if gap < threshold:
                        w = 1.5 + 8.0 * base_r[i]
                        normals[i] += w * direction
                        weights[i] += w
            for i in range(n - 1):
                for j in range(i + 1, n):
                    delta = base_c[i] - base_c[j]
                    distance = float(np.linalg.norm(delta))
                    gap = distance - base_r[i] - base_r[j]
                    if gap < threshold and distance > 1e-12:
                        direction = delta / distance
                        w = 1.0 + 10.0 * (base_r[i] + base_r[j])
                        normals[i] += w * direction
                        normals[j] -= w * direction
                        weights[i] += w
                        weights[j] += w

            displacement = np.zeros_like(base_c)
            # Bias the opening move toward larger circles, which are usually
            # the most valuable and most radius-constrained members of the
            # active contact graph.  The absolute displacement remains capped.
            largest_radius = max(float(np.max(base_r)), 1e-12)
            for i in range(n):
                length = float(np.linalg.norm(normals[i]))
                if length > 1e-12:
                    radius_bias = 0.85 + 0.15 * base_r[i] / largest_radius
                    displacement[i] = min(0.006, 0.006 * radius_bias) * (
                        normals[i] / length
                    )

            # The two alternatives use opposite bounded tangent-cone
            # perturbations, while the first preserves the incumbent exactly.
            starts = [incumbent.copy()]
            for sign in (-1.0, 1.0):
                c = base_c.copy()
                for i in range(n):
                    d = displacement[i]
                    if weights[i] > 0.0:
                        tangent = np.array((-d[1], d[0]))
                        c[i] += d + sign * 0.20 * tangent
                c = np.clip(c, 0.002, 0.998)
                starts.append(np.r_[c.ravel(), base_r.copy()])

            bounds = ([(0.0, 1.0)] * (2*n) + [(0.002, 0.25)] * n)
            best_local = None
            for start in starts:
                first = minimize(
                    lambda z: -float(np.sum(z[2*n:])),
                    start, method="SLSQP", bounds=bounds,
                    constraints={"type": "ineq", "fun": cons},
                    options={"maxiter": 180, "ftol": 2e-10, "disp": False},
                )
                if (not np.all(np.isfinite(first.x)) or
                        np.min(cons(first.x)) < -2e-8):
                    continue

                # A short second solve repairs residual active constraints
                # after the radius-expanding pass without adding starts.
                second = minimize(
                    lambda z: -float(np.sum(z[2*n:])),
                    first.x, method="SLSQP", bounds=bounds,
                    constraints={"type": "ineq", "fun": cons},
                    options={"maxiter": 100, "ftol": 2e-10, "disp": False},
                )
                if (second.success and np.all(np.isfinite(second.x)) and
                        np.min(cons(second.x)) >= -2e-8 and
                        (best_local is None or second.fun < best_local.fun)):
                    best_local = second
            return None if best_local is None else best_local.x

        polished = active_contact_polish(centers, radii)
        if polished is not None:
            centers = polished[:2*n].reshape(n, 2).copy()
            radii = polished[2*n:].copy()
    except Exception:
        centers = seeds[0].copy()
        radii = np.full(n, 0.055, dtype=float)

    centers = np.clip(np.asarray(centers, dtype=float), 0.0, 1.0)
    radii = np.maximum(np.asarray(radii, dtype=float), 0.001)

    margin = 1.0
    margin = min(margin, float(np.min(centers / radii[:, None])))
    margin = min(margin, float(np.min((1.0 - centers) / radii[:, None])))
    for i in range(n):
        for j in range(i):
            d = np.linalg.norm(centers[i] - centers[j])
            margin = min(margin, float(d / (radii[i] + radii[j])))
    # Apply the requested final uniform safety shrink while preserving the
    # optimized contact geometry and guaranteed non-overlap.
    radii *= min(1.0, 0.9999998 * margin)
    return centers, radii, float(np.sum(radii))


def compute_max_radii(centers):
    n = centers.shape[0]
    radii = np.array(
        [min(x, y, 1-x, 1-y) for x, y in centers], dtype=float
    )
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if radii[i] + radii[j] > d:
                scale = d / (radii[i] + radii[j])
                radii[i] *= scale
                radii[j] *= scale
    return radii


def run_packing():
    return construct_packing()


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
    c, r, s = run_packing()
    print(f"Sum of radii: {s}")