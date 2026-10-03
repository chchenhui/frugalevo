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

        counts = (4, 5, 4, 5, 4, 4)
        seeds = []
        for mirror, reverse, skew in (
            (0, 0, 0.000), (1, 0, 0.000), (0, 1, 0.000), (1, 1, 0.000),
            (0, 0, 0.012), (1, 0, -0.012), (0, 1, 0.009), (1, 1, -0.009),
        ):
            pts = []
            for row, count in enumerate(counts):
                yy = 0.075 + row * 0.165
                width = (count - 1) * 0.155
                offset = (0.025 if row % 2 else 0.0) + skew * (row - 2.5)
                xx = (1.0 - width) * 0.5 + offset
                xs = np.linspace(xx, xx + width, count)
                if mirror:
                    xs = 1.0 - xs
                pts.extend((float(x), yy) for x in xs)
            if reverse:
                pts = [(x, 1.0 - y) for x, y in pts[::-1]]
            # Start closer to the useful large-radius basin.  The row pitch
            # and horizontal spacing still leave positive separation slack,
            # while the increased radii reduce the optimizer's tendency to
            # settle near the small-radius boundary solution.
            rr = np.full(26, 0.072, dtype=float)
            seeds.append(np.r_[np.asarray(pts, dtype=float).ravel(), rr])

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