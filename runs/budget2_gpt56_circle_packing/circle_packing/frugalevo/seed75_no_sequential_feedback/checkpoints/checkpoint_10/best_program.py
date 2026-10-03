"""Constructor-based circle packing for n=26 circles."""
import numpy as np


def compute_max_radii(centers):
    """Conservative feasible radii for a fixed set of centers."""
    n = len(centers)
    radii = np.min(np.c_[centers, 1.0 - centers], axis=1).astype(float)
    for _ in range(12):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.linalg.norm(centers[i] - centers[j]))
                excess = radii[i] + radii[j] - d
                if excess > 0.0:
                    changed = True
                    radii[i] = max(0.0, radii[i] - 0.5 * excess - 1e-11)
                    radii[j] = max(0.0, radii[j] - 0.5 * excess - 1e-11)
        if not changed:
            break
    return radii


def construct_packing():
    """Use several geometrically distinct layered contact topologies."""
    n = 26

    def row_seed(counts, ys, phase):
        pts = []
        for k, (m, y) in enumerate(zip(counts, ys)):
            if m == 4:
                xs = np.linspace(0.13, 0.87, 4)
            elif m == 5:
                xs = np.linspace(0.09, 0.91, 5)
            else:
                xs = np.linspace(0.055, 0.945, 6)

            # Alternate row phase, but never move a nominal boundary point
            # outside a safely positive initial clearance.
            if phase:
                shift = phase if (k & 1) else -phase
                xs = np.clip(xs + shift, 0.04, 0.96)
            pts.extend((float(x), float(y)) for x in xs)
        return np.asarray(pts, dtype=float)

    starts = []

    # The incumbent's most productive central-six family remains represented.
    for phase in (0.0, -0.012, 0.012):
        starts.append(row_seed(
            (5, 5, 6, 5, 5),
            (0.090, 0.295, 0.500, 0.705, 0.910),
            phase
        ))

    # A denser three-row core and sparse top/bottom rows.
    for phase in (0.0, -0.012, 0.012):
        starts.append(row_seed(
            (4, 6, 6, 6, 4),
            (0.075, 0.285, 0.500, 0.715, 0.925),
            phase
        ))

    # Two inverted-density families explore boundary-ring/core alternatives.
    for phase in (0.0, 0.012):
        starts.append(row_seed(
            (6, 5, 4, 5, 6),
            (0.060, 0.270, 0.500, 0.730, 0.940),
            phase
        ))
    for phase in (0.0, -0.012):
        starts.append(row_seed(
            (5, 6, 4, 6, 5),
            (0.085, 0.285, 0.500, 0.715, 0.915),
            phase
        ))

    try:
        from scipy.optimize import minimize

        def objective(z):
            return -float(np.sum(z[2 * n:]))

        def constraints(z):
            p = z[:2 * n].reshape(n, 2)
            r = z[2 * n:]
            ans = [
                p[:, 0] - r,
                p[:, 1] - r,
                1.0 - p[:, 0] - r,
                1.0 - p[:, 1] - r,
            ]
            for i in range(n - 1):
                d = p[i + 1:] - p[i]
                ans.append(np.sqrt(np.sum(d * d, axis=1) + 1e-20)
                           - r[i] - r[i + 1:])
            return np.concatenate(ans)

        def initial_radii(p):
            wall = np.min(np.c_[p, 1.0 - p], axis=1)
            near = np.full(n, np.inf)
            for i in range(n):
                d = np.sqrt(np.sum((p - p[i]) ** 2, axis=1))
                d[i] = np.inf
                near[i] = np.min(d)
            return np.clip(0.40 * np.minimum(wall, 0.5 * near), 0.003, 0.08)

        def repair(p, r):
            """Make numerical solver output strictly feasible at negligible cost."""
            r = np.maximum(0.0, np.minimum(r, np.min(np.c_[p, 1.0 - p], axis=1)
                                                 - 2e-10))
            for _ in range(8):
                changed = False
                for i in range(n - 1):
                    for j in range(i + 1, n):
                        d = float(np.linalg.norm(p[i] - p[j]))
                        excess = r[i] + r[j] - d
                        if excess > 0.0:
                            cut = 0.5 * excess + 2e-10
                            r[i] = max(0.0, r[i] - cut)
                            r[j] = max(0.0, r[j] - cut)
                            changed = True
                if not changed:
                    break
            return r

        bounds = [(0.0, 1.0)] * (2 * n) + [(1e-7, 0.25)] * n
        best = None
        best_value = -np.inf

        # Every start is strictly feasible before SLSQP is called.
        for p0 in starts:
            z0 = np.r_[p0.ravel(), initial_radii(p0)]
            res = minimize(
                objective, z0, method="SLSQP", bounds=bounds,
                constraints={"type": "ineq", "fun": constraints},
                options={"maxiter": 1400, "ftol": 2e-10, "disp": False},
            )
            if np.all(np.isfinite(res.x)):
                p = res.x[:2 * n].reshape(n, 2)
                r = repair(p, res.x[2 * n:])
                if np.min(constraints(np.r_[p.ravel(), r])) >= -1e-8:
                    value = float(np.sum(r))
                    if value > best_value:
                        best, best_value = np.r_[p.ravel(), r], value

        # Spend a dedicated local budget on the best discovered contact basin.
        if best is not None:
            polished = minimize(
                objective, best, method="SLSQP", bounds=bounds,
                constraints={"type": "ineq", "fun": constraints},
                options={"maxiter": 1200, "ftol": 5e-11, "disp": False},
            )
            if np.all(np.isfinite(polished.x)):
                p = polished.x[:2 * n].reshape(n, 2)
                r = repair(p, polished.x[2 * n:])
                value = float(np.sum(r))
                if np.min(constraints(np.r_[p.ravel(), r])) >= -1e-8 and value > best_value:
                    best, best_value = np.r_[p.ravel(), r], value

        if best is not None:
            return (best[:2 * n].reshape(n, 2), best[2 * n:],
                    float(np.sum(best[2 * n:])))

    except Exception:
        pass

    centers = starts[0]
    radii = compute_max_radii(centers)
    return centers, radii, float(np.sum(radii))


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