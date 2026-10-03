"""Constructor-based circle packing for n=26 circles."""
import numpy as np


def compute_max_radii(centers):
    """Conservative feasible radii for a fixed set of centers."""
    n = len(centers)
    radii = np.min(np.c_[centers, 1.0 - centers], axis=1).astype(float)
    for _ in range(16):
        changed = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                d = float(np.linalg.norm(centers[i] - centers[j]))
                excess = radii[i] + radii[j] - d
                if excess > 0.0:
                    changed = True
                    cut = 0.5 * excess + 1e-11
                    radii[i] = max(0.0, radii[i] - cut)
                    radii[j] = max(0.0, radii[j] - cut)
        if not changed:
            break
    return radii


def construct_packing():
    """Jointly optimize several layered and chevron-core constructions."""
    n = 26

    def row_seed(counts, ys, phase=0.0):
        pts = []
        for k, (m, y) in enumerate(zip(counts, ys)):
            if m == 4:
                xs = np.linspace(0.13, 0.87, 4)
            elif m == 5:
                xs = np.linspace(0.09, 0.91, 5)
            else:
                xs = np.linspace(0.055, 0.945, 6)
            if phase:
                xs = np.clip(xs + (phase if (k & 1) else -phase), 0.035, 0.965)
            pts.extend((float(x), float(y)) for x in xs)
        return np.asarray(pts, dtype=float)

    def chevron_seed(amplitude, phase):
        """Five bands, with the central six split into alternating pockets."""
        pts = []
        for k, y in enumerate((0.090, 0.295)):
            xs = np.linspace(0.09, 0.91, 5)
            shift = -phase if (k & 1) == 0 else phase
            pts.extend((float(x + shift), float(y)) for x in xs)

        xs = np.linspace(0.055, 0.945, 6)
        for k, x in enumerate(xs):
            pts.append((float(x), float(0.500 + amplitude * (-1.0 if k & 1 else 1.0))))

        for k, y in enumerate((0.705, 0.910), start=3):
            xs = np.linspace(0.09, 0.91, 5)
            shift = -phase if (k & 1) == 0 else phase
            pts.extend((float(x + shift), float(y)) for x in xs)
        return np.asarray(pts, dtype=float)

    starts = []
    for phase in (0.0, -0.012, 0.012):
        starts.append(row_seed(
            (5, 5, 6, 5, 5),
            (0.090, 0.295, 0.500, 0.705, 0.910),
            phase
        ))
    for phase in (0.0, -0.012, 0.012):
        starts.append(row_seed(
            (4, 6, 6, 6, 4),
            (0.075, 0.285, 0.500, 0.715, 0.925),
            phase
        ))
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

    # A non-collinear central band is a distinct local contact topology.
    for amp in (0.020, 0.032, 0.044):
        starts.append(chevron_seed(amp, 0.010))

    try:
        from scipy.optimize import minimize

        def objective(z):
            return -float(np.sum(z[2 * n:]))

        def constraints(z):
            p = z[:2 * n].reshape(n, 2)
            r = z[2 * n:]
            values = [
                p[:, 0] - r,
                p[:, 1] - r,
                1.0 - p[:, 0] - r,
                1.0 - p[:, 1] - r,
            ]
            for i in range(n - 1):
                delta = p[i + 1:] - p[i]
                values.append(np.sqrt(np.sum(delta * delta, axis=1) + 1e-24)
                              - r[i] - r[i + 1:])
            return np.concatenate(values)

        def initial_radii(p):
            wall = np.min(np.c_[p, 1.0 - p], axis=1)
            near = np.full(n, np.inf)
            for i in range(n):
                d = np.sqrt(np.sum((p - p[i]) ** 2, axis=1))
                d[i] = np.inf
                near[i] = np.min(d)
            return np.clip(0.40 * np.minimum(wall, 0.5 * near), 0.002, 0.08)

        def repair(p, r):
            wall = np.min(np.c_[p, 1.0 - p], axis=1)
            r = np.maximum(0.0, np.minimum(r, wall - 3e-10))
            for _ in range(18):
                changed = False
                for i in range(n - 1):
                    for j in range(i + 1, n):
                        d = float(np.linalg.norm(p[i] - p[j]))
                        excess = r[i] + r[j] - d
                        if excess > 0.0:
                            cut = 0.5 * excess + 3e-10
                            r[i] = max(0.0, r[i] - cut)
                            r[j] = max(0.0, r[j] - cut)
                            changed = True
                if not changed:
                    break
            return r

        bounds = [(0.0, 1.0)] * (2 * n) + [(1e-8, 0.25)] * n
        best = None
        best_value = -np.inf

        def keep_candidate(x):
            nonlocal best, best_value
            if x is None or not np.all(np.isfinite(x)):
                return
            p = x[:2 * n].reshape(n, 2)
            r = repair(p, x[2 * n:])
            z = np.r_[p.ravel(), r]
            if np.min(constraints(z)) >= -2e-8:
                value = float(np.sum(r))
                if value > best_value:
                    best = z
                    best_value = value

        # Similar aggregate work to the prior portfolio despite three extra seeds.
        for p0 in starts:
            z0 = np.r_[p0.ravel(), initial_radii(p0)]
            res = minimize(
                objective, z0, method="SLSQP", bounds=bounds,
                constraints={"type": "ineq", "fun": constraints},
                options={"maxiter": 1100, "ftol": 2e-10, "disp": False},
            )
            keep_candidate(getattr(res, "x", None))

        if best is not None:
            def curved_continuation(base, sign, reflected):
                """Create a feasible asymmetric curved-boundary continuation of the incumbent."""
                p = base[:2 * n].reshape(n, 2).copy()
                if reflected:
                    p[:, 0] = 1.0 - p[:, 0]
                x = p[:, 0]
                edge = ((x - 0.5) / 0.5) ** 2
                # Add a small cubic skew as well as the main quadratic bend.
                # This breaks the rigid left/right row symmetry while keeping
                # the displacement bounded and concentrated near the corners.
                skew = ((x - 0.5) / 0.5) ** 3
                for i, y in enumerate(p[:, 1]):
                    if y < 0.22:
                        p[i, 1] += sign * (0.012 * edge[i] + 0.0025 * skew[i])
                    elif y > 0.78:
                        p[i, 1] -= sign * (0.012 * edge[i] - 0.0025 * skew[i])
                    elif y < 0.42:
                        p[i, 1] -= sign * 0.006 * edge[i]
                    elif y > 0.58:
                        p[i, 1] += sign * 0.006 * edge[i]
                p = np.clip(p, 0.002, 0.998)
                r = compute_max_radii(p)
                r = np.clip(r, 1e-8, 0.25)
                return np.r_[p.ravel(), r]

            for sign in (-1.0, 1.0):
                for reflected in (False, True):
                    seed = curved_continuation(best, sign, reflected)
                    polished = minimize(
                        objective, seed, method="SLSQP", bounds=bounds,
                        constraints={"type": "ineq", "fun": constraints},
                        options={"maxiter": 900, "ftol": 5e-11, "disp": False},
                    )
                    keep_candidate(getattr(polished, "x", None))

        if best is not None:
            p = best[:2 * n].reshape(n, 2)
            r = repair(p, best[2 * n:])
            z = np.r_[p.ravel(), r]
            if np.min(constraints(z)) >= -2e-8:
                return p, r, float(np.sum(r))

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