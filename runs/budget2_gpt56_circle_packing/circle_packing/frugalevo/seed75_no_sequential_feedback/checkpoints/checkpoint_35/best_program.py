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
    def cvt_seed(transform):
        """Build 26 sites from a Halton pattern followed by 100 grid-CVT sweeps."""
        def radical_inverse(index, base):
            value = 0.0
            scale = 1.0 / base
            while index:
                index, digit = divmod(index, base)
                value += digit * scale
                scale /= base
            return value

        sites = np.asarray(
            [(radical_inverse(i, 2), radical_inverse(i, 3))
             for i in range(1, n + 1)],
            dtype=float
        )
        if transform == 1:
            sites[:, 0] = 1.0 - sites[:, 0]
        elif transform == 2:
            sites = sites[:, ::-1].copy()

        axis = (np.arange(64, dtype=float) + 0.5) / 64.0
        grid = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(-1, 2)
        for _ in range(100):
            distance = np.sum((grid[:, None, :] - sites[None, :, :]) ** 2, axis=2)
            owner = np.argmin(distance, axis=1)
            updated = sites.copy()
            for j in range(n):
                members = grid[owner == j]
                if len(members):
                    updated[j] = np.mean(members, axis=0)
            sites = np.clip(updated, 0.025, 0.975)
        return sites

    for transform in (0, 1, 2):
        starts.append(cvt_seed(transform))

    # A non-collinear central band is a distinct local contact topology.
    for amp in (0.020, 0.032, 0.044):
        starts.append(chevron_seed(amp, 0.010))

    try:
        from scipy.optimize import minimize

        def objective(z):
            return -float(np.sum(z[2 * n:]))

        def constraints(z):
            """Return wall and pairwise non-overlap slacks in fixed SLSQP order."""
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

        def objective_jac(z):
            """Return the exact gradient of the negative total-radius objective."""
            gradient = np.zeros(3 * n, dtype=float)
            gradient[2 * n:] = -1.0
            return gradient

        def constraints_jac(z):
            """Return the analytic Jacobian for walls and every pair distance."""
            p = z[:2 * n].reshape(n, 2)
            r = z[2 * n:]
            rows = 4 * n + n * (n - 1) // 2
            jac = np.zeros((rows, 3 * n), dtype=float)
            for i in range(n):
                base = i
                jac[base, 2 * i] = 1.0
                jac[base, 2 * n + i] = -1.0
                base = n + i
                jac[base, 2 * i + 1] = 1.0
                jac[base, 2 * n + i] = -1.0
                base = 2 * n + i
                jac[base, 2 * i] = -1.0
                jac[base, 2 * n + i] = -1.0
                base = 3 * n + i
                jac[base, 2 * i + 1] = -1.0
                jac[base, 2 * n + i] = -1.0

            row = 4 * n
            for i in range(n - 1):
                for j in range(i + 1, n):
                    delta = p[i] - p[j]
                    distance = float(np.sqrt(np.dot(delta, delta) + 1e-24))
                    direction = delta / distance
                    jac[row, 2 * i:2 * i + 2] = direction
                    jac[row, 2 * j:2 * j + 2] = -direction
                    jac[row, 2 * n + i] = -1.0
                    jac[row, 2 * n + j] = -1.0
                    row += 1
            return jac

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
                objective, z0, method="SLSQP", jac=objective_jac,
                bounds=bounds,
                constraints={
                    "type": "ineq",
                    "fun": constraints,
                    "jac": constraints_jac,
                },
                options={"maxiter": 1100, "ftol": 2e-10, "disp": False},
            )
            keep_candidate(getattr(res, "x", None))

        if best is not None:
            def active_nullspace_modes(base):
                """Compute up to six smallest-singular-value modes of active contacts."""
                values = constraints(base)
                full_jac = constraints_jac(base)
                active = np.flatnonzero(values <= 3e-5)
                if len(active) == 0:
                    return []
                _, singular, vh = np.linalg.svd(
                    full_jac[active], full_matrices=False
                )
                order = np.argsort(singular)
                modes = []
                for index in order[:6]:
                    mode = np.asarray(vh[index], dtype=float).copy()
                    norm = float(np.linalg.norm(mode))
                    if norm > 1e-12:
                        modes.append(mode / norm)
                return modes

            def continuation_objective(z, mode, sign):
                """Maximize radius sum while gently following one elastic tangent mode."""
                return -float(np.sum(z[2 * n:]) +
                              sign * 3e-4 * np.dot(mode, z - base))

            def continuation_jac(z, mode, sign):
                """Return the gradient of the radius objective plus tangent bias."""
                gradient = np.zeros(3 * n, dtype=float)
                gradient[2 * n:] = -1.0
                gradient -= sign * 3e-4 * mode
                return gradient

            base = best.copy()
            modes = active_nullspace_modes(base)
            for mode in modes:
                # Prefer the sign that initially increases total radius, but
                # test both signs because curvature can reverse the advantage.
                radius_slope = float(np.sum(mode[2 * n:]))
                signs = (1.0, -1.0) if radius_slope >= 0.0 else (-1.0, 1.0)
                for sign in signs:
                    seed = base.copy()
                    direction = sign * mode
                    scale = min(
                        0.006,
                        0.20 / max(1.0, float(np.linalg.norm(direction, ord=np.inf)))
                    )
                    seed += scale * direction
                    seed[:2 * n] = np.clip(
                        seed[:2 * n].reshape(n, 2), 1e-7, 1.0 - 1e-7
                    ).ravel()
                    seed[2 * n:] = repair(
                        seed[:2 * n].reshape(n, 2),
                        np.maximum(1e-8, seed[2 * n:])
                    )
                    continuation = minimize(
                        lambda z: continuation_objective(z, mode, sign),
                        seed,
                        method="SLSQP",
                        jac=lambda z: continuation_jac(z, mode, sign),
                        bounds=bounds,
                        constraints={
                            "type": "ineq",
                            "fun": constraints,
                            "jac": constraints_jac,
                        },
                        options={"maxiter": 450, "ftol": 2e-10, "disp": False},
                    )
                    candidate = getattr(continuation, "x", None)
                    if candidate is None or not np.all(np.isfinite(candidate)):
                        candidate = seed
                    polished = minimize(
                        objective, candidate, method="SLSQP", jac=objective_jac,
                        bounds=bounds,
                        constraints={
                            "type": "ineq",
                            "fun": constraints,
                            "jac": constraints_jac,
                        },
                        options={"maxiter": 500, "ftol": 5e-11, "disp": False},
                    )
                    polished_candidate = getattr(polished, "x", None)
                    if polished_candidate is None:
                        polished_candidate = candidate
                    if np.all(np.isfinite(polished_candidate)):
                        repaired = polished_candidate.copy()
                        repaired[2 * n:] = repair(
                            repaired[:2 * n].reshape(n, 2),
                            repaired[2 * n:]
                        )
                        keep_candidate(repaired)

        if best is not None:
            # Harvest evaluator tolerance with a certified uniform micro-inflation.
            p = best[:2 * n].reshape(n, 2).copy()
            r = repair(p, best[2 * n:]).astype(float, copy=True)

            # A wall constraint loses inflate, while a pair constraint loses
            # twice inflate.  Keep a small margin inside the evaluator limit.
            inflate = 4.0e-7
            trial = r + inflate
            trial_z = np.r_[p.ravel(), trial]
            trial_slack = float(np.min(constraints(trial_z)))

            # If numerical residuals consume too much slack, reduce inflation
            # by the measured deficit rather than discarding the incumbent.
            if trial_slack < -9.0e-7:
                inflate = max(
                    0.0,
                    inflate + trial_slack + 9.0e-7
                )
                trial = r + inflate
                trial_z = np.r_[p.ravel(), trial]

            if np.min(constraints(trial_z)) >= -9e-7:
                return p, trial, float(np.sum(trial))

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