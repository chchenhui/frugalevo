import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """Optimize layered two-heptagon starts, release them to annealing, and polish."""
    from scipy.optimize import minimize

    n = 14
    d = 3
    rng = np.random.default_rng(271828)
    pairs = np.array([(i, j) for i in range(n) for j in range(i)], dtype=int)

    def normalize(x):
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    def ratio2(x):
        q = x[pairs[:, 0]] - x[pairs[:, 1]]
        dsq = np.einsum("ij,ij->i", q, q)
        return float(np.min(dsq) / np.max(dsq))

    # Optimize a five-parameter pair of staggered heptagons before releasing
    # the coordinates to the unrestricted annealer.  The offset uses opposite
    # signs on the two layers, so it changes the geometry rather than merely
    # translating the whole configuration.
    angles = 2.0 * np.pi * np.arange(7) / 7.0
    starts = []
    layered_seeds = (
        np.array([0.34, 0.92, 0.92, np.pi / 7.0, 0.00]),
        np.array([0.42, 0.86, 0.94, np.pi / 7.0, 0.08]),
        np.array([0.28, 0.98, 0.88, 2.0 * np.pi / 7.0, -0.08]),
    )

    def layered(params):
        h, rp, rm, alpha, offset = params
        upper = np.column_stack((
            rp * np.cos(angles) + offset,
            rp * np.sin(angles),
            np.full(7, h),
        ))
        lower = np.column_stack((
            rm * np.cos(angles + alpha) - offset,
            rm * np.sin(angles + alpha),
            np.full(7, -h),
        ))
        p = np.vstack((upper, lower))
        diameter = np.sqrt(np.max(np.sum(
            (p[pairs[:, 0]] - p[pairs[:, 1]]) ** 2, axis=1
        )))
        if not np.isfinite(diameter) or diameter <= 1.0e-12:
            return np.zeros((n, d), dtype=float)
        return p / diameter

    def layered_objective(params):
        p = layered(params)
        if not np.any(p):
            return 1.0e6
        return -ratio2(p)

    for seed in layered_seeds:
        try:
            fit = minimize(
                layered_objective,
                seed,
                method="Nelder-Mead",
                options={"maxiter": 180, "xatol": 1.0e-8, "fatol": 1.0e-10},
            )
            candidate = layered(fit.x)
            if candidate.shape == (n, d) and np.all(np.isfinite(candidate)):
                starts.append(candidate.copy())
        except Exception:
            pass

    # Retain a deterministic spherical fallback if a local layered solve fails.
    z = 1.0 - 2.0 * (np.arange(n) + 0.5) / n
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    theta = 2.0 * np.pi * np.arange(n) / phi
    base = np.column_stack((
        np.sqrt(np.maximum(0.0, 1.0 - z * z)) * np.cos(theta),
        np.sqrt(np.maximum(0.0, 1.0 - z * z)) * np.sin(theta),
        z,
    ))
    if not starts:
        starts.append(base.copy())

    best = None
    best_value = -np.inf
    restarts = len(starts)
    iterations = 25000

    for restart in range(restarts):
        points = starts[restart].copy()
        current = ratio2(points)
        local_best = points.copy()
        local_value = current

        for it in range(iterations):
            frac = it / (iterations - 1)
            temperature = 0.010 * (1.0 - frac) ** 2 + 0.000035
            step = 0.24 * (1.0 - frac) + 0.008

            # Release the layered ansatz in unrestricted Cartesian space.
            # The layered initializer is diameter-normalized, not
            # unit-sphere-normalized, so projecting every moved point onto
            # its tangent sphere would discard its independently optimized
            # radii and heights.  Cartesian moves preserve that structure
            # while the subsequent exact constrained polish handles scale.
            k = int(rng.integers(n))
            candidate = points.copy()
            displacement = rng.normal(size=3)
            norm = np.linalg.norm(displacement)
            if norm <= 1.0e-14:
                continue
            candidate[k] += displacement * (step / norm)

            value = ratio2(candidate)
            delta = value - current
            if delta >= 0.0 or rng.random() < np.exp(delta / temperature):
                points = candidate
                current = value
            if current > local_value:
                local_value = current
                local_best = points.copy()

        if local_value > best_value:
            best_value = local_value
            best = local_best

    # Fix translation and scale: point zero is the origin and dmax is one.
    q = best - best[0]
    initial_dsq = np.sum((q[pairs[:, 0]] - q[pairs[:, 1]]) ** 2, axis=1)
    dmax = np.sqrt(np.max(initial_dsq))
    q /= dmax
    initial_dsq /= dmax * dmax
    t0 = float(np.min(initial_dsq) * 0.999)

    # Variables are points 1..13 followed by the common lower bound t.
    x0 = np.concatenate((q[1:].ravel(), [t0]))

    def unpack(x):
        p = np.zeros((n, 3), dtype=float)
        p[1:] = x[:-1].reshape(n - 1, 3)
        return p

    def objective(x):
        return -x[-1]

    def objective_jac(x):
        g = np.zeros_like(x)
        g[-1] = -1.0
        return g

    def constraints(x):
        p = unpack(x)
        diff = p[pairs[:, 0]] - p[pairs[:, 1]]
        dsq = np.einsum("ij,ij->i", diff, diff)
        return np.concatenate((dsq - x[-1], 1.0 - dsq))

    def constraints_jac(x):
        p = unpack(x)
        diff = p[pairs[:, 0]] - p[pairs[:, 1]]
        m = len(pairs)
        jac = np.zeros((2 * m, 3 * (n - 1) + 1), dtype=float)
        for row, (i, j) in enumerate(pairs):
            v = 2.0 * diff[row]
            if i:
                a = 3 * (i - 1)
                jac[row, a:a + 3] += v
            if j:
                a = 3 * (j - 1)
                jac[row, a:a + 3] -= v
            jac[row, -1] = -1.0
            jac[m + row, :-1] = -jac[row, :-1]
        return jac

    polished = best
    try:
        result = minimize(
            objective,
            x0,
            jac=objective_jac,
            method="SLSQP",
            constraints={"type": "ineq", "fun": constraints, "jac": constraints_jac},
            options={"maxiter": 350, "ftol": 1.0e-11, "disp": False},
        )
        if result.x.shape == x0.shape and np.all(np.isfinite(result.x)):
            trial = unpack(result.x)
            if ratio2(trial) > ratio2(polished):
                polished = trial
    except Exception:
        pass

    polished = polished - np.mean(polished, axis=0, keepdims=True)
    scale = np.max(np.linalg.norm(polished, axis=1))
    if not np.isfinite(scale) or scale <= 0.0:
        polished = best
        polished = polished - np.mean(polished, axis=0, keepdims=True)
        scale = np.max(np.linalg.norm(polished, axis=1))
    polished /= scale
    return np.asarray(polished, dtype=float)