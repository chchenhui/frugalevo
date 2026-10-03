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

    """Refine active contacts with deterministic nullspace-displacement starts."""
    q = np.asarray(best, dtype=float).copy()
    q -= q[0]
    dsq0 = np.sum((q[pairs[:, 0]] - q[pairs[:, 1]]) ** 2, axis=1)
    q /= np.sqrt(np.max(dsq0))
    dsq0 /= np.max(dsq0)

    def unpack(x):
        """Recover the fourteen points from the gauge-fixed SLSQP vector."""
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
        v = p[pairs[:, 0]] - p[pairs[:, 1]]
        z = np.einsum("ij,ij->i", v, v)
        return np.concatenate((z - x[-1], 1.0 - z))

    def constraints_jac(x):
        p = unpack(x)
        v = p[pairs[:, 0]] - p[pairs[:, 1]]
        m = len(pairs)
        j = np.zeros((2 * m, 3 * (n - 1) + 1), dtype=float)
        for r, (i, k) in enumerate(pairs):
            w = 2.0 * v[r]
            if i:
                j[r, 3 * (i - 1):3 * i] += w
            if k:
                j[r, 3 * (k - 1):3 * k] -= w
            j[r, -1] = -1.0
            j[m + r, :-1] = -j[r, :-1]
        return j

    # Signed contact Jacobian: short contacts are pushed apart and long
    # contacts are contracted.  Removing point zero also removes translation.
    lo = np.min(dsq0)
    hi = np.max(dsq0)
    active = np.flatnonzero((dsq0 <= lo * 1.004) | (dsq0 >= hi * 0.996))
    rows = []
    for r in active:
        i, k = pairs[r]
        row = np.zeros(3 * (n - 1), dtype=float)
        w = 2.0 * (q[i] - q[k])
        sign = 1.0 if dsq0[r] <= lo * 1.004 else -1.0
        if i:
            row[3 * (i - 1):3 * i] += sign * w
        if k:
            row[3 * (k - 1):3 * k] -= sign * w
        rows.append(row)
    direction = np.zeros(3 * (n - 1), dtype=float)
    if rows:
        _, _, vh = np.linalg.svd(np.asarray(rows), full_matrices=True)
        # The last right-singular direction is the least constrained collective
        # motion; its sign is selected by the contact objective.
        direction = vh[-1].copy()
        if np.dot(np.asarray(rows).mean(axis=0), direction) < 0.0:
            direction *= -1.0
        direction /= max(np.linalg.norm(direction), 1.0e-15)

    def diameter_normalize(p):
        """Recenter a trial and divide by its measured positive diameter."""
        p = np.asarray(p, dtype=float).copy()
        p -= np.mean(p, axis=0, keepdims=True)
        v = p[pairs[:, 0]] - p[pairs[:, 1]]
        den = np.sqrt(np.max(np.einsum("ij,ij->i", v, v)))
        return p / den if np.isfinite(den) and den > 1.0e-14 else None

    polished = q.copy()
    incumbent_value = ratio2(polished)

    # Build two deterministic starts by backtracking along the active-contact
    # nullspace direction, retaining the best valid step for each sign.
    starts = [q.copy()]
    if np.linalg.norm(direction) > 0.0:
        direction3 = direction.reshape(n - 1, 3)
        for sign in (1.0, -1.0):
            chosen = None
            chosen_value = -np.inf
            for step in (0.003, 0.0015, 0.00075, 0.000375,
                         0.0001875, 0.00009375):
                trial = q.copy()
                trial[1:] += sign * step * direction3
                trial = diameter_normalize(trial)
                if trial is None or not np.all(np.isfinite(trial)):
                    continue
                value = ratio2(trial)
                if np.isfinite(value) and value > chosen_value:
                    chosen = trial.copy()
                    chosen_value = value
            if chosen is not None and chosen_value >= incumbent_value * (1.0 - 2.0e-4):
                starts.append(chosen)

    try:
        from scipy.optimize import NonlinearConstraint

        for start in starts[:3]:
            vv = start[pairs[:, 0]] - start[pairs[:, 1]]
            zz = np.einsum("ij,ij->i", vv, vv)
            xstart = np.concatenate((
                np.asarray(start[1:], dtype=float).ravel(),
                [float(np.min(zz) * 0.9995)],
            ))

            sqp = minimize(
                objective, xstart, jac=objective_jac, method="SLSQP",
                constraints={"type": "ineq", "fun": constraints,
                             "jac": constraints_jac},
                options={"maxiter": 120, "ftol": 1.0e-11, "disp": False},
            )
            if sqp.x.shape != xstart.shape or not np.all(np.isfinite(sqp.x)):
                continue

            # Refine the best SLSQP point on the full exact pairwise
            # inequalities, while retaining the better of both local solves.
            candidates = [sqp.x]
            try:
                nonlinear = NonlinearConstraint(
                    constraints, 0.0, np.inf, jac=constraints_jac
                )
                tc = minimize(
                    objective, sqp.x, jac=objective_jac,
                    method="trust-constr", constraints=(nonlinear,),
                    options={"maxiter": 80, "gtol": 2.0e-9,
                             "xtol": 2.0e-9, "verbose": 0},
                )
                if tc.x.shape == xstart.shape and np.all(np.isfinite(tc.x)):
                    candidates.append(tc.x)
            except Exception:
                pass

            for candidate in candidates:
                trial = diameter_normalize(unpack(candidate))
                if trial is None or not np.all(np.isfinite(trial)):
                    continue
                value = ratio2(trial)
                if np.isfinite(value) and value > incumbent_value:
                    polished = trial.copy()
                    incumbent_value = value
    except Exception:
        pass

    # Recenter and normalize by the actual pairwise diameter so the returned
    # representative remains finite and has exactly the evaluator's scale.
    polished -= np.mean(polished, axis=0, keepdims=True)
    diff = polished[pairs[:, 0]] - polished[pairs[:, 1]]
    dsq = np.einsum("ij,ij->i", diff, diff)
    diameter = float(np.sqrt(np.max(dsq)))
    if not np.isfinite(diameter) or diameter <= 1.0e-14:
        polished = np.asarray(best, dtype=float).copy()
        polished -= np.mean(polished, axis=0, keepdims=True)
        diff = polished[pairs[:, 0]] - polished[pairs[:, 1]]
        diameter = float(np.sqrt(np.max(np.einsum("ij,ij->i", diff, diff))))
    return np.asarray(polished / diameter, dtype=float)