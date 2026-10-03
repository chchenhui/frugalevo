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

    # Optimize a dimerized pentagonal-bipyramid contact topology before
    # releasing the coordinates to unrestricted Cartesian annealing.
    starts = []

    def dimerized(params):
        """Construct seven oriented dimers around a pentagonal bipyramid."""
        params = np.asarray(params, dtype=float)
        if params.shape != (7,) or not np.all(np.isfinite(params)):
            return np.zeros((n, d), dtype=float)

        r, h, le, lp, beta, azp, azm = params
        if min(r, h, le, lp) <= 1.0e-8:
            return np.zeros((n, d), dtype=float)

        angles = 2.0 * np.pi * np.arange(5) / 5.0
        centers = np.zeros((7, 3), dtype=float)
        centers[:5, 0] = r * np.cos(angles)
        centers[:5, 1] = r * np.sin(angles)
        centers[5, 2] = h
        centers[6, 2] = -h

        directions = np.zeros((7, 3), dtype=float)
        for k, a in enumerate(angles):
            tangent = np.array([-np.sin(a), np.cos(a), 0.0])
            vertical = np.array([0.0, 0.0, 1.0])
            directions[k] = (
                np.cos(beta + a) * tangent + np.sin(beta + a) * vertical
            )
        directions[5] = np.array([np.cos(azp), np.sin(azp), 0.0])
        directions[6] = np.array([np.cos(azm), np.sin(azm), 0.0])

        lengths = np.array([le] * 5 + [lp, lp], dtype=float)
        plus = centers + lengths[:, None] * directions
        minus = centers - lengths[:, None] * directions
        p = np.empty((14, 3), dtype=float)
        p[0::2] = plus
        p[1::2] = minus

        diff = p[pairs[:, 0]] - p[pairs[:, 1]]
        diameter = float(np.sqrt(np.max(np.einsum("ij,ij->i", diff, diff))))
        if not np.isfinite(diameter) or diameter <= 1.0e-12:
            return np.zeros((n, d), dtype=float)
        return np.asarray(p / diameter, dtype=float)

    def dimerized_objective(params):
        """Evaluate the exact negative squared minimum-to-diameter ratio."""
        p = dimerized(params)
        return 1.0e6 if not np.any(p) else -ratio2(p)

    # Bounds prevent Nelder-Mead simplex excursions from creating degenerate
    # dimers or numerically useless huge centers; angles remain unrestricted
    # modulo periodicity through a broad fixed interval.
    dimer_seeds = (
        # Compact-height basin with nearly tangential belt dimers.
        np.array([0.86, 0.76, 0.145, 0.125, 0.04, 0.00, np.pi / 2.0]),
        # Tall basin with a staggered belt and oblique polar dimers.
        np.array([0.98, 0.88, 0.105, 0.175, 0.30,
                  np.pi / 5.0, 4.0 * np.pi / 5.0]),
        # Short polar dimers and a shallow pentagonal bipyramid.
        np.array([0.62, 0.96, 0.205, 0.085, -0.24,
                  np.pi / 2.0, -np.pi / 2.0]),
        # Large-height basin with a substantially different belt twist.
        np.array([1.08, 0.70, 0.125, 0.145, 0.58,
                  0.00, np.pi]),
    )
    dimer_bounds = (
        (0.20, 1.80), (0.20, 1.80), (0.025, 0.65), (0.025, 0.65),
        (-2.0 * np.pi, 2.0 * np.pi),
        (-2.0 * np.pi, 2.0 * np.pi), (-2.0 * np.pi, 2.0 * np.pi),
    )
    candidates = []
    for seed in dimer_seeds:
        try:
            fit = minimize(
                dimerized_objective, seed, method="Nelder-Mead",
                bounds=dimer_bounds,
                options={"maxiter": 260, "xatol": 1.0e-8, "fatol": 1.0e-10},
            )
            candidate = dimerized(fit.x)
            if candidate.shape == (n, d) and np.all(np.isfinite(candidate)):
                candidates.append((ratio2(candidate), candidate.copy()))
        except Exception:
            pass
    candidates.sort(key=lambda item: item[0], reverse=True)
    # The dimerized family is the primary construction.  Only its two best
    # exact candidates are released, preventing weaker parameter basins from
    # consuming the unrestricted annealing budget.
    starts.extend(candidate.copy() for _, candidate in candidates[:2])

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

    # Use a collective smooth-min/smooth-max continuation in gauge-fixed
    # Cartesian coordinates.  This replaces discontinuous one-point annealing
    # with bounded L-BFGS-B solves while retaining exact-ratio selection.
    def smooth_normalize(x):
        """Unpack, recenter, and diameter-normalize a gauge-fixed vector."""
        p = np.zeros((n, 3), dtype=float)
        p[1:] = np.asarray(x, dtype=float).reshape(n - 1, 3)
        p -= np.mean(p, axis=0, keepdims=True)
        v = p[pairs[:, 0]] - p[pairs[:, 1]]
        z = np.einsum("ij,ij->i", v, v)
        den = float(np.sqrt(np.max(z)))
        if not np.isfinite(den) or den <= 1.0e-14:
            return None
        return p / den

    def smooth_objective(x, tau_min, tau_max):
        """Evaluate the negative soft bottleneck diameter objective."""
        p = smooth_normalize(x)
        if p is None or not np.all(np.isfinite(p)):
            return 1.0e6
        v = p[pairs[:, 0]] - p[pairs[:, 1]]
        z = np.einsum("ij,ij->i", v, v)
        a = -z / tau_min
        b = z / tau_max
        ma = float(np.max(a))
        mb = float(np.max(b))
        soft_min = -tau_min * (ma + np.log(np.sum(np.exp(a - ma))))
        soft_max = tau_max * (mb + np.log(np.sum(np.exp(b - mb))))
        value = soft_min - soft_max
        return float(-value) if np.isfinite(value) else 1.0e6

    # Always include the deterministic Fibonacci configuration as a third
    # bounded continuation start, even when the dimerized family succeeds.
    continuation_starts = [np.asarray(s, dtype=float).copy()
                           for s in starts[:2]]
    continuation_starts.append(base.copy())

    best = None
    best_value = -np.inf
    schedules = (
        (0.0300, 0.0300),
        (0.0120, 0.0120),
        (0.0040, 0.0040),
        (0.0012, 0.0012),
        (0.0004, 0.0004),
    )

    for initial in continuation_starts[:3]:
        current = np.asarray(initial, dtype=float).copy()
        current_value = ratio2(current)
        for tau_min, tau_max in schedules:
            x0 = np.asarray(current[1:], dtype=float).ravel()
            try:
                fit = minimize(
                    smooth_objective, x0,
                    args=(tau_min, tau_max),
                    method="L-BFGS-B",
                    options={
                        "maxiter": 90,
                        "maxls": 20,
                        "ftol": 1.0e-12,
                        "gtol": 1.0e-8,
                    },
                )
                trial = smooth_normalize(fit.x)
                if trial is not None and np.all(np.isfinite(trial)):
                    value = ratio2(trial)
                    if value >= current_value:
                        current = trial.copy()
                        current_value = value
            except Exception:
                pass

        if current_value > best_value:
            best_value = current_value
            best = current.copy()

    if best is None:
        best = np.asarray(base, dtype=float).copy()

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