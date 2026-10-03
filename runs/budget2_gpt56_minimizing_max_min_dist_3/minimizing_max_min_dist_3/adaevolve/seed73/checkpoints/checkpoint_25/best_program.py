# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """Maximize the exact squared minimum/diameter ratio with SLSQP.

    The diameter is fixed to one.  A scalar t is maximized subject to all
    squared pair distances satisfying t <= ||p_i-p_j||^2 <= 1.  Translation
    is eliminated by deriving the last point from the centroid condition,
    while three coordinate gauges remove arbitrary rotations.
    """
    from scipy.optimize import minimize

    rng = np.random.default_rng(20260912)
    n = 14
    ii, jj = np.triu_indices(n, 1)
    m = len(ii)

    a = np.sqrt(3.0)
    seed = np.vstack((
        np.array([
            [a, 0.0, 0.0], [-a, 0.0, 0.0],
            [0.0, a, 0.0], [0.0, -a, 0.0],
            [0.0, 0.0, a], [0.0, 0.0, -a],
        ]),
        np.array([
            [sx, sy, sz]
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ]),
    ))

    # p[0,y], p[0,z], and p[1,z] are fixed to zero.  Point 13 is minus the
    # sum of points 0,...,12, imposing the centroid constraint exactly.
    free = np.array([k for k in range(39) if k not in (1, 2, 5)])
    B = np.zeros((42, 36))
    B[free, np.arange(36)] = 1.0
    B[39:42] = -B[:39].sum(axis=0)
    BP = B.reshape(14, 3, 36)
    pair_map = BP[ii] - BP[jj]

    def gauge_and_scale(p):
        """Center, orient p0 along x, p1 into xy, and set diameter to one."""
        p = p - p.mean(axis=0)
        e1 = p[0] / np.linalg.norm(p[0])
        v = p[1] - np.dot(p[1], e1) * e1
        if np.linalg.norm(v) < 1.0e-10:
            v = np.array([0.0, 1.0, 0.0])
            v -= np.dot(v, e1) * e1
        e2 = v / np.linalg.norm(v)
        e3 = np.cross(e1, e2)
        p = p @ np.column_stack((e1, e2, e3))
        d = p[ii] - p[jj]
        p /= np.sqrt(np.einsum("ij,ij->i", d, d).max())
        return p

    def unpack(y):
        return (B @ y[:36]).reshape(14, 3)

    def distances(y):
        p = unpack(y)
        d = p[ii] - p[jj]
        return p, d, np.einsum("ij,ij->i", d, d)

    def constraints(y):
        _, _, q = distances(y)
        return np.concatenate((q - y[36], 1.0 - q))

    def constraint_jacobian(y):
        _, d, _ = distances(y)
        dq = 2.0 * np.einsum("pi,piv->pv", d, pair_map)
        jac = np.zeros((2 * m, 37))
        jac[:m, :36] = dq
        jac[:m, 36] = -1.0
        jac[m:, :36] = -dq
        return jac

    def ratio(p):
        d = p[ii] - p[jj]
        q = np.einsum("ij,ij->i", d, d)
        return q.min() / q.max()

    def initial_vector(p):
        p = gauge_and_scale(p)
        q = np.einsum("ij,ij->i", p[ii] - p[jj], p[ii] - p[jj])
        return np.concatenate((p.ravel()[free], [0.999 * q.min()]))

    best = gauge_and_scale(seed)
    best_value = ratio(best)
    constraint = {
        "type": "ineq",
        "fun": constraints,
        "jac": constraint_jacobian,
    }

    # These are all deterministic perturbations of the geometrically relevant
    # rhombic-dodecahedral arrangement.  Each solve uses the exact epigraph
    # problem rather than a soft surrogate or a stochastic acceptance rule.
    starts = [seed]
    for noise, count in ((0.025, 3), (0.060, 4), (0.120, 4), (0.220, 3)):
        for _ in range(count):
            starts.append(seed + noise * rng.standard_normal((n, 3)))

    for p0 in starts:
        result = minimize(
            lambda y: -y[36],
            initial_vector(p0),
            jac=lambda y: np.r_[np.zeros(36), -1.0],
            method="SLSQP",
            bounds=[(None, None)] * 36 + [(0.0, 1.0)],
            constraints=constraint,
            options={"ftol": 1.0e-12, "maxiter": 1800, "disp": False},
        )
        candidate = unpack(result.x)
        value = ratio(candidate)
        if np.isfinite(value) and value > best_value:
            best = candidate
            best_value = value

    # Exact diameter scaling avoids any tiny SLSQP upper-constraint tolerance.
    d = best[ii] - best[jj]
    best /= np.sqrt(np.einsum("ij,ij->i", d, d).max())
    return np.asarray(best, dtype=float)

    def normalize(x):
        x = x - x.mean(axis=0)
        return x / np.sqrt(np.mean(np.sum(x * x, axis=1)))

    def ratio_squared(x):
        delta = x[ii] - x[jj]
        q = np.einsum("ij,ij->i", delta, delta)
        return q.min() / q.max()

    best = normalize(seed)
    best_value = ratio_squared(best)

    # A range of perturbation sizes explores both nearby symmetry-breaking
    # basins and genuinely distinct configurations.  The exact symmetric
    # seed remains available in case no perturbation improves it.
    starts = [best]
    for noise in (0.045, 0.080, 0.120, 0.180, 0.260, 0.360):
        for _ in range(4):
            starts.append(
                normalize(seed + noise * rng.standard_normal((n, 3)))
            )

    for x in starts:
        x = x.copy()
        for it in range(4200):
            delta = x[ii] - x[jj]
            q = np.einsum("ij,ij->i", delta, delta)

            # Differentiable soft minimum and maximum of squared distances.
            tau = 0.045 * (0.12 ** (it / 4199.0))
            qmin = q.min()
            emin = np.exp(-(q - qmin) / tau)
            wmin = emin / emin.sum()
            smooth_min = qmin - tau * np.log(emin.sum())

            qmax = q.max()
            emax = np.exp((q - qmax) / tau)
            wmax = emax / emax.sum()
            smooth_max = qmax + tau * np.log(emax.sum())

            # Gradient of log(smooth_min / smooth_max).
            weights = wmin / smooth_min - wmax / smooth_max
            grad = np.zeros_like(x)
            contribution = 2.0 * weights[:, None] * delta
            np.add.at(grad, ii, contribution)
            np.add.at(grad, jj, -contribution)

            step = 0.030 * (1.0 - 0.70 * it / 4199.0)
            x = normalize(x + step * grad)

            value = ratio_squared(x)
            if value > best_value:
                best_value = value
                best = x.copy()

    # Exact-objective polishing.  Limiting contacts often cannot be improved
    # by moving one point alone: several vertices must move coherently before
    # a new contact graph becomes favorable.  Therefore mix ordinary local
    # moves with pair and small active-set perturbations.
    current = best.copy()
    current_value = best_value
    radius = 0.0045
    epoch = 30000

    for trial in range(180000):
        delta = current[ii] - current[jj]
        q = np.einsum("ij,ij->i", delta, delta)
        lo, hi = q.min(), q.max()

        # A slightly wider active band keeps vertices available immediately
        # after a tied shortest/diameter contact splits.
        active = (q <= lo * 1.003) | (q >= hi * 0.997)
        vertices = np.unique(np.concatenate((ii[active], jj[active])))

        candidate = current.copy()
        mode = trial % 10

        if mode < 6:
            # Fine one-vertex coordinate move.
            chosen = vertices[rng.integers(len(vertices), size=1)]
            candidate[chosen] += radius * rng.standard_normal((1, 3))
        elif mode < 9:
            # Two active vertices may move independently, allowing a contact
            # edge to rotate without requiring an unfavorable intermediate.
            count = min(2, len(vertices))
            chosen = rng.choice(vertices, size=count, replace=False)
            candidate[chosen] += radius * rng.standard_normal((count, 3))
        else:
            # Coherent cluster move explores changes of the active graph while
            # retaining nearby internal distances.
            count = min(4, len(vertices))
            chosen = rng.choice(vertices, size=count, replace=False)
            shift = radius * rng.standard_normal(3)
            candidate[chosen] += shift
            candidate[chosen] += 0.35 * radius * rng.standard_normal((count, 3))

        candidate = normalize(candidate)
        value = ratio_squared(candidate)

        phase = trial % epoch
        temperature = 2.0e-5 * (0.02 ** (phase / (epoch - 1)))
        if value >= current_value or rng.random() < np.exp(
            (value - current_value) / temperature
        ):
            current = candidate
            current_value = value
            radius = min(0.018, radius * 1.004)
            if value > best_value:
                best = candidate.copy()
                best_value = value
        else:
            radius = max(8.0e-6, radius * 0.99970)

        # Independent annealing epochs are restarted from the incumbent,
        # retaining gains while repeatedly attempting different ridge crossings.
        if phase == epoch - 1:
            current = best.copy()
            current_value = best_value
            radius = 0.0035

    return np.asarray(best, dtype=float)


# EVOLVE-BLOCK-END
