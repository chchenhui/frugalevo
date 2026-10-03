# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """Construct 11 points by deterministic multistart maximin optimization.

    An auxiliary variable is maximized subject to all 165 triangle areas being
    at least that value.  Starts place two points on each side of the outer
    equilateral triangle and five points in its interior; SLSQP then releases
    all coordinates and optimizes the complete arrangement.
    """
    n = 11
    h = np.sqrt(3.0) / 2.0
    r3 = np.sqrt(3.0)

    # This boundary-active scaffold has no three points forced onto one side.
    seed = np.array([
        [0.250000, 0.000000],
        [0.750000, 0.000000],
        [0.125000, 0.25 * h],
        [0.375000, 0.75 * h],
        [0.875000, 0.25 * h],
        [0.625000, 0.75 * h],
        [0.500000, 0.270000],
        [0.330000, 0.360000],
        [0.670000, 0.360000],
        [0.410000, 0.550000],
        [0.590000, 0.550000],
    ], dtype=float)

    try:
        from scipy.optimize import minimize
    except Exception:
        # A deterministic feasible fallback is retained if SciPy is absent.
        return seed

    triples = np.array([
        (i, j, k)
        for i in range(n)
        for j in range(i + 1, n)
        for k in range(j + 1, n)
    ], dtype=np.intp)
    ii, jj, kk = triples.T
    m = len(triples)

    def doubled_area(p):
        u = p[jj] - p[ii]
        v = p[kk] - p[ii]
        return np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])

    def value(p):
        return 0.5 * float(np.min(doubled_area(p)))

    def area_constraint(z):
        return 0.5 * doubled_area(z[:-1].reshape(n, 2)) - z[-1]

    def area_jacobian(z):
        p = z[:-1].reshape(n, 2)
        u = p[jj] - p[ii]
        v = p[kk] - p[ii]
        det = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
        sign = np.where(det >= 0.0, 1.0, -1.0)

        jac = np.zeros((m, 2 * n + 1))
        rows = np.arange(m)
        gi = np.column_stack((u[:, 1] - v[:, 1], v[:, 0] - u[:, 0]))
        gj = np.column_stack((v[:, 1], -v[:, 0]))
        gk = np.column_stack((-u[:, 1], u[:, 0]))

        for ind, grad in ((ii, gi), (jj, gj), (kk, gk)):
            jac[rows, 2 * ind] += 0.5 * sign * grad[:, 0]
            jac[rows, 2 * ind + 1] += 0.5 * sign * grad[:, 1]
        jac[:, -1] = -1.0
        return jac

    # For each point these are x >= 0, y >= 0, y <= sqrt(3)x, and
    # y <= sqrt(3)(1-x), respectively.
    def domain_constraint(z):
        p = z[:-1].reshape(n, 2)
        return np.column_stack((
            p[:, 0],
            p[:, 1],
            r3 * p[:, 0] - p[:, 1],
            r3 - r3 * p[:, 0] - p[:, 1],
        )).ravel()

    domain_jac = np.zeros((4 * n, 2 * n + 1))
    for q in range(n):
        domain_jac[4 * q + 0, 2 * q] = 1.0
        domain_jac[4 * q + 1, 2 * q + 1] = 1.0
        domain_jac[4 * q + 2, 2 * q] = r3
        domain_jac[4 * q + 2, 2 * q + 1] = -1.0
        domain_jac[4 * q + 3, 2 * q] = -r3
        domain_jac[4 * q + 3, 2 * q + 1] = -1.0

    rng = np.random.default_rng(11092026)
    starts = [seed]

    # Structured feasible restarts explore multiple side spacings and interior
    # oriented-area patterns while preserving the productive six-side family.
    for trial in range(28):
        p = seed.copy()
        bottom = np.sort(rng.uniform(0.10, 0.90, 2))
        left = np.sort(rng.uniform(0.10, 0.90, 2))
        right = np.sort(rng.uniform(0.10, 0.90, 2))

        p[0:2] = np.column_stack((bottom, np.zeros(2)))
        p[2:4] = np.column_stack((0.5 * left, h * left))
        p[4:6] = np.column_stack((1.0 - 0.5 * right, h * right))

        bary = rng.dirichlet((1.1, 1.1, 1.1), size=5)
        random_interior = np.column_stack((
            bary[:, 1] + 0.5 * bary[:, 2],
            h * bary[:, 2],
        ))
        alpha = 0.10 + 0.82 * trial / 27.0
        p[6:] = (1.0 - alpha) * seed[6:] + alpha * random_interior
        starts.append(p)

    constraints = [
        {"type": "ineq", "fun": domain_constraint, "jac": lambda z: domain_jac},
        {"type": "ineq", "fun": area_constraint, "jac": area_jacobian},
    ]
    objective_jac = np.r_[np.zeros(2 * n), -1.0]

    best = seed.copy()
    best_value = value(best)

    for start in starts:
        z0 = np.r_[start.ravel(), max(0.0, 0.80 * value(start))]
        try:
            result = minimize(
                lambda z: -z[-1],
                z0,
                jac=lambda z: objective_jac,
                method="SLSQP",
                constraints=constraints,
                bounds=[(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)],
                options={"maxiter": 1400, "ftol": 2e-12, "disp": False},
            )
        except Exception:
            continue

        if result.x is None or not np.all(np.isfinite(result.x)):
            continue
        candidate = result.x[:-1].reshape(n, 2)
        candidate_value = value(candidate)

        # Do not trust the optimizer's reported auxiliary variable without an
        # independent domain check and direct triangle-area recomputation.
        valid = (
            np.all(candidate[:, 0] >= -1e-8)
            and np.all(candidate[:, 1] >= -1e-8)
            and np.all(r3 * candidate[:, 0] - candidate[:, 1] >= -1e-8)
            and np.all(r3 * candidate[:, 0] + candidate[:, 1] <= r3 + 1e-8)
        )
        if valid and np.isfinite(candidate_value) and candidate_value > best_value:
            best = candidate
            best_value = candidate_value

    # A high-accuracy continuation is particularly useful once the active
    # collection of small triangles has been identified by the restart phase.
    try:
        result = minimize(
            lambda z: -z[-1],
            np.r_[best.ravel(), 0.995 * best_value],
            jac=lambda z: objective_jac,
            method="SLSQP",
            constraints=constraints,
            bounds=[(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)],
            options={"maxiter": 2000, "ftol": 5e-13, "disp": False},
        )
        if result.x is not None and np.all(np.isfinite(result.x)):
            candidate = result.x[:-1].reshape(n, 2)
            candidate_value = value(candidate)
            valid = (
                np.all(candidate[:, 0] >= -1e-8)
                and np.all(candidate[:, 1] >= -1e-8)
                and np.all(r3 * candidate[:, 0] - candidate[:, 1] >= -1e-8)
                and np.all(r3 * candidate[:, 0] + candidate[:, 1] <= r3 + 1e-8)
            )
            if valid and candidate_value > best_value:
                best = candidate
    except Exception:
        pass

    return np.asarray(best, dtype=float).reshape(11, 2)


# EVOLVE-BLOCK-END
