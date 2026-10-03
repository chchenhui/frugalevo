# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """Use deterministic multistart SLSQP to maximize the minimum of all
    165 triangle areas, subject to equilateral-triangle half-plane constraints."""
    n = 11
    h = np.sqrt(3.0) / 2.0
    root3 = np.sqrt(3.0)

    # Two points on each side provides a useful boundary-active scaffold,
    # while avoiding the three collinear points produced by side grids.
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
        from scipy.optimize import minimize, LinearConstraint
    except Exception:
        return seed

    triples = np.array([
        (i, j, k)
        for i in range(n)
        for j in range(i + 1, n)
        for k in range(j + 1, n)
    ], dtype=np.intp)
    ii, jj, kk = triples.T
    m = len(triples)

    def twice_areas(points):
        u = points[jj] - points[ii]
        v = points[kk] - points[ii]
        return np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])

    def score(points):
        return 0.5 * float(np.min(twice_areas(points)))

    # Variables are x0,y0,...,x10,y10,t.  The final variable t is the
    # guaranteed minimum triangle area.
    rows, lower, upper = [], [], []
    for q in range(n):
        for cx, cy, lo, hi in (
            (1.0, 0.0, 0.0, np.inf),       # x >= 0
            (0.0, 1.0, 0.0, np.inf),       # y >= 0
            (root3, -1.0, 0.0, np.inf),    # y <= sqrt(3) x
            (root3, 1.0, -np.inf, root3),  # y <= sqrt(3)(1-x)
        ):
            row = np.zeros(2 * n + 1)
            row[2 * q] = cx
            row[2 * q + 1] = cy
            rows.append(row)
            lower.append(lo)
            upper.append(hi)

    domain = LinearConstraint(
        np.asarray(rows), np.asarray(lower), np.asarray(upper)
    )

    def area_constraints(z):
        return 0.5 * twice_areas(z[:-1].reshape(n, 2)) - z[-1]

    def area_jacobian(z):
        points = z[:-1].reshape(n, 2)
        u = points[jj] - points[ii]
        v = points[kk] - points[ii]
        signed = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
        sign = np.where(signed >= 0.0, 1.0, -1.0)

        jac = np.zeros((m, 2 * n + 1))
        r = np.arange(m)
        grad_i = np.column_stack((u[:, 1] - v[:, 1], v[:, 0] - u[:, 0]))
        grad_j = np.column_stack((v[:, 1], -v[:, 0]))
        grad_k = np.column_stack((-u[:, 1], u[:, 0]))

        for indices, grad in ((ii, grad_i), (jj, grad_j), (kk, grad_k)):
            jac[r, 2 * indices] += 0.5 * sign * grad[:, 0]
            jac[r, 2 * indices + 1] += 0.5 * sign * grad[:, 1]
        jac[:, -1] = -1.0
        return jac

    rng = np.random.default_rng(11092026)
    starts = [seed]

    # Side-parameterized starts preserve feasibility and explore distinct
    # active-set arrangements around the productive boundary scaffold.
    for trial in range(24):
        p = seed.copy()
        bottom = np.sort(rng.uniform(0.10, 0.90, 2))
        left = np.sort(rng.uniform(0.10, 0.90, 2))
        right = np.sort(rng.uniform(0.10, 0.90, 2))
        p[:2] = np.column_stack((bottom, np.zeros(2)))
        p[2:4] = np.column_stack((0.5 * left, h * left))
        p[4:6] = np.column_stack((1.0 - 0.5 * right, h * right))

        bary = rng.dirichlet((1.1, 1.1, 1.1), size=5)
        interior = np.column_stack((
            bary[:, 1] + 0.5 * bary[:, 2],
            h * bary[:, 2],
        ))
        alpha = 0.12 + 0.75 * trial / 23.0
        p[6:] = (1.0 - alpha) * seed[6:] + alpha * interior
        starts.append(p)

    nonlinear = {
        "type": "ineq",
        "fun": area_constraints,
        "jac": area_jacobian,
    }
    objective_jac = np.r_[np.zeros(2 * n), -1.0]
    best = seed.copy()
    best_value = score(best)

    for start in starts:
        z0 = np.r_[start.ravel(), max(0.0, 0.80 * score(start))]
        try:
            result = minimize(
                fun=lambda z: -z[-1],
                x0=z0,
                jac=lambda z: objective_jac,
                method="SLSQP",
                constraints=[domain, nonlinear],
                bounds=[(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)],
                options={"maxiter": 1300, "ftol": 2e-12, "disp": False},
            )
        except Exception:
            continue

        if result.x is None or not np.all(np.isfinite(result.x)):
            continue
        candidate = result.x[:-1].reshape(n, 2)
        value = score(candidate)
        inside = (
            np.all(candidate[:, 0] >= -1e-8)
            and np.all(candidate[:, 1] >= -1e-8)
            and np.all(root3 * candidate[:, 0] - candidate[:, 1] >= -1e-8)
            and np.all(root3 * candidate[:, 0] + candidate[:, 1] <= root3 + 1e-8)
        )
        if inside and np.isfinite(value) and value > best_value:
            best = candidate
            best_value = value

    # A final solve from the best candidate can improve the active constraints.
    z0 = np.r_[best.ravel(), 0.995 * best_value]
    try:
        result = minimize(
            fun=lambda z: -z[-1],
            x0=z0,
            jac=lambda z: objective_jac,
            method="SLSQP",
            constraints=[domain, nonlinear],
            bounds=[(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)],
            options={"maxiter": 1800, "ftol": 5e-13, "disp": False},
        )
        if result.x is not None and np.all(np.isfinite(result.x)):
            candidate = result.x[:-1].reshape(n, 2)
            value = score(candidate)
            if value > best_value:
                best = candidate
    except Exception:
        pass

    return np.asarray(best, dtype=float).reshape(11, 2)


# EVOLVE-BLOCK-END
