# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """Maximize the minimum of all 165 triangle areas by deterministic
    boundary-aware multistart SLSQP with analytic area derivatives."""
    h = np.sqrt(3.0) / 2.0
    n = 11

    # This seed deliberately has two points on each side but no three points
    # on one side.  It is substantially better than a regular triangular grid,
    # which necessarily contains many zero-area triples.
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
        from scipy.optimize import LinearConstraint, minimize
    except Exception:
        # The seed is feasible and contains no intentionally collinear triple.
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

    # Variables are x_0,y_0,...,x_10,y_10,t.  The four half planes are
    # x >= 0, y >= 0, sqrt(3)x-y >= 0, sqrt(3)x+y <= sqrt(3).
    rows, lo, hi = [], [], []
    root3 = np.sqrt(3.0)
    for q in range(n):
        for cx, cy, lower, upper in (
            (1.0, 0.0, 0.0, np.inf),
            (0.0, 1.0, 0.0, np.inf),
            (root3, -1.0, 0.0, np.inf),
            (root3, 1.0, -np.inf, root3),
        ):
            row = np.zeros(2 * n + 1)
            row[2 * q] = cx
            row[2 * q + 1] = cy
            rows.append(row)
            lo.append(lower)
            hi.append(upper)
    domain = LinearConstraint(np.asarray(rows), np.asarray(lo), np.asarray(hi))

    def constraints(z):
        p = z[:-1].reshape(n, 2)
        return 0.5 * twice_areas(p) - z[-1]

    def constraint_jacobian(z):
        p = z[:-1].reshape(n, 2)
        u = p[jj] - p[ii]
        v = p[kk] - p[ii]
        signed = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
        s = np.where(signed >= 0.0, 1.0, -1.0)

        jac = np.zeros((m, 2 * n + 1))
        r = np.arange(m)

        # Derivatives of (p_j-p_i) cross (p_k-p_i).
        gi = np.column_stack((u[:, 1] - v[:, 1], v[:, 0] - u[:, 0]))
        gj = np.column_stack((v[:, 1], -v[:, 0]))
        gk = np.column_stack((-u[:, 1], u[:, 0]))

        for ind, grad in ((ii, gi), (jj, gj), (kk, gk)):
            jac[r, 2 * ind] += 0.5 * s * grad[:, 0]
            jac[r, 2 * ind + 1] += 0.5 * s * grad[:, 1]
        jac[:, -1] = -1.0
        return jac

    # Good local optima usually have several active side constraints.  Rather
    # than fixing those six locations in every structured restart, vary their
    # side parameters independently.  This explores different combinatorial
    # active sets while retaining two, and only two, points per side.
    rng = np.random.default_rng(11092026)
    starts = [seed]
    for trial in range(32):
        p = seed.copy()

        # Parameters s give (s,0), (.5*s,h*s), and (1-.5*s,h*s) on the
        # bottom, left, and right sides respectively.  Sorting each pair
        # avoids duplicate/order-degenerate initial configurations.
        bottom = np.sort(rng.uniform(0.12, 0.88, 2))
        left = np.sort(rng.uniform(0.12, 0.88, 2))
        right = np.sort(rng.uniform(0.12, 0.88, 2))
        p[0:2] = np.column_stack((bottom, np.zeros(2)))
        p[2:4] = np.column_stack((0.5 * left, h * left))
        p[4:6] = np.column_stack((1.0 - 0.5 * right, h * right))

        # Convex mixing keeps every interior start feasible.  Early starts
        # deliberately stay close to the productive seed; later ones permit
        # substantially different oriented-area patterns.
        bary = rng.dirichlet((1.15, 1.15, 1.15), size=5)
        interior = np.column_stack((
            bary[:, 1] + 0.5 * bary[:, 2],
            h * bary[:, 2],
        ))
        alpha = 0.16 + 0.72 * trial / 31.0
        p[6:] = (1.0 - alpha) * seed[6:] + alpha * interior
        starts.append(p)

    # A few entirely interior starts remain useful for sign patterns not
    # reachable continuously from a six-side-point configuration.
    for _ in range(8):
        bary = rng.dirichlet((0.9, 0.9, 0.9), size=n)
        starts.append(np.column_stack((
            bary[:, 1] + 0.5 * bary[:, 2],
            h * bary[:, 2],
        )))

    best = seed.copy()
    best_score = score(best)
    nonlinear = {
        "type": "ineq",
        "fun": constraints,
        "jac": constraint_jacobian,
    }
    objective_gradient = np.r_[np.zeros(2 * n), -1.0]

    for start in starts:
        initial_t = max(0.0, 0.85 * score(start))
        z0 = np.r_[start.ravel(), initial_t]
        try:
            result = minimize(
                fun=lambda z: -z[-1],
                x0=z0,
                jac=lambda z: objective_gradient,
                method="SLSQP",
                constraints=[domain, nonlinear],
                bounds=[(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)],
                options={"maxiter": 1100, "ftol": 2e-12, "disp": False},
            )
        except Exception:
            continue

        if result.x is None or not np.all(np.isfinite(result.x)):
            continue
        candidate = result.x[:-1].reshape(n, 2)
        candidate_score = score(candidate)

        # Explicit validation protects against a failed/inaccurate optimizer
        # termination being selected merely because its reported t is large.
        inside = (
            np.all(candidate[:, 0] >= -1e-8)
            and np.all(candidate[:, 1] >= -1e-8)
            and np.all(root3 * candidate[:, 0] - candidate[:, 1] >= -1e-8)
            and np.all(root3 * candidate[:, 0] + candidate[:, 1] <= root3 + 1e-8)
        )
        if inside and np.isfinite(candidate_score) and candidate_score > best_score:
            best = candidate
            best_score = candidate_score

    return best


# EVOLVE-BLOCK-END
