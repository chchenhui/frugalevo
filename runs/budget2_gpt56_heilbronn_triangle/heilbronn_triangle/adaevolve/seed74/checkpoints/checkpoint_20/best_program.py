# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """Globally optimize an 11-point reflection-symmetric maximin layout.

    Five triangle-safe left-half points are decoded from nested coordinates,
    reflected about x=0.5, and combined with one symmetry-axis point.
    Differential evolution directly maximizes the evaluator's normalized
    minimum unsigned doubled determinant over this feasible parameterization.
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

    # A left-half point is (x,y)=(u/2, h*u*v), so 0<=v<=1 guarantees
    # y<=sqrt(3)x.  Its reflection automatically lies in the right half.
    # The final parameter is the height fraction of the axis point.
    triples = np.array([
        (i, j, k)
        for i in range(n)
        for j in range(i + 1, n)
        for k in range(j + 1, n)
    ], dtype=np.intp)
    ii, jj, kk = triples.T

    def decode(z):
        uv = np.asarray(z[:10], dtype=float).reshape(5, 2)
        left = np.empty((5, 2), dtype=float)
        left[:, 0] = 0.5 * uv[:, 0]
        left[:, 1] = h * uv[:, 0] * uv[:, 1]
        points = np.empty((11, 2), dtype=float)
        points[:5] = left
        points[5:10] = left[::-1].copy()
        points[5:10, 0] = 1.0 - points[5:10, 0]
        points[10] = (0.5, h * z[10])
        return points

    def normalized_value(points):
        u = points[jj] - points[ii]
        v = points[kk] - points[ii]
        twice = np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])
        return float(np.min(twice) / np.sqrt(3.0))

    # The first three left points yield one bottom pair and two points on
    # each sloping side after reflection: a useful six-boundary scaffold.
    scaffold = np.array([
        0.44, 0.00,             # (0.22, 0)
        0.28, 1.00,             # left side
        0.68, 1.00,             # left side
        0.62, 0.82,             # interior
        0.86, 0.98,             # near upper interior
        0.72,                   # symmetry-axis point
    ], dtype=float)

    try:
        from scipy.optimize import differential_evolution, minimize

        rng = np.random.default_rng(11092026)
        # A supplied population allows both focused scaffold perturbations and
        # broad coverage while retaining entirely deterministic evolution.
        population = rng.uniform(0.0, 1.0, size=(352, 11))
        population[:, ::2][:, :5] = rng.uniform(0.015, 0.985, size=(352, 5))
        population[:144] = scaffold + rng.normal(0.0, 0.075, size=(144, 11))
        population[:144] = np.clip(population[:144], 0.0001, 0.9999)
        population[:144, ::2][:, :5] = np.clip(
            population[:144, ::2][:, :5], 0.015, 0.985
        )

        bounds = [(0.0001, 0.9999)] * 11
        for q in range(0, 10, 2):
            bounds[q] = (0.015, 0.985)

        result = differential_evolution(
            lambda z: -normalized_value(decode(z)),
            bounds=bounds,
            init=population,
            strategy="best1bin",
            maxiter=900,
            mutation=(0.45, 1.35),
            recombination=0.82,
            tol=1e-9,
            polish=False,
            seed=11092026,
            workers=1,
            updating="immediate",
        )

        # Differential evolution is the global stage.  Its final population
        # normally identifies the correct active-triangle basin, but its raw
        # maximin objective is nonsmooth.  Refine only this one basin with an
        # epigraph SLSQP solve in physical, reflection-symmetric coordinates.
        raw = decode(result.x)
        candidate = raw.copy()
        candidate_value = normalized_value(candidate)

        # Variables are the five left-half Cartesian points and the axis
        # height.  The reflected partners are reconstructed exactly, so this
        # polish retains the same 11-point symmetry class as DE.
        def unpack(w):
            left = w[:10].reshape(5, 2)
            p = np.empty((11, 2), dtype=float)
            p[:5] = left
            p[5:10] = left[::-1]
            p[5:10, 0] = 1.0 - p[5:10, 0]
            p[10] = (0.5, w[10])
            return p

        def epigraph_constraint(w):
            p = unpack(w)
            u = p[jj] - p[ii]
            v = p[kk] - p[ii]
            a = np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]) / np.sqrt(3.0)
            left = w[:10].reshape(5, 2)
            domain = np.column_stack((
                left[:, 0],
                left[:, 1],
                r3 * left[:, 0] - left[:, 1],
            )).ravel()
            return np.r_[a - w[11], domain, w[10], h - w[10]]

        def epigraph_jacobian(w):
            p = unpack(w)
            u = p[jj] - p[ii]
            v = p[kk] - p[ii]
            det = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
            s = np.where(det >= 0.0, 1.0, -1.0) / np.sqrt(3.0)
            jac = np.zeros((len(ii) + 17, 12), dtype=float)

            # Gradients of det with respect to the three physical points.
            grads = (
                np.column_stack((u[:, 1] - v[:, 1], v[:, 0] - u[:, 0])),
                np.column_stack((v[:, 1], -v[:, 0])),
                np.column_stack((-u[:, 1], u[:, 0])),
            )
            for inds, g in zip((ii, jj, kk), grads):
                g = g * s[:, None]
                for row, point in enumerate(inds):
                    if point < 5:
                        jac[row, 2 * point:2 * point + 2] += g[row]
                    elif point < 10:
                        # Point 9-point is the reflection of left point.
                        q = 9 - point
                        jac[row, 2 * q] -= g[row, 0]
                        jac[row, 2 * q + 1] += g[row, 1]
                    else:
                        jac[row, 10] += g[row, 1]

            jac[:len(ii), 11] = -1.0
            offset = len(ii)
            for q in range(5):
                jac[offset + 3 * q, 2 * q] = 1.0
                jac[offset + 3 * q + 1, 2 * q + 1] = 1.0
                jac[offset + 3 * q + 2, 2 * q] = r3
                jac[offset + 3 * q + 2, 2 * q + 1] = -1.0
            jac[offset + 15, 10] = 1.0
            jac[offset + 16, 10] = -1.0
            return jac

        try:
            z0 = np.r_[raw[:5].ravel(), raw[10, 1], 0.999 * candidate_value]
            polished = minimize(
                lambda w: -w[11],
                z0,
                jac=lambda w: np.r_[np.zeros(11), -1.0],
                method="SLSQP",
                bounds=([(0.0, 0.5), (0.0, h)] * 5
                        + [(0.0, h), (0.0, 0.5)]),
                constraints={
                    "type": "ineq",
                    "fun": epigraph_constraint,
                    "jac": epigraph_jacobian,
                },
                options={"maxiter": 1200, "ftol": 2e-13, "disp": False},
            )
            if polished.x is not None and np.all(np.isfinite(polished.x)):
                refined = unpack(polished.x)
                refined_value = normalized_value(refined)
                if refined_value > candidate_value:
                    candidate = refined
                    candidate_value = refined_value
        except Exception:
            # The DE output is already feasible and remains the safe result.
            pass

        # A tiny homothety about the centroid makes boundary comparisons
        # robust while preserving reflection symmetry and distinctness.
        centroid = np.array([0.5, h / 3.0])
        inward = (1.0 - 1e-10) * candidate + 1e-10 * centroid
        inward_value = normalized_value(inward)
        if inward_value >= candidate_value * (1.0 - 1e-8):
            candidate = inward
            candidate_value = inward_value

        # Preserve the independent asymmetric fallback if either optimizer
        # fails unexpectedly or the symmetric family is inferior.
        if np.all(np.isfinite(candidate)) and candidate_value > normalized_value(seed):
            return candidate
    except Exception:
        pass

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
