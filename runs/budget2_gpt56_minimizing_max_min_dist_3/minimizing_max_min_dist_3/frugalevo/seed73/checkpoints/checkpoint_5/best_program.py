import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """Soft spherical search followed by one exact fixed-diameter epigraph polish."""
    n = 14
    rng = np.random.default_rng(20250314)
    pi, pj = np.triu_indices(n, 1)
    m = len(pi)

    k = np.arange(n, dtype=float)
    z = 1.0 - 2.0 * (k + 0.5) / n
    theta = np.pi * (3.0 - np.sqrt(5.0)) * k
    base = np.column_stack((
        np.sqrt(1.0 - z * z) * np.cos(theta),
        np.sqrt(1.0 - z * z) * np.sin(theta),
        z,
    ))

    def ratio_of(p):
        q = p[pi] - p[pj]
        ds2 = np.einsum("ij,ij->i", q, q)
        if not np.all(np.isfinite(ds2)) or ds2.max() <= 0.0:
            return -np.inf
        return float(ds2.min() / ds2.max())

    best = base.copy()
    best_ratio = ratio_of(best)

    for restart in range(12):
        if restart == 0:
            points = base.copy()
        else:
            points = base + 0.32 * rng.normal(size=(n, 3))
            points /= np.linalg.norm(points, axis=1, keepdims=True)

        for iteration in range(2600):
            delta = points[pi] - points[pj]
            dist2 = np.einsum("ij,ij->i", delta, delta)
            beta = 8.0 + 150.0 * iteration / 2599.0
            scaled = -beta * (dist2 - dist2.min())
            weights = np.exp(np.maximum(scaled, -45.0))
            weights /= weights.sum()

            gradient = np.zeros_like(points)
            contribution = 2.0 * weights[:, None] * delta
            np.add.at(gradient, pi, contribution)
            np.add.at(gradient, pj, -contribution)
            tangent = gradient - np.sum(
                gradient * points, axis=1, keepdims=True
            ) * points

            step = 0.055 * (1.0 - 0.72 * iteration / 2599.0)
            points += step * tangent
            points /= np.linalg.norm(points, axis=1, keepdims=True)

        value = ratio_of(points)
        if value > best_ratio:
            best_ratio = value
            best = points.copy()

    # Directly optimize the measured maximin formulation at the incumbent
    # diameter.  This can exploit radial freedom omitted by sphere updates.
    try:
        from scipy.optimize import minimize

        delta = best[pi] - best[pj]
        ds2 = np.einsum("ij,ij->i", delta, delta)
        diameter2 = float(ds2.max())
        x0 = np.r_[best.ravel(), float(ds2.min())]

        def objective(x):
            return -x[-1]

        def objective_jac(x):
            g = np.zeros_like(x)
            g[-1] = -1.0
            return g

        def constraints(x):
            p = x[:-1].reshape(n, 3)
            d = p[pi] - p[pj]
            q = np.einsum("ij,ij->i", d, d)
            return np.r_[q - x[-1], diameter2 - q]

        def constraints_jac(x):
            p = x[:-1].reshape(n, 3)
            d = p[pi] - p[pj]
            jac = np.zeros((2 * m, 3 * n + 1))
            rows = np.arange(m)
            for axis in range(3):
                jac[rows, 3 * pi + axis] = 2.0 * d[:, axis]
                jac[rows, 3 * pj + axis] = -2.0 * d[:, axis]
                jac[m + rows, 3 * pi + axis] = -2.0 * d[:, axis]
                jac[m + rows, 3 * pj + axis] = 2.0 * d[:, axis]
            jac[:m, -1] = -1.0
            return jac

        # Translation is an exactly flat direction of every distance
        # constraint.  Remove it explicitly so SLSQP can spend its limited
        # iterations on the shape variables rather than wandering in that
        # nullspace.
        def centroid_constraint(x):
            return x[:-1].reshape(n, 3).mean(axis=0)

        def centroid_jac(x):
            jac = np.zeros((3, 3 * n + 1))
            for axis in range(3):
                jac[axis, axis::3] = 1.0 / n
            return jac

        result = minimize(
            objective,
            x0,
            jac=objective_jac,
            method="SLSQP",
            bounds=[(None, None)] * (3 * n) + [(0.0, diameter2)],
            constraints=[
                {"type": "ineq", "fun": constraints,
                 "jac": constraints_jac},
                {"type": "eq", "fun": centroid_constraint,
                 "jac": centroid_jac},
            ],
            options={"maxiter": 220, "ftol": 1e-11, "disp": False},
        )
        candidate = result.x[:-1].reshape(n, 3)
        candidate_ratio = ratio_of(candidate)
        if np.isfinite(candidate_ratio) and candidate_ratio > best_ratio:
            best = candidate
    except Exception:
        pass

    return np.asarray(best, dtype=float)