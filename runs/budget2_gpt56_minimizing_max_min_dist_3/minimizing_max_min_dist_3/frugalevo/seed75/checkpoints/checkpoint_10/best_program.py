import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """Annealed spherical construction followed by exact diameter-constrained polish."""
    n = 14
    rng = np.random.default_rng(20260912)
    ii, jj = np.triu_indices(n, 1)
    m = len(ii)

    index = np.arange(n, dtype=float)
    z = 1.0 - 2.0 * (index + 0.5) / n
    golden = np.pi * (3.0 - np.sqrt(5.0))
    seed = np.column_stack((
        np.sqrt(1.0 - z * z) * np.cos(golden * index),
        np.sqrt(1.0 - z * z) * np.sin(golden * index),
        z,
    ))

    def distances2(x):
        d = x[ii] - x[jj]
        return np.einsum("ij,ij->i", d, d)

    def ratio(x):
        q = distances2(x)
        return float(q.min() / q.max())

    best = seed.copy()
    best_value = ratio(best)

    # Explore unrestricted radial configurations with smooth Riesz-energy
    # continuation, retaining the best state for the exact polishing stage.
    for restart in range(8):
        if restart == 0:
            points = seed.copy()
        else:
            points = rng.normal(size=(n, 3))
        points = points.astype(float, copy=True)
        points -= points.mean(axis=0, keepdims=True)
        diameter = np.sqrt(distances2(points).max())
        if not np.isfinite(diameter) or diameter <= 0.0:
            continue
        points /= diameter

        # Increasing powers progressively replace smooth repulsion by a
        # max-min separation objective while preserving unrestricted radii.
        for power in (2, 4, 8, 16):
            epsilon = 2.0e-3 / power
            for iteration in range(180):
                delta = points[ii] - points[jj]
                q = np.einsum("ij,ij->i", delta, delta)
                weight = -2.0 * power * (q + epsilon) ** (-power - 1)
                gradient = np.zeros_like(points)
                np.add.at(gradient, ii, weight[:, None] * delta)
                np.add.at(gradient, jj, -weight[:, None] * delta)

                scale = np.max(np.linalg.norm(gradient, axis=1))
                if not np.isfinite(scale) or scale <= 1.e-14:
                    break
                learning_rate = 0.075 * (1.0 - 0.55 *
                                         iteration / 179.0)
                points -= learning_rate * gradient / scale
                points -= points.mean(axis=0, keepdims=True)
                diameter = np.sqrt(distances2(points).max())
                if not np.isfinite(diameter) or diameter <= 0.0:
                    break
                points /= diameter

        value = ratio(points)
        if np.isfinite(value) and value > best_value:
            best, best_value = points.copy(), value

    # Directly optimize the evaluator-visible min/diameter objective by
    # fixing squared diameter to one and maximizing the common lower bound.
    try:
        from scipy.optimize import minimize

        def normalize(x):
            x = np.asarray(x, dtype=float).reshape(n, 3).copy()
            x -= x.mean(axis=0, keepdims=True)
            qmax = distances2(x).max()
            if not np.isfinite(qmax) or qmax <= 0.0:
                return None
            return x / np.sqrt(qmax)

        def constraints(v):
            x = v[:-1].reshape(n, 3)
            t = v[-1]
            q = distances2(x)
            return np.concatenate((q - t, 1.0 - q))

        def constraint_jacobian(v):
            x = v[:-1].reshape(n, 3)
            delta = x[ii] - x[jj]
            jac = np.zeros((2 * m, 3 * n + 1), dtype=float)
            rows = np.arange(m)
            for coord in range(3):
                jac[rows, 3 * ii + coord] = 2.0 * delta[:, coord]
                jac[rows, 3 * jj + coord] = -2.0 * delta[:, coord]
            jac[:m, -1] = -1.0
            jac[m:, :-1] = -jac[:m, :-1]
            return jac

        starts = [normalize(best)]
        # Small deterministic tangent perturbations provide nearby contact
        # topologies without requiring another full global search.
        for scale in (0.004, 0.010):
            trial = best + scale * rng.normal(size=(n, 3))
            starts.append(normalize(trial))

        for start in starts:
            if start is None:
                continue
            q = distances2(start)
            v0 = np.concatenate((start.ravel(), [float(q.min())]))
            result = minimize(
                lambda v: -v[-1],
                v0,
                method="SLSQP",
                jac=lambda v: np.r_[np.zeros(3 * n), -1.0],
                constraints={"type": "ineq", "fun": constraints,
                             "jac": constraint_jacobian},
                bounds=[(None, None)] * (3 * n) + [(0.0, 1.0)],
                options={"maxiter": 350, "ftol": 1.e-11, "disp": False},
            )
            if result.x is not None and np.all(np.isfinite(result.x)):
                candidate = normalize(result.x[:-1])
                if candidate is not None:
                    candidate_value = ratio(candidate)
                    if candidate_value > best_value:
                        best, best_value = candidate, candidate_value
    except Exception:
        pass

    return np.asarray(best, dtype=float)