import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """Deterministic annealing followed by signed maximin constrained polishing."""
    n = 11
    h = np.sqrt(3.0) / 2.0
    rng = np.random.default_rng(11011)

    triples = np.array(
        [(i, j, k) for i in range(n) for j in range(i + 1, n)
         for k in range(j + 1, n)],
        dtype=np.int32,
    )

    def determinants(points):
        a = points[triples[:, 0]]
        b = points[triples[:, 1]]
        c = points[triples[:, 2]]
        return ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))

    def objective(points):
        return float(np.min(np.abs(determinants(points))))

    def inside(points):
        x, y = points[:, 0], points[:, 1]
        return bool(np.all((x >= 0.0) & (y >= 0.0) &
                           (y <= np.sqrt(3.0) * x) &
                           (y <= np.sqrt(3.0) * (1.0 - x))))

    def random_points():
        u = rng.random(n)
        v = rng.random(n)
        mask = u + v > 1.0
        u[mask] = 1.0 - u[mask]
        v[mask] = 1.0 - v[mask]
        return np.column_stack((
            v + 0.5 * (1.0 - u - v),
            h * (1.0 - u - v),
        ))

    best = None
    best_value = -1.0

    # This preserves the robust global component of the parent construction.
    for _ in range(24):
        points = random_points()
        value = objective(points)
        for iteration in range(40000):
            q = iteration / 40000.0
            scale = 0.16 * (1.0 - q) ** 1.7 + 0.001
            idx = int(rng.integers(n))
            trial = points.copy()
            trial[idx] += rng.normal(0.0, scale, 2)
            if not inside(trial):
                continue
            trial_value = objective(trial)
            temperature = max(1e-9, 0.0025 * (1.0 - q) ** 1.25)
            if (trial_value >= value or
                    rng.random() < np.exp((trial_value - value) / temperature)):
                points, value = trial, trial_value
                if value > best_value:
                    best, best_value = points.copy(), value

    if best is None:
        return random_points().astype(np.float64)

    # SLSQP is optional so the construction still executes on minimal NumPy setups.
    try:
        from scipy.optimize import minimize

        seed = best.copy()
        signs = np.sign(determinants(seed))
        signs[signs == 0.0] = 1.0
        x0 = np.concatenate((seed.ravel(), [best_value * 0.999]))

        def constraints(z):
            p = z[:-1].reshape(n, 2)
            t = z[-1]
            x, y = p[:, 0], p[:, 1]
            edge = np.concatenate((
                x, y, np.sqrt(3.0) * x - y,
                np.sqrt(3.0) * (1.0 - x) - y,
            ))
            return np.concatenate((edge, signs * determinants(p) - t))

        candidates = [x0]
        for s in range(3):
            z = x0.copy()
            perturb = np.random.default_rng(9200 + s).normal(0.0, 2e-4, 22)
            z[:-1] += perturb
            candidates.append(z)

        for z in candidates:
            result = minimize(
                lambda q: -q[-1],
                z,
                method="SLSQP",
                bounds=[(0.0, 1.0)] * 22 + [(0.0, 1.0)],
                constraints={"type": "ineq", "fun": constraints},
                options={"maxiter": 1200, "ftol": 1e-11, "disp": False},
            )
            if result.x is None:
                continue
            p = result.x[:-1].reshape(n, 2)
            value = objective(p)
            if inside(p) and np.all(np.isfinite(p)) and value > best_value:
                best, best_value = p.copy(), value
    except Exception:
        pass

    return np.asarray(best, dtype=np.float64)