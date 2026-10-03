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
    endpoints = []

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
        endpoints.append((value, points.copy()))
        if len(endpoints) > 3:
            endpoints.sort(key=lambda item: item[0], reverse=True)
            del endpoints[3:]

    if best is None:
        return random_points().astype(np.float64)

    # Repair sparse Delaunay bottlenecks before the signed maximin polish.
    try:
        from scipy.optimize import minimize
        from scipy.spatial import Delaunay

        """Build three deterministic Delaunay-patch repaired seeds."""
        ranked = sorted(endpoints, key=lambda item: item[0], reverse=True)
        selected = []
        for endpoint_value, seed in ranked[:3]:
            seed = np.asarray(seed, dtype=np.float64).copy()
            det = determinants(seed)
            worst = np.argsort(np.abs(det))[:12]
            try:
                triangulation = Delaunay(seed)
                edge_data = {}
                for simplex in triangulation.simplices:
                    for a in range(3):
                        u = int(simplex[a])
                        v = int(simplex[(a + 1) % 3])
                        opposite = int(simplex[(a + 2) % 3])
                        key = tuple(sorted((u, v)))
                        edge_data.setdefault(key, set()).add(opposite)

                scored = []
                for edge, opposites in edge_data.items():
                    if len(opposites) != 2:
                        continue
                    u, v = edge
                    local = [abs(det[k]) for k in worst
                             if u in triples[k] or v in triples[k]]
                    score = min(local) if local else np.inf
                    scored.append((score, edge, tuple(opposites)))
                scored.sort(key=lambda item: item[0])
            except Exception:
                scored = []

            repaired = []
            for _, edge, opposites in scored[:4]:
                patch = set(edge) | set(opposites)
                for tri_index in worst:
                    if any(int(q) in patch for q in triples[tri_index]):
                        patch.update(int(q) for q in triples[tri_index])
                patch = sorted(patch)
                if len(patch) < 4:
                    continue

                signs = np.sign(det)
                signs[signs == 0.0] = 1.0
                z0 = np.concatenate((seed.ravel(),
                                     [max(0.0, endpoint_value * .999)]))
                fixed = np.ones(2 * n + 1, dtype=bool)
                for q in patch:
                    fixed[2 * q:2 * q + 2] = False
                fixed[-1] = False

                def patch_constraints(z, signs=signs, fixed=fixed):
                    p = z[:-1].reshape(n, 2)
                    t = z[-1]
                    x, y = p[:, 0], p[:, 1]
                    edge_values = np.concatenate((
                        x, y, np.sqrt(3.0) * x - y,
                        np.sqrt(3.0) * (1.0 - x) - y,
                    ))
                    equalities = np.abs(z[fixed] - z0[fixed])
                    return np.concatenate((
                        edge_values, signs * determinants(p) - t,
                        1e-10 - equalities,
                    ))

                starts = [z0]
                alternate = z0.copy()
                u, v = edge
                ou, ov = opposites
                midpoint = .5 * (seed[u] + seed[v])
                alternate[2 * u:2 * u + 2] = (
                    .65 * seed[u] + .35 * seed[ov])
                alternate[2 * v:2 * v + 2] = (
                    .65 * seed[v] + .35 * seed[ou])
                starts.append(alternate)

                for start in starts:
                    result = minimize(
                        lambda z: -z[-1],
                        start,
                        method="SLSQP",
                        bounds=[(0.0, 1.0)] * (2 * n) + [(0.0, 1.0)],
                        constraints={"type": "ineq",
                                     "fun": patch_constraints},
                        options={"maxiter": 350, "ftol": 1e-11,
                                 "disp": False},
                    )
                    if result.x is None:
                        continue
                    candidate = result.x[:-1].reshape(n, 2)
                    candidate_value = objective(candidate)
                    if (inside(candidate) and
                            np.all(np.isfinite(candidate)) and
                            candidate_value > best_value):
                        best = candidate.copy()
                        best_value = candidate_value
                        repaired.append((candidate_value, candidate.copy()))

            if repaired:
                selected.append(max(repaired, key=lambda item: item[0]))
            else:
                selected.append((endpoint_value, seed))
        selected.sort(key=lambda item: item[0], reverse=True)

        vertices = np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.5, h]], dtype=np.float64
        )

        for endpoint_value, seed in selected:
            seed = np.asarray(seed, dtype=np.float64).copy()
            signs = np.sign(determinants(seed))
            signs[signs == 0.0] = 1.0
            x0 = np.concatenate((
                seed.ravel(), [max(0.0, endpoint_value * 0.999)]
            ))

            def constraints(z, signs=signs):
                p = z[:-1].reshape(n, 2)
                t = z[-1]
                x, y = p[:, 0], p[:, 1]
                edge = np.concatenate((
                    x, y, np.sqrt(3.0) * x - y,
                    np.sqrt(3.0) * (1.0 - x) - y,
                ))
                return np.concatenate((edge, signs * determinants(p) - t))

            candidates = [x0]
            bary = np.column_stack((
                1.0 - seed[:, 0] - seed[:, 1] / h,
                seed[:, 0] - seed[:, 1] / (2.0 * h),
                seed[:, 1] / h,
            ))
            perturb = np.random.default_rng(
                9200 + len(candidates)
            ).normal(0.0, 2e-4, seed.shape)
            perturbed = seed + perturb
            bary = np.column_stack((
                1.0 - perturbed[:, 0] - perturbed[:, 1] / h,
                perturbed[:, 0] - perturbed[:, 1] / (2.0 * h),
                perturbed[:, 1] / h,
            ))
            bary = np.maximum(bary, 2e-8)
            bary /= bary.sum(axis=1, keepdims=True)
            projected = bary @ vertices
            candidates.append(np.concatenate((
                projected.ravel(), [max(0.0, endpoint_value * 0.999)]
            )))

            for z in candidates:
                result = minimize(
                    lambda q: -q[-1],
                    z,
                    method="SLSQP",
                    bounds=[(0.0, 1.0)] * 22 + [(0.0, 1.0)],
                    constraints={"type": "ineq", "fun": constraints},
                    options={"maxiter": 900, "ftol": 1e-11, "disp": False},
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