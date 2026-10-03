import numpy as np


def heilbronn_convex13() -> np.ndarray:
    """Deterministic soft-min search followed by a signed maximin epigraph polish."""
    n = 13
    rng = np.random.default_rng(42)
    triples = np.array(
        [(i, j, k) for i in range(n) for j in range(i + 1, n)
         for k in range(j + 1, n)], dtype=int
    )

    # Build deterministic cyclic-hull seeds for every hull cardinality from
    # five through twelve.  Each family uses a different angular phase and
    # barycentric low-discrepancy interior pattern.
    topology_seeds = []
    center = np.array([0.5, 0.5], dtype=float)

    for m in range(5, 13):
        angles = np.linspace(0.0, 2.0 * np.pi, m, endpoint=False)
        vertices = np.column_stack((np.cos(angles), np.sin(angles)))
        for family in range(2):
            phase = (-0.14 if family == 0 else 0.21) + 0.011 * m
            radii = 0.435 + 0.017 * np.cos(3.0 * angles + phase)
            hull_seed = center + radii[:, None] * vertices
            count = n - m
            inner = np.empty((count, 2), dtype=float)

            for r in range(count):
                q = r + 1 + 13 * m + 17 * family

                # Van der Corput coordinates in bases two and three.
                u = 0.0
                scale = 1.0
                t = q
                while t:
                    scale *= 2.0
                    u += (t % 2) / scale
                    t //= 2

                v = 0.0
                scale = 1.0
                t = q
                while t:
                    scale *= 3.0
                    v += (t % 3) / scale
                    t //= 3

                sector = int(m * u) % m
                nxt = (sector + 1) % m
                blend = 0.20 + 0.60 * v

                # Start with the polygon centroid and add a controlled
                # barycentric mass on one edge.  The radial factor varies
                # independently, avoiding the central clustering of purely
                # centroid-pulled seeds.
                weights = np.full(m, 0.62 / m, dtype=float)
                edge_mass = 0.38
                weights[sector] += edge_mass * (1.0 - blend)
                weights[nxt] += edge_mass * blend
                point = weights @ hull_seed

                radial = 0.40 + 0.40 * (0.5 * u + 0.5 * v)
                inner[r] = center + radial * (point - center)

            topology_seeds.append(np.vstack((hull_seed, inner)))

    base = topology_seeds[0].copy()

    try:
        from scipy.optimize import minimize
        from scipy.spatial import ConvexHull
        from scipy.special import logsumexp
    except Exception:
        return base

    def dets(p):
        a, b, c = p[triples[:, 0]], p[triples[:, 1]], p[triples[:, 2]]
        return ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))

    def score(p):
        try:
            h = ConvexHull(p).volume
        except Exception:
            return 0.0
        return float(0.5 * np.min(np.abs(dets(p))) / h) if h > 1e-12 else 0.0

    def soft_objective(x, temperature):
        p = x.reshape(n, 2)
        try:
            h = ConvexHull(p).volume
        except Exception:
            return 1e4
        if h < 1e-12:
            return 1e4
        ar = 0.5 * np.abs(dets(p)) / h
        return float(temperature * logsumexp(-ar / temperature))

    best = base.copy()
    best_value = score(best)

    # Rank the eight topology/reflection starts cheaply, then continue only
    # with the four most promising cyclic-hull contact patterns.
    ranked = []
    for start in topology_seeds:
        try:
            surrogate = soft_objective(start.ravel(), 0.001)
        except Exception:
            surrogate = 1e4
        ranked.append((surrogate, start))
    ranked.sort(key=lambda item: item[0])
    for _, start in ranked[:5]:
        x = np.asarray(start, dtype=float).copy().ravel()
        for temp in (0.004, 0.002, 0.001, 0.0005, 0.00025,
                     0.00012, 0.00006):
            r = minimize(
                lambda z, t=temp: soft_objective(z, t), x,
                method="L-BFGS-B", bounds=[(0.002, 0.998)] * (2 * n),
                options={"maxiter": 400, "ftol": 1e-13,
                         "gtol": 1e-8, "maxls": 40}
            )
            if np.isfinite(r.fun):
                x = r.x
        candidate = x.reshape(n, 2)
        v = score(candidate)
        if v > best_value:
            best, best_value = candidate.copy(), v

    # Affine-gauged exact contact refinement.  The best-conditioned
    # noncollinear hull triple is fixed to (0,0), (1,0), and (0,1), removing
    # translation, rotation, shear, and scale-like affine null directions.
    try:
        hull = ConvexHull(best)
        order = np.asarray(hull.vertices, dtype=int)
        hull_triples = [(order[i], order[j], order[k])
                        for i in range(len(order))
                        for j in range(i + 1, len(order))
                        for k in range(j + 1, len(order))]
        anchor = np.asarray(max(
            hull_triples,
            key=lambda abc: abs(np.linalg.det(np.column_stack(
                (best[abc[1]] - best[abc[0]],
                 best[abc[2]] - best[abc[0]]))))
        ), dtype=int)

        origin = best[anchor[0]].copy()
        matrix = np.column_stack((best[anchor[1]] - origin,
                                  best[anchor[2]] - origin))
        determinant = float(np.linalg.det(matrix))
        if abs(determinant) < 1e-10:
            raise ValueError("ill-conditioned affine gauge")

        inverse = np.linalg.inv(matrix)
        gauged = (best - origin) @ inverse
        signed = np.sign(dets(gauged))
        signed[signed == 0.0] = 1.0

        anchor_set = set(anchor.tolist())
        free = np.asarray([i for i in range(n) if i not in anchor_set],
                          dtype=int)
        fixed = gauged[anchor].copy()

        def unpack(z):
            """Reconstruct all gauged points from free coordinates and t."""
            p = np.empty((n, 2), dtype=float)
            p[anchor] = fixed
            p[free] = z[:-1].reshape(len(free), 2)
            return p

        def restore(p):
            """Map gauged coordinates back to the original affine frame."""
            return p @ matrix + origin

        def polygon_area(p):
            """Evaluate the signed shoelace area in the fixed hull order."""
            q = p[order]
            return 0.5 * (np.dot(q[:, 0], np.roll(q[:, 1], -1))
                          - np.dot(q[:, 1], np.roll(q[:, 0], -1)))

        def constraints(z):
            """Return signed triangle, containment, convexity, and area margins."""
            p = unpack(z)
            q = p[order]
            area = polygon_area(p)
            tri = 0.5 * signed * dets(p) - z[-1] * area

            edges = np.roll(q, -1, axis=0) - q
            contain = (edges[:, None, 0] * (p[None, :, 1] - q[:, None, 1])
                       - edges[:, None, 1] * (p[None, :, 0] - q[:, None, 0]))

            turns = np.roll(q, -1, axis=0) - q
            nxt = np.roll(turns, -1, axis=0)
            convex = turns[:, 0] * nxt[:, 1] - turns[:, 1] * nxt[:, 0]
            return np.concatenate((tri, contain.ravel(), convex,
                                   [area - 1e-5]))

        z0 = np.concatenate((gauged[free].ravel(),
                             [best_value * 0.999]))
        starts = [z0.copy()]
        # Tangent-space perturbations use alternating free coordinates, so the
        # three affine gauge contacts remain exactly fixed.
        for delta in (1.5e-4, -1.5e-4):
            zz = z0.copy()
            zz[:-1:2] += delta
            starts.append(zz)

        for start in starts:
            result = minimize(
                lambda z: -z[-1], start, method="SLSQP",
                constraints={"type": "ineq", "fun": constraints},
                options={"maxiter": 350, "ftol": 1e-12, "disp": False}
            )
            if np.isfinite(result.fun):
                candidate = restore(unpack(result.x))
                value = score(candidate)
                if value > best_value:
                    best = candidate.copy()
                    best_value = value
    except Exception:
        pass

    return np.asarray(np.clip(best, 0.0, 1.0), dtype=float)