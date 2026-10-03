import numpy as np


def heilbronn_convex13() -> np.ndarray:
    """Deterministic soft-min search followed by a signed maximin epigraph polish."""
    n = 13
    rng = np.random.default_rng(42)
    triples = np.array(
        [(i, j, k) for i in range(n) for j in range(i + 1, n)
         for k in range(j + 1, n)], dtype=int
    )

    # Build deterministic variable-angle star-polygon seeds.  Positive,
    # nonuniform angular gaps expose boundary-contact patterns unavailable to
    # equally spaced cyclic seeds, while the small radial amplitudes preserve
    # strict convexity in the initial configurations.
    topology_seeds = []
    center = np.array([0.5, 0.5], dtype=float)

    for m in (6, 7, 8, 9, 10, 11):
        for profile in range(2 if m in (8, 9, 10) else 1):
            for gap_profile in range(3 if m in (8, 9, 10) else 2):
                phase = 0.17 * (profile + 1) + 0.09 * gap_profile
                indices = np.arange(m, dtype=float)

                gap = (
                    1.0
                    + 0.13 * np.cos(3.0 * indices + phase)
                    + (0.055 if gap_profile != 1 else -0.055)
                    * np.cos(2.0 * indices - 0.7 * phase)
                )
                gap *= (2.0 * np.pi) / np.sum(gap)
                angles = np.concatenate(([0.0], np.cumsum(gap[:-1])))

                radial_phase = phase + (0.23 if profile else -0.19)
                radial = 0.435 * (
                    1.0
                    + (0.030 if profile == 0 else 0.023)
                    * np.cos(3.0 * angles + radial_phase)
                    + (0.012 if profile == 0 else -0.012)
                    * np.cos(2.0 * angles - 0.6 * radial_phase)
                )
                hull_seed = center + radial[:, None] * np.column_stack(
                    (np.cos(angles), np.sin(angles))
                )
                count = n - m
                inner = np.empty((count, 2), dtype=float)

                for r in range(count):
                    q = r + 1 + 19 * m + 7 * profile + 11 * gap_profile

                    # Deterministic van der Corput coordinates in bases two
                    # and three provide reproducible interpolation weights.
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

                    a = int(m * u) % m
                    separation = 2 + int((m - 3) * v)
                    b = (a + separation) % m
                    c = (b + 1 + int((m - 3) * u)) % m
                    if c == a or c == b:
                        c = (a + m // 2 + 1) % m
                        if c == b:
                            c = (c + 1) % m

                    # Low-contact hulls use shifted, mutually separated
                    # cavities; the eleven-vertex case uses opposite
                    # alternating cavities for its two interior points.
                    if m == 6:
                        a = (2 * r + 1) % m
                        b = (a + 3) % m
                        c = (a + 2) % m
                    elif m == 7:
                        a = (2 * r + 1) % m
                        b = (a + 2) % m
                        c = (a + 4) % m
                    elif m == 11:
                        a = (1 + 5 * r) % m
                        b = (a + 2) % m
                        c = (a + 4) % m

                    # Strictly positive barycentric weights keep every
                    # interior point inside the convex hull.
                    wa = 0.25 + 0.20 * u
                    wb = 0.25 + 0.20 * v
                    wc = 1.0 - wa - wb
                    inner[r] = wa * hull_seed[a] + wb * hull_seed[b]
                    inner[r] += wc * hull_seed[c]

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

    # Rank all hull-cardinality starts cheaply, then continue only with the
    # six most promising contact patterns.
    ranked = []
    for start in topology_seeds:
        try:
            surrogate = soft_objective(start.ravel(), 0.001)
        except Exception:
            surrogate = 1e4
        ranked.append((surrogate, start))
    ranked.sort(key=lambda item: item[0])
    for _, start in ranked[:6]:
        x = np.asarray(start, dtype=float).copy().ravel()
        for temp in (0.004, 0.002, 0.001, 0.0005, 0.00025,
                     0.00012, 0.00006):
            r = minimize(
                lambda z, t=temp: soft_objective(z, t), x,
                method="L-BFGS-B", bounds=[(0.002, 0.998)] * (2 * n),
                options={"maxiter": 300, "ftol": 1e-13,
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