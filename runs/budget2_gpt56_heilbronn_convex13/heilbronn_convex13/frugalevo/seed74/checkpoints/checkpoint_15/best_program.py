import numpy as np


def heilbronn_convex13() -> np.ndarray:
    """Deterministic soft-min search followed by a signed maximin epigraph polish."""
    n = 13
    rng = np.random.default_rng(42)
    triples = np.array(
        [(i, j, k) for i in range(n) for j in range(i + 1, n)
         for k in range(j + 1, n)], dtype=int
    )

    # Build explicit cyclic-hull topology seeds.  Each seed has m ordered
    # boundary contacts and 13-m staggered interior points.
    topology_seeds = []
    center = np.array([0.5, 0.5], dtype=float)
    interior = np.array([
        [0.37, 0.37], [0.63, 0.37], [0.50, 0.50],
        [0.37, 0.63], [0.63, 0.63], [0.50, 0.31],
        [0.50, 0.69], [0.31, 0.50], [0.69, 0.50],
        [0.43, 0.50], [0.57, 0.50]
    ], dtype=float)
    for m in (7, 8, 9, 10):
        angles = np.linspace(0.0, 2.0 * np.pi, m, endpoint=False)
        for reflected in (False, True):
            phase = 0.17 if reflected else -0.11
            radii = 0.435 + 0.018 * np.cos(3.0 * angles + phase)
            if reflected:
                radii = radii[::-1]
            hull_seed = center + radii[:, None] * np.column_stack(
                (np.cos(angles), np.sin(angles))
            )
            if reflected:
                hull_seed[:, 1] = 1.0 - hull_seed[:, 1]
            # Use a distinct staggered prefix for each topology so that
            # interior crowding is not tied to the incumbent 11-contact seed.
            count = n - m
            inner = interior[:count].copy()
            inner[:, 0] += 0.012 * np.sin(np.arange(count) + m)
            inner[:, 1] += 0.010 * np.cos(1.7 * np.arange(count) + m)
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
    for _, start in ranked[:4]:
        x = np.asarray(start, dtype=float).copy().ravel()
        for temp in (0.004, 0.002, 0.001, 0.0005, 0.00025,
                     0.00012, 0.00006):
            r = minimize(
                lambda z, t=temp: soft_objective(z, t), x,
                method="L-BFGS-B", bounds=[(0.002, 0.998)] * (2 * n),
                options={"maxiter": 500, "ftol": 1e-13,
                         "gtol": 1e-8, "maxls": 40}
            )
            if np.isfinite(r.fun):
                x = r.x
        candidate = x.reshape(n, 2)
        v = score(candidate)
        if v > best_value:
            best, best_value = candidate.copy(), v

    # The signed epigraph polish below is the sole local refinement stage.
    # It directly maximizes the normalized minimum triangle area instead of
    # perturbing the soft-min incumbent with coordinate-wise hill climbing.

    # Signed epigraph polish.  The hull order and all determinant signs are
    # fixed locally, eliminating the Qhull discontinuity from the optimizer.
    try:
        hull = ConvexHull(best)
        order = hull.vertices
        poly = best[order]
        signed = np.sign(dets(best))
        signed[signed == 0.0] = 1.0

        def polygon_area(p):
            q = p[order]
            return 0.5 * (np.dot(q[:, 0], np.roll(q[:, 1], -1))
                          - np.dot(q[:, 1], np.roll(q[:, 0], -1)))

        def constraints(z):
            p, t = z[:-1].reshape(n, 2), z[-1]
            q = p[order]
            area = polygon_area(p)
            # All triangle contacts, expressed with their incumbent orientation.
            tri = 0.5 * signed * dets(p) - t * area
            # CCW hull edges contain every returned point.
            edges = np.roll(q, -1, axis=0) - q
            contain = (edges[:, None, 0] * (p[None, :, 1] - q[:, None, 1])
                       - edges[:, None, 1] * (p[None, :, 0] - q[:, None, 0]))
            # Strictly nonnegative turns retain convex cyclic hull order.
            turns = (np.roll(q, -1, axis=0) - q)
            nxt = np.roll(turns, -1, axis=0)
            convex = turns[:, 0] * nxt[:, 1] - turns[:, 1] * nxt[:, 0]
            return np.concatenate((tri, contain.ravel(), convex, [area - 1e-5]))

        z0 = np.concatenate((best.ravel(), [best_value * 0.999]))
        for offset in (0.0, 2e-5):
            zz = z0.copy()
            if offset:
                # Deterministic alternate start affecting only interior candidates.
                mask = np.ones(n, dtype=bool)
                mask[order] = False
                zz[:-1].reshape(n, 2)[mask, 0] += offset
            r = minimize(
                lambda z: -z[-1], zz, method="SLSQP",
                bounds=[(0.0001, 0.9999)] * (2 * n) + [(0.0, 0.1)],
                constraints={"type": "ineq", "fun": constraints},
                options={"maxiter": 450, "ftol": 1e-12, "disp": False}
            )
            if np.isfinite(r.fun):
                candidate = r.x[:-1].reshape(n, 2)
                v = score(candidate)
                if v > best_value:
                    best, best_value = candidate.copy(), v
    except Exception:
        pass

    return np.asarray(np.clip(best, 0.0, 1.0), dtype=float)