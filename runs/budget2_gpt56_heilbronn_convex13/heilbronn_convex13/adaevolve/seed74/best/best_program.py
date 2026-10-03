# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_convex13() -> np.ndarray:
    """Use deterministic annealing followed by SLSQP epigraph maximin polishing.

    Annealing supplies a nondegenerate combinatorial type.  The final SLSQP
    solve jointly optimizes all coordinates and an epigraph variable against
    every normalized triangle-area constraint while preserving the discovered
    convex-hull cycle and all determinant signs.
    """
    n = 13
    rng = np.random.default_rng(13031957)
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    tri = np.array(
        [(i, j, k) for i in range(n - 2)
         for j in range(i + 1, n - 1)
         for k in range(j + 1, n)],
        dtype=np.intp,
    )

    def project_simplex(p):
        p = np.abs(p)
        if p.sum() > 1.0:
            p = np.abs(1.0 - p)
            if p.sum() > 1.0:
                p /= p.sum()
        return p

    def determinants(p):
        a, b, c = p[tri[:, 0]], p[tri[:, 1]], p[tri[:, 2]]
        return ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))

    def quality(p):
        ar = 0.5 * np.abs(determinants(p))
        low = np.partition(ar, 11)[:12]
        return low[0] + .12 * low.mean(), low[0]

    best, best_minimum = None, -1.0
    for restart in range(12):
        points = np.vstack((corners, rng.dirichlet((1.15, 1.15, 1.15), 10)[:, 1:]))
        score, current_min = quality(points)
        for iteration in range(18000):
            f = iteration / 17999.0
            step = .115 * (1.0 - f) ** 1.65 + .0012
            temp = .0018 * (1.0 - f) ** 2.4 + .000002
            index = int(rng.integers(3, n))
            old = points[index].copy()
            points[index] = project_simplex(old + rng.normal(0.0, step, 2))
            new_score, new_min = quality(points)
            if new_score >= score or rng.random() < np.exp((new_score - score) / temp):
                score, current_min = new_score, new_min
            else:
                points[index] = old
            if current_min > best_minimum:
                best, best_minimum = points.copy(), current_min

    if best is None or not np.all(np.isfinite(best)):
        return np.vstack((corners, rng.dirichlet((1.0, 1.0, 1.0), 10)[:, 1:]))

    # Monotone-chain hull, retained as a fixed differentiable hull topology.
    def hull_order(p):
        order = sorted(range(n), key=lambda i: (p[i, 0], p[i, 1], i))
        def cross(i, j, k):
            return np.cross(p[j] - p[i], p[k] - p[i])
        lower = []
        for i in order:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], i) <= 0:
                lower.pop()
            lower.append(i)
        upper = []
        for i in reversed(order):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], i) <= 0:
                upper.pop()
            upper.append(i)
        return np.asarray(lower[:-1] + upper[:-1], dtype=np.intp)

    hull = hull_order(best)
    signs = np.sign(determinants(best))
    if len(hull) < 3 or np.any(signs == 0):
        return best

    def hull_area(p):
        q = p[hull]
        return .5 * np.sum(q[:, 0] * np.roll(q[:, 1], -1)
                           - q[:, 1] * np.roll(q[:, 0], -1))

    initial_area = hull_area(best)
    initial_r = np.min(np.abs(determinants(best))) / (2.0 * initial_area)

    # Every non-hull point must remain left of every directed hull edge.
    interior = np.setdiff1d(np.arange(n), hull)
    def constraints(z):
        p, r = z[:-1].reshape(n, 2), z[-1]
        area = hull_area(p)
        det = determinants(p)
        normalized = signs * det / (2.0 * area) - r
        q, nxt = p[hull], p[np.roll(hull, -1)]
        edge_constraints = []
        if len(interior):
            edge_constraints = np.concatenate([
                (nxt[k, 0] - q[k, 0]) * (p[interior, 1] - q[k, 1])
                - (nxt[k, 1] - q[k, 1]) * (p[interior, 0] - q[k, 0])
                for k in range(len(hull))
            ])
        # Positive turns prevent a hull edge from disappearing or reversing.
        turns = np.array([
            np.cross(q[(k + 1) % len(hull)] - q[k],
                     q[(k + 2) % len(hull)] - q[(k + 1) % len(hull)])
            for k in range(len(hull))
        ])
        return np.concatenate((normalized, np.asarray(edge_constraints), turns,
                               np.array([area - 1e-7])))

    try:
        from scipy.optimize import minimize
        result = minimize(
            lambda z: -z[-1],
            np.r_[best.ravel(), initial_r * .999],
            method="SLSQP",
            bounds=[(-.25, 1.25)] * (2 * n) + [(0.0, .2)],
            constraints={"type": "ineq", "fun": constraints},
            options={"maxiter": 350, "ftol": 1e-11, "disp": False},
        )
        if result.success and np.all(np.isfinite(result.x)):
            candidate = result.x[:-1].reshape(n, 2)
            # Exact post-validation protects against hull changes or SLSQP
            # feasibility tolerances; only a genuine evaluator improvement wins.
            candidate_hull = hull_order(candidate)
            candidate_area = hull_area(candidate) if np.array_equal(candidate_hull, hull) else 0.0
            old_value = np.min(np.abs(determinants(best))) / (2.0 * initial_area)
            if candidate_area > 0.0:
                new_value = np.min(np.abs(determinants(candidate))) / (2.0 * candidate_area)
                if new_value > old_value + 1e-10:
                    best = candidate
    except Exception:
        pass
    return best


# EVOLVE-BLOCK-END
