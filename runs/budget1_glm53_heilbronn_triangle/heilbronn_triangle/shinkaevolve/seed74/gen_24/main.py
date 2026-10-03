# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 11.

    Uses a deterministic softmin-annealed gradient ascent (Adam) on the minimum
    triangle area over all C(11,3) triplets, with exact analytic gradients and a
    barycentric projection to keep all points inside the equilateral triangle.

    Returns:
        points: np.ndarray of shape (11,2) with the x,y coordinates of the points.
    """
    n = 11
    H = np.sqrt(3) / 2.0
    V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, H]])

    try:
        from itertools import combinations

        idx = np.array(list(combinations(range(n), 3)))
        I, J, K = idx[:, 0], idx[:, 1], idx[:, 2]

        # Deterministic initialization: the 3 vertices + 8 seeded interior points
        rng = np.random.default_rng(20240611)
        u = rng.random((8, 3)) + 0.35
        u = u / u.sum(axis=1, keepdims=True)
        lam = np.zeros((n, 3))
        lam[0] = [1.0, 0.0, 0.0]
        lam[1] = [0.0, 1.0, 0.0]
        lam[2] = [0.0, 0.0, 1.0]
        lam[3:] = u
        pts = lam @ V

        # Adam state
        m = np.zeros_like(pts)
        v = np.zeros_like(pts)
        lr, b1, b2, eps = 0.02, 0.9, 0.999, 1e-8

        def exact_min_area(P):
            x, y = P[:, 0], P[:, 1]
            c = (x[J] - x[I]) * (y[K] - y[I]) - (y[J] - y[I]) * (x[K] - x[I])
            return 0.5 * np.abs(c)

        def polish(P, iters=600):
            """Adaptive-threshold bottleneck polish: move vertices of all
            'tight' triangles (area <= thresh * min) along the gradient of
            their signed area; thresh narrows from 3.0 to 1.2 over time."""
            P = P.copy()
            cur = float(exact_min_area(P).min())
            for it in range(iters):
                thresh = 3.0 - 1.8 * (it / iters)  # 3.0 -> 1.2
                x, y = P[:, 0], P[:, 1]
                xi, yi = x[I], y[I]
                xj, yj = x[J], y[J]
                xk, yk = x[K], y[K]
                c = (xj - xi) * (yk - yi) - (yj - yi) * (xk - xi)
                a = 0.5 * np.abs(c)
                amin = a.min()
                tight = a <= thresh * amin
                if not np.any(tight):
                    tight = a <= 1.2 * amin
                s = np.sign(c)
                coef = np.where(tight, 0.5 * s, 0.0)  # ascent on signed area
                g = np.zeros_like(P)
                np.add.at(g, I, np.stack([coef * (yj - yk), coef * (xk - xj)], axis=1))
                np.add.at(g, J, np.stack([coef * (yk - yi), coef * (xi - xk)], axis=1))
                np.add.at(g, K, np.stack([coef * (yi - yj), coef * (xj - xi)], axis=1))
                # temperature-scaled step: broad early, tiny late
                step = 0.02 * (1.0 - it / iters) + 0.0005
                gn = np.linalg.norm(g, axis=1, keepdims=True)
                gn[gn < 1e-15] = 1.0
                moved = False
                s_try = step
                for _ in range(6):
                    cand = P + s_try * g / gn
                    lam3 = cand[:, 1] / H
                    lam2 = cand[:, 0] - 0.5 * lam3
                    lam1 = 1.0 - lam2 - lam3
                    L = np.stack([lam1, lam2, lam3], axis=1)
                    L = np.clip(L, 1e-9, 1.0)
                    L = L / L.sum(axis=1, keepdims=True)
                    cand = L @ V
                    v = float(exact_min_area(cand).min())
                    if v > cur + 1e-14:
                        P, cur, moved = cand, v, True
                        break
                    s_try *= 0.5
                if not moved and step < 1e-6:
                    break
            return P, cur

        # Deterministic multi-start: run Adam + polish from several seeds,
        # keep the configuration with the largest exact minimum area.
        best_pts = None
        best_val = -1.0
        for seed in (20240611, 7, 1234):
            rng = np.random.default_rng(seed)
            u = rng.random((8, 3)) + 0.35
            u = u / u.sum(axis=1, keepdims=True)
            lam = np.zeros((n, 3))
            lam[0] = [1.0, 0.0, 0.0]
            lam[1] = [0.0, 1.0, 0.0]
            lam[2] = [0.0, 0.0, 1.0]
            lam[3:] = u
            pts = lam @ V
            m = np.zeros_like(pts)
            v = np.zeros_like(pts)
            lr, b1, b2, eps = 0.02, 0.9, 0.999, 1e-8
            for it in range(1, 801):
                tau = max(0.0004, 0.02 * np.exp(-it / 200.0))
                x, y = pts[:, 0], pts[:, 1]
                xi, yi = x[I], y[I]
                xj, yj = x[J], y[J]
                xk, yk = x[K], y[K]
                c = (xj - xi) * (yk - yi) - (yj - yi) * (xk - xi)
                a = 0.5 * np.abs(c)
                s = np.sign(c)
                z = np.exp(-(a - a.min()) / tau)
                w = z / z.sum()
                coef = -0.5 * w * s
                g = np.zeros_like(pts)
                np.add.at(g, I, np.stack([coef * (yj - yk), coef * (xk - xj)], axis=1))
                np.add.at(g, J, np.stack([coef * (yk - yi), coef * (xi - xk)], axis=1))
                np.add.at(g, K, np.stack([coef * (yi - yj), coef * (xj - xi)], axis=1))
                m = b1 * m + (1 - b1) * g
                v = b2 * v + (1 - b2) * (g * g)
                mh = m / (1 - b1 ** it)
                vh = v / (1 - b2 ** it)
                pts = pts - lr * mh / (np.sqrt(vh) + eps)
                lam3 = pts[:, 1] / H
                lam2 = pts[:, 0] - 0.5 * lam3
                lam1 = 1.0 - lam2 - lam3
                L = np.stack([lam1, lam2, lam3], axis=1)
                L = np.clip(L, 1e-9, 1.0)
                L = L / L.sum(axis=1, keepdims=True)
                pts = L @ V
            pts, val = polish(pts)
            if val > best_val:
                best_val, best_pts = val, pts.copy()

        return best_pts

        for it in range(1, 1201):
            # Anneal the softmin temperature
            tau = max(0.0004, 0.02 * np.exp(-it / 300.0))

            x, y = pts[:, 0], pts[:, 1]
            xi, yi = x[I], y[I]
            xj, yj = x[J], y[J]
            xk, yk = x[K], y[K]

            # Signed doubled areas of all triplets
            c = (xj - xi) * (yk - yi) - (yj - yi) * (xk - xi)
            a = 0.5 * np.abs(c)
            s = np.sign(c)

            # Softmin weights (numerically stable)
            z = np.exp(-(a - a.min()) / tau)
            w = z / z.sum()

            # df/dc = -w * 0.5 * sign(c) for f = tau*log(sum(exp(-a/tau)))
            coef = -0.5 * w * s

            g = np.zeros_like(pts)
            # Exact partials of c w.r.t. each vertex of each triplet
            np.add.at(g, I, np.stack([coef * (yj - yk), coef * (xk - xj)], axis=1))
            np.add.at(g, J, np.stack([coef * (yk - yi), coef * (xi - xk)], axis=1))
            np.add.at(g, K, np.stack([coef * (yi - yj), coef * (xj - xi)], axis=1))

            # Adam descent on f (i.e., ascent on min area)
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * (g * g)
            mh = m / (1 - b1 ** it)
            vh = v / (1 - b2 ** it)
            pts = pts - lr * mh / (np.sqrt(vh) + eps)

            # Project back into the triangle via barycentric clipping
            lam3 = pts[:, 1] / H
            lam2 = pts[:, 0] - 0.5 * lam3
            lam1 = 1.0 - lam2 - lam3
            L = np.stack([lam1, lam2, lam3], axis=1)
            L = np.clip(L, 1e-9, 1.0)
            L = L / L.sum(axis=1, keepdims=True)
            pts = L @ V

        return pts

    except Exception:
        # Fallback: simple deterministic feasible configuration
        lam = np.zeros((n, 3))
        lam[0] = [1.0, 0.0, 0.0]
        lam[1] = [0.0, 1.0, 0.0]
        lam[2] = [0.0, 0.0, 1.0]
        k = 0
        for i in range(3):
            for j in range(3):
                if i + j <= 2 and k < n:
                    a_, b_ = 0.25 + 0.25 * i, 0.25 + 0.25 * j
                    cc = 1.0 - a_ - b_
                    if cc >= 0.25:
                        lam[k] = [cc, a_, b_]
                        k += 1
        return lam @ V


# EVOLVE-BLOCK-END