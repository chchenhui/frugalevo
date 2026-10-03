# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 11.

    Returns:
        points: np.ndarray of shape (11,2) with the x,y coordinates of the points.
    """
    n = 11
    sqrt3 = np.sqrt(3.0)

    # Container: equilateral triangle A(0,0), B(1,0), C(0.5, sqrt(3)/2).
    # Barycentric-like parameters (u, v): point = u*B + v*C + (1-u-v)*A,
    # feasible iff u >= 0, v >= 0, u + v <= 1.  Mapping is affine:
    #   x = u + 0.5 v,  y = (sqrt3/2) v.

    def bary_to_xy(b):
        u = b[:, 0]
        v = b[:, 1]
        x = u + 0.5 * v
        y = 0.5 * sqrt3 * v
        return np.column_stack([x, y])

    triples = None

    def areas_xy(xy):
        # vectorized absolute areas of all point triples
        p = xy[triples[:, 0]]
        q = xy[triples[:, 1]]
        r = xy[triples[:, 2]]
        return 0.5 * np.abs((q[:, 0] - p[:, 0]) * (r[:, 1] - p[:, 1])
                            - (q[:, 1] - p[:, 1]) * (r[:, 0] - p[:, 0]))

    try:
        from itertools import combinations

        triples = np.array(list(combinations(range(n), 3)))

        def project(b):
            # project onto simplex u >= 0, v >= 0, u + v <= 1
            b = np.maximum(b, 0.0)
            s = b.sum(axis=1)
            over = s > 1.0
            if np.any(over):
                b[over] = b[over] / s[over, None]
            return b

        def pin_vertices(b):
            # pin the three container vertices to guarantee large-scale spread
            b[0] = [0.0, 0.0]
            b[1] = [1.0, 0.0]
            b[2] = [0.0, 1.0]
            return b

        def grad_bary(b, wts):
            # gradient (wrt barycentric coords) of sum_t wts[t] * area[t],
            # using the analytic cross-product chain rule
            xy = bary_to_xy(b)
            p = xy[triples[:, 0]]
            q = xy[triples[:, 1]]
            r = xy[triples[:, 2]]
            cross = ((q[:, 0] - p[:, 0]) * (r[:, 1] - p[:, 1])
                     - (q[:, 1] - p[:, 1]) * (r[:, 0] - p[:, 0]))
            s = np.where(cross >= 0.0, 1.0, -1.0)
            c = (0.5 * s * wts)[:, None]
            g = np.zeros((n, 2))
            np.add.at(g, triples[:, 0],
                      c * np.column_stack([q[:, 1] - r[:, 1], r[:, 0] - q[:, 0]]))
            np.add.at(g, triples[:, 1],
                      c * np.column_stack([r[:, 1] - p[:, 1], p[:, 0] - r[:, 0]]))
            np.add.at(g, triples[:, 2],
                      c * np.column_stack([p[:, 1] - q[:, 1], q[:, 0] - p[:, 0]]))
            # chain rule through x = u + 0.5 v, y = (sqrt3/2) v
            gb = np.empty_like(b)
            gb[:, 0] = g[:, 0]
            gb[:, 1] = 0.5 * g[:, 0] + 0.5 * sqrt3 * g[:, 1]
            return gb

        def make_start(seed):
            rng = np.random.default_rng(seed)
            b = project(0.9 * rng.random((n, 2)))
            return pin_vertices(b)

        def adam(b, lrs, betas, iters):
            # Adam ascent on soft-min lower bound of the minimum area
            m = np.zeros_like(b)
            v = np.zeros_like(b)
            t = 0
            for lr, beta, it in zip(lrs, betas, iters):
                for _ in range(it):
                    t += 1
                    xy = bary_to_xy(b)
                    a = np.maximum(areas_xy(xy), 1e-12)
                    w = np.exp(-beta * (a - a.min()))
                    w /= w.sum()
                    g = -grad_bary(b, w)  # gradient of loss = -softmin
                    m = 0.9 * m + 0.1 * g
                    v = 0.999 * v + 0.001 * g * g
                    mh = m / (1.0 - 0.9 ** t)
                    vh = v / (1.0 - 0.999 ** t)
                    b = project(b - lr * mh / (np.sqrt(vh) + 1e-9))
                    b = pin_vertices(b)
            return b

        def polish(b, rounds=150, lr=0.002, tol=1.05):
            # active-set polish: gradient ascent using only the binding
            # (near-minimal) triangles; keep the best configuration seen
            best_b = b.copy()
            best_a = areas_xy(bary_to_xy(b)).min()
            for _ in range(rounds):
                xy = bary_to_xy(b)
                a = areas_xy(xy)
                amin = a.min()
                if amin > best_a:
                    best_a = amin
                    best_b = b.copy()
                mask = a <= amin * tol
                g = grad_bary(b, mask.astype(float))
                gmax = np.abs(g).max()
                if gmax < 1e-15:
                    break
                b = project(b + lr * g / gmax)
                b = pin_vertices(b)
            amin = areas_xy(bary_to_xy(b)).min()
            if amin > best_a:
                best_a = amin
                best_b = b.copy()
            return best_b, best_a

        best_xy = None
        best_val = -1.0
        for seed in (11, 42, 2024, 7):
            try:
                b = make_start(seed)
                b = adam(b, (0.03, 0.01, 0.003), (20.0, 80.0, 400.0), (250, 250, 400))
                b, val = polish(b)
                xy = bary_to_xy(b)
                val = areas_xy(xy).min()
                if val > best_val:
                    best_val = val
                    best_xy = xy
            except Exception:
                continue

        if best_xy is not None:
            return best_xy

    except Exception:
        pass

    # Fallback: deterministic triangular-grid-ish arrangement (always feasible)
    b = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.5, 0.0], [0.5, 0.5], [0.0, 0.5],
        [0.25, 0.0], [0.25, 0.25], [0.75, 0.0],
        [0.0, 0.25], [0.375, 0.25],
    ])
    return bary_to_xy(b)


# EVOLVE-BLOCK-END