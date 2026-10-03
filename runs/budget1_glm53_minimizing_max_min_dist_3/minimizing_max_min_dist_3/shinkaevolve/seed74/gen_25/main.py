# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3D maximizing dmin/dmax.

    Centrally symmetric construction: 7 well-separated unit directions
    plus their antipodes. Diameter is exactly 2, so ratio equals the
    min chord distance of the 7-direction spherical code (Tammes N=7).
    """
    from scipy.optimize import minimize

    n = 14
    m = 7
    rng = np.random.default_rng(20240517)
    I, J = np.triu_indices(m, 1)

    def dir_min_dist(X):
        return np.linalg.norm(X[I] - X[J], axis=1).min()

    def anneal(X0, ks=(4.0, 8.0, 16.0, 32.0, 64.0, 128.0),
               iters=160, lr=0.35):
        X = np.asarray(X0, float).copy()
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        for k in ks:
            for _ in range(iters):
                diff = X[I] - X[J]
                d = np.maximum(np.linalg.norm(diff, axis=1), 1e-12)
                w = np.exp(-k * (d - d.min()))
                w /= w.sum()
                u = diff / d[:, None]
                G = np.zeros_like(X)
                np.add.at(G, I, w[:, None] * u)
                np.add.at(G, J, -w[:, None] * u)
                # project onto tangent plane of sphere
                G -= X * (X * G).sum(axis=1, keepdims=True)
                gn = np.linalg.norm(G)
                if gn < 1e-14:
                    break
                X = X + (lr / np.sqrt(k)) * G / gn
                X /= np.linalg.norm(X, axis=1, keepdims=True)
        return X

    def polish(X0, maxiter=400):
        X0 = X0 / np.linalg.norm(X0, axis=1, keepdims=True)
        t0 = dir_min_dist(X0)
        z0 = np.concatenate([X0.ravel(), [t0]])

        def fobj(z):
            return -z[-1]

        def c_dist(z):
            X = z[:3 * m].reshape(m, 3)
            return ((X[I] - X[J]) ** 2).sum(axis=1) - z[-1] ** 2

        def j_dist(z):
            X = z[:3 * m].reshape(m, 3)
            diff = X[I] - X[J]
            g = np.zeros((len(I), 3 * m + 1))
            g[:, 3 * I[:, None] + np.arange(3)] = 2 * diff
            g[:, 3 * J[:, None] + np.arange(3)] = -2 * diff
            g[:, -1] = -2 * z[-1]
            return g

        def c_norm(z):
            X = z[:3 * m].reshape(m, 3)
            return (X ** 2).sum(axis=1) - 1.0

        def j_norm(z):
            X = z[:3 * m].reshape(m, 3)
            g = np.zeros((m, 3 * m + 1))
            for i in range(m):
                g[i, 3 * i:3 * i + 3] = 2 * X[i]
            return g

        try:
            res = minimize(fobj, z0, method="SLSQP",
                           constraints=[
                               {"type": "ineq", "fun": c_dist, "jac": j_dist},
                               {"type": "eq", "fun": c_norm, "jac": j_norm},
                           ],
                           options={"maxiter": maxiter, "ftol": 1e-12})
            X = res.x[:3 * m].reshape(m, 3)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        except Exception:
            X = X0
        return X

    # ---- diverse starts for the 7-direction code ----
    starts = []
    # pentagonal bipyramid (guaranteed >= 72 deg)
    bp = [[0, 0, 1], [0, 0, -1]]
    for t in range(5):
        a = 2 * np.pi * t / 5
        bp.append([np.cos(a), np.sin(a), 0])
    starts.append(np.array(bp, float))
    # fibonacci 7
    kk = np.arange(m) + 0.5
    ga = np.pi * (3 - np.sqrt(5))
    z = 1 - 2 * kk / m
    r_ = np.sqrt(np.maximum(0.0, 1 - z ** 2))
    starts.append(np.stack([r_ * np.cos(ga * kk),
                            r_ * np.sin(ga * kk), z], 1))
    # octahedron + extra
    starts.append(np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0],
                            [0, -1, 0], [0, 0, 1], [0, 0, -1],
                            [0.5, 0.5, 0.7]], float))
    # jittered bipyramids at various latitudes
    for beta in (55, 65, 75, 85, 95):
        s = [[0, 0, 1], [0, 0, -1]]
        for t in range(5):
            a = 2 * np.pi * t / 5 + 0.3
            sb = np.sin(np.radians(beta))
            cb = np.cos(np.radians(beta))
            s.append([sb * np.cos(a), sb * np.sin(a), cb])
        starts.append(np.array(s, float) + 0.05 * rng.standard_normal((m, 3)))
    # random starts
    for _ in range(16):
        starts.append(rng.standard_normal((m, 3)))

    # ---- search: anneal then polish, keep best chord distance ----
    annealed = []
    for X0 in starts:
        X = anneal(X0)
        annealed.append(X)
    annealed.sort(key=dir_min_dist, reverse=True)

    best_X, best_d = None, -1.0
    for X in annealed[:6]:
        Xp = polish(X)
        d = dir_min_dist(Xp)
        if d > best_d:
            best_d, best_X = d, Xp
    # extra jittered restarts around the best
    if best_X is not None:
        for _ in range(6):
            Xp = polish(best_X + 0.03 * rng.standard_normal((m, 3)))
            d = dir_min_dist(Xp)
            if d > best_d:
                best_d, best_X = d, Xp
    else:
        best_X = annealed[0]

    # ---- build centrally symmetric 14-point set ----
    pts = np.vstack([best_X, -best_X])

    # verify true ratio on the full 14-point set
    I14, J14 = np.triu_indices(n, 1)
    dd = np.linalg.norm(pts[I14] - pts[J14], axis=1)
    dmax = dd.max()
    if dmax <= 0:
        pts = rng.standard_normal((n, 3))
    else:
        pts = pts - pts.mean(axis=0)
        pts = pts / dmax

    return pts


# EVOLVE-BLOCK-END