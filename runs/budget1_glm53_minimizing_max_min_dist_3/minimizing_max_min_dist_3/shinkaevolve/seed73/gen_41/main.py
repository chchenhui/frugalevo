# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y) coordinates of the 14 points.

    """

    n = 14
    d = 3

    from scipy.optimize import minimize

    iu = np.triu_indices(n, 1)

    def score(P):
        D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
        dmin = D[iu].min()
        dmax = D[iu].max()
        if dmax <= 0:
            return -1.0
        return (dmin / dmax) ** 2

    def objective(flat):
        P = flat.reshape(n, d)
        diff = P[:, None, :] - P[None, :, :]
        dist = np.sqrt((diff ** 2).sum(-1) + 1e-12)
        dm = dist[iu]
        beta = 40.0
        # smooth min and max of pairwise distances
        wmin = np.exp(-beta * dm)
        dmin_s = (dm * wmin).sum() / wmin.sum()
        wmax = np.exp(beta * dm)
        dmax_s = (dm * wmax).sum() / wmax.sum()
        # objective: maximize dmin/dmax  ->  minimize -ratio (smooth)
        return -(dmin_s / dmax_s)

    best_s, best_P = -1.0, None
    for seed in range(60):
        rng = np.random.RandomState(seed)
        P0 = rng.randn(n, d)
        if seed == 0:
            # hint: cube vertices + antipodal-ish start
            cube = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)
            P0 = np.vstack([cube[:7], -cube[:7]])
        elif seed == 1:
            # hint: start from random points in a ball (not on a sphere)
            P0 = rng.randn(n, d) * rng.rand(n, 1)
        elif seed == 2:
            # hint: two concentric shells (outer antipodal pairs, inner ring)
            outer = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)
            ang = 2 * np.pi * np.arange(7) / 7
            inner = np.stack([0.6 * np.cos(ang), 0.6 * np.sin(ang), 0.6 * np.sin(2 * ang)], axis=1)
            P0 = np.vstack([outer, inner])
        elif seed == 3:
            # hint: points on a sphere, slightly perturbed inward
            V = rng.randn(n, d)
            V /= np.linalg.norm(V, axis=1, keepdims=True)
            P0 = V * (0.8 + 0.4 * rng.rand(n, 1))
        res = minimize(objective, P0.ravel(), method="L-BFGS-B",
                       options={"maxiter": 800, "maxfun": 3000})
        if not (np.all(np.isfinite(res.x)) and res.x.size == n * d):
            continue
        P = res.x.reshape(n, d)
        s = score(P)
        if s > best_s:
            best_s, best_P = s, P

    # ---- polishing stage: exact-ratio SLSQP (Tammes-style) ----
    def polish(P, rounds=6):
        P = P.copy()
        for _ in range(rounds):
            # rescale so dmax == 1
            D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
            dm = D[iu]
            dmax = dm.max()
            if not np.isfinite(dmax) or dmax <= 0:
                break
            P = P / dmax
            t0 = dm.min() / dmax

            def cons_t(x, t):
                return _mindist(x) - t

            def _mindist(x):
                Q = x.reshape(n, d)
                diff = Q[:, None, :] - Q[None, :, :]
                dist = np.sqrt((diff ** 2).sum(-1) + 1e-12)
                return dist[iu].min()

            def cons_all(x):
                # all pairwise distances <= 1 (keeps dmax at 1)
                Q = x.reshape(n, d)
                diff = Q[:, None, :] - Q[None, :, :]
                dist = np.sqrt((diff ** 2).sum(-1) + 1e-12)
                return 1.0 - dist[iu]

            x0 = P.ravel()
            # maximize t: encode t as extra variable, maximize t
            def neg_t(y):
                return -y[0]

            def cons_pairs(y):
                t = y[0]
                Q = y[1:].reshape(n, d)
                diff = Q[:, None, :] - Q[None, :, :]
                dist = np.sqrt((diff ** 2).sum(-1) + 1e-12)
                return dist[iu] - t

            def cons_dmax(y):
                Q = y[1:].reshape(n, d)
                diff = Q[:, None, :] - Q[None, :, :]
                dist = np.sqrt((diff ** 2).sum(-1) + 1e-12)
                return 1.0 - dist[iu]

            y0 = np.concatenate([[t0], x0])
            r = minimize(neg_t, y0, method="SLSQP",
                         constraints=[{"type": "ineq", "fun": cons_pairs},
                                      {"type": "ineq", "fun": cons_dmax}],
                         options={"maxiter": 300, "ftol": 1e-10})
            if np.all(np.isfinite(r.x)):
                Pn = r.x[1:].reshape(n, d)
                sn = score(Pn)
                if sn > score(P):
                    P = Pn
            D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
            dmx = D[iu].max()
            if dmx > 0:
                P = P / dmx
        return P

    if best_P is not None and np.all(np.isfinite(best_P)) and best_P.shape == (n, d):
        Pf = polish(best_P)
        if score(Pf) > best_s:
            best_s, best_P = score(Pf), Pf

    # center and normalize for a clean output
    best_P = best_P - best_P.mean(axis=0)
    best_P = best_P / np.abs(best_P).max()
    return best_P


# EVOLVE-BLOCK-END