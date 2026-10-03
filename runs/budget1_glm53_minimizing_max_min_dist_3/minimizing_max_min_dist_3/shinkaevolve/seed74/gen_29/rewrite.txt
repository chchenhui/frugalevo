# EVOLVE-BLOCK-START
import time

import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3

    from scipy.optimize import minimize

    rng = np.random.default_rng(42)
    i_arr, j_arr = np.triu_indices(n, k=1)

    def true_ratio_pts(pts):
        dd = np.linalg.norm(pts[i_arr] - pts[j_arr], axis=1)
        mx = dd.max()
        if mx <= 0:
            return 0.0
        return dd.min() / mx

    # ---------- exact SLSQP polish: maximize t s.t. t^2 <= d^2 <= 1 ----------
    m = len(i_arr)

    def polish(pts, maxiter=300):
        P = pts - pts.mean(axis=0)
        mx = np.linalg.norm(P[i_arr] - P[j_arr], axis=1).max()
        if mx <= 1e-12:
            return pts, true_ratio_pts(pts)
        P = P / mx
        t0 = np.linalg.norm(P[i_arr] - P[j_arr], axis=1).min()
        x0 = np.concatenate([P.flatten(), [t0]])

        def fobj(x):
            return -x[-1]

        def fjac(x):
            J = np.zeros(3 * n + 1)
            J[-1] = -1.0
            return J

        def cons(x):
            P2 = x[:3 * n].reshape(n, d)
            dv = np.sum((P2[i_arr] - P2[j_arr]) ** 2, axis=1)
            return np.concatenate([1.0 - dv, dv - x[-1] ** 2])

        def cjac(x):
            P2 = x[:3 * n].reshape(n, d)
            diff = P2[i_arr] - P2[j_arr]
            J = np.zeros((2 * m, 3 * n + 1))
            J[:m, 3 * i_arr] = -2 * diff
            # scatter-add for b indices (vectorized via np.add.at pattern)
            for dim in range(d):
                np.add.at(J[:m], 3 * j_arr + dim, 2 * diff[:, dim])
                np.add.at(J[:m], 3 * i_arr + dim, -2 * diff[:, dim])
                np.add.at(J[m:, 3 * i_arr + dim], 2 * diff[:, dim])
                np.add.at(J[m:, 3 * j_arr + dim], -2 * diff[:, dim])
            J[m:, -1] = -2 * x[-1]
            return J

        res = minimize(fobj, x0, method="SLSQP",
                       constraints=[{"type": "ineq", "fun": cons, "jac": cjac}],
                       options={"maxiter": maxiter, "ftol": 1e-12})
        P2 = res.x[:3 * n].reshape(n, d)
        P2 = P2 - P2.mean(axis=0)
        return P2, true_ratio_pts(P2)

    # ---------- soft objectives ----------
    def soft_obj_factory(k):
        def obj(flat):
            pts = flat.reshape(n, d)
            pts = pts - pts.mean(axis=0)
            dd = np.linalg.norm(pts[i_arr] - pts[j_arr], axis=1)
            mx = dd.max()
            if mx <= 1e-12:
                return 1e6
            s = dd / mx
            smin = -np.log(np.sum(np.exp(-k * s))) / k
            return -smin
        return obj

    def repulsion_obj(flat):
        # Howard-style: maximize sum of log distances (spreads points)
        pts = flat.reshape(n, d)
        dd = np.linalg.norm(pts[i_arr] - pts[j_arr], axis=1)
        if np.any(dd < 1e-9):
            return 1e6
        return -np.sum(np.log(dd))

    ks = [8.0, 20.0, 50.0, 120.0]

    def anneal(x, iters=(150, 150, 150, 150)):
        for k, it in zip(ks, iters):
            res = minimize(soft_obj_factory(k), x, method="L-BFGS-B",
                           options={"maxiter": it})
            x = res.x
        return x

    # ---------- structured starts ----------
    phi = (1 + np.sqrt(5)) / 2
    ico = []
    for a in (-1, 1):
        for b in (-phi, phi):
            ico.append([a, b, 0])
            ico.append([0, a, b])
            ico.append([b, 0, a])
    ico = np.array(ico, dtype=float)

    cubo = []
    for a in (-1, 1):
        for b in (-1, 1):
            cubo.append([a, b, 0])
            cubo.append([a, 0, b])
            cubo.append([0, a, b])
    cubo = np.array(cubo, dtype=float)[:12]

    kk = np.arange(n) + 0.5
    ga = np.pi * (3 - np.sqrt(5))
    fib = np.zeros((n, 3))
    fib[:, 2] = 1 - 2 * kk / n
    r_ = np.sqrt(np.maximum(0.0, 1 - fib[:, 2] ** 2))
    fib[:, 0] = r_ * np.cos(ga * kk)
    fib[:, 1] = r_ * np.sin(ga * kk)

    starts = [fib.copy()]

    for base, name in ((ico, 'ico'), (cubo, 'cubo')):
        # rotate base so poles go along z, x, y axes
        for axis in range(3):
            B = np.roll(base, axis, axis=1)
            for pr in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
                pole = np.zeros((2, 3))
                pole[:, 2] = pr
                starts.append(np.vstack([B, pole]))

    for _ in range(6):
        p = rng.standard_normal((n, d))
        p = p / np.linalg.norm(p, axis=1, keepdims=True)
        starts.append(p * (0.5 + rng.random()))
    for _ in range(4):
        starts.append(rng.standard_normal((n, d)))

    best_pts = None
    best_r = -1.0

    def consider(pts):
        nonlocal best_pts, best_r
        r = true_ratio_pts(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()

    t0 = time.time()
    budget = 32.0

    for pts0 in starts:
        if time.time() - t0 > budget:
            break
        x = pts0.flatten()
        # light repulsion spread
        res = minimize(repulsion_obj, x, method="L-BFGS-B",
                       options={"maxiter": 100})
        x = res.x
        x = anneal(x)
        P, r = polish(x.reshape(n, d))
        consider(P)

    # ---------- budgeted restart loop around incumbent ----------
    restart_idx = 0
    while time.time() - t0 < budget and restart_idx < 40:
        restart_idx += 1
        if restart_idx % 3 == 0:
            pts0 = rng.standard_normal((n, d))
        else:
            scale = 0.05 if restart_idx % 2 else 0.02
            pts0 = best_pts + scale * rng.standard_normal((n, d))
        x = pts0.flatten()
        x = anneal(x, iters=(100, 100, 100, 100))
        P, r = polish(x.reshape(n, d))
        consider(P)
        P, r = polish(P + 0.01 * rng.standard_normal((n, d)))
        consider(P)

    # final Nelder-Mead on the true ratio
    def true_obj(flat):
        return -true_ratio_pts(flat.reshape(n, d))

    if best_pts is not None:
        res = minimize(true_obj, best_pts.flatten(), method="Nelder-Mead",
                        options={"maxiter": 8000, "maxfev": 12000,
                                 "xatol": 1e-12, "fatol": 1e-14})
        consider(res.x.reshape(n, d))

    if best_pts is None:
        best_pts = rng.standard_normal((n, d))

    best_pts = np.asarray(best_pts, dtype=float)
    best_pts = best_pts - best_pts.mean(axis=0)
    dists = np.linalg.norm(best_pts[:, None, :] - best_pts[None, :, :], axis=-1)
    dmax = dists.max()
    if dmax > 0:
        best_pts = best_pts / dmax

    return best_pts


# EVOLVE-BLOCK-END