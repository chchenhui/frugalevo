# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of
    minimum to maximum pairwise distance.

    Returns
        points: np.ndarray of shape (14,3)
    """
    n = 14
    d = 3
    from scipy.optimize import minimize
    from scipy.special import logsumexp

    i_arr, j_arr = np.triu_indices(n, k=1)
    m = len(i_arr)

    def true_ratio_pts(pts):
        dd = np.linalg.norm(pts[i_arr] - pts[j_arr], axis=1)
        mx = dd.max()
        if mx <= 1e-12:
            return 0.0
        return dd.min() / mx

    # ---------- smooth ratio objective with exact gradient ----------
    def make_obj_grad(k):
        def fg(flat):
            pts = flat.reshape(n, d)
            pts = pts - pts.mean(axis=0)
            diff = pts[i_arr] - pts[j_arr]            # (m,3)
            dd = np.linalg.norm(diff, axis=1)         # (m,)
            if np.any(dd <= 1e-12):
                return 1e6, np.zeros_like(flat)
            # stable softmax weights
            lse_min = logsumexp(-k * dd)
            lse_max = logsumexp(k * dd)
            smin = -lse_min / k
            smax = lse_max / k
            f = -(smin / smax)
            w_min = np.exp(-k * dd - lse_min)          # sums to 1
            w_max = np.exp(k * dd - lse_max)
            # df/dd_m
            dfd = -((w_min / 1.0) * smax - smin * w_max) / (smax ** 2)
            u = diff / dd[:, None]                     # unit vectors a->b
            g = np.zeros((n, d))
            contrib = dfd[:, None] * u
            np.add.at(g, i_arr, contrib)
            np.add.at(g, j_arr, -contrib)
            return f, (g - g.mean(axis=0)).flatten()
        return fg

    # ---------- exact SLSQP polish: max t s.t. t^2 <= |pi-pj|^2 <= 1 ----------
    def polish(pts):
        P = pts - pts.mean(axis=0)
        dd = np.linalg.norm(P[i_arr] - P[j_arr], axis=1)
        mx = dd.max()
        if mx <= 1e-12:
            return pts, 0.0
        P = P / mx
        t0 = dd.min() / mx
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
            # upper constraints: 1 - |diff|^2 >= 0
            J[:m, 3 * i_arr] = -2 * diff
            # fancy indexing for column blocks; do explicitly:
            J[:m] = 0.0
            for idx in range(m):
                a, b = i_arr[idx], j_arr[idx]
                J[idx, 3 * a:3 * a + 3] = -2 * diff[idx]
                J[idx, 3 * b:3 * b + 3] = 2 * diff[idx]
                J[m + idx, 3 * a:3 * a + 3] = 2 * diff[idx]
                J[m + idx, 3 * b:3 * b + 3] = -2 * diff[idx]
            J[m:, -1] = -2 * x[-1]
            return J

        res = minimize(fobj, x0, method="SLSQP",
                       constraints=[{"type": "ineq", "fun": cons, "jac": cjac}],
                       options={"maxiter": 500, "ftol": 1e-14})
        P2 = res.x[:3 * n].reshape(n, d)
        P2 = P2 - P2.mean(axis=0)
        return P2, true_ratio_pts(P2)

    rng = np.random.default_rng(7)

    # ---------- structured starts ----------
    phi = (1 + np.sqrt(5)) / 2
    ico = []
    for a in (-1, 1):
        for b in (-phi, phi):
            ico.append([a, b, 0])
            ico.append([0, a, b])
            ico.append([b, 0, a])
    ico = np.array(ico[:12], dtype=float)
    ico_poles = np.vstack([ico, [[0, 0, 2.0], [0, 0, -2.0]]])

    kk = np.arange(n) + 0.5
    ga = np.pi * (3 - np.sqrt(5))
    fib = np.zeros((n, 3))
    fib[:, 2] = 1 - 2 * kk / n
    r_ = np.sqrt(np.maximum(0.0, 1 - fib[:, 2] ** 2))
    fib[:, 0] = r_ * np.cos(ga * kk)
    fib[:, 1] = r_ * np.sin(ga * kk)

    starts = [ico_poles.copy(), fib.copy()]
    for scale in (0.05, 0.15, 0.3):
        starts.append(ico_poles + scale * rng.standard_normal((n, d)))
        starts.append(fib + scale * rng.standard_normal((n, d)))
    for _ in range(12):
        p = rng.standard_normal((n, d))
        p = p / np.linalg.norm(p, axis=1, keepdims=True)
        starts.append(p * (0.5 + rng.random()))
    for _ in range(8):
        starts.append(rng.standard_normal((n, d)))

    best_pts = None
    best_r = -1.0

    ks = [15.0, 60.0, 200.0]
    for pts0 in starts:
        x = pts0.flatten()
        ok = True
        for k in ks:
            fg = make_obj_grad(k)
            res = minimize(fg, x, jac=True, method="L-BFGS-B",
                           options={"maxiter": 500})
            x = res.x
        pts = x.reshape(n, d)
        r = true_ratio_pts(pts)
        # polish each decent candidate
        if r > 0.6 * best_r or best_r < 0:
            pts, r = polish(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()

    # jittered re-polish of the leader
    for _ in range(6):
        Pp = best_pts + 0.03 * rng.standard_normal((n, d))
        P2, r2 = polish(Pp)
        if r2 > best_r:
            best_r = r2
            best_pts = P2.copy()

    best_pts = np.asarray(best_pts, dtype=float)
    # normalize: center and scale so max pairwise distance = 1
    best_pts = best_pts - best_pts.mean(axis=0)
    dists = np.linalg.norm(best_pts[:, None, :] - best_pts[None, :, :], axis=-1)
    dmax = dists.max()
    if dmax > 0:
        best_pts = best_pts / dmax

    return best_pts


# EVOLVE-BLOCK-END