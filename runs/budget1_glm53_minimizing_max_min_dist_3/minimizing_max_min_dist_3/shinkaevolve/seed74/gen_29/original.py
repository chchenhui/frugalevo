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

    def true_ratio(pts):
        dists = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        iu = np.triu_indices(n, k=1)
        dd = dists[iu]
        dmax = dd.max()
        if dmax <= 0:
            return 0.0
        return dd.min() / dmax

    def objective(flat):
        pts = flat.reshape(n, d)
        # center (translation invariance)
        pts = pts - pts.mean(axis=0)
        dists = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        iu = np.triu_indices(n, k=1)
        dd = dists[iu]
        if np.any(dd <= 1e-12):
            return 1e6
        k = 50.0
        # smooth min and max of squared-ish distances
        smin = -np.log(np.sum(np.exp(-k * dd))) / k
        smax = np.log(np.sum(np.exp(k * dd))) / k
        return -(smin / smax)

    def grad(flat):
        pts = flat.reshape(n, d).copy()
        pts_c = pts - pts.mean(axis=0)
        diff = pts_c[:, None, :] - pts_c[None, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        i, j = np.triu_indices(n, k=1)
        dd = dists[i, j]
        k = 50.0
        w_min = np.exp(-k * dd)
        w_max = np.exp(k * dd)
        smin = -np.log(w_min.sum()) / k
        smax = np.log(w_max.sum()) / k
        # d(smin)/dp, d(smax)/dp
        g = np.zeros_like(pts_c)
        # smin grad: smin = -log(sum e^{-k dd})/k
        # dsmin/dp_l = sum_m wmin_m/(k*W) * unit vector contribution
        Wm = w_min.sum()
        Wx = w_max.sum()
        for a, b, wm, wx, dval in zip(i, j, w_min, w_max, dd):
            if dval < 1e-12:
                continue
            u = (pts_c[a] - pts_c[b]) / dval
            # dsmin/da = (wm/(k*Wm)) * k * u = wm/Wm * u ... sign:
            # d/dd e^{-k dd} = -k e^{-k dd}; d smin/d dd = w/(W k)
            # so d smin/d p_a = (w/(W k)) * k * u = w/W * u
            coef_min = wm / Wm
            coef_max = wx / Wx
            g[a] += (coef_min - 0) * u * 0  # placeholder, computed below
            g[a] += (coef_min / k) * k * u
            g[b] -= (coef_min / k) * k * u
            g[a] += (coef_max / k) * k * u
            g[b] -= (coef_max / k) * k * u
        # combine: f = -smin/smax, df/dp = -(dsmin*smax - smin*dsmax)/smax^2
        return (-(g) / (smax ** 2) * smax).flatten()  # refined below

    best_pts = None
    best_r = -1.0

    rng = np.random.default_rng(42)

    # Structured starts: icosahedron vertices plus 2 poles, and random starts
    phi = (1 + np.sqrt(5)) / 2
    ico = []
    for a in (-1, 1):
        for b in (-phi, phi):
            ico.append([a, b, 0])
            ico.append([0, a, b])
            ico.append([b, 0, a])
    ico = np.array(ico, dtype=float)
    starts = []
    pts0 = np.vstack([ico, [[0, 0, 2.0], [0, 0, -2.0]]])
    starts.append(pts0 + 0.1 * rng.standard_normal((n, d)))

    for t in range(6):
        starts.append(rng.standard_normal((n, d)))

    for pts0 in starts:
        res = minimize(objective, pts0.flatten(), method="L-BFGS-B",
                       options={"maxiter": 800})
        pts = res.x.reshape(n, d)
        r = true_ratio(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()

    # Polish the best one with a direct Nelder-Mead on the true ratio
    def true_obj(flat):
        return -true_ratio(flat.reshape(n, d))

    res = minimize(true_obj, best_pts.flatten(), method="Nelder-Mead",
                   options={"maxiter": 4000, "maxfev": 6000, "xatol": 1e-10,
                            "fatol": 1e-12})
    pts = res.x.reshape(n, d)
    if true_ratio(pts) > best_r:
        best_pts = pts

    # Normalize: center and scale so max pairwise distance = 1 (convenience)
    best_pts = best_pts - best_pts.mean(axis=0)
    dists = np.linalg.norm(best_pts[:, None, :] - best_pts[None, :, :], axis=-1)
    dmax = dists.max()
    if dmax > 0:
        best_pts = best_pts / dmax

    return best_pts


# EVOLVE-BLOCK-END