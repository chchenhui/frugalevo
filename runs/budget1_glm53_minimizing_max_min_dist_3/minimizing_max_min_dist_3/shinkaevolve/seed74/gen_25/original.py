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

    rng = np.random.default_rng(42)
    I, J = np.triu_indices(n, k=1)

    def true_ratio(P):
        dd = np.linalg.norm(P[I] - P[J], axis=1)
        mx = dd.max()
        if mx <= 0:
            return 0.0
        return dd.min() / mx

    def normalize(P):
        P = P - P.mean(axis=0)
        mx = np.linalg.norm(P[I] - P[J], axis=1).max()
        if mx <= 1e-12:
            return P
        return P / mx

    # Fast annealed analytic-gradient ascent on a soft-minimum of the
    # pairwise distances. Points are re-centered and re-scaled to dmax=1
    # after every step, so the ratio equals the minimum distance.
    def grad_ascend(P0, ks=(3.0, 6.0, 12.0, 25.0, 50.0), iters=200,
                    lr=0.02):
        P = normalize(np.asarray(P0, dtype=float).copy())
        for k in ks:
            for _ in range(iters):
                diff = P[I] - P[J]
                dd = np.linalg.norm(diff, axis=1)
                dd = np.maximum(dd, 1e-9)
                w = np.exp(-k * (dd - dd.min()))
                W = w.sum()
                u = diff / dd[:, None]
                coef = (w / W)[:, None]
                G = np.zeros_like(P)
                np.add.at(G, I, coef * u)
                np.add.at(G, J, -coef * u)
                P = normalize(P + lr * G)
        return P

    # Structured starts: icosahedron + poles, Fibonacci sphere, jittered,
    # random-sphere and Gaussian starts.
    phi = (1 + np.sqrt(5)) / 2
    ico = []
    for a in (-1, 1):
        for b in (-phi, phi):
            ico.append([a, b, 0])
            ico.append([0, a, b])
            ico.append([b, 0, a])
    ico = np.array(ico, dtype=float)
    ico_poles = np.vstack([ico, [[0, 0, 2.0], [0, 0, -2.0]]])

    kk = np.arange(n) + 0.5
    ga = np.pi * (3 - np.sqrt(5))
    fib = np.zeros((n, 3))
    fib[:, 2] = 1 - 2 * kk / n
    r_ = np.sqrt(np.maximum(0.0, 1 - fib[:, 2] ** 2))
    fib[:, 0] = r_ * np.cos(ga * kk)
    fib[:, 1] = r_ * np.sin(ga * kk)

    starts = [ico_poles.copy(), fib.copy()]
    for scale in (0.05, 0.2):
        starts.append(ico_poles + scale * rng.standard_normal((n, d)))
        starts.append(fib + scale * rng.standard_normal((n, d)))
    for _ in range(10):
        p = rng.standard_normal((n, d))
        p = p / np.linalg.norm(p, axis=1, keepdims=True)
        starts.append(p * (0.5 + rng.random()))
    for _ in range(4):
        starts.append(rng.standard_normal((n, d)))

    best_pts = None
    best_r = -1.0
    for pts0 in starts:
        P = grad_ascend(pts0)
        r = true_ratio(P)
        if r > best_r:
            best_r = r
            best_pts = P.copy()

    # Exact SLSQP polish: maximize t s.t. t^2 <= ||pi-pj||^2 <= 1
    m = len(I)

    def polish(P):
        P = normalize(P)
        if np.linalg.norm(P[I] - P[J], axis=1).max() <= 1e-12:
            return P, true_ratio(P)
        t0 = np.linalg.norm(P[I] - P[J], axis=1).min()
        x0 = np.concatenate([P.flatten(), [t0]])

        def fobj(x):
            return -x[-1]

        def fjac(x):
            Jc = np.zeros(43)
            Jc[-1] = -1.0
            return Jc

        def cons(x):
            P2 = x[:42].reshape(n, d)
            dv = np.sum((P2[I] - P2[J]) ** 2, axis=1)
            return np.concatenate([1.0 - dv, dv - x[-1] ** 2])

        def cjac(x):
            P2 = x[:42].reshape(n, d)
            diff = P2[I] - P2[J]
            Jc = np.zeros((2 * m, 43))
            Jc[:m, 3 * I[:, None] + np.arange(3)] = -2 * diff
            Jc[:m, 3 * J[:, None] + np.arange(3)] = 2 * diff
            Jc[m:, 3 * I[:, None] + np.arange(3)] = 2 * diff
            Jc[m:, 3 * J[:, None] + np.arange(3)] = -2 * diff
            Jc[m:, -1] = -2 * x[-1]
            return Jc

        res = minimize(fobj, x0, method="SLSQP",
                       constraints=[{"type": "ineq", "fun": cons,
                                     "jac": cjac}],
                       options={"maxiter": 500, "ftol": 1e-12})
        P2 = res.x[:42].reshape(n, d)
        P2 = P2 - P2.mean(axis=0)
        return P2, true_ratio(P2)

    P2, r2 = polish(best_pts)
    if r2 > best_r:
        best_r = r2
        best_pts = P2
    for _ in range(5):
        Pp = best_pts + 0.02 * rng.standard_normal((n, d))
        P2, r2 = polish(Pp)
        if r2 > best_r:
            best_r = r2
            best_pts = P2

    # Normalize: center and scale so max pairwise distance = 1 (convenience)
    best_pts = best_pts - best_pts.mean(axis=0)
    dists = np.linalg.norm(best_pts[:, None, :] - best_pts[None, :, :], axis=-1)
    dmax = dists.max()
    if dmax > 0:
        best_pts = best_pts / dmax

    return best_pts


# EVOLVE-BLOCK-END