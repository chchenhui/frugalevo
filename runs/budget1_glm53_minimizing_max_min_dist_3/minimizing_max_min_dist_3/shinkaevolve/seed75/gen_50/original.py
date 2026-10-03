# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y) coordinates of the 14 points.

    """

    from scipy.optimize import minimize
    from scipy.special import logsumexp

    n = 14
    d = 3
    iu = np.triu_indices(n, 1)

    def unit(x):
        p = x.reshape(n, d)
        return p / np.linalg.norm(p, axis=1, keepdims=True)

    def dists(p):
        diff = p[:, None, :] - p[None, :, :]
        D2 = (diff * diff).sum(-1)
        return np.sqrt(np.maximum(D2[iu], 1e-12))

    def ratio(p):
        D = dists(p)
        return D.min() / D.max()

    def tammes14_seed():
        # Two staggered hexagonal rings at height +-h plus two poles.
        # For 14 points the optimal ring height satisfies
        # (2/3)*r^2 = (1-h)^2  with r^2 = 1 - h^2, giving
        # h = 1/sqrt(5) and r = 2/sqrt(5).
        h = 1.0 / np.sqrt(5.0)
        r = 2.0 / np.sqrt(5.0)
        ang = np.pi / 6.0
        pts = []
        pts.append([0.0, 0.0, 1.0])
        pts.append([0.0, 0.0, -1.0])
        for k in range(6):
            a = k * (np.pi / 3.0)
            pts.append([r * np.cos(a), r * np.sin(a), h])
            pts.append([r * np.cos(a + ang), r * np.sin(a + ang), -h])
        return np.array(pts)

    rng = np.random.default_rng(0)
    best_pts = None
    best_r = -1.0

    def d6_family(R, h, z0):
        th = np.arange(6) * (np.pi / 3.0)
        r1 = np.stack([R * np.cos(th), R * np.sin(th), np.full(6, h)], axis=1)
        r2 = np.stack([R * np.cos(th + np.pi / 6), R * np.sin(th + np.pi / 6),
                       np.full(6, -h)], axis=1)
        poles = np.array([[0.0, 0.0, z0], [0.0, 0.0, -z0]])
        return np.vstack([r1, r2, poles])

    seeds = [tammes14_seed()]
    # Fine grid over the D6 family: this family contains the known
    # near-optimal configurations for 14 points.
    grid_best_r, grid_best = -1.0, None
    for h in np.linspace(0.15, 1.4, 26):
        for z0 in np.linspace(max(h, 0.2) + 0.05, 2.6, 48):
            P = d6_family(1.0, h, z0)
            P = P / np.linalg.norm(P, axis=1, keepdims=True)
            r = ratio(P)
            if r > grid_best_r:
                grid_best_r, grid_best = r, P
    if grid_best is not None:
        seeds.append(grid_best.copy())
    for _ in range(6):
        p = rng.normal(size=(n, d))
        seeds.append(p / np.linalg.norm(p, axis=1, keepdims=True))

    for pts in seeds:

        # Annealed soft-min maximization of minimum distance on the unit sphere.
        for T in (0.08, 0.03, 0.012, 0.005, 0.002):
            def loss(x, T=T):
                p = unit(x)
                D = dists(p)
                return T * logsumexp(-D / T)

            res = minimize(loss, pts.ravel(), method="L-BFGS-B",
                           options={"maxiter": 500, "maxfun": 2000})
            pts = unit(res.x)

        # Final refinement on the true objective dmin/dmax (normalized to
        # unit radius of gyration keeps scale fixed while allowing the
        # diameter to shrink, exactly what the metric rewards).
        def loss2(x):
            p = x.reshape(n, d)
            p = p - p.mean(0)
            D = dists(p)
            lo = T * logsumexp(-D / T)
            hi = 0.005 * logsumexp(D / 0.005)
            return lo + hi

        res = minimize(loss2, pts.ravel(), method="L-BFGS-B",
                       options={"maxiter": 800, "maxfun": 3000})
        p2 = res.x.reshape(n, d)
        p2 -= p2.mean(0)
        r = ratio(p2)
        if r > best_r:
            best_r = r
            best_pts = p2.copy()

        r = ratio(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()

    # Greedy hill-climb on the true ratio: move the closest pair apart
    # along its separating direction, pull the farthest pair together,
    # and try single-coordinate nudges. Accept only strict improvements.
    def climb(P, steps=(0.02, 0.006, 0.002, 6e-4, 2e-4, 8e-5)):
        P = (P - P.mean(axis=0)).copy()
        P = P / dists(P).max()
        best = ratio(P)
        for step in steps:
            for _ in range(2000):
                D = dists(P)
                imin = int(np.argmin(D))
                imax = int(np.argmax(D))
                i0, j0 = int(iu[0][imin]), int(iu[1][imin])
                i1, j1 = int(iu[0][imax]), int(iu[1][imax])
                cands = []
                u = P[j0] - P[i0]
                nu = np.linalg.norm(u)
                if nu > 1e-12:
                    u = u / nu
                    Q = P.copy(); Q[i0] -= 0.5 * step * u; Q[j0] += 0.5 * step * u
                    cands.append(Q)
                    Q = P.copy(); Q[i0] -= step * u; cands.append(Q)
                    Q = P.copy(); Q[j0] += step * u; cands.append(Q)
                v = P[j1] - P[i1]
                nv = np.linalg.norm(v)
                if nv > 1e-12:
                    v = v / nv
                    Q = P.copy(); Q[i1] += 0.5 * step * v; Q[j1] -= 0.5 * step * v
                    cands.append(Q)
                    Q = P.copy(); Q[i1] += step * v; cands.append(Q)
                    Q = P.copy(); Q[j1] -= step * v; cands.append(Q)
                for a in range(n):
                    for c in range(d):
                        for s in (step, -step):
                            Q = P.copy(); Q[a, c] += s; cands.append(Q)
                improved = False
                for Q in cands:
                    r = ratio(Q)
                    if r > best + 1e-15:
                        best = r
                        P = Q
                        improved = True
                        break
                if not improved:
                    break
        return P, best

    Pc, rc = climb(best_pts)
    if rc > best_r:
        best_r, best_pts = rc, Pc
    # a few jittered climb restarts around the best
    for k in range(6):
        mag = 0.03 * (0.7 ** k)
        Q = best_pts + mag * rng.normal(size=best_pts.shape)
        Qc, rc = climb(Q)
        if rc > best_r:
            best_r, best_pts = rc, Qc

    # Perturb-and-reanneal restarts around the incumbent: each restart
    # runs a full annealing chain (soft-min on sphere, then free-space
    # soft-min/soft-max) from a perturbed copy of the best found so far,
    # which can escape basins that greedy climbing cannot.
    def anneal_chain(pts):
        for T in (0.06, 0.02, 0.01, 0.005, 0.002):
            def loss(x, T=T):
                p = unit(x)
                D = dists(p)
                return T * logsumexp(-D / T)

            res = minimize(loss, pts.ravel(), method="L-BFGS-B",
                           options={"maxiter": 500, "maxfun": 2000})
            pts = unit(res.x)

        def loss2(x):
            p = x.reshape(n, d)
            p = p - p.mean(0)
            D = dists(p)
            lo = 0.002 * logsumexp(-D / 0.002)
            hi = 0.005 * logsumexp(D / 0.005)
            return lo + hi

        res = minimize(loss2, pts.ravel(), method="L-BFGS-B",
                       options={"maxiter": 800, "maxfun": 3000})
        p2 = res.x.reshape(n, d)
        p2 -= p2.mean(0)
        return p2

    for rep in range(8):
        prng = np.random.default_rng(5000 + rep)
        start = best_pts + 0.08 * prng.normal(size=best_pts.shape)
        p2 = anneal_chain(start)
        r = ratio(p2)
        if r > best_r:
            best_r = r
            best_pts = p2.copy()
        # also hill-climb the annealed candidate for fine polish
        Qc, rc = climb(p2)
        if rc > best_r:
            best_r, best_pts = rc, Qc

    return best_pts


# EVOLVE-BLOCK-END