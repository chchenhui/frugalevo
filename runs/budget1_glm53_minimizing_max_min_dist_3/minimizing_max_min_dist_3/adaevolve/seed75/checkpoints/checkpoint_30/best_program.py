# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Constructs 14 points in 3D maximizing dmin/dmax.

    Approach: multi-start annealed soft-min optimization. All points are
    kept inside the unit-diameter region (penalty on dmax > 1), and we
    maximize a smooth lower bound on dmin (soft-min with decreasing
    temperature). Starts include random spherical configurations and a
    structured icosahedron-based seed (12 icosa vertices + 2 poles).
    After annealing, an incumbent polish loop perturbs the best found
    configuration. The final configuration is rescaled so dmax = 1.
    """

    n, d = 14, 3
    iu = np.triu_indices(n, 1)
    rng = np.random.default_rng(2024)

    def pairwise(P):
        return np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)[iu]

    def ratio_of(P):
        D = pairwise(P)
        return D.min() / D.max()

    def direct_polish(P, rounds=8):
        """Exact min-max polish via SLSQP: maximize t subject to
        |pi-pj|^2 >= t^2 (all pairs) and |pi-pj|^2 <= 1 (diameter),
        with analytic Jacobians. Each round rescales so dmax = 1 first;
        keeps the result only if the true dmin/dmax ratio improves."""
        P = np.asarray(P, dtype=float).copy()
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        m = len(pairs)
        for _ in range(rounds):
            dm = pairwise(P).max()
            if dm <= 0:
                break
            P = P / dm

            def dist2(Pp):
                return ((Pp[:, None, :] - Pp[None, :, :]) ** 2).sum(-1)[iu]

            d2 = dist2(P)
            t0 = np.sqrt(d2.min()) * 0.999
            y0 = np.concatenate([P.ravel(), [t0]])

            def obj(y):
                return -y[-1]

            def obj_g(y):
                g = np.zeros(n * d + 1)
                g[-1] = -1.0
                return g

            def cons_f(y):
                x = y[:n * d].reshape(n, d)
                t = y[-1]
                dd = dist2(x)
                return np.concatenate([dd - t * t, 1.0 - dd])

            def cons_j(y):
                x = y[:n * d].reshape(n, d)
                t = y[-1]
                J = np.zeros((2 * m, n * d + 1))
                for idx, (i, j) in enumerate(pairs):
                    u = 2.0 * (x[i] - x[j])
                    J[idx, i * d:i * d + d] = u
                    J[idx, j * d:j * d + d] = -u
                    J[idx, -1] = -2.0 * t
                    J[m + idx, i * d:i * d + d] = -u
                    J[m + idx, j * d:j * d + d] = u
                return J

            try:
                res = minimize(obj, y0, jac=obj_g, method='SLSQP',
                               constraints=[{'type': 'ineq',
                                             'fun': cons_f,
                                             'jac': cons_j}],
                               options={'maxiter': 300, 'ftol': 1e-12})
            except Exception:
                break
            Pn = np.asarray(res.x[:n * d], dtype=float).reshape(n, d)
            if ratio_of(Pn) > ratio_of(P):
                P = Pn
            else:
                break
        return P

    def push_polish(P, iters=600, step=0.02):
        """Greedy ratio polish: each iteration pushes the closest pair
        apart and pulls the farthest pair together along their connecting
        lines; moves are accepted only if the true dmin/dmax ratio
        improves, otherwise the step size decays."""
        P = np.asarray(P, dtype=float).copy()
        best = ratio_of(P)
        tri = np.triu_indices(n, 1)
        for _ in range(iters):
            if step < 1e-7:
                break
            diff = P[:, None, :] - P[None, :, :]
            D = np.sqrt((diff ** 2).sum(-1))
            Dp = D[tri]
            a = int(np.argmin(Dp)); i, j = tri[0][a], tri[1][a]
            b = int(np.argmax(Dp)); k, l = tri[0][b], tri[1][b]
            Q = P.copy()
            u = P[i] - P[j]
            u = u / np.linalg.norm(u)
            Q[i] += step * u
            Q[j] -= step * u
            v = P[k] - P[l]
            v = v / np.linalg.norm(v)
            Q[k] -= step * v
            Q[l] += step * v
            r = ratio_of(Q)
            if r > best:
                best, P = r, Q
            else:
                step *= 0.7
        return P

    def make_obj(tau):
        def obj(x):
            P = x.reshape(n, d)
            D = pairwise(P)
            dmin = D.min()
            smin = dmin - tau * np.log(np.sum(np.exp(-(D - dmin) / tau)))
            dmax = D.max()
            pen = 500.0 * max(0.0, dmax - 1.0) ** 2
            return -smin + pen
        return obj

    taus = [0.08, 0.04, 0.02, 0.01, 0.005, 0.002, 0.001]

    def refine(x):
        x = np.asarray(x, dtype=float).ravel()
        if x.size != n * d:
            return None
        for tau in taus:
            try:
                res = minimize(make_obj(tau), x, method='L-BFGS-B',
                               options={'maxiter': 200})
            except Exception:
                break
            if res.x is None or np.asarray(res.x).size != n * d:
                break
            x = np.asarray(res.x, dtype=float).ravel()
        return x.reshape(n, d) if np.size(x) == n * d else None

    best_pts = None
    best_r = -1.0

    def consider(x):
        nonlocal best_pts, best_r
        if x is None:
            return
        x = np.asarray(x, dtype=float).reshape(n, d)
        x = push_polish(x)
        x = direct_polish(x)
        D = pairwise(x)
        if D.max() <= 0:
            return
        r = ratio_of(x)
        if r > best_r:
            best_r, best_pts = r, x.copy()

    # structured start: icosa vertices (12) + 2 antipodal extra points
    p = (1 + np.sqrt(5)) / 2
    icosa = np.array([
        [-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0],
        [0, 0, -1], [0, 0, 1],
        [-p, p, 0], [p, -p, 0], [p, p, 0], [-p, -p, 0],
        [0, -p, p], [0, p, -p],
    ], dtype=float) / np.sqrt(1 + p * p)
    icosa *= 0.4
    for extra in (np.array([[0.0, 0.45, 0.0], [0.0, -0.45, 0.0]]),
                  np.array([[0.45, 0.0, 0.0], [-0.45, 0.0, 0.0]])):
        seed = np.vstack([icosa, extra])
        for _ in range(3):
            x = seed + 0.02 * rng.normal(size=seed.shape)
            consider(refine(x))

    # random multi-start on the sphere
    for _ in range(35):
        v = rng.normal(size=(n, d))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        r0 = 0.4 + 0.1 * rng.random()
        x = r0 * v
        consider(refine(x))

    # perturbation polish around the incumbent
    if best_pts is not None:
        for scale in (0.05, 0.02, 0.01):
            for _ in range(10):
                x = best_pts + scale * rng.normal(size=best_pts.shape)
                consider(refine(x))

    # extended greedy polish of the incumbent with perturbation escapes
    if best_pts is not None:
        for scale in (0.01, 0.005, 0.002):
            for _ in range(6):
                cand = best_pts + scale * rng.normal(size=best_pts.shape)
                consider(push_polish(cand, iters=1500, step=0.01))
        consider(push_polish(best_pts, iters=4000, step=0.005))
        # final exact constrained polish rounds on the incumbent
        for _ in range(3):
            cand = best_pts + 0.002 * rng.normal(size=best_pts.shape)
            consider(direct_polish(cand, rounds=12))
        consider(direct_polish(best_pts, rounds=20))

    if best_pts is None:
        # absolute fallback: random points
        np.random.seed(0)
        best_pts = np.random.randn(n, d)
    else:
        best_pts = direct_polish(best_pts, rounds=15)

    D = pairwise(best_pts)
    best_pts = best_pts / D.max()
    return np.asarray(best_pts, dtype=float).reshape(n, d)


# EVOLVE-BLOCK-END
