# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def min_max_dist_dim3_14():
    N, D = 14, 3
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    npair = len(pairs)
    rng = np.random.default_rng(20240514)

    def sq_dist_matrix(P):
        d = P[:, None, :] - P[None, :, :]
        return np.einsum('ijk,ijk->ij', d, d)

    def exact_stats(P):
        M = sq_dist_matrix(P)
        iu = np.triu_indices(N, 1)
        vals = M[iu]
        return vals.min(), vals.max()

    def ratio_sq(P):
        dmin, dmax = exact_stats(P)
        if dmax <= 0:
            return 0.0
        return dmin / dmax

    def rescale_unit_diam(P):
        _, dmax = exact_stats(P)
        if dmax <= 0:
            return P
        return P / dmax

    def solve_epigraph(P0, rounds=3):
        """Maximize s s.t. d_ij^2 >= s, d_ij^2 <= 1, with rescale-refit rounds."""
        P = rescale_unit_diam(np.asarray(P0, dtype=float))
        best_P = P.copy()
        best_r = ratio_sq(P)
        for _ in range(rounds):
            M = sq_dist_matrix(P)
            dmin2 = min(M[i, j] for i, j in pairs)
            s0 = max(dmin2, 1e-6)
            z = np.concatenate([P.ravel(), [s0]])

            def obj(z):
                return -z[-1]

            def obj_grad(z):
                g = np.zeros_like(z)
                g[-1] = -1.0
                return g

            cons = []
            # d_ij^2 - s >= 0
            for (i, j) in pairs:
                def make_c(i, j):
                    def c(z):
                        Pm = z[:N * D].reshape(N, D)
                        v = Pm[i] - Pm[j]
                        return float(v @ v - z[-1])
                    def gc(z):
                        g = np.zeros(N * D + 1)
                        Pm = z[:N * D].reshape(N, D)
                        v = Pm[i] - Pm[j]
                        g[i * D:i * D + D] = 2 * v
                        g[j * D:j * D + D] = -2 * v
                        g[-1] = -1.0
                        return g
                    return {'type': 'ineq', 'fun': c, 'jac': gc}
                cons.append(make_c(i, j))
            # 1 - d_ij^2 >= 0
            for (i, j) in pairs:
                def make_d(i, j):
                    def c(z):
                        Pm = z[:N * D].reshape(N, D)
                        v = Pm[i] - Pm[j]
                        return float(1.0 - v @ v)
                    def gc(z):
                        g = np.zeros(N * D + 1)
                        Pm = z[:N * D].reshape(N, D)
                        v = Pm[i] - Pm[j]
                        g[i * D:i * D + D] = -2 * v
                        g[j * D:j * D + D] = 2 * v
                        return g
                    return {'type': 'ineq', 'fun': c, 'jac': gc}
                cons.append(make_d(i, j))

            try:
                res = minimize(obj, z, jac=obj_grad, constraints=cons,
                               method='SLSQP',
                               options={'maxiter': 250, 'ftol': 1e-12})
                Pn = res.x[:N * D].reshape(N, D)
            except Exception:
                Pn = P
            if not np.all(np.isfinite(Pn)):
                Pn = P
            Pn = rescale_unit_diam(Pn)
            r = ratio_sq(Pn)
            if r > best_r:
                best_r = r
                best_P = Pn.copy()
            P = Pn
        return best_P, best_r

    def fibonacci_sphere(n, jitter=0.0):
        k = np.arange(n) + 0.5
        phi = np.arccos(1 - 2 * k / n)
        theta = np.pi * (1 + 5 ** 0.5) * k
        P = np.stack([np.cos(theta) * np.sin(phi),
                      np.sin(theta) * np.sin(phi),
                      np.cos(phi)], axis=1)
        if jitter > 0:
            P = P + jitter * rng.standard_normal((n, 3))
        return P

    def icosahedron_plus():
        t = (1 + 5 ** 0.5) / 2
        V = np.array([
            [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
            [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
            [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
        ], dtype=float)
        V /= np.linalg.norm(V[0])
        extra = fibonacci_sphere(2) * 0.9
        return np.vstack([V, extra])

    def two_rings():
        pts = [np.array([0, 0, 1.0]), np.array([0, 0, -1.0])]
        for zc, r in ((0.5, 0.55), (-0.5, 0.55)):
            for k in range(6):
                th = 2 * np.pi * k / 6 + (0.3 if zc < 0 else 0.0)
                pts.append([r * np.cos(th), r * np.sin(th), zc])
        return np.array(pts)

    def rings_staggered():
        pts = []
        for zc, r, off in ((0.6, 0.6, 0.0), (-0.6, 0.6, np.pi / 5)):
            for k in range(5):
                th = 2 * np.pi * k / 5 + off
                pts.append([r * np.cos(th), r * np.sin(th), zc])
        for zc in (1.2, -1.2, 0.0):
            pts.append([0, 0, zc] if zc != 0 else [0.0, 0.0, 0.0])
        P = np.array(pts, dtype=float)
        while len(P) < N:
            P = np.vstack([P, fibonacci_sphere(1)[0]])
        return P[:N]

    starts = []
    starts.append(fibonacci_sphere(N))
    starts.append(fibonacci_sphere(N, jitter=0.05))
    starts.append(icosahedron_plus())
    starts.append(two_rings())
    starts.append(rings_staggered())
    starts.append(rng.standard_normal((N, 3)) * 0.5)
    starts.append(rng.uniform(-0.5, 0.5, (N, 3)))
    # random perturbations of the structured starts
    for base in (fibonacci_sphere(N), icosahedron_plus(), two_rings()):
        for _ in range(4):
            starts.append(base + 0.1 * rng.standard_normal((N, 3)))
    for _ in range(10):
        starts.append(0.4 * rng.standard_normal((N, 3)))

    best_P = None
    best_r = -1.0
    for P0 in starts:
        P, r = solve_epigraph(P0)
        if r > best_r:
            best_r = r
            best_P = P.copy()

    best_P = rescale_unit_diam(best_P)
    # tiny polish pass on the overall best
    P, r = solve_epigraph(best_P + 0.01 * rng.standard_normal((N, 3)), rounds=2)
    if r > best_r:
        best_P = rescale_unit_diam(P)
    return np.ascontiguousarray(best_P, dtype=float)
# EVOLVE-BLOCK-END