# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    n, d = 14, 3
    from scipy.optimize import minimize

    iu = np.triu_indices(n, 1)
    rng = np.random.default_rng(12345)

    # ---------- geometry utilities ----------
    def pdist_sq(pts):
        diff = pts[:, None, :] - pts[None, :, :]
        return np.sum(diff * diff, axis=-1)

    def score_sq(pts):
        u = pdist_sq(pts)[iu]
        mx = u.max()
        if mx <= 1e-15:
            return 0.0
        return u.min() / mx

    def sphere(p):
        p = p - p.mean(axis=0)
        r = np.linalg.norm(p, axis=1, keepdims=True)
        r[r == 0] = 1.0
        return p / r

    # ---------- seed bank (diversified families) ----------
    def build_seeds():
        seeds = []
        phi = (1 + np.sqrt(5)) / 2
        ico = sphere(np.array([
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
        ], dtype=float))
        cubo = sphere(np.array([
            [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
            [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
            [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1],
        ], dtype=float))

        # family A: icosahedron + 2 poles, swept pole radius
        for rp in (0.5, 0.8, 1.0, 1.3, 1.7, 2.2, 3.0):
            for ax in range(3):
                e = np.zeros((2, 3))
                e[0, ax] = rp
                e[1, ax] = -rp
                seeds.append(np.vstack([ico, e]))

        # family B: cuboctahedron + 2 poles, swept pole radius
        for rp in (0.5, 0.8, 1.0, 1.3, 1.7, 2.2, 3.0):
            for ax in range(3):
                e = np.zeros((2, 3))
                e[0, ax] = rp
                e[1, ax] = -rp
                seeds.append(np.vstack([cubo, e]))

        # family C: heptagonal antiprisms (a classic 14-point motif)
        for h in (0.3, 0.6, 0.9, 1.2, 1.6):
            for r in (0.6, 1.0, 1.4):
                t = np.arange(7) * 2 * np.pi / 7
                top = np.stack([r * np.cos(t), r * np.sin(t),
                                np.full(7, h)], axis=1)
                bot = np.stack([r * np.cos(t + np.pi / 7),
                                r * np.sin(t + np.pi / 7),
                                np.full(7, -h)], axis=1)
                seeds.append(np.vstack([top, bot]))

        # family D: fcc cluster (center + 12 shell) + 1 extra point
        fcc = np.vstack([[[0, 0, 0]], cubo])
        for extra in ([2, 0, 0], [0, 0, 2], [1, 1, 1], [0, 0, 0.5]):
            seeds.append(np.vstack([fcc, [np.array(extra, dtype=float)]]))

        # family E: fibonacci sphere, several twists
        k = np.arange(n)
        for tw in (0, 1, 2):
            ga = np.pi * (3 - np.sqrt(5)) + tw * 0.05
            z = 1 - 2 * (k + 0.5) / n
            r = np.sqrt(np.maximum(0.0, 1 - z * z))
            seeds.append(np.stack([r * np.cos(ga * k), r * np.sin(ga * k),
                                   z], axis=1))

        # family F: random clouds + antipodal clouds (free radii)
        for _ in range(16):
            seeds.append(sphere(rng.standard_normal((n, d))))
        for _ in range(6):
            dirs = sphere(rng.standard_normal((n // 2, d)))
            rad = rng.uniform(0.4, 1.6, size=(n // 2, 1))
            seeds.append(np.vstack([dirs * rad, -dirs * rad]))

        # family G: cube vertices + center + 5 near-center randoms
        lat = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1)
                        for z in (-1, 1)], dtype=float)
        for sc in (0.3, 0.6, 1.0):
            extra5 = sphere(rng.standard_normal((5, d))) * sc
            seeds.append(np.vstack([lat, [[0, 0, 0]], extra5]))
        return seeds

    # ---------- stage 1: Adam on smoothed ratio ----------
    def smooth_ratio_grad(pts, tau):
        D2 = pdist_sq(pts)
        u = D2[iu]
        smax = float(u.max())
        if smax <= 1e-12:
            return 0.0, np.zeros_like(pts)
        mn = smax - tau * np.log(np.sum(np.exp(-(u - smax) / tau)))
        mx = smax + tau * np.log(np.sum(np.exp((u - smax) / tau)))
        if mx < 1e-12:
            return 0.0, np.zeros_like(pts)
        w = np.exp(-(u - smax) / tau)
        w /= w.sum()
        v = np.exp((u - smax) / tau)
        v /= v.sum()
        dhdu = (w * mx - mn * v) / (mx * mx)
        S = np.zeros((n, n))
        S[iu] = dhdu
        S = S + S.T
        grad = 2.0 * np.einsum('ij,ijk->ik', S,
                               pts[:, None, :] - pts[None, :, :])
        grad -= grad.mean(axis=0, keepdims=True)
        return mn / mx, grad

    def adam_smooth(pts, iters=300, lr=0.02, tau0=0.3, tau1=0.04):
        m = np.zeros_like(pts)
        v = np.zeros_like(pts)
        b1, b2, eps = 0.9, 0.999, 1e-8
        best_pts = pts.copy()
        best_s = score_sq(pts)
        for it in range(iters):
            tau = tau0 * (tau1 / tau0) ** (it / max(1, iters - 1))
            _, g = smooth_ratio_grad(pts, tau)
            gn = np.linalg.norm(g)
            if gn > 1e-12:
                g = g / gn
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g * g
            mh = m / (1 - b1 ** (it + 1))
            vh = v / (1 - b2 ** (it + 1))
            pts = pts + lr * mh / (np.sqrt(vh) + eps)
            s = score_sq(pts)
            if s > best_s:
                best_s = s
                best_pts = pts.copy()
        return best_pts, best_s

    # ---------- stage 2: SLSQP on exact objective ----------
    def make_constraints():
        cons = []
        for i in range(n):
            for j in range(i + 1, n):
                def fmin(z, i=i, j=j):
                    return np.sum((z[3*i:3*i+3] - z[3*j:3*j+3])**2) - z[-1]
                def gmin(z, i=i, j=j):
                    g = np.zeros(len(z))
                    di = 2.0 * (z[3*i:3*i+3] - z[3*j:3*j+3])
                    g[3*i:3*i+3] = di
                    g[3*j:3*j+3] = -di
                    g[-1] = -1.0
                    return g
                def fmax(z, i=i, j=j):
                    return 1.0 - np.sum((z[3*i:3*i+3] - z[3*j:3*j+3])**2)
                def gmax(z, i=i, j=j):
                    g = np.zeros(len(z))
                    di = 2.0 * (z[3*i:3*i+3] - z[3*j:3*j+3])
                    g[3*i:3*i+3] = -di
                    g[3*j:3*j+3] = di
                    return g
                cons.append({'type': 'ineq', 'fun': fmin, 'jac': gmin})
                cons.append({'type': 'ineq', 'fun': fmax, 'jac': gmax})
        return cons

    constraints = make_constraints()

    def obj(z):
        return -z[-1]

    def obj_grad(z):
        g = np.zeros_like(z)
        g[-1] = -1.0
        return g

    def slsqp(pts):
        mx = np.sqrt(pdist_sq(pts).max())
        if mx <= 0:
            return None, 0.0
        p = pts / mx
        t0 = pdist_sq(p)[iu].min()
        z0 = np.concatenate([p.ravel(), [t0]])
        try:
            res = minimize(obj, z0, jac=obj_grad, constraints=constraints,
                           method='SLSQP',
                           options={'maxiter': 300, 'ftol': 1e-12})
            p2 = res.x[:-1].reshape(n, d)
            return p2, score_sq(p2)
        except Exception:
            return None, 0.0

    def normalize(pts):
        mx = np.sqrt(max(pdist_sq(pts).max(), 1e-15))
        return pts / mx

    # ---------- pipeline ----------
    seeds = build_seeds()
    results = []
    for sd in seeds:
        p, s = adam_smooth(sd.copy())
        p = normalize(p)
        results.append((score_sq(p), p))
    results.sort(key=lambda c: -c[0])

    best_pts, best_s = results[0]
    for s0, p0 in results[:12]:
        if s0 > best_s:
            best_s, best_pts = s0, p0.copy()
        p2, s2 = slsqp(p0)
        if p2 is not None and s2 > best_s:
            best_s, best_pts = s2.copy(), s2

    # ---------- basin hopping around incumbent ----------
    for hop in range(8):
        scale = 0.15 * (0.7 ** hop)
        cand = best_pts + rng.standard_normal(best_pts.shape) * scale
        cand = normalize(cand)
        p, s = adam_smooth(cand, iters=200, lr=0.01, tau0=0.08, tau1=0.02)
        p = normalize(p)
        s = score_sq(p)
        if s > best_s:
            best_s, best_pts = s, p.copy()
        p2, s2 = slsqp(p)
        if p2 is not None and s2 > best_s:
            best_s, best_pts = p2.copy(), s2
        # local fine polish
        p3, s3 = adam_smooth(best_pts.copy(), iters=150, lr=0.005,
                             tau0=0.03, tau1=0.01)
        p3 = normalize(p3)
        s3 = score_sq(p3)
        if s3 > best_s:
            best_s, best_pts = s3.copy(), s3
        p4, s4 = slsqp(p3)
        if p4 is not None and s4 > best_s:
            best_s, best_pts = p4.copy(), s4

    best_pts = normalize(best_pts)
    return np.asarray(best_pts, dtype=float)


# EVOLVE-BLOCK-END