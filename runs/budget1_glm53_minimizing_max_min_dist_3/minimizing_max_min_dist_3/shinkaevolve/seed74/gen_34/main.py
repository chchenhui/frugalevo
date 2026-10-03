# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def min_max_dist_dim3_14() -> np.ndarray:
    n, d = 14, 3
    iu = np.triu_indices(n, 1)
    pi, pj = iu
    m = len(pi)
    rng = np.random.default_rng(42)

    def pdist_sq(pts):
        diff = pts[:, None, :] - pts[None, :, :]
        return np.sum(diff * diff, axis=-1)

    def score_sq(pts):
        u = pdist_sq(pts)[iu]
        mx = u.max()
        if mx <= 0:
            return 0.0
        return u.min() / mx

    def sphere(p):
        p = p - p.mean(axis=0)
        r = np.linalg.norm(p, axis=1, keepdims=True)
        r[r == 0] = 1.0
        return p / r

    # ---------- fast softmin spreader (crossover from partner program) ----------
    def softmin_grad(X, t):
        diff = X[:, None, :] - X[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-12)
        dd = dist[iu]
        w = np.exp(-(dd - dd.min()) / t)
        w /= w.sum()
        u = (X[pi] - X[pj]) / dd[:, None]
        G = np.zeros_like(X)
        contrib = w[:, None] * u
        np.add.at(G, pi, contrib)
        np.add.at(G, pj, -contrib)
        return float((w * dd).sum()), G

    def softmin_spread(X0, steps=200, lr=0.03, t0=0.3, t1=0.05):
        X = X0.copy()
        for k in range(steps):
            t = t0 + (t1 - t0) * k / steps
            _, G = softmin_grad(X, t)
            gn = np.linalg.norm(G)
            if gn < 1e-12:
                break
            X = X + lr * (t / 0.1) * (G / gn) * np.sqrt((X ** 2).sum() / n)
        return X

    # ---------------- seeds ----------------
    seeds = []
    for _ in range(18):
        seeds.append(sphere(rng.standard_normal((n, d))))
    for _ in range(4):
        dirs = sphere(rng.standard_normal((n // 2, d)))
        seeds.append(np.vstack([dirs, -dirs]))
    phi = (1 + np.sqrt(5)) / 2
    ico = sphere(np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ], dtype=float))
    for ax in range(3):
        e = np.zeros((2, 3)); e[0, ax] = 1.3; e[1, ax] = -1.3
        seeds.append(np.vstack([ico, e]))
    k = np.arange(n)
    ga = np.pi * (3 - np.sqrt(5))
    z = 1 - 2 * (k + 0.5) / n
    r = np.sqrt(np.maximum(0.0, 1 - z * z))
    fib = np.stack([r * np.cos(ga * k), r * np.sin(ga * k), z], axis=1)
    seeds.append(fib)
    lat = np.array([[x, y, zz] for x in (-1, 1) for y in (-1, 1)
                    for zz in (-1, 1)], dtype=float)
    extra5 = sphere(rng.standard_normal((5, d))) * 0.5
    seeds.append(sphere(np.vstack([lat, [[0, 0, 0]], extra5])))

    # -------- smooth (log-sum-exp) ratio with analytic gradient --------
    def smooth_ratio_grad(pts, tau):
        D2 = pdist_sq(pts)
        u = D2[iu]
        smax = float(u.max())
        if smax <= 1e-12:
            return 0.0, np.zeros_like(pts)
        mn = -tau * np.log(np.sum(np.exp(-(u - smax) / tau))) + smax
        mx = tau * np.log(np.sum(np.exp((u - smax) / tau))) + smax
        if mx < 1e-12:
            return 0.0, np.zeros_like(pts)
        w = np.exp(-(u - smax) / tau)
        w = w / w.sum()
        v = np.exp((u - smax) / tau)
        v = v / v.sum()
        dhdu = (w * mx - mn * v) / (mx * mx)
        S = np.zeros((n, n))
        S[iu] = dhdu
        S = S + S.T
        grad = 2.0 * np.einsum('ij,ijk->ik', S, pts[:, None, :] - pts[None, :, :])
        grad -= grad.mean(axis=0, keepdims=True)
        return mn / mx, grad

    def adam_smooth(pts, iters=350, lr=0.02, tau0=0.3, tau1=0.04):
        m_v = np.zeros_like(pts)
        v_v = np.zeros_like(pts)
        b1, b2, eps = 0.9, 0.999, 1e-8
        best_pts = pts.copy()
        best_s = score_sq(pts)
        for it in range(iters):
            tau = tau0 * (tau1 / tau0) ** (it / max(1, iters - 1))
            _, g = smooth_ratio_grad(pts, tau)
            gn = np.linalg.norm(g)
            if gn > 1e-12:
                g = g / gn
            m_v = b1 * m_v + (1 - b1) * g
            v_v = b2 * v_v + (1 - b2) * g * g
            mh = m_v / (1 - b1 ** (it + 1))
            vh = v_v / (1 - b2 ** (it + 1))
            pts = pts + lr * mh / (np.sqrt(vh) + eps)
            s = score_sq(pts)
            if s > best_s:
                best_s = s
                best_pts = pts.copy()
        return best_pts, best_s

    # ---------------- vectorized SLSQP refinement ----------------
    nz = 3 * n + 1

    def obj(z):
        return -z[-1]

    def obj_grad(z):
        g = np.zeros(nz)
        g[-1] = -1.0
        return g

    def cons_f(z):
        P = z[:3 * n].reshape(n, 3)
        u = pdist_sq(P)[iu]
        return np.concatenate([u - z[-1], 1.0 - u])

    def cons_j(z):
        P = z[:3 * n].reshape(n, 3)
        diff = P[pi] - P[pj]
        J = np.zeros((2 * m, nz))
        rows = np.arange(m)
        ci = 3 * pi[:, None] + np.arange(3)
        cj = 3 * pj[:, None] + np.arange(3)
        # min constraints: d2_ij - t >= 0
        J[rows[:, None], ci] = 2.0 * diff
        J[rows[:, None], cj] = -2.0 * diff
        J[:m, -1] = -1.0
        # max constraints: 1 - d2_ij >= 0
        J[m + rows[:, None], ci] = -2.0 * diff
        J[m + rows[:, None], cj] = 2.0 * diff
        return J

    cons = [{'type': 'ineq', 'fun': cons_f, 'jac': cons_j}]

    def slsqp(pts):
        try:
            mx = np.sqrt(max(pdist_sq(pts).max(), 1e-12))
            p = sphere(pts / mx)
            u = pdist_sq(p)[iu]
            z0 = np.concatenate([p.ravel(), [u.min()]])
            res = minimize(obj, z0, jac=obj_grad, constraints=cons,
                           method='SLSQP', options={'maxiter': 400, 'ftol': 1e-14})
            p2 = res.x[:3 * n].reshape(n, 3)
            if not np.all(np.isfinite(p2)):
                return None, 0.0
            return p2, score_sq(p2)
        except Exception:
            return None, 0.0

    # ---------------- stage 1: Adam on all seeds ----------------
    trials = []
    for sd in seeds:
        p, s = adam_smooth(sd.copy())
        trials.append((s, p))
    # crossover: softmin-preconditioned random starts
    for _ in range(8):
        Xp = softmin_spread(rng.standard_normal((n, d)))
        p, s = adam_smooth(sphere(Xp.copy()))
        trials.append((s, p))
        p2, s2 = adam_smooth(Xp.copy(), iters=250)
        if s2 > 0.0:
            trials.append((s2, p2))
    trials.sort(key=lambda c: -c[0])
    if not trials:
        trials = [(0.0, np.eye(n, d))]

    # ---------------- stage 2: SLSQP on top candidates ----------------
    best_pts = None
    best_s = -1.0
    for s0, p0 in trials[:10]:
        p0 = p0 / np.sqrt(max(pdist_sq(p0).max(), 1e-12))
        s = score_sq(p0)
        if s > best_s:
            best_s, best_pts = s, p0.copy()
        p2, s2 = slsqp(p0)
        if p2 is not None and s2 > best_s:
            best_s, best_pts = s2, p2.copy()
    if best_pts is None:
        best_pts, best_s = trials[0][1].copy(), trials[0][0]

    # ---------------- stage 3: fine polish + perturbation restarts ----------------
    for _ in range(3):
        p, s = adam_smooth(best_pts.copy(), iters=250, lr=0.008,
                           tau0=0.05, tau1=0.015)
        if s > best_s:
            best_s, best_pts = s, p.copy()
        p2, s2 = slsqp(p)
        if p2 is not None and s2 > best_s:
            best_s, best_pts = s2, p2.copy()
        p3, s3 = slsqp(best_pts.copy())
        if p3 is not None and s3 > best_s:
            best_s, best_pts = s3, p3.copy()

    for noise in (0.02, 0.05, 0.10):
        for _ in range(3):
            cand = best_pts + noise * rng.standard_normal(best_pts.shape)
            p2, s2 = slsqp(sphere(cand))
            if p2 is not None and s2 > best_s:
                best_s, best_pts = s2, p2.copy()

    # normalize: diameter = 1
    best_pts = sphere(best_pts)
    D2 = pdist_sq(best_pts)
    mx = float(D2.max())
    if mx > 0:
        best_pts = best_pts / np.sqrt(mx)
    return np.asarray(best_pts, dtype=float)


# EVOLVE-BLOCK-END