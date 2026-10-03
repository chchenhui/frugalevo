# EVOLVE-BLOCK-START
import numpy as np

try:
    from scipy.optimize import minimize
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


def min_max_dist_dim3_14() -> np.ndarray:
    n, d = 14, 3
    iu = np.triu_indices(n, 1)
    ii, jj = iu

    def true_score(P):
        dd = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1) + 1e-18)
        dists = dd[iu]
        dmax = dists.max()
        if dmax <= 0:
            return 0.0
        return (dists.min() / dmax) ** 2

    def project(P):
        return P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)

    # ---------- Phase 1: annealed soft-min ascent on the unit sphere ----------
    def sphere_ascent(P0, steps=350):
        P = project(P0.copy())
        T0, T1 = 0.15, 0.004
        for t in range(steps):
            frac = t / steps
            T = T0 * (T1 / T0) ** frac
            step = 0.08 * (1.0 - frac) + 5e-4
            diff = P[:, None, :] - P[None, :, :]
            dist = np.sqrt((diff ** 2).sum(-1) + 1e-18)
            dm = dist[iu]
            e = np.exp(-(dm - dm.min()) / T)
            w = np.zeros((n, n))
            w[ii, jj] = e
            w = w + w.T
            # force pushing the closest pairs apart
            F = (w[:, :, None] * diff / np.maximum(dist, 1e-12)[:, :, None]).sum(axis=1)
            # prevent collapse of the max distance: mild attraction for farthest pairs
            P = P + step * F / e.sum()
            P = project(P)
        return P

    # ---------- Phase 2: free-space smooth (dmin/dmax)^2 refinement ----------
    def smooth_obj(flat, beta):
        P = flat.reshape(n, 3)
        diff = P[:, None, :] - P[None, :, :]
        d2 = (diff ** 2).sum(-1)
        dij2 = d2[iu]
        # soft min of squared distances
        a = -beta * dij2
        am = a.max()
        smin = -(am + np.log(np.exp(a - am).sum())) / beta
        # soft max
        b = beta * dij2
        bm = b.max()
        smax = (bm + np.log(np.exp(b - bm).sum())) / beta
        F = smin / smax  # maximize
        obj = -F
        wmin = np.exp(a - am); wmin /= wmin.sum()
        wmax = np.exp(b - bm); wmax /= wmax.sum()
        g = (wmin / smax + (smin / smax ** 2) * wmax)  # d F / d dij2, sign-fixed below
        # dF/ddij2 = wmin/smax + smin*wmax/smax^2 ... careful signs:
        g = (wmin / smax) - (smin * wmax) / (smax ** 2)
        grad = np.zeros((n, 3))
        contrib = 2.0 * g[:, None] * diff[ii, jj]
        np.add.at(grad, ii, contrib)
        np.add.at(grad, jj, -contrib)
        return obj, grad.ravel()

    # ---------- structured + random seeds ----------
    seeds = []
    t = (1.0 + 5.0 ** 0.5) / 2.0
    ico = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], dtype=float)
    ico = project(ico)
    # icosahedron + 2 poles, several rotations
    for k in range(6):
        rng = np.random.RandomState(500 + k)
        Q, _ = np.linalg.qr(rng.randn(3, 3))
        base = np.vstack([ico, [[0, 0, 1.0], [0, 0, -1.0]]])
        seeds.append(project(base @ Q.T + 0.05 * rng.randn(n, 3)))
    # two staggered hex rings + poles (known good basin for 14)
    for zr, off in [(0.40, 0.0), (0.45, np.pi / 6), (0.35, np.pi / 6)]:
        a1 = np.arange(6) * (np.pi / 3) + off
        a2 = a1 + np.pi / 3
        r = np.sqrt(1 - zr ** 2)
        pts = [[0, 0, 1.0], [0, 0, -1.0]]
        for a in a1:
            pts.append([r * np.cos(a), r * np.sin(a), zr])
        for a in a2:
            pts.append([r * np.cos(a), r * np.sin(a), -zr])
        seeds.append(np.array(pts, dtype=float))
    # cube antipodal hints
    cube = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)
    seeds.append(project(np.vstack([cube[:7], -cube[:7]])))
    # random spherical seeds
    for s in range(28):
        rng = np.random.RandomState(1000 + s)
        v = rng.randn(n, 3)
        seeds.append(project(v))

    # ---------- run phase 1 for every seed ----------
    cands = []
    for P0 in seeds:
        P = sphere_ascent(P0)
        cands.append((true_score(P), P))
    cands.sort(key=lambda c: -c[0])

    best_s, best_P = cands[0]

    # ---------- phase 2 on top candidates ----------
    if HAVE_SCIPY:
        for s0, P0 in cands[:6]:
            P = P0.copy()
            for beta in (30.0, 100.0, 300.0):
                res = minimize(smooth_obj, P.ravel(), args=(beta,), jac=True,
                                method="L-BFGS-B",
                                options={"maxiter": 400, "maxfun": 1600})
                P = res.x.reshape(n, 3)
            s = true_score(P)
            if s > best_s:
                best_s, best_P = s, P

    # ---------- exact micro-polish: rotate bottleneck point away ----------
    def micro(P, rounds=6):
        P = P.copy()
        s = true_score(P)
        rng = np.random.RandomState(7)
        for _ in range(rounds):
            dd = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1) + 1e-18)
            dists = dd[iu]
            dmax = dists.max()
            k = np.argmin(dists)
            i, j = ii[k], jj[k]
            improved = False
            for amp in (0.02, 0.05, 0.1):
                u = P[i] - P[j]
                u = u / max(np.linalg.norm(u), 1e-12)
                for trial in range(8):
                    pert = amp * (u + 0.5 * rng.randn(3))
                    Q = P.copy()
                    Q[i] = Q[i] + pert
                    Q = Q - Q.mean(axis=0)
                    st = true_score(Q)
                    if st > s + 1e-9:
                        s, P, improved = st, Q, True
                        break
                if improved:
                    break
            if not improved:
                break
        return s, P

    if HAVE_SCIPY:
        s, P = micro(best_P)
        if s > best_s:
            best_s, best_P = s, P
        # one last exact-objective SLSQP pass
        def exact_obj(flat):
            P = flat.reshape(n, 3)
            return -true_score(P)
        res = minimize(exact_obj, best_P.ravel(), method="Nelder-Mead",
                       options={"maxiter": 2000, "fatol": 1e-10, "xatol": 1e-8})
        P = res.x.reshape(n, 3)
        s = true_score(P)
        if s > best_s:
            best_s, best_P = s, P

    best_P = best_P - best_P.mean(axis=0)
    sc = np.abs(best_P).max()
    if sc > 0:
        best_P = best_P / sc
    return np.asarray(best_P, dtype=float)
# EVOLVE-BLOCK-END