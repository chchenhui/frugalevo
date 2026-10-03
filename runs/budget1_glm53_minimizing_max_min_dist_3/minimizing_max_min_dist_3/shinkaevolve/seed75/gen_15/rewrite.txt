# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the coordinates of the 14 points.
    """

    n = 14
    d = 3
    rng = np.random.default_rng(12345)
    iu = np.triu_indices(n, 1)

    def ratio(P):
        Dp = np.linalg.norm(P[:, None] - P[None, :], axis=-1)[iu]
        return Dp.min() / Dp.max()

    def d6_points(p, h, r):
        t = 2 * np.pi * np.arange(6) / 6.0
        ring1 = np.stack([r * np.cos(t), r * np.sin(t), np.full(6, h)], axis=1)
        ring2 = np.stack([r * np.cos(t + np.pi / 6), r * np.sin(t + np.pi / 6),
                          np.full(6, -h)], axis=1)
        poles = np.array([[0.0, 0.0, p], [0.0, 0.0, -p]])
        return np.vstack([ring1, ring2, poles])

    def d6_ratio(p, h, r):
        # closed-form pairwise distances for the D6 configuration
        d_adj = r                                # adjacent in ring
        d_cross = np.sqrt(r * r * (2 - np.sqrt(3)) + 4 * h * h)  # rings, min twist
        d_pt = np.sqrt(r * r + (p - h) ** 2)     # pole to near ring
        d_pb = np.sqrt(r * r + (p + h) ** 2)     # pole to far ring
        d_pp = 2 * p                             # pole-pole
        dmin = min(d_adj, d_cross, d_pt, d_pb)
        dmax = max(d_pp, d_pb, 2 * r, r * np.sqrt(3))
        return dmin / max(dmax, 1e-12)

    # ---- coarse grid then local refinement over (p, h, r) ----
    best_par, best_val = None, -1.0
    for p in np.linspace(0.4, 1.6, 40):
        for h in np.linspace(0.1, 1.2, 30):
            for r in np.linspace(0.3, 1.5, 30):
                v = d6_ratio(p, h, r)
                if v > best_val:
                    best_val, best_par = v, (p, h, r)
    # local random-refine around best
    par = np.array(best_par)
    step = 0.02
    for it in range(4000):
        cand = par + step * rng.normal(size=3)
        v = d6_ratio(*cand)
        if v > best_val:
            best_val, par = v, cand
        if it % 1000 == 999:
            step *= 0.5
    p_opt, h_opt, r_opt = par
    d6_seed = d6_points(p_opt, h_opt, r_opt)

    # ---- soft-min / soft-max projected gradient polish ----
    def polish(P, lr, iters, tau0, tau_min):
        P = P / np.linalg.norm(P, axis=1, keepdims=True)
        for it in range(iters):
            Diff = P[:, None] - P[None, :]
            D = np.linalg.norm(Diff, axis=-1) + 1e-12
            Dp = D[iu]
            tau = max(tau_min, tau0 * (0.995 ** it))
            w = np.exp(-(Dp - Dp.min()) / tau)
            w /= w.sum()
            wM = np.exp((Dp - Dp.max()) / tau)
            wM /= wM.sum()
            Wm = np.zeros((n, n)); WM = np.zeros((n, n))
            Wm[iu] = w; Wm.T[iu] = w
            WM[iu] = wM; WM.T[iu] = wM
            U = Diff / D[:, :, None]
            G = (Wm[:, :, None] * U).sum(axis=1) - 0.5 * (WM[:, :, None] * U).sum(axis=1)
            gn = np.linalg.norm(G)
            if gn > 1e-12:
                P = P + lr * np.sqrt(n) * G / gn
            P /= np.linalg.norm(P, axis=1, keepdims=True)
        return P

    best = None
    best_r = -1.0

    seeds = [d6_seed]
    # jittered variants of the D6 seed (small symmetry-breaking perturbations)
    for _ in range(6):
        Q = d6_seed + 0.02 * rng.normal(size=d6_seed.shape)
        Q /= np.linalg.norm(Q, axis=1, keepdims=True)
        seeds.append(Q)
    # icosahedron + 2 poles as fallback trial
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ], dtype=float)
    ico /= np.linalg.norm(ico, axis=1, keepdims=True)
    seeds.append(np.vstack([ico, [[0, 0, 1.2], [0, 0, -1.2]]]))

    for idx, S in enumerate(seeds):
        P = polish(S.astype(float).copy(), 0.06, 400, 0.08, 0.01)
        r = ratio(P)
        if r > best_r:
            best_r, best = r, P.copy()

    # perturb-and-polish rounds around the best
    for round_i in range(10):
        mag = 0.06 * (0.7 ** round_i)
        Q = best + mag * rng.normal(size=best.shape)
        Q /= np.linalg.norm(Q, axis=1, keepdims=True)
        Q = polish(Q, 0.03, 300, 0.04, 0.004)
        if ratio(Q) > best_r:
            best_r = ratio(Q)
            best = Q.copy()
    best = polish(best, 0.02, 300, 0.02, 0.002)

    points = np.asarray(best, dtype=float)
    if not np.isfinite(points).all() or points.std() == 0:
        np.random.seed(42)
        points = np.random.randn(n, d)
    return points


# EVOLVE-BLOCK-END