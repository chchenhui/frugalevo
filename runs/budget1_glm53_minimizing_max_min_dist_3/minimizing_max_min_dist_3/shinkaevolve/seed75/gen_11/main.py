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
    rng = np.random.default_rng(12345)
    iu = np.triu_indices(n, 1)
    i0, i1 = iu

    def ratio(P):
        Dp = np.linalg.norm(P[:, None] - P[None, :], axis=-1)[iu]
        return Dp.min() / Dp.max()

    def soft_refine(P, iters, lr, tau0):
        for it in range(iters):
            Diff = P[:, None] - P[None, :]
            D = np.linalg.norm(Diff, axis=-1) + 1e-12
            Dp = D[iu]
            tau = max(0.008, tau0 * (0.99 ** it))
            w = np.exp(-(Dp - Dp.min()) / tau)
            w /= w.sum()
            wM = np.exp((Dp - Dp.max()) / tau)
            wM /= wM.sum()
            Wm = np.zeros((n, n))
            WM = np.zeros((n, n))
            Wm[iu] = w
            Wm.T[iu] = w
            WM[iu] = wM
            WM.T[iu] = wM
            U = Diff / D[:, :, None]
            G = (Wm[:, :, None] * U).sum(axis=1) - 0.5 * (WM[:, :, None] * U).sum(axis=1)
            gn = np.linalg.norm(G)
            if gn > 1e-12:
                P = P + lr * np.sqrt(n) * G / gn
                P /= np.linalg.norm(P, axis=1, keepdims=True)
        return P

    def hard_polish(P, iters, step):
        # exact subgradient moves on the current min / max pairs
        for it in range(iters):
            D = np.linalg.norm(P[:, None] - P[None, :], axis=-1) + 1e-12
            Dp = D[iu]
            ka = int(np.argmin(Dp))
            kb = int(np.argmax(Dp))
            a, b = i0[ka], i1[ka]
            u = (P[b] - P[a]) / D[a, b]
            P[a] -= step * u
            P[b] += step * u
            c, e = i0[kb], i1[kb]
            u = (P[e] - P[c]) / D[c, e]
            P[c] += 0.5 * step * u
            P[e] -= 0.5 * step * u
            P /= np.linalg.norm(P, axis=1, keepdims=True)
            step *= 0.995
        return P

    # candidate families: staggered parallel rings on the unit sphere
    inits = []
    for (k1, k2) in [(7, 7), (6, 8), (8, 6), (5, 9)]:
        for h in (0.25, 0.45, 0.65, 0.85, 1.05):
            for tw in np.linspace(0.0, np.pi / max(k1, k2), 4, endpoint=False):
                r1 = np.sqrt(max(1e-6, 1 - h * h))
                t1 = 2 * np.pi * np.arange(k1) / k1
                ring1 = np.stack([r1 * np.cos(t1), r1 * np.sin(t1), np.full(k1, -h)], axis=1)
                t2 = 2 * np.pi * np.arange(k2) / k2 + tw
                ring2 = np.stack([r1 * np.cos(t2), r1 * np.sin(t2), np.full(k2, h)], axis=1)
                inits.append(np.vstack([ring1, ring2]))
    # ring + two poles families
    for k in (10, 11, 12):
        t = 2 * np.pi * np.arange(k) / k
        ring = np.stack([np.cos(t), np.sin(t), np.zeros(k)], axis=1)
        inits.append(np.vstack([ring, [[0, 0, 1.0], [0, 0, -1.0]]]))
    # icosahedron + 2 points
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ], dtype=float)
    ico /= np.linalg.norm(ico, axis=1, keepdims=True)
    inits.append(np.vstack([ico, [[0, 0, 1.2], [0, 0, -1.2]]]))
    # a few random starts
    for _ in range(8):
        inits.append(rng.normal(size=(n, d)))

    results = []
    for P0 in inits:
        P = np.asarray(P0, dtype=float).copy()
        P /= np.linalg.norm(P, axis=1, keepdims=True)
        P = soft_refine(P, 300, 0.08, 0.15)
        results.append((ratio(P), P))
    # a few random multi-start annealing runs
    for _ in range(6):
        P = rng.normal(size=(n, d))
        P /= np.linalg.norm(P, axis=1, keepdims=True)
        for it in range(200):
            P = P + 0.05 * rng.normal(size=P.shape) * (0.98 ** it)
            P /= np.linalg.norm(P, axis=1, keepdims=True)
            P = soft_refine(P, 30, 0.05, 0.1)
        results.append((ratio(P), P))

    results.sort(key=lambda x: -x[0])
    best_r, best = results[0]

    # polish the top candidates with exact min/max pair moves + low-temp soft refine
    for r0, P in results[:4]:
        Q = hard_polish(P.copy(), 250, 0.02)
        Q = soft_refine(Q, 250, 0.02, 0.03)
        Q = hard_polish(Q, 250, 0.01)
        rQ = ratio(Q)
        if rQ > best_r:
            best_r = rQ
            best = Q

    points = np.asarray(best, dtype=float)
    # sanity: ensure positive max distance
    if not np.isfinite(points).all():
        np.random.seed(42)
        points = np.random.randn(n, d)
    return points


# EVOLVE-BLOCK-END