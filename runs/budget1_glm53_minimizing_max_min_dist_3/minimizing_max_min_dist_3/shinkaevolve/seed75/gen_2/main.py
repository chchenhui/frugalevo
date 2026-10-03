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

    def ratio(P):
        Dp = np.linalg.norm(P[:, None] - P[None, :], axis=-1)[iu]
        return Dp.min() / Dp.max()

    best = None
    best_r = -1.0

    for trial in range(24):
        # random init on unit sphere
        P = rng.normal(size=(n, d))
        P /= np.linalg.norm(P, axis=1, keepdims=True)
        lr = 0.08
        for it in range(300):
            Diff = P[:, None] - P[None, :]
            D = np.linalg.norm(Diff, axis=-1) + 1e-12
            Dp = D[iu]
            tau = max(0.02, 0.15 * (0.99 ** it))
            # soft-min weights (maximize min distance)
            w = np.exp(-(Dp - Dp.min()) / tau)
            w /= w.sum()
            # soft-max weights (penalize diameter)
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
            P -= P.mean(axis=0)
            if it % 50 == 49:
                lr *= 0.7
        r = ratio(P)
        if r > best_r:
            best_r = r
            best = P.copy()

    points = np.asarray(best, dtype=float)
    # sanity: ensure positive max distance
    if not np.isfinite(points).all():
        np.random.seed(42)
        points = np.random.randn(n, d)
    return points


# EVOLVE-BLOCK-END