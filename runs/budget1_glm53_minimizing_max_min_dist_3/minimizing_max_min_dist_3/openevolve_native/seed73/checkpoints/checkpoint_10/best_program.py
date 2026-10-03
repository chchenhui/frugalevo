# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Construct 14 points in 3D maximizing dmin/dmax.

    Approach: multi-start annealed ascent on a smooth surrogate of the
    TRUE objective log(dmin) - log(dmax):
        f(X) = 0.5*log(softmin d^2) - 0.5*log(softmax d^2),
    with softmin/softmax weights W^-_ij = exp(-(d_ij^2 - min)/tau) and
    W^+_ij = exp((d_ij^2 - max)/tau). Its gradient REPELS the closest
    pairs (growing dmin) and ATTRACTS the farthest pairs (shrinking
    dmax), directly targeting the ratio instead of only dmin. tau is
    annealed downward; the objective is scale invariant so each step
    renormalizes (unit sphere stage or RMS normalization for the free
    stage, since the optimum is not spherical).

    Restarts mix Gaussian free starts, spherical Tammes-like starts,
    icosahedron+2 starts and staggered two-ring (antiprism) starts.
    The best configuration is then refined by "kick-and-reoptimize":
    random perturbations of decreasing magnitude, each followed by a
    low-tau ascent, kept only if the true ratio improves. Returned
    scaled so that dmax = 1. Fixed seeds throughout.
    """
    n = 14
    rng = np.random.default_rng(12345)
    iu = np.triu_indices(n, 1)

    def sq_dists(X):
        return np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)

    def ratio_of(X):
        D2 = sq_dists(X)
        dm = float(np.sqrt(D2[iu].min()))
        dM = float(np.sqrt(D2[iu].max()))
        return dm / dM if dM > 0 else 0.0

    def rms_norm(X):
        r = np.sqrt(np.mean(np.sum(X * X, axis=1)))
        return X / r if (r > 0 and np.isfinite(r)) else X

    def softmin_ascent(X, iters, tau0, tau1, lr, sphere):
        """Ascent on softmin of squared distances (grows dmin)."""
        tau = tau0
        decay = (tau1 / tau0) ** (1.0 / iters)
        for _ in range(iters):
            D2 = sq_dists(X)
            np.fill_diagonal(D2, np.inf)
            m = D2.min()
            W = np.exp(-(D2 - m) / tau)
            S = W.sum()
            if not np.isfinite(S) or S <= 0:
                break
            grad = 2.0 * (W.sum(axis=1)[:, None] * X - W @ X) / S
            X = X + lr * grad
            if sphere:
                X = X / np.linalg.norm(X, axis=1, keepdims=True)
            else:
                X = rms_norm(X)
            tau *= decay
        return X

    def ratio_ascent(X, iters, tau0, tau1, lr):
        """Ascent on 0.5*log(softmin d^2) - 0.5*log(softmax d^2).

        Gradient wrt x_i: (1/m_soft) * sum_j W^-_ij (x_i - x_j)
                          - (1/M_soft) * sum_j W^+_ij (x_i - x_j)
        (up to the common factor 1/2 absorbed into lr). Close pairs
        repel; far pairs attract, shrinking the diameter.
        """
        tau = tau0
        decay = (tau1 / tau0) ** (1.0 / iters)
        for _ in range(iters):
            D2 = sq_dists(X)
            np.fill_diagonal(D2, np.inf)
            m = D2.min()
            M = D2.max()
            Wn = np.exp(-(D2 - m) / tau)
            Wx = np.exp((D2 - M) / tau)
            Sn = Wn.sum()
            Sx = Wx.sum()
            if not (np.isfinite(Sn) and np.isfinite(Sx) and Sn > 0 and Sx > 0):
                break
            Gn = (Wn.sum(axis=1)[:, None] * X - Wn @ X) / Sn
            Gx = (Wx.sum(axis=1)[:, None] * X - Wx @ X) / Sx
            X = X + lr * (Gn / m - Gx / M)
            X = rms_norm(X)
            tau *= decay
        return X

    # Icosahedron vertices (unit sphere) for structured starts.
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ], dtype=float)
    ico /= np.linalg.norm(ico, axis=1, keepdims=True)

    best_X = None
    best_r = -1.0

    def consider(cand):
        nonlocal best_r, best_X
        if cand is not None and np.all(np.isfinite(cand)):
            r = ratio_of(cand)
            if r > best_r:
                best_r = r
                best_X = cand.copy()

    # Staggered two-ring (antiprism) structured start.
    def ring_start(rng_):
        thetas = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        ring1 = np.stack([np.cos(thetas), np.sin(thetas),
                          np.full(7, 0.6)], axis=1)
        thetas2 = thetas + np.pi / 7
        ring2 = np.stack([np.cos(thetas2), np.sin(thetas2),
                          np.full(7, -0.6)], axis=1)
        return np.vstack([ring1, ring2]) + 0.05 * rng_.standard_normal((14, 3))

    # ---- Multi-start search: diverse starts, two-stage refinement. ----
    for trial in range(400):
        mode = trial % 5
        if mode == 0:
            X = rms_norm(rng.standard_normal((n, 3)))
        elif mode == 4:
            extra = rng.standard_normal((2, 3)) * 0.4
            extra /= np.linalg.norm(extra, axis=1, keepdims=True)
            X = np.vstack([ico, extra])
            X = softmin_ascent(X, iters=300, tau0=0.20, tau1=0.01,
                               lr=0.03, sphere=True)
        else:
            X = rng.standard_normal((n, 3))
            X = X / np.linalg.norm(X, axis=1, keepdims=True)
            X = softmin_ascent(X, iters=300, tau0=0.20, tau1=0.01,
                               lr=0.03, sphere=True)
        # Free-position refinement: first grow dmin, then the ratio.
        Y = softmin_ascent(X, iters=250, tau0=0.05, tau1=0.002,
                           lr=0.015, sphere=False)
        consider(Y)
        Z = ratio_ascent(Y, iters=250, tau0=0.05, tau1=0.001,
                         lr=0.008)
        consider(Z)

    for _ in range(40):
        X = rms_norm(ring_start(rng))
        Y = softmin_ascent(X, iters=300, tau0=0.20, tau1=0.01,
                           lr=0.03, sphere=False)
        consider(Y)
        consider(ratio_ascent(Y, iters=250, tau0=0.05, tau1=0.001, lr=0.008))

    # ---- Kick-and-reoptimize polishing on the best configuration. ----
    # Perturbations of decreasing magnitude, each followed by low-tau
    # ratio ascent; keep only genuine improvements of the true ratio.
    scales = [0.08, 0.05, 0.03, 0.02, 0.01, 0.005]
    for scale in scales:
        for _ in range(30):
            X = best_X + scale * rng.standard_normal((n, 3))
            X = rms_norm(X)
            Z = ratio_ascent(X, iters=300, tau0=0.02, tau1=0.0005,
                             lr=0.004)
            consider(Z)
            # Occasionally also run a pure dmin stage first (helps when
            # the kick created a too-close pair).
            if rng.random() < 0.5:
                Y = softmin_ascent(X, iters=200, tau0=0.05, tau1=0.002,
                                   lr=0.01, sphere=False)
                consider(ratio_ascent(Y, iters=250, tau0=0.02,
                                      tau1=0.0005, lr=0.004))

    # Scale so that the maximum pairwise distance equals 1.
    dM = float(np.sqrt(sq_dists(best_X)[iu].max()))
    best_X = best_X / dM
    return np.asarray(best_X, dtype=float)


# EVOLVE-BLOCK-END
