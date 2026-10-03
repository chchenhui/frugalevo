# EVOLVE-BLOCK-START
import numpy as np


def _repulse(X0, steps=200, lr=0.002):
    """Coulomb-style repulsion: minimize sum 1/d_ij^2 via gradient descent.

    Cheap O(n^2) preprocessing that turns a clustered random start into a
    well-spread quasi-uniform configuration. Normalized steps keep it stable.
    """
    X = X0.copy()
    n = X.shape[0]
    ii, jj = np.triu_indices(n, 1)
    for _ in range(steps):
        diff = X[ii] - X[jj]
        d2 = np.sum(diff ** 2, axis=1) + 1e-9
        # grad of sum 1/d^2 wrt pair diff is -2*diff/d^4
        coef = -2.0 / (d2 ** 2)[:, None]
        G = np.zeros_like(X)
        contrib = coef * diff
        np.add.at(G, ii, contrib)
        np.add.at(G, jj, -contrib)
        gn = np.sqrt((G ** 2).sum())
        if gn < 1e-12:
            break
        # normalize so each step moves a bounded amount; decay learning rate
        X = X - (lr / (1.0 + 0.01 * _)) * G / gn
        # re-center to avoid drift
        X = X - X.mean(axis=0)
    return X


def _polish(X0, steps=350, lr=0.02, t0=0.15, t1=0.01):
    """Maximize dmin/dmax by maximizing a soft-min of pair distances while
    penalizing pairs that exceed the current diameter target (scale-invariant
    reformulation: keep dmax pinned, push dmin up).

    Objective (to maximize):
        softmin_t(d_ij) - lambda * softplus_excess(max-side)
    implemented as: weighted (softmax on -d/t) mean of distances, minus a
    quadratic penalty on pairs with d > diam_target, where diam_target tracks
    the current max distance.
    """
    X = X0.copy()
    n = X.shape[0]
    ii, jj = np.triu_indices(n, 1)
    lam = 4.0
    for k in range(steps):
        t = t0 * (t1 / t0) ** (k / max(steps - 1, 1))
        diff = X[ii] - X[jj]
        d = np.sqrt(np.sum(diff ** 2, axis=1) + 1e-12)
        diam = d.max()
        # soft-min weights (maximize weighted mean distance)
        w = np.exp(-(d - d.min()) / max(t, 1e-6))
        w /= w.sum()
        u = diff / d[:, None]
        # diameter penalty: pairs above diam get pushed together
        over = np.maximum(d - diam, 0.0)
        # gradient contributions
        Gi = np.zeros_like(X)
        np.add.at(Gi, ii, w[:, None] * u)
        np.add.at(Gi, jj, -(w[:, None] * u))
        # penalize the single farthest pair (moves to reduce diameter)
        kfar = np.argmax(d)
        pf = (X[ii[kfar]] - X[jj[kfar]]) / d[kfar]
        Gi[ii[kfar]] -= lam * pf
        Gi[jj[kfar]] += lam * pf
        gn = np.sqrt((Gi ** 2).sum())
        if gn < 1e-12:
            break
        step = lr * (0.5 ** (k / steps))  # decaying steps
        X = X + step * Gi / gn * diam
        X = X - X.mean(axis=0)
    return X


def min_max_dist_dim3_14() -> np.ndarray:
    n, d = 14, 3
    rng = np.random.default_rng(12345)
    best_pts, best_ratio = None, -1.0
    for trial in range(12):
        X0 = rng.standard_normal((n, d))
        # stage 1: repulsion to spread points (Coulomb energy descent)
        X = _repulse(X0, steps=200)
        # stage 2: constrained polish maximizing dmin with dmax pinned
        X = _polish(X, steps=350)
        # evaluate true ratio
        D = np.linalg.norm(X[:, None] - X[None, :], axis=-1)
        iu = np.triu_indices(n, 1)
        dmin = D[iu].min()
        dmax = D[iu].max()
        if dmax <= 0:
            continue
        r = dmin / dmax
        if r > best_ratio:
            best_ratio = r
            best_pts = X
    # normalize: center and scale so max pairwise distance = 1
    P = best_pts - best_pts.mean(axis=0)
    Dm = np.linalg.norm(P[:, None] - P[None, :], axis=-1)
    diam = Dm.max()
    if diam > 0:
        P = P / diam
    return P


# EVOLVE-BLOCK-END