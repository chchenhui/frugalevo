# EVOLVE-BLOCK-START
import numpy as np


def _pairwise(P):
    """Pairwise distance matrix and unit direction tensors for point set P."""
    diff = P[:, None, :] - P[None, :, :]
    D = np.sqrt((diff ** 2).sum(-1)) + np.eye(len(P))
    return D, diff


def _ratio(P):
    """Exact objective (dmin/dmax)^2."""
    D, _ = _pairwise(P)
    iu = np.triu_indices(len(P), 1)
    d = D[iu]
    return (d.min() / d.max()) ** 2


def _soft_step(P, p=40.0, lr=0.01):
    """One ascent step on softmin_p(d)/softmax_p(d) with stable weights."""
    n = len(P)
    D, diff = _pairwise(P)
    iu = np.triu_indices(n, 1)
    d = D[iu]
    # log-sum-exp style weights: w ~ d^-p (soft min), v ~ d^p (soft max)
    lw = -p * np.log(d)
    lv = p * np.log(d)
    lw -= lw.max()
    lv -= lv.max()
    w = np.exp(lw)
    v = np.exp(lv)
    W = np.zeros((n, n)); W[iu] = w; W += W.T
    V = np.zeros((n, n)); V[iu] = v; V += V.T
    coef = (W / W.sum() - V / V.sum()) / D
    np.fill_diagonal(coef, 0.0)
    grad = (coef[:, :, None] * diff).sum(axis=1)
    gn = np.linalg.norm(grad)
    if gn > 1e-14:
        P = P + lr * grad / gn
    return P


def _refine(P, iters=3000, lr=0.005):
    """Free-space exact-ratio polish: subgradient ascent on
    dmin - (dmin/dmax) * dmax, i.e. push apart pairs within 2% of the
    minimum distance while pulling together pairs within 2% of the maximum.
    No sphere projection (radii are free), linear lr decay, best incumbent
    tracked by the exact squared ratio."""
    n = len(P)
    best = _ratio(P)
    bestP = P.copy()
    for it in range(iters):
        cur_lr = lr * (1.0 - it / iters) + 1e-4
        D, diff = _pairwise(P)
        iu = np.triu_indices(n, 1)
        d = D[iu]
        dmin, dmax = d.min(), d.max()
        if dmax <= 0.0:
            continue
        r = dmin / dmax
        mmin = (D < dmin * 1.02).astype(float) - np.eye(n)
        mmax = (D > dmax * 0.98).astype(float) - np.eye(n)
        coef = (mmin - r * mmax) / D
        grad = (coef[:, :, None] * diff).sum(axis=1)
        gn = np.linalg.norm(grad)
        if gn > 1e-14:
            P = P + cur_lr * grad / gn
        r2 = r * r
        if r2 > best:
            best = r2
            bestP = P.copy()
    return bestP, best


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Construct 14 points in 3D maximizing (dmin/dmax)^2.

    Approach: deterministic multistart. Seeds are symmetric polyhedral
    configurations (cube+octahedron, icosahedron+poles) plus fixed-seed
    random starts. Each start ascends a smooth soft-min/soft-max surrogate
    (stable exp-weighted gradients), then is polished with an exact max-min
    subgradient step. Points are kept on the unit sphere (scale-invariant
    convenience). The best incumbent by the exact ratio is returned; a
    random fallback guarantees a valid (14,3) finite array.
    """
    n = 14
    starts = []

    # cube corners + octahedron vertices (14 points exactly)
    cube = np.array([[sx, sy, sz] for sx in (-1, 1)
                     for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
    octa = np.vstack([np.eye(3), -np.eye(3)])
    starts.append(np.vstack([cube, octa]))

    # icosahedron (12) + 2 poles
    phi = (1 + 5 ** 0.5) / 2
    ico = []
    for a in (-1, 1):
        for b in (-phi, phi):
            ico += [[0, a, b], [a, b, 0], [b, 0, a]]
    ico = np.array(ico, dtype=float)
    ico /= np.linalg.norm(ico, axis=1, keepdims=True)
    starts.append(np.vstack([ico, [[0, 0, 1.0], [0, 0, -1.0]]]))

    rng = np.random.RandomState(7)
    for _ in range(8):
        starts.append(rng.randn(n, 3))

    def _optimize(P0, soft_iters=2000, ref_iters=3000):
        P0 = P0 - P0.mean(axis=0)
        P = P0
        for _ in range(soft_iters):
            P = _soft_step(P)
        return _refine(P, iters=ref_iters)

    bestP, bestR = None, -1.0
    for P0 in starts:
        try:
            P, r = _optimize(P0)
        except Exception:
            continue
        if r > bestR:
            bestR = r
            bestP = P

    # Differential evolution over free 3D coordinates with the translation
    # gauge removed: point 0 fixed at the origin, point 1 on the +x axis at
    # a free radius. 37 variables. Objective: -(dmin/dmax)^2 via pdist.
    # Result is polished in free space by _refine and accepted only if the
    # exact squared ratio beats the incumbent.
    try:
        from scipy.optimize import differential_evolution
        from scipy.spatial.distance import pdist

        def _decode(x):
            P = np.zeros((n, 3))
            P[0] = [0.0, 0.0, 0.0]
            P[1] = [max(float(x[0]), 1e-3), 0.0, 0.0]
            P[2:] = np.asarray(x[1:], dtype=float).reshape(n - 2, 3)
            return P

        def _neg_obj(x):
            P = _decode(x)
            d = pdist(P)
            dM = d.max()
            if dM <= 0.0:
                return 1.0
            return -((d.min() / dM) ** 2)

        bounds = [(0.05, 3.0)] + [(-2.0, 2.0)] * (3 * (n - 2))
        de = differential_evolution(
            _neg_obj, bounds, seed=0, maxiter=120, popsize=18,
            tol=1e-8, polish=True, disp=False,
        )
        Pde = _decode(de.x)
        Ppol, rpol = _refine(Pde, iters=2000, lr=0.005)
        if rpol > bestR:
            bestR = rpol
            bestP = Ppol
    except Exception:
        pass

    if bestP is None or not np.all(np.isfinite(bestP)):
        return np.random.RandomState(42).randn(n, 3)
    return bestP


# EVOLVE-BLOCK-END
