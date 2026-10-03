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

    Approach (staggered-ring-orbit): seeds come from a parametric family of
    two 7-point rings at heights +-h with twist theta and radii r1, r2
    (4 DOFs instead of 42 coordinates). A deterministic coarse (h, theta)
    grid (<=63 starts, cheap) seeds L-BFGS-B on the ring parameters
    (maxiter 400). Each ring optimum then receives the FULL polish pipeline:
    soft-min/soft-max ascent followed by the exact 3000-iter max-min
    subgradient refine, so the 7-fold contact topology can relax beyond the
    ring manifold. The best ring configuration additionally gets 10
    deterministic jitter restarts (seeded, small Gaussian symmetry breaking)
    through the same pipeline, letting _refine escape the 7-fold saddle.
    Incumbent tracked by exact squared ratio; finite random fallback kept.
    """
    from scipy.optimize import minimize

    n = 14

    def _ring_config(p):
        h, th, r1, r2 = [float(x) for x in p]
        a = 2.0 * np.pi * np.arange(7) / 7.0
        up = np.stack([r1 * np.cos(a), r1 * np.sin(a), np.full(7, h)], axis=1)
        lo = np.stack([r2 * np.cos(a + th), r2 * np.sin(a + th),
                       np.full(7, -h)], axis=1)
        return np.vstack([up, lo])

    def _neg_ratio(p):
        return -_ratio(_ring_config(p))

    def _full_polish(P0, soft_iters=1500, ref_iters=3000):
        P = P0 - P0.mean(axis=0)
        for _ in range(soft_iters):
            P = _soft_step(P)
        return _refine(P, iters=ref_iters)

    bounds = [(0.05, 2.0), (0.0, 2.0 * np.pi / 7.0),
              (0.1, 2.0), (0.1, 2.0)]

    bestP, bestR = None, -1.0

    def _consider(P):
        nonlocal bestP, bestR
        if np.all(np.isfinite(P)):
            r = _ratio(P)
            if r > bestR:
                bestR = r
                bestP = P.copy()

    # Coarse deterministic grid over (h, theta, radius ratio r2/r1).
    pool = []
    for h in np.linspace(0.2, 1.0, 9):
        for th in np.linspace(0.0, np.pi / 7.0, 7):
            for rr in (0.85, 1.0, 1.15):
                pool.append([float(h), float(th), 1.0, float(rr)])
    starts = sorted(pool, key=_neg_ratio)[:20]

    ring_best_p, ring_best_r = None, -np.inf
    for p0 in starts:
        try:
            res = minimize(_neg_ratio, np.asarray(p0, dtype=float),
                           method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": 400})
            P = _ring_config(res.x)
        except Exception:
            continue
        if not np.all(np.isfinite(P)):
            continue
        if -_neg_ratio(res.x) > ring_best_r:
            ring_best_r = -_neg_ratio(res.x)
            ring_best_p = np.asarray(res.x, dtype=float)
        try:
            Pr, r = _full_polish(P)
        except Exception:
            continue
        _consider(Pr)

    # Deterministic symmetry-breaking jitter restarts around the best ring,
    # with a decreasing sigma schedule (coarse exploration -> fine tuning).
    if ring_best_p is not None:
        jr = np.random.RandomState(11)
        for k in range(20):
            sigma = 0.05 - 0.04 * (k / 19.0)
            pj = ring_best_p + sigma * jr.randn(4)
            pj = np.clip(pj, [b[0] for b in bounds], [b[1] for b in bounds])
            P = _ring_config(pj)
            try:
                Pr, r = _full_polish(P)
            except Exception:
                continue
            _consider(Pr)

    # Final extended polish of the best incumbent: alternating rounds of
    # deep soft-min/soft-max ascent, long exact max-min refine, and
    # coordinate-space L-BFGS-B on the exact squared ratio (lets the
    # solution leave the ring manifold entirely). Each round seeds the
    # next from the current incumbent, tracked by exact ratio.
    if bestP is not None:
        try:
            def _neg(x):
                return -_ratio(x.reshape(n, 3))

            for rnd in range(3):
                # Soft ascent with a round-dependent sharpness.
                Ps = bestP.copy()
                p_exp = 40.0 + 30.0 * rnd
                for _ in range(2000):
                    Ps = _soft_step(Ps, p=p_exp, lr=0.008)
                Pr, rr = _refine(Ps, iters=6000, lr=0.005)
                _consider(Pr)
                # Coordinate-space L-BFGS-B from the refreshed incumbent.
                res2 = minimize(_neg, bestP.ravel().copy(), method="L-BFGS-B",
                                options={"maxiter": 500, "maxfun": 12000})
                if np.all(np.isfinite(res2.x)):
                    Pc = res2.x.reshape(n, 3).astype(float)
                    Pc = Pc - Pc.mean(axis=0)
                    _consider(Pc)
                # Short exact refine after L-BFGS to convert its progress.
                Pq, rq = _refine(bestP.copy(), iters=2000, lr=0.003)
                _consider(Pq)
        except Exception:
            pass

        # Seeded per-point jitter symmetry breaking around the incumbent
        # (decreasing sigma, fine exploration), each through a solid refine.
        jr = np.random.RandomState(23)
        for k in range(16):
            sigma = 0.03 - 0.025 * (k / 15.0)
            Pj = bestP + sigma * jr.randn(n, 3)
            try:
                Prj, rj = _refine(Pj, iters=3000, lr=0.004)
                _consider(Prj)
            except Exception:
                continue

    if bestP is None or not np.all(np.isfinite(bestP)):
        return np.random.RandomState(42).randn(n, 3)
    return bestP


# EVOLVE-BLOCK-END
