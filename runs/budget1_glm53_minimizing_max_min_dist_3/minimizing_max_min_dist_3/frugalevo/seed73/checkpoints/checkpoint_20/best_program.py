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


def _epigraph_solve(P0, maxiter=400, rounds=4):
    """Epigraph max-min NLP solved with SLSQP plus warm-started continuation.

    Normalizes the seed so dmax = 1, then minimizes -t subject to
    d(i,j)^2 - t >= 0 and 1 - d(i,j)^2 >= 0 over variables (42 coords, t),
    with analytic constraint Jacobians (grad of d^2 is linear in the
    coordinates). Continuation = warm-started re-solves of the FULL 182-
    constraint set (dropping constraints lets dmax grow and destroys the
    ratio); each round re-normalizes and re-warm-starts, sharpening the
    contact set. Guarded: the result is returned only if its exact squared
    ratio beats the seed's, otherwise the seed is returned unchanged, so
    acceptance is monotone in the scored objective.
    """
    from scipy.optimize import minimize
    n = len(P0)
    iu = np.triu_indices(n, 1)
    ia, ja = iu[0], iu[1]
    m = len(ia)
    rows = np.arange(m)
    diff0 = P0[:, None, :] - P0[None, :, :]
    D0 = np.sqrt((diff0 ** 2).sum(-1))
    dmax0 = float(D0[iu].max())
    if not np.isfinite(dmax0) or dmax0 <= 1e-12:
        return P0.copy()
    seed_r = _ratio(P0)
    P = (P0 / dmax0).astype(float)

    def _d2(Pm):
        diff = Pm[:, None, :] - Pm[None, :, :]
        return (diff ** 2).sum(-1)[iu]

    def cons_f(x):
        d2 = _d2(x[:3 * n].reshape(n, 3))
        return np.concatenate([d2 - x[-1], 1.0 - d2])

    def cons_jac(x):
        Pm = x[:3 * n].reshape(n, 3)
        G = 2.0 * (Pm[ia] - Pm[ja])  # (m, 3)
        J = np.zeros((2 * m, 3 * n + 1))
        for r in range(3):
            J[rows, 3 * ia + r] = G[:, r]
            J[rows, 3 * ja + r] = -G[:, r]
            J[m + rows, 3 * ia + r] = -G[:, r]
            J[m + rows, 3 * ja + r] = G[:, r]
        J[rows, -1] = -1.0
        J[m + rows, -1] = 1.0
        return J

    obj = lambda x: -x[-1]
    obj_jac = lambda x: np.concatenate([np.zeros(3 * n), [-1.0]])
    cons = [{"type": "ineq", "fun": cons_f, "jac": cons_jac}]
    x = np.concatenate([P.ravel(), [float(_d2(P).min())]])
    bestP, bestR = P0.copy(), seed_r
    for _ in range(rounds):
        try:
            res = minimize(obj, x, jac=obj_jac, method="SLSQP",
                           constraints=cons,
                           options={"maxiter": maxiter, "ftol": 1e-12})
            x = res.x
        except Exception:
            break
        if not np.all(np.isfinite(x)):
            break
        Pc = x[:3 * n].reshape(n, 3).astype(float)
        if not np.all(np.isfinite(Pc)):
            break
        rc = _ratio(Pc)
        if rc > bestR:
            bestR = rc
            bestP = Pc.copy()
        # Re-normalize and warm-start the next continuation round.
        dd = _d2(Pc)
        dmx = float(np.sqrt(dd.max()))
        if dmx <= 1e-12:
            break
        x = np.concatenate([(Pc / dmx).ravel(),
                            [float(dd.min()) / dmx ** 2]])
    return bestP


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Construct 14 points in 3D maximizing (dmin/dmax)^2.

    Approach (epigraph-SLSQP with soft basin-finding): cheap ring-parameter
    seeds (two 7-point rings, 4 DOFs) come from a deterministic coarse grid
    + L-BFGS-B. Each seed gets a SHORT bounded soft-min/soft-max ascent
    (300 iters, basin finding only) and is then polished by an epigraph
    NLP solve: minimize -t s.t. d(i,j)^2 >= t, d(i,j)^2 <= 1 over 43
    variables with analytic Jacobians, plus warm-started continuation
    rounds. Every solve is guarded by the exact squared ratio (monotone
    acceptance). Seeded ring jitters and per-point jitters of the incumbent
    go through the same soft-then-epigraph pipeline; a final deep epigraph
    continuation on the best incumbent squeezes the last digits. Finite
    random fallback kept.
    """
    from scipy.optimize import minimize

    n = 14

    def _ring_config(p, k=7):
        """Two staggered rings with sizes (k, n-k), heights +/-h, independent
        radii r1, r2, relative twist th. Generalizes the 7+7 family to
        unequal splits, adding independent angular spacings as new DOFs."""
        h, th, r1, r2 = [float(x) for x in p]
        a1 = 2.0 * np.pi * np.arange(k) / float(k)
        k2 = n - k
        a2 = 2.0 * np.pi * np.arange(k2) / float(k2) + th
        up = np.stack([r1 * np.cos(a1), r1 * np.sin(a1), np.full(k, h)], axis=1)
        lo = np.stack([r2 * np.cos(a2), r2 * np.sin(a2),
                       np.full(k2, -h)], axis=1)
        return np.vstack([up, lo])

    def _neg_ratio(p, k=7):
        return -_ratio(_ring_config(p, k))

    def _polish(P0, soft_iters=300, rounds=4, maxiter=400):
        """Short soft ascent (basin finding) then epigraph SLSQP polish."""
        P = P0 - P0.mean(axis=0)
        for _ in range(soft_iters):
            P = _soft_step(P, p=50.0, lr=0.01)
        return _epigraph_solve(P, maxiter=maxiter, rounds=rounds)

    bestP, bestR = None, -1.0

    def _consider(P):
        nonlocal bestP, bestR
        if np.all(np.isfinite(P)):
            r = _ratio(P)
            if r > bestR:
                bestR = r
                bestP = P.copy()

    # Coarse deterministic grid over (h, theta, radius ratio r2/r1),
    # run per unequal-split k in {4,5,6,7}; polish top-5 per split.
    for k in (4, 5, 6, 7):
        bounds = [(0.05, 2.0), (0.0, 2.0 * np.pi / max(k, 1)),
                  (0.1, 2.0), (0.1, 2.0)]
        bb = ([b[0] for b in bounds], [b[1] for b in bounds])
        pool = []
        for h in np.linspace(0.2, 1.0, 9):
            for th in np.linspace(0.0, np.pi / max(k, 1), 7):
                for rr in (0.85, 1.0, 1.15):
                    pool.append([float(h), float(th), 1.0, float(rr)])
        pool.sort(key=lambda q: _neg_ratio(np.asarray(q, dtype=float), k))
        starts = pool[:5]

        ring_best_p = None
        for p0 in starts:
            try:
                res = minimize(_neg_ratio, np.asarray(p0, dtype=float),
                               args=(k,), method="L-BFGS-B", bounds=bounds,
                               options={"maxiter": 400})
                P = _ring_config(res.x, k)
            except Exception:
                continue
            if not np.all(np.isfinite(P)):
                continue
            if ring_best_p is None:
                ring_best_p = np.asarray(res.x, dtype=float)
            try:
                _consider(_polish(P))
            except Exception:
                continue

        # Deterministic symmetry-breaking jitter restarts around the best
        # ring for this split, decreasing sigma schedule.
        if ring_best_p is not None:
            jr = np.random.RandomState(11 + k)
            for j in range(10):
                sigma = 0.05 - 0.04 * (j / 9.0)
                pj = ring_best_p + sigma * jr.randn(4)
                pj = np.clip(pj, bb[0], bb[1])
                try:
                    _consider(_polish(_ring_config(pj, k)))
                except Exception:
                    continue

        # Seeded per-point jitter symmetry breaking around the incumbent,
        # each through the soft-then-epigraph pipeline.
        if bestP is not None:
            jr = np.random.RandomState(23)
            for k in range(12):
                sigma = 0.03 - 0.025 * (k / 11.0)
                Pj = bestP + sigma * jr.randn(n, 3)
                try:
                    _consider(_polish(Pj, soft_iters=200, rounds=3))
                except Exception:
                    continue

            # Final deep epigraph continuation on the best incumbent:
            # longer SLSQP runs and more warm-started rounds.
            try:
                _consider(_polish(bestP.copy(), soft_iters=400,
                                  rounds=6, maxiter=600))
            except Exception:
                pass

    if bestP is None or not np.all(np.isfinite(bestP)):
        return np.random.RandomState(42).randn(n, 3)
    return bestP


# EVOLVE-BLOCK-END
