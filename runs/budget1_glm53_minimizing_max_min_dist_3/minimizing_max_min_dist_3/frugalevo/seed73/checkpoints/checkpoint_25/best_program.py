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


def _kkt_refine(P0, max_newton=60, max_cycles=5):
    """Exact active-set KKT Newton solve on the incumbent's contact set.

    Normalizes so dmax = 1, builds active sets A_min (d^2 <= t*+delta) and
    A_max (d^2 >= 1-delta) with growing delta until enough equations, then
    solves the KKT equality system (coordinate stationarity, dL/dt = 0,
    active-edge equalities) over (42 coords, t, multipliers) with damped
    Newton (lstsq steps, residual line search). Negative multipliers cause
    those edges to be dropped from the SAME sets and the system re-solved
    (<= max_cycles). Accepts only if the exact squared ratio strictly
    improves and every pairwise d^2 lies in [t_new - 1e-9, 1 + 1e-9];
    otherwise returns the (normalized) incumbent unchanged.
    """
    n = len(P0)
    iu = np.triu_indices(n, 1)
    ia, ja = np.asarray(iu[0]), np.asarray(iu[1])
    P = np.asarray(P0, dtype=float).copy()
    D, _ = _pairwise(P)
    d0 = D[iu]
    dmx = float(d0.max())
    if not np.isfinite(dmx) or dmx <= 1e-12:
        return P0.copy()
    P = P / dmx
    bestP, bestR = P.copy(), _ratio(P)

    def _d2all(Pm):
        return ((Pm[:, None, :] - Pm[None, :, :]) ** 2).sum(-1)[iu]

    A = B = None
    for _cycle in range(max_cycles):
        if A is None:
            d2 = _d2all(P)
            t = float(d2.min())
            delta = 1e-7
            A = B = None
            while delta <= 1e-3:
                A = np.where(d2 <= t + delta)[0]
                B = np.where(d2 >= 1.0 - delta)[0]
                if len(A) + len(B) >= 3 * n + 1:
                    break
                delta *= 10.0
            if len(A) == 0:
                A = np.array([int(np.argmin(d2))])
            if len(B) == 0:
                B = np.array([int(np.argmax(d2))])
        mA, mB = len(A), len(B)
        iaA, jaA = ia[A], ja[A]
        iaB, jaB = ia[B], ja[B]
        nv = 3 * n + 1 + mA + mB
        iAt = 3 * n + 1
        iBt = iAt + mA

        def _res(x):
            Pm = x[:3 * n].reshape(n, 3)
            tm = x[3 * n]
            lamA = x[iAt:iBt]
            lamB = x[iBt:]
            gA = 2.0 * (Pm[iaA] - Pm[jaA])
            gB = 2.0 * (Pm[iaB] - Pm[jaB])
            d2A = (gA * (Pm[iaA] - Pm[jaA])).sum(1)
            d2B = (gB * (Pm[iaB] - Pm[jaB])).sum(1)
            stat = np.zeros(3 * n)
            for e in range(mA):
                stat[3 * iaA[e]:3 * iaA[e] + 3] += lamA[e] * gA[e]
                stat[3 * jaA[e]:3 * jaA[e] + 3] -= lamA[e] * gA[e]
            for e in range(mB):
                stat[3 * iaB[e]:3 * iaB[e] + 3] += lamB[e] * gB[e]
                stat[3 * jaB[e]:3 * jaB[e] + 3] -= lamB[e] * gB[e]
            return np.concatenate([stat, [-1.0 + lamA.sum()],
                                   d2A - tm, d2B - 1.0])

        def _jac(x):
            Pm = x[:3 * n].reshape(n, 3)
            gA = 2.0 * (Pm[iaA] - Pm[jaA])
            gB = 2.0 * (Pm[iaB] - Pm[jaB])
            lamA = x[iAt:iBt]
            lamB = x[iBt:]
            J = np.zeros((nv, nv))
            for lam, ii, jj in ((lamA, iaA, jaA), (lamB, iaB, jaB)):
                for lam_e, i, j in zip(lam, ii, jj):
                    for r in range(3):
                        J[3 * i + r, 3 * i + r] += 2.0 * lam_e
                        J[3 * j + r, 3 * j + r] += 2.0 * lam_e
                        J[3 * i + r, 3 * j + r] -= 2.0 * lam_e
                        J[3 * j + r, 3 * i + r] -= 2.0 * lam_e
                for e, (i, j) in enumerate(zip(ii, jj)):
                    for r in range(3):
                        J[3 * i + r, iAt + e if ii is iaA else iBt + e] += gA[e, r] if ii is iaA else gB[e, r]
                        J[3 * j + r, iAt + e if ii is iaA else iBt + e] -= gA[e, r] if ii is iaA else gB[e, r]
            J[3 * n, iAt:iBt] = 1.0
            for e in range(mA):
                for r in range(3):
                    J[3 * n + 1 + e, 3 * iaA[e] + r] += gA[e, r]
                    J[3 * n + 1 + e, 3 * jaA[e] + r] -= gA[e, r]
                J[3 * n + 1 + e, 3 * n] = -1.0
            for e in range(mB):
                for r in range(3):
                    J[3 * n + 1 + mA + e, 3 * iaB[e] + r] += gB[e, r]
                    J[3 * n + 1 + mA + e, 3 * jaB[e] + r] -= gB[e, r]
            return J

        x = np.concatenate([P.ravel(), [float(_d2all(P).min())],
                            np.full(mA, 1.0 / max(mA, 1)),
                            np.full(mB, 1.0 / max(mB, 1))])
        for _it in range(max_newton):
            F = _res(x)
            f0 = float(np.linalg.norm(F))
            if not np.isfinite(f0) or f0 <= 1e-14:
                break
            try:
                step = np.linalg.lstsq(_jac(x), -F, rcond=None)[0]
            except Exception:
                break
            improved = False
            alpha = 1.0
            for _ls in range(8):
                xn = x + alpha * step
                fn = float(np.linalg.norm(_res(xn)))
                if np.isfinite(fn) and fn < f0:
                    x = xn
                    improved = True
                    break
                alpha *= 0.5
            if not improved:
                break
        Pn = x[:3 * n].reshape(n, 3).astype(float)
        if np.all(np.isfinite(Pn)):
            d2n = _d2all(Pn)
            tn = float(d2n.min())
            if (np.all(d2n >= tn - 1e-9) and np.all(d2n <= 1.0 + 1e-9)):
                rn = _ratio(Pn)
                if rn > bestR:
                    bestR = rn
                    bestP = Pn.copy()
        # Drop negative multipliers from the SAME sets and re-solve.
        lamA = x[iAt:iBt]
        lamB = x[iBt:]
        keepA = np.where(lamA >= -1e-12)[0]
        keepB = np.where(lamB >= -1e-12)[0]
        if len(keepA) == mA and len(keepB) == mB:
            break
        A, B = A[keepA], B[keepB]
        if len(A) == 0 or len(B) == 0:
            break
        P = bestP.copy()
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

        # Discrete contact-graph swap walk with escape candidates. The
        # walking point Pcur hops between neighboring contact graphs:
        # each iteration normalizes Pcur (dmax = 1), finds active min/max
        # edges, and builds <= 9 deterministic candidates — pair-separation
        # moves on the tightest min-edges / dmax pair plus mirror moves,
        # AND 3 seeded Gaussian escape candidates around the incumbent
        # (decreasing sigma) so the walk can cross barriers pure pair
        # moves cannot. Every candidate is polished by the existing
        # soft+epigraph pipeline and scored by the exact squared ratio.
        # The best candidate is accepted into the walk if it improves OR
        # ties the incumbent (drift lets the walk point escape plateaus);
        # the incumbent bestP itself is updated only on strict improvement
        # via _consider, so acceptance stays monotone. Bounded: <= 40
        # iterations, <= 9 short NLP solves each.
        if bestP is not None:
            delta = 0.03
            jr = np.random.RandomState(23)
            Pcur = bestP.copy()
            for _step in range(40):
                Dc, _ = _pairwise(Pcur)
                dw = Dc[np.triu_indices(n, 1)]
                dmax_c = float(dw.max())
                if dmax_c <= 1e-12:
                    break
                Pw = Pcur / dmax_c
                Dw, _ = _pairwise(Pw)
                iuw = np.triu_indices(n, 1)
                dw = Dw[iuw]
                dmin_w, dmax_w = float(dw.min()), float(dw.max())
                ia_w, ja_w = iuw[0], iuw[1]

                def _active(target, tol):
                    sel = np.abs(dw - target) <= tol
                    if not np.any(sel):
                        sel = np.abs(dw - target) <= 1e-4
                    order = np.argsort(np.abs(dw[sel] - target))
                    return np.where(sel)[0][order][:3]

                min_ids = _active(dmin_w, 1e-6)
                max_ids = _active(dmax_w, 1e-6)

                def _move(i, j, s):
                    """Push pair (i,j) apart (s=+1) or together (s=-1)."""
                    Q = Pw.copy()
                    u = Q[j] - Q[i]
                    un = np.linalg.norm(u)
                    if un <= 1e-12:
                        return None
                    u = u / un
                    Q[i] -= s * delta * u
                    Q[j] += s * delta * u
                    return Q

                cands = []
                for e in min_ids:
                    cands.append(_move(int(ia_w[e]), int(ja_w[e]), +1.0))
                for e in max_ids:
                    cands.append(_move(int(ia_w[e]), int(ja_w[e]), -1.0))
                if len(dw) > 1:
                    o2 = int(np.argsort(-dw)[1])
                    cands.append(_move(int(ia_w[o2]), int(ja_w[o2]), +1.0))
                    o1 = int(np.argsort(dw)[1])
                    cands.append(_move(int(ia_w[o1]), int(ja_w[o1]), -1.0))
                # Escape candidates: seeded Gaussian jitters of the
                # incumbent with decreasing sigma, part of the walk's
                # candidate generation (not a separate restart loop).
                sigma = 0.03 - 0.02 * (_step / 39.0)
                for _e in range(3):
                    cands.append(bestP + sigma * jr.randn(n, 3))

                scored = []
                for Q in cands:
                    if Q is None or not np.all(np.isfinite(Q)):
                        continue
                    try:
                        Qp = _polish(Q, soft_iters=100, rounds=3,
                                     maxiter=300)
                    except Exception:
                        continue
                    if np.all(np.isfinite(Qp)):
                        scored.append((_ratio(Qp), Qp))
                if not scored:
                    break
                scored.sort(key=lambda q: -q[0])
                r_top, P_top = scored[0]
                if r_top > bestR:
                    _consider(P_top)
                    Pcur = P_top.copy()
                elif r_top >= bestR - 1e-9:
                    # Tie drift: move the walk point, incumbent unchanged.
                    Pcur = P_top.copy()
                else:
                    break  # no candidate reaches the incumbent: done

            # Final deep epigraph continuation on the best incumbent:
            # longer SLSQP runs and more warm-started rounds.
            try:
                _consider(_polish(bestP.copy(), soft_iters=400,
                                  rounds=6, maxiter=600))
            except Exception:
                pass

    # Final exact active-set KKT Newton refinement of the incumbent's
    # contact set (runs once, on the normal return path, after all SLSQP
    # stages). SLSQP with ftol=1e-12 returns a truncated iterate, not an
    # exact KKT point; solving the active-set equality system to residual
    # ~1e-14 harvests the remaining digits. Strictly guarded, so the
    # incumbent can only improve.
    if bestP is not None:
        try:
            _consider(_kkt_refine(bestP.copy()))
        except Exception:
            pass

    if bestP is None or not np.all(np.isfinite(bestP)):
        return np.random.RandomState(42).randn(n, 3)
    return bestP


# EVOLVE-BLOCK-END
