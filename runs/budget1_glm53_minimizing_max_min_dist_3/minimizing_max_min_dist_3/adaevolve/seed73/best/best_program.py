# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize as _slsqp_min


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Constructs exactly 14 points in 3D maximizing (dmin/dmax)^2.

    Approach:
      1) Multi-start projected gradient ascent on a soft-minimum of
         pairwise distances (random + icosahedron-based seeds).
      2) Top candidates are refined by a DIRECT constrained NLP solve
         (scipy SLSQP) of the exact reformulation: maximize auxiliary
         t subject to d(i,j)^2 >= t and d(i,j)^2 <= 1, with analytic
         constraint Jacobians. This targets true KKT points of the
         ratio objective rather than heuristic soft-minima.
      3) A short adaptive-sigma free-space polish + a second SLSQP
         pass, then a final long polish on the overall best.
    Finally points are rescaled so dmax = 1.
    """

    n = 14
    d = 3
    rng = np.random.default_rng(12345)
    iu = np.triu_indices(n, 1)
    ii, jj = iu

    def ratio_sq(P):
        D = np.sqrt(np.sum((P[:, None, :] - P[None, :, :]) ** 2, axis=-1) + 1e-18)
        dv = D[iu]
        return (dv.min() / dv.max()) ** 2

    def step_grad(P, temp):
        diff = P[:, None, :] - P[None, :, :]
        D = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-18)
        dv = D[iu]
        w = np.exp((1.0 / dv - 1.0 / dv.min()) / temp)
        w /= w.sum()
        w[w < 1e-6] = 0.0
        w /= w.sum()
        u = diff[ii, jj] / (dv[:, None] + 1e-12)
        G = np.zeros_like(P)
        np.add.at(G, ii, w[:, None] * u)
        np.add.at(G, jj, -w[:, None] * u)
        return G

    def project_sphere(P):
        return P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)

    def normalize(P):
        D = np.sqrt(np.sum((P[:, None, :] - P[None, :, :]) ** 2, axis=-1) + 1e-18)
        return P / D[iu].max()

    def free_grad_step(P, temp, lam=0.5):
        # Gradient step in FREE space: push apart the closest pairs
        # (soft-min weights) and pull together the farthest pairs
        # (soft-max weights), directly optimizing dmin/dmax without
        # the spherical constraint.
        diff = P[:, None, :] - P[None, :, :]
        D = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-18)
        dv = D[iu]
        # soft-min weights (repel)
        wmin = np.exp((1.0 / dv - 1.0 / dv.min()) / temp)
        wmin /= wmin.sum()
        wmin[wmin < 1e-6] = 0.0
        wmin /= wmin.sum()
        # soft-max weights (attract)
        wmax = np.exp((dv - dv.max()) / temp)
        wmax /= wmax.sum()
        wmax[wmax < 1e-6] = 0.0
        wmax /= wmax.sum()
        u = diff[ii, jj] / (dv[:, None] + 1e-12)
        w = wmin - lam * wmax
        G = np.zeros_like(P)
        np.add.at(G, ii, w[:, None] * u)
        np.add.at(G, jj, -w[:, None] * u)
        return G

    # Icosahedron vertices (12 points) for structured seeds
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ], dtype=float)
    ico /= np.linalg.norm(ico, axis=1, keepdims=True)

    def structured_seed(k):
        # Structured seed families (good basins for n=14):
        #  fam 0: 12 icosahedron vertices (jittered) + 2 random points
        #  fam 1: icosahedron + one antipodal pair along a random axis
        #  fam 2: 7 random antipodal pairs (dmax is a diameter by design)
        P = np.empty((n, d))
        m = min(12, n)
        fam = k % 3
        if fam == 0:
            P[:m] = ico[:m] + 0.02 * rng.standard_normal((m, d))
            P[m:] = rng.standard_normal((n - m, d))
        elif fam == 1:
            P[:m] = ico[:m] + 0.02 * rng.standard_normal((m, d))
            ax = rng.standard_normal(d)
            ax /= np.linalg.norm(ax)
            P[m] = ax
            if m + 1 < n:
                P[m + 1] = -ax
        else:
            V = rng.standard_normal((n // 2, d))
            V /= np.linalg.norm(V, axis=1, keepdims=True)
            P[0::2] = V
            P[1::2] = -V
            if n % 2:
                P[-1] = rng.standard_normal(d)
        return project_sphere(P)

    # ---- Stage 0: gradient ascent from many starts ----
    candidates = []
    n_random = 60
    n_struct = 30
    steps = 700
    for s in range(n_random + n_struct):
        P = structured_seed(s) if s >= n_random else project_sphere(
            rng.standard_normal((n, d)))
        lr = 0.02
        for it in range(steps):
            temp = 0.05 * (0.995 ** it) + 0.005
            P = project_sphere(P + lr * step_grad(P, temp))
        candidates.append((ratio_sq(P), P))

    candidates.sort(key=lambda t: -t[0])

    # ---- Stage 1: direct constrained NLP refinement via SLSQP ----
    # Exact reformulation of the ratio objective: introduce auxiliary
    # variable t and solve  max t  s.t.  d(i,j)^2 >= t  and  d(i,j)^2 <= 1
    # over the 42 point coordinates + t. SLSQP converges to KKT points
    # of the true ratio objective (unlike soft-min heuristics which
    # stall near saddle regions). Analytic Jacobians are supplied.
    npair = len(ii)

    def slsqp_refine(P):
        P = normalize(P)
        dv2 = np.sum((P[ii] - P[jj]) ** 2, axis=1)
        t0 = dv2.min()
        z0 = np.concatenate([P.ravel(), [t0]])

        def cons(z):
            Q = z[:n * d].reshape(n, d)
            t = z[-1]
            dv2 = np.sum((Q[ii] - Q[jj]) ** 2, axis=1)
            return np.concatenate([dv2 - t, 1.0 - dv2])

        def cons_jac(z):
            # Vectorized analytic Jacobian (np.add.at handles the
            # duplicate coordinate indices correctly).
            Q = z[:n * d].reshape(n, d)
            diff = Q[ii] - Q[jj]
            J = np.zeros((2 * npair, n * d + 1))
            J[:npair, -1] = -1.0
            rows = np.repeat(np.arange(npair), d)
            ci = (ii[:, None] * d + np.arange(d)).ravel()
            cj = (jj[:, None] * d + np.arange(d)).ravel()
            df = diff.ravel()
            np.add.at(J[:npair], (rows, ci), 2.0 * df)
            np.add.at(J[:npair], (rows, cj), -2.0 * df)
            np.add.at(J[npair:], (rows, ci), -2.0 * df)
            np.add.at(J[npair:], (rows, cj), 2.0 * df)
            return J

        def obj(z):
            return -z[-1]

        def obj_grad(z):
            g = np.zeros(n * d + 1)
            g[-1] = -1.0
            return g

        res = _slsqp_min(
            obj, z0, jac=obj_grad, method='SLSQP',
            constraints=[{'type': 'ineq', 'fun': cons, 'jac': cons_jac}],
            options={'maxiter': 300, 'ftol': 1e-12})
        Q = normalize(res.x[:n * d].reshape(n, d))
        return ratio_sq(Q), Q

    def free_polish(P, iters, seed):
        # Adaptive-sigma polish in free space (safety net / fine tuning):
        # dmax renormalized to 1 each step.
        r3 = np.random.default_rng(seed)
        P = normalize(P)
        best = ratio_sq(P)
        sigma = 0.004
        for _ in range(iters):
            Q = normalize(P + sigma * r3.standard_normal(P.shape))
            rq = ratio_sq(Q)
            if rq > best:
                best = rq
                P = Q
                sigma = min(sigma * 1.3, 0.02)
            else:
                sigma *= 0.9995
                if sigma < 1e-6:
                    sigma = 1e-6
        return best, P

    # ---- Stage 2: SLSQP-refine top gradient candidates, then polish ----
    best_ratio = -1.0
    best_P = None
    for idx, (r0, P0) in enumerate(candidates[:6]):
        Q0 = normalize(P0)
        seed_r = ratio_sq(Q0)
        try:
            r, Q = slsqp_refine(Q0)
        except Exception:
            r, Q = seed_r, Q0
        if r < seed_r:  # fall back if SLSQP failed / infeasible
            r, Q = seed_r, Q0
        if r > best_ratio:
            best_ratio = r
            best_P = Q.copy()
        # short free-space polish, then a second SLSQP pass from there
        r1, Q1 = free_polish(Q, 3000, 1000 + idx)
        if r1 > best_ratio:
            best_ratio = r1
            best_P = Q1.copy()
        try:
            r2, Q2 = slsqp_refine(Q1)
        except Exception:
            r2, Q2 = -1.0, Q1
        if r2 > best_ratio:
            best_ratio = r2
            best_P = Q2.copy()

    # ---- Stage 2.5: multi-restart perturbed SLSQP around the best ----
    # Escape local optima of the nonconvex ratio landscape by perturbing
    # the current best configuration and re-solving the EXACT constrained
    # reformulation (max t s.t. d^2 >= t, d^2 <= 1) from each perturbed
    # start, followed by a short free-space polish and a second exact
    # pass. All acceptance is on the true ratio objective.
    bh_rng = np.random.default_rng(2024)
    for trial in range(30):
        sig = 0.002 + 0.004 * (trial % 8)
        Pt = normalize(best_P + sig * bh_rng.standard_normal(best_P.shape))
        try:
            rt, Qt = slsqp_refine(Pt)
        except Exception:
            continue
        if rt > best_ratio:
            best_ratio = rt
            best_P = Qt.copy()
        rt2, Qt2 = free_polish(Qt, 2000, 3000 + trial)
        if rt2 > best_ratio:
            best_ratio = rt2
            best_P = Qt2.copy()
        try:
            rt3, Qt3 = slsqp_refine(Qt2)
        except Exception:
            continue
        if rt3 > best_ratio:
            best_ratio = rt3
            best_P = Qt3.copy()

    # ---- Stage 2.6: contact-graph root solving (scipy.optimize.root) ----
    # At a true optimum of (dmin/dmax)^2, many pairwise distances equal
    # dmin exactly (the contact graph) and several equal dmax (diameter
    # pairs). Exploit this KKT structure directly: extract the active
    # sets from the incumbent at several tolerance levels, then solve
    # the polynomial system
    #     d(i,j)^2 = t   for contact edges,
    #     d(i,j)^2 = 1   for diameter edges,
    #     gauge fixing: p0 = origin (3 eqs), p1_z = 0, p2_y = 0 (6 eqs)
    # as a nonlinear least-squares root problem (method 'lm', which
    # handles over/under-determined systems) with randomized starts.
    # Converged solutions satisfying all inactive-edge inequalities
    # (d^2 >= t, d^2 <= 1) are exact stationary candidates of the
    # ratio objective; keep any that improve the true ratio, then
    # polish with the exact SLSQP reformulation.
    from scipy.optimize import root as _root

    def contact_root_refine(P, ctol, dtol, r3, ntries=8):
        """Solve the contact-graph polynomial system for the active
        sets implied by the incumbent P (thresholds ctol/dtol)."""
        P = normalize(P)
        dv2 = np.sum((P[ii] - P[jj]) ** 2, axis=1)
        dmin2, dmax2 = dv2.min(), dv2.max()
        ci = np.where(dv2 <= dmin2 * (1.0 + ctol))[0]
        di = np.where(dv2 >= dmax2 * (1.0 - dtol))[0]
        # Need enough equations vs. 42 coords + t unknowns to pin a
        # rigid configuration; skip degenerate active sets.
        if len(ci) < 10 or len(ci) + len(di) + 6 < 20:
            return -1.0, None
        best_r, best_Q = -1.0, None
        for k in range(ntries):
            # first attempt: incumbent itself; then randomized starts
            P0 = normalize(P + (0.01 * r3.standard_normal(P.shape)
                                if k else 0.0))
            dv20 = np.sum((P0[ii] - P0[jj]) ** 2, axis=1)
            z0 = np.concatenate([P0.ravel(), [dv20.min()]])

            def fun(z):
                Q = z[:n * d].reshape(n, d)
                t = z[-1]
                dd = np.sum((Q[ii] - Q[jj]) ** 2, axis=1)
                # contact residuals, diameter residuals, gauge fixing
                return np.concatenate([dd[ci] - t, dd[di] - 1.0,
                                       Q[0].ravel(), [Q[1, 2]],
                                       [Q[2, 1]]])

            try:
                sol = _root(fun, z0, method='lm',
                            options={'maxiter': 400, 'ftol': 1e-12})
            except Exception:
                continue
            Q = normalize(sol.x[:n * d].reshape(n, d))
            dd = np.sum((Q[ii] - Q[jj]) ** 2, axis=1)
            # verify inactive-edge inequalities of the KKT system
            if dd.min() < sol.x[-1] - 1e-6 or dd.max() > 1.0 + 1e-6:
                continue
            rq = ratio_sq(Q)
            if rq > best_r:
                best_r, best_Q = rq, Q
        return best_r, best_Q

    # Contact-graph enumeration across multiple top candidates and a
    # finer tolerance sweep: each distinct candidate carries a distinct
    # contact-graph topology, so refining from the top-3 configurations
    # effectively enumerates several plausible KKT graph structures.
    cr_rng = np.random.default_rng(555)
    bases = [best_P.copy()]
    for c in candidates[:6]:
        Qc0 = normalize(c[1])
        if all(np.min(np.sum((Qc0 - b) ** 2, axis=1)) > 1e-3 or True
               for b in bases):
            # keep geometrically distinct bases (ratio-based dedup)
            if not any(abs(ratio_sq(Qc0) - ratio_sq(b)) < 1e-9
                       for b in bases):
                bases.append(Qc0)
    for base in bases[:3]:
        for ct in (0.005, 0.01, 0.02, 0.03, 0.06, 0.10, 0.15):
            for dt in (0.005, 0.01, 0.03, 0.05):
                try:
                    rc, Qc = contact_root_refine(base, ct, dt, cr_rng, 6)
                except Exception:
                    continue
                if rc > best_ratio and Qc is not None:
                    best_ratio = rc
                    best_P = Qc.copy()
                # exact SLSQP polish on the root solution (KKT refinement)
                if Qc is not None:
                    try:
                        rcs, Qcs = slsqp_refine(Qc)
                        if rcs > best_ratio:
                            best_ratio = rcs
                            best_P = Qcs.copy()
                    except Exception:
                        pass
                    # randomized-restart root solve seeded by the polished
                    # solution: re-derive the exact contact system from the
                    # refined active sets for a second-order KKT correction
                    try:
                        rc2, Qc2 = contact_root_refine(best_P, 0.01, 0.02,
                                                       cr_rng, 4)
                        if rc2 > best_ratio and Qc2 is not None:
                            best_ratio = rc2
                            best_P = Qc2.copy()
                            rcs2, Qcs2 = slsqp_refine(Qc2)
                            if rcs2 > best_ratio:
                                best_ratio = rcs2
                                best_P = Qcs2.copy()
                    except Exception:
                        pass

    # ---- Stage 3: cyclic-annealed polish + exact SLSQP on overall best ----
    # Cyclic sigma resets keep the long polish from freezing after the
    # adaptive sigma collapses; each cycle re-explores at coarse scale
    # then refines, all accepted on the true ratio objective.
    def cyclic_polish(P, cycles, iters_per, seed):
        r4 = np.random.default_rng(seed)
        P = normalize(P)
        best = ratio_sq(P)
        for _ in range(cycles):
            sigma = 0.004
            for _ in range(iters_per):
                Q = normalize(P + sigma * r4.standard_normal(P.shape))
                rq = ratio_sq(Q)
                if rq > best:
                    best = rq
                    P = Q
                    sigma = min(sigma * 1.3, 0.02)
                else:
                    sigma *= 0.999
                    if sigma < 2e-6:
                        sigma = 2e-6
        return best, P

    _, P = cyclic_polish(best_P, 6, 5000, 7)
    try:
        rf, Qf = slsqp_refine(normalize(P))
        if rf > ratio_sq(normalize(P)):
            P = Qf
    except Exception:
        pass
    _, P = free_polish(P, 15000, 77)
    try:
        rf, Qf = slsqp_refine(normalize(P))
        if rf > ratio_sq(normalize(P)):
            P = Qf
    except Exception:
        pass
    # final micro-polish at very small sigma only
    r5 = np.random.default_rng(777)
    P = normalize(P)
    best = ratio_sq(P)
    sigma = 1e-3
    for _ in range(8000):
        Q = normalize(P + sigma * r5.standard_normal(P.shape))
        rq = ratio_sq(Q)
        if rq > best:
            best = rq
            P = Q
        else:
            sigma *= 0.9999
            if sigma < 1e-7:
                sigma = 1e-7

    return np.asarray(normalize(P), dtype=float)


# EVOLVE-BLOCK-END
