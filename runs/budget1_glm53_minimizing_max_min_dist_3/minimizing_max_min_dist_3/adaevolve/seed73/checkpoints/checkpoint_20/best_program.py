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
    for trial in range(15):
        sig = 0.005 + 0.015 * (trial % 4)
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

    # ---- Stage 3: final long polish + exact SLSQP on overall best ----
    _, P = free_polish(best_P, 30000, 7)
    try:
        rf, Qf = slsqp_refine(normalize(P))
        if rf > ratio_sq(normalize(P)):
            P = Qf
    except Exception:
        pass
    _, P = free_polish(P, 10000, 77)

    return np.asarray(normalize(P), dtype=float)


# EVOLVE-BLOCK-END
