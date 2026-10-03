# EVOLVE-BLOCK-START
import numpy as np


def _ratio(pts):
    """(dmin/dmax)^2 of a point set."""
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    iu = np.triu_indices(len(pts), 1)
    dm = d[iu]
    return dm.min() / dm.max()


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Constructs 14 points in 3D maximizing (dmin/dmax)^2.

    Approach: repeatedly run a physical "electron repulsion" relaxation on
    the unit sphere (repulsive ~1/d^p forces with increasing stiffness p,
    re-normalizing points onto the sphere after each step). This spreads
    points like charges on a conductor, which is an excellent heuristic for
    the Tammes-like problem behind this ratio. Multiple random restarts;
    the configuration with the best actual dmin/dmax ratio is returned,
    followed by multi-basin simulated annealing on the true objective, a
    soft log-sum-exp SLSQP polish, and a final exact epigraph SLSQP stage
    that fixes dmax = 1 via constraints and maximizes the squared minimum
    distance directly (analytic Jacobians, perturbation restarts).

    Returns
        points: np.ndarray of shape (14, 3)
    """
    n, dim = 14, 3
    rng = np.random.default_rng(12345)

    def relax(pts, steps=250, lr=0.02):
        for p in (1.0, 2.0, 4.0, 8.0):
            for _ in range(steps):
                diff = pts[:, None, :] - pts[None, :, :]
                dist = np.sqrt((diff ** 2).sum(-1) + 1e-12)
                inv = dist ** (-(p + 1))
                np.fill_diagonal(inv, 0.0)
                force = (diff * inv[:, :, None]).sum(axis=1)
                nrm = np.linalg.norm(force, axis=1, keepdims=True)
                nrm[nrm == 0] = 1.0
                pts = pts + lr * force / nrm
                pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        return pts

    best_pts, best = None, -1.0

    def consider(pts):
        nonlocal best, best_pts
        r = _ratio(pts)
        if r > best:
            best, best_pts = r, pts.copy()

    # Structured seeds: antiprisms (two staggered m-gons at z = ±h) are
    # natural near-optimal sphere configurations for even n; try several
    # aspect ratios for m = 6 (plus 2 poles) and m = 7.
    seeds = []
    for m, poles in ((6, True), (7, False)):
        for h in np.linspace(0.15, 0.65, 11):
            r = np.sqrt(max(1.0 - h * h, 1e-6))
            ang = 2.0 * np.pi / m
            pts = [[r * np.cos(i * ang), r * np.sin(i * ang), h]
                   for i in range(m)]
            pts += [[r * np.cos((i + 0.5) * ang),
                     r * np.sin((i + 0.5) * ang), -h]
                    for i in range(m)]
            if poles:
                pts = [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]] + pts
            pts = np.array(pts, dtype=float)
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
            seeds.append(pts)

    # Seeding is done below into the candidate pool.

    # Simulated annealing on the true objective (dmin/dmax)^2 over
    # UNCONSTRAINED coordinates (the ratio is scale/translation
    # invariant). Unlike the previous greedy hill-climb, worsening moves
    # are accepted with probability exp((r - cur) / T) so the search can
    # escape the local optimum; per-point adaptive step sizes are kept.
    def anneal(pts, iters, step0, t0, t1, seed):
        rng = np.random.default_rng(seed)
        pts = pts.copy()
        cur = _ratio(pts)
        loc_pts, loc = pts.copy(), cur
        steps = np.full(n, step0)
        for t in range(iters):
            frac = t / iters
            T = t0 * (t1 / t0) ** frac
            i = rng.integers(n)
            old = pts[i].copy()
            cand = old + steps[i] * rng.normal(size=dim)
            if np.linalg.norm(cand) < 1e-9:
                continue
            pts[i] = cand
            r = _ratio(pts)
            if r >= cur or rng.random() < np.exp((r - cur) / max(T, 1e-12)):
                cur = r
                steps[i] = min(steps[i] * 1.2, step0 * 4)
                if r > loc:
                    loc, loc_pts = r, pts.copy()
            else:
                pts[i] = old
                steps[i] = max(steps[i] * 0.8, step0 * 0.01)
        return loc, loc_pts

    # Keep a pool of good candidates from distinct relaxation basins;
    # annealing from several distinct starts escapes single-basin local
    # optima far better than annealing one seed.
    pool = []

    def consider_pool(pts):
        pool.append((_ratio(pts), pts.copy()))

    for pts0 in seeds:
        consider_pool(relax(pts0.copy()))

    for _ in range(200):
        pts = rng.normal(size=(n, dim))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        consider_pool(relax(pts))

    pool.sort(key=lambda t: -t[0])
    best, best_pts = pool[0]

    # Anneal from the top few distinct seeds, then refine the overall
    # best with a coarse-to-fine schedule and a final cold polish.
    seed = 777
    for _, pts0 in pool[:6]:
        for iters, s0, t0 in ((80000, 0.05, 0.004),
                              (80000, 0.02, 0.0015),
                              (60000, 0.005, 0.0004)):
            seed += 13
            cur, pts = anneal(pts0, iters, s0, t0, 1e-7, seed)
            if cur > best:
                best, best_pts = cur, pts.copy()
            pts0 = pts

    # Cold (T -> 0) greedy polish with tiny adaptive steps; several
    # independent restarts with progressively smaller step sizes
    # squeeze out the last digits of the ratio.
    for s0, sd in ((0.003, 424242), (0.0015, 11111),
                   (0.0008, 22222), (0.0004, 33333)):
        cur, pts = anneal(best_pts, 80000, s0, 1e-9, 1e-12, sd)
        if cur > best:
            best, best_pts = cur, pts.copy()

    # Deterministic gradient polish: SLSQP on a smooth log-sum-exp
    # surrogate of (dmin/dmax)^2, with temperature sharpened over
    # rounds so soft-min/soft-max converge to the hard min/max. This
    # escapes the stochastic stagnation of the annealer.
    from scipy.optimize import minimize
    from scipy.special import logsumexp

    iu = np.triu_indices(n, 1)

    def smooth_obj(x, tau):
        p = x.reshape(n, dim)
        d2 = ((p[:, None, :] - p[None, :, :]) ** 2).sum(-1)[iu]
        smin = -logsumexp(-d2 / tau) * tau
        smax = logsumexp(d2 / tau) * tau
        return -smin / smax

    for tau in (0.05, 0.02, 0.008, 0.003, 0.001):
        res = minimize(
            smooth_obj, best_pts.ravel(), args=(tau,), method='SLSQP',
            options={'maxiter': 400, 'ftol': 1e-14})
        cand = res.x.reshape(n, dim)
        r = _ratio(cand)
        if r > best:
            best, best_pts = r, cand.copy()

    # Final tiny cold anneal on top of the SLSQP output.
    cur, pts = anneal(best_pts, 60000, 0.0004, 1e-10, 1e-13, 90909)
    if cur > best:
        best, best_pts = cur, pts.copy()

    # Exact epigraph SLSQP: normalize (centroid 0, dmax = 1) and solve
    # the true problem directly — variables are the 42 coordinates plus
    # an epigraph variable t, constraints d_ij^2 - t >= 0 and
    # 1 - d_ij^2 >= 0 for all 91 pairs, objective maximize t. At
    # feasibility t IS the exact squared ratio (dmin/dmax)^2, so this
    # converges onto the true optimum far tighter than the soft
    # surrogate. Analytic constraint Jacobians (vectorized scatter over
    # the pairs) make SLSQP fast and precise.
    iup = np.array(np.triu_indices(n, 1)).T
    m = len(iup)
    nvar = n * dim + 1
    rows = np.arange(m)

    def epigraph(pts):
        P = pts - pts.mean(0)
        d = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
        P = P / d.max()
        dif = P[iup[:, 0]] - P[iup[:, 1]]
        z = np.concatenate([P.ravel(), [(dif * dif).sum(1).min()]])

        def cons(z):
            P = z[:n * dim].reshape(n, dim)
            dif = P[iup[:, 0]] - P[iup[:, 1]]
            dd = (dif * dif).sum(1)
            return np.concatenate([dd - z[-1], 1.0 - dd])

        def cjac(z):
            P = z[:n * dim].reshape(n, dim)
            dif = P[iup[:, 0]] - P[iup[:, 1]]
            J = np.zeros((2 * m, nvar))
            for d_ in range(dim):
                J[rows, iup[:, 0] * dim + d_] = 2 * dif[:, d_]
                J[rows, iup[:, 1] * dim + d_] = -2 * dif[:, d_]
                J[m + rows, iup[:, 0] * dim + d_] = -2 * dif[:, d_]
                J[m + rows, iup[:, 1] * dim + d_] = 2 * dif[:, d_]
            J[:m, -1] = -1.0
            return J

        for _ in range(4):
            res = minimize(
                lambda z: -z[-1], z,
                jac=lambda z: np.concatenate([np.zeros(n * dim), [-1.0]]),
                constraints=[{'type': 'ineq', 'fun': cons, 'jac': cjac}],
                method='SLSQP', options={'maxiter': 3000, 'ftol': 1e-18})
            z = res.x
            # Re-normalize (centroid 0, dmax = 1) between rounds to
            # restore constraint conditioning; the epigraph variable is
            # reset to the current squared min distance.
            P = z[:n * dim].reshape(n, dim)
            P = P - P.mean(0)
            d = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
            P = P / d.max()
            dif = P[iup[:, 0]] - P[iup[:, 1]]
            z = np.concatenate([P.ravel(), [(dif * dif).sum(1).min()]])
        P2 = z[:n * dim].reshape(n, dim)
        return _ratio(P2), P2

    r, pts = epigraph(best_pts)
    if r > best:
        best, best_pts = r, pts.copy()

    # The exact epigraph solver converges onto the true optimum far
    # tighter than the annealer, so also run it from the runner-up
    # relaxation basins: their exact optima are sometimes higher than
    # the incumbent's, and only genuine improvements are kept.
    for _, pts0 in pool[1:5]:
        r, pts = epigraph(pts0)
        if r > best:
            best, best_pts = r, pts.copy()

    # Basin-hopping tail: alternate tiny cold anneals with the exact
    # epigraph SLSQP, and restart the epigraph from multi-scale
    # Gaussian perturbations of the incumbent (fixed seed), from
    # coarse (0.02) down to fine (0.002). Only improvements are
    # accepted, so this can only squeeze the ratio upward.
    rng2 = np.random.default_rng(2024)
    for rnd in range(16):
        cur, pts = anneal(best_pts, 40000, 0.0006, 1e-10, 1e-13, 500 + rnd)
        if cur > best:
            best, best_pts = cur, pts.copy()
        r, pts = epigraph(best_pts)
        if r > best:
            best, best_pts = r, pts.copy()
        for scale in (0.02, 0.008, 0.003, 0.002, 0.001, 0.0005,
                      0.0002, 0.0001):
            for _ in range(3):
                pert = best_pts + scale * rng2.normal(size=best_pts.shape)
                r, pts = epigraph(pert)
                if r > best:
                    best, best_pts = r, pts.copy()

    # Final high-precision lock-in: one long epigraph solve on the
    # incumbent with many re-normalization rounds. Each round resets
    # the centroid to 0 and dmax to 1 and re-seeds the epigraph
    # variable with the current squared min distance, restoring
    # constraint conditioning so SLSQP can grind the active pairs
    # (those pinned at dmin and dmax) to machine precision. Only
    # strict improvements are kept, so this stage is a pure squeeze.
    def epigraph_long(pts, rounds=8):
        P = pts - pts.mean(0)
        d = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
        P = P / d.max()
        dif = P[iup[:, 0]] - P[iup[:, 1]]
        z = np.concatenate([P.ravel(), [(dif * dif).sum(1).min()]])
        for _ in range(rounds):
            res = minimize(
                lambda z: -z[-1], z,
                jac=lambda z: np.concatenate([np.zeros(n * dim), [-1.0]]),
                constraints=[{'type': 'ineq',
                             'fun': lambda z: np.concatenate([
                                 ((z[:n * dim].reshape(n, dim)[iup[:, 0]]
                                   - z[:n * dim].reshape(n, dim)[iup[:, 1]])
                                  ** 2).sum(1) - z[-1],
                                 1.0 - ((z[:n * dim].reshape(n, dim)[iup[:, 0]]
                                         - z[:n * dim].reshape(n, dim)[iup[:, 1]])
                                        ** 2).sum(1)])}],
                method='SLSQP', options={'maxiter': 6000, 'ftol': 1e-20})
            z = res.x
            P = z[:n * dim].reshape(n, dim)
            P = P - P.mean(0)
            d = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
            P = P / d.max()
            dif = P[iup[:, 0]] - P[iup[:, 1]]
            z = np.concatenate([P.ravel(), [(dif * dif).sum(1).min()]])
        P2 = z[:n * dim].reshape(n, dim)
        return _ratio(P2), P2

    r, pts = epigraph_long(best_pts)
    if r > best:
        best, best_pts = r, pts.copy()

    return best_pts


# EVOLVE-BLOCK-END
