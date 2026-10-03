# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

N = 14
D = 3
_IU = np.triu_indices(N, 1)


def pair_dists(P):
    Dm = np.linalg.norm(P[:, None] - P[None, :], axis=-1)
    return Dm[_IU]


def ratio(P):
    d = pair_dists(P)
    return d.min() / d.max()


def d6_family(R, h, z0):
    """Two staggered hexagonal rings (30 deg offset) at z=+-h plus two poles."""
    th = np.arange(6) * (np.pi / 3.0)
    r1 = np.stack([R * np.cos(th), R * np.sin(th), np.full(6, h)], axis=1)
    r2 = np.stack([R * np.cos(th + np.pi / 6), R * np.sin(th + np.pi / 6),
                   np.full(6, -h)], axis=1)
    poles = np.array([[0.0, 0.0, z0], [0.0, 0.0, -z0]])
    return np.vstack([r1, r2, poles])


def soft_ratio_obj(z, tau):
    R, h, z0 = z
    if R <= 0 or h <= 0 or z0 <= h:
        return 1e6
    P = d6_family(R, h, z0)
    d = pair_dists(P)
    # maximize soft-min - soft-max (negative for minimize)
    smin = -tau * logsumexp(-d / tau)
    smax = tau * logsumexp(d / tau)
    return -(smin - smax)


def repulsion(P0, iters=120, step=0.05):
    """Force-based relaxation: push each point away from its nearest neighbor."""
    P = P0.copy()
    for _ in range(iters):
        P = P - P.mean(axis=0)
        nrm = np.linalg.norm(P, axis=1, keepdims=True)
        P = P / np.maximum(nrm, 1e-12)
        Dm = np.linalg.norm(P[:, None] - P[None, :], axis=-1)
        np.fill_diagonal(Dm, np.inf)
        j = Dm.argmin(axis=1)
        disp = P - P[j]
        lens = np.maximum(np.linalg.norm(disp, axis=1, keepdims=True), 1e-12)
        P = P + step * disp / lens
        step *= 0.97
    P = P - P.mean(axis=0)
    return P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)


def polish(P0, taus=(0.05, 0.02, 0.008, 0.003, 0.001, 0.0004)):
    """Annealed soft-min/soft-max L-BFGS-B polish, scale-invariant objective."""
    P = (P0 - P0.mean(axis=0)).copy()
    P = P / pair_dists(P).max()
    x = P.ravel()
    IJ = np.array([(i, j) for i in range(N) for j in range(i + 1, N)])

    def obj(z, tau=0.01):
        X = z.reshape(N, D)
        diff = X[IJ[:, 0]] - X[IJ[:, 1]]
        d2 = np.sum(diff * diff, axis=1)
        return tau * (logsumexp(-d2 / tau) + logsumexp(d2 / tau))

    for tau in taus:
        res = minimize(lambda z, t=tau: obj(z, t), x, method="L-BFGS-B",
                       options={"maxiter": 300, "ftol": 1e-14, "gtol": 1e-12})
        if np.isfinite(res.x).all():
            x = res.x
    X = x.reshape(N, D)
    X = X - X.mean(axis=0)
    X = X / pair_dists(X).max()
    return X


def icosa_plus_poles():
    """Icosahedron vertices (12) + 2 points near opposite z poles."""
    phi = (1 + np.sqrt(5)) / 2
    base = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ], dtype=float)
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    extra = np.array([[0.0, 0.0, 1.05], [0.1, -0.1, -1.05]])
    P = np.vstack([base, extra])
    P = P - P.mean(axis=0)
    return P / pair_dists(P).max()


def greedy_adaptive(P0, iters=600):
    """Greedy coordinate descent on the exact ratio with adaptive per-point steps.

    Points involved in the current min-distance or max-distance pair get
    larger steps (0.05-0.1); other points get small steps (0.01). This
    concentrates effort on the binding constraints.
    """
    P = P0 - P0.mean(axis=0)
    P = P / pair_dists(P).max()
    r_best = ratio(P)
    iu = np.array([(i, j) for i in range(N) for j in range(i + 1, N)])
    # per-point step sizes
    step = np.full(N, 0.01)
    for _ in range(iters):
        d = pair_dists(P)
        ia, ib = iu[d.argmin()]
        ja, jb = iu[d.argmax()]
        # adaptive per-point steps: binding points get big steps
        step[:] = 0.01
        for b in (ia, ib):
            step[b] = 0.1
        for b in (ja, jb):
            step[b] = max(step[b], 0.05)
        improved = False
        for i in range(N):
            for ax in range(D):
                for sgn in (1, -1):
                    Q = P.copy()
                    Q[i, ax] += sgn * step[i]
                    Q = Q - Q.mean(axis=0)
                    Q = Q / pair_dists(Q).max()
                    r = ratio(Q)
                    if r > r_best + 1e-9:
                        P, r_best, improved = Q, r, True
        if not improved:
            step *= 0.5
            if step.max() < 1e-6:
                break
    return P, r_best


def min_max_dist_dim3_14() -> np.ndarray:
    rng = np.random.default_rng(7)

    # --- Candidate A: D6 parametric family (previous best basin) ---
    best_param, best_r = None, -1.0
    for h in np.linspace(0.15, 1.4, 20):
        for z0 in np.linspace(max(h, 0.2) + 0.05, 2.6, 36):
            P = d6_family(1.0, h, z0)
            r = ratio(P)
            if r > best_r:
                best_r, best_param = r, (1.0, h, z0)
    res = minimize(lambda z: soft_ratio_obj(z, 0.002), np.array(best_param),
                   method="Nelder-Mead",
                   options={"xatol": 1e-10, "fatol": 1e-14, "maxiter": 2000})
    if np.isfinite(res.x).all():
        R, h, z0 = res.x
        if R > 0 and h > 0 and z0 > h:
            r = ratio(d6_family(R, h, z0))
            if r > best_r:
                best_r, best_param = r, (R, h, z0)
    P0 = d6_family(*best_param)
    P0 = (P0 - P0.mean(axis=0)) / pair_dists(P0).max()
    A = polish(P0)
    rA = ratio(A)
    if rA > best_r:
        best_r, best = rA, A.copy()
    else:
        best = P0.copy()

    # --- Candidate B: icosahedron + 2 poles (best known basin) ---
    B = icosa_plus_poles()
    B = polish(B)
    rB = ratio(B)
    if rB > best_r:
        best_r, best = rB, B.copy()

    # --- Candidate C: jittered icosahedron restarts ---
    base_ico = icosa_plus_poles()
    for k in range(3):
        mag = 0.08 * (0.6 ** k)
        Q0 = base_ico + mag * rng.normal(size=base_ico.shape)
        Q = polish(Q0)
        r = ratio(Q)
        if r > best_r + 1e-12:
            best_r, best = r, Q.copy()

    # --- Candidate D: random sphere starts with repulsion ---
    for _ in range(2):
        Q0 = rng.normal(size=(N, D))
        Q0 /= np.maximum(np.linalg.norm(Q0, axis=1, keepdims=True), 1e-12)
        Q = repulsion(Q0)
        Q = polish(Q)
        r = ratio(Q)
        if r > best_r + 1e-12:
            best_r, best = r, Q.copy()

    # --- High-precision soft polish of the winner ---
    Q = polish(best, taus=(0.001, 0.0003, 0.0001))
    if ratio(Q) > best_r:
        best, best_r = Q, ratio(Q)

    # --- Stage 6: targeted pair/triple hill-climb on min- and max-distance pairs ---
    def renorm(P):
        P = P - P.mean(axis=0)
        return P / pair_dists(P).max()

    def hillclimb(P0, iters=400, step0=0.02):
        P = renorm(P0.copy())
        r_best = ratio(P)
        step = step0
        iu = np.array([(i, j) for i in range(N) for j in range(i + 1, N)])
        for it in range(iters):
            d = pair_dists(P)
            ia, ib = iu[d.argmin()]
            ja, jb = iu[d.argmax()]
            improved = False
            u_min = P[ia] - P[ib]
            n_min = np.linalg.norm(u_min)
            if n_min > 1e-12:
                u_min /= n_min
            u_max = P[ja] - P[jb]
            n_max = np.linalg.norm(u_max)
            if n_max > 1e-12:
                u_max /= n_max
            for mag in (step, step / 2, step / 4):
                # move 1: push min pair apart
                Q = P.copy()
                Q[ia] += mag * u_min
                Q[ib] -= mag * u_min
                Q = renorm(Q)
                r = ratio(Q)
                if r > r_best + 1e-14:
                    P, r_best, improved = Q, r, True
                    break
                # move 2: pull max pair inward
                Q = P.copy()
                Q[ja] -= mag * u_max
                Q[jb] += mag * u_max
                Q = renorm(Q)
                r = ratio(Q)
                if r > r_best + 1e-14:
                    P, r_best, improved = Q, r, True
                    break
                # move 3 (triple): do both simultaneously
                if (ia, ib) != (ja, jb):
                    Q = P.copy()
                    Q[ia] += mag * u_min
                    Q[ib] -= mag * u_min
                    Q[ja] -= mag * u_max
                    Q[jb] += mag * u_max
                    Q = renorm(Q)
                    r = ratio(Q)
                    if r > r_best + 1e-14:
                        P, r_best, improved = Q, r, True
                        break
            if not improved:
                step *= 0.6
                if step < 1e-7:
                    break
        return P, r_best

    # --- Adaptive greedy polish on exact ratio ---
    Q, r = greedy_adaptive(best)
    if r > best_r + 1e-14:
        best, best_r = Q, r
    # one more fine pass from the improved configuration
    Q, r = greedy_adaptive(best, iters=300)
    if r > best_r + 1e-14:
        best, best_r = Q, r

    points = np.asarray(best, dtype=float)
    assert points.shape == (N, D) and np.isfinite(points).all()
    assert pair_dists(points).max() > 0
    return points
# EVOLVE-BLOCK-END