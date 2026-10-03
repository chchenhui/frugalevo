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


def ring_poles_family(n_ring, h, z0, twist=0.0):
    """Single ring of n_ring points at z=h plus two poles at +-z0."""
    th = np.arange(n_ring) * (2 * np.pi / n_ring) + twist
    ring = np.stack([np.cos(th), np.sin(th), np.full(n_ring, h)], axis=1)
    poles = np.array([[0.0, 0.0, z0], [0.0, 0.0, -z0]])
    return np.vstack([ring, poles])


def hept_antiprism_family(R, h):
    """Heptagonal antiprism: two staggered 7-rings at z=+-h."""
    th = np.arange(7) * (2 * np.pi / 7)
    r1 = np.stack([R * np.cos(th), R * np.sin(th), np.full(7, h)], axis=1)
    r2 = np.stack([R * np.cos(th + np.pi / 7), R * np.sin(th + np.pi / 7),
                   np.full(7, -h)], axis=1)
    return np.vstack([r1, r2])


def _batch_ratios(cands, dmax0):
    """Ratios dmin/dmax for a stack of candidate point sets, vectorized."""
    # cands: (m, N, D)
    diff = cands[:, :, None, :] - cands[:, None, :, :]
    D2 = (diff * diff).sum(-1)
    m = D2.shape[0]
    dd = np.sqrt(np.maximum(D2[:, _IU[0], _IU[1]], 1e-24))
    return dd.min(axis=1) / dd.max(axis=1)


def climb(P, steps=(0.02, 0.006, 0.002, 6e-4, 2e-4, 8e-5, 3e-5)):
    """Steepest-ascent hill-climb on the true ratio with coordinated pair moves.

    All candidate moves (min-pair separation along u, perpendicular shears
    of the min pair, max-pair contraction, single-coordinate nudges) are
    evaluated in one vectorized batch and the best-improving move is taken.
    """
    P = (P - P.mean(axis=0)).copy()
    P = P / pair_dists(P).max()
    best_r = ratio(P)
    for step in steps:
        max_acc = 4000 if step > 1e-3 else 1500
        n_acc = 0
        while n_acc < max_acc:
            d = pair_dists(P)
            imin = int(np.argmin(d))
            imax = int(np.argmax(d))
            i0, j0 = int(_IU[0][imin]), int(_IU[1][imin])
            i1, j1 = int(_IU[0][imax]), int(_IU[1][imax])
            cands = []
            u = P[j0] - P[i0]
            nu = np.linalg.norm(u)
            if nu > 1e-12:
                u = u / nu
                for w in (0.5, 1.0):
                    Q = P.copy(); Q[i0] -= w * step * u; Q[j0] += w * step * u
                    cands.append(Q)
                Q = P.copy(); Q[i0] -= step * u; cands.append(Q)
                Q = P.copy(); Q[j0] += step * u; cands.append(Q)
                # perpendicular shears of the min pair (rotate pair apart)
                w1 = np.array([1.0, 0.0, 0.0])
                if abs(u[0]) > 0.9:
                    w1 = np.array([0.0, 1.0, 0.0])
                p1 = np.cross(u, w1); p1 /= np.linalg.norm(p1)
                p2 = np.cross(u, p1)
                for pp in (p1, p2):
                    for s in (step, -step):
                        Q = P.copy()
                        Q[i0] += 0.5 * s * pp
                        Q[j0] -= 0.5 * s * pp
                        cands.append(Q)
            v = P[j1] - P[i1]
            nv = np.linalg.norm(v)
            if nv > 1e-12:
                v = v / nv
                for w in (0.5, 1.0):
                    Q = P.copy(); Q[i1] += w * step * v; Q[j1] -= w * step * v
                    cands.append(Q)
                Q = P.copy(); Q[i1] += step * v; cands.append(Q)
                Q = P.copy(); Q[j1] -= step * v; cands.append(Q)
            for a in range(N):
                for c in range(D):
                    for s in (step, -step):
                        Q = P.copy(); Q[a, c] += s; cands.append(Q)
            C = np.stack(cands, axis=0)
            rs = _batch_ratios(C, None)
            k = int(np.argmax(rs))
            if rs[k] > best_r + 1e-15:
                best_r = float(rs[k])
                P = C[k].copy()
                n_acc += 1
            else:
                break
    return P, best_r


def min_max_dist_dim3_14() -> np.ndarray:
    rng = np.random.default_rng(7)

    # --- Stage 1: grid search over several symmetric families ---
    fam_best = []  # list of (ratio, points)
    # D6 family: two staggered hex rings + two poles
    best_param, best_r = None, -1.0
    for h in np.linspace(0.15, 1.4, 26):
        for z0 in np.linspace(max(h, 0.2) + 0.05, 2.6, 48):
            P = d6_family(1.0, h, z0)
            r = ratio(P)
            if r > best_r:
                best_r, best_param = r, (1.0, h, z0)
    fam_best.append((best_r, d6_family(*best_param)))
    # 12-ring + two poles family
    best_r2, best_p2 = -1.0, None
    for h in np.linspace(-0.9, 0.9, 37):
        for z0 in np.linspace(max(abs(h), 0.2) + 0.05, 2.6, 48):
            P = ring_poles_family(12, h, z0)
            r = ratio(P)
            if r > best_r2:
                best_r2, best_p2 = r, (12, h, z0, 0.0)
    fam_best.append((best_r2, ring_poles_family(*best_p2)))
    # heptagonal antiprism family
    best_r3, best_p3 = -1.0, None
    for R in np.linspace(0.3, 1.6, 27):
        for h in np.linspace(0.05, 1.2, 24):
            P = hept_antiprism_family(R, h)
            r = ratio(P)
            if r > best_r3:
                best_r3, best_p3 = r, (R, h)
    fam_best.append((best_r3, hept_antiprism_family(*best_p3)))

    # refine and polish each family's best seed, keep global best
    best, best_r = None, -1.0
    for r0, P0 in fam_best:
        P0 = (P0 - P0.mean(axis=0)) / pair_dists(P0).max()
        Q = polish(P0)
        rq = ratio(Q)
        if rq > best_r:
            best_r, best = rq, Q
        Qc, rc = climb(Q)
        if rc > best_r:
            best_r, best = rc, Qc
        Qc, rc = climb(P0)
        if rc > best_r:
            best_r, best = rc, Qc
    # NM-refined D6 candidate as well
    res = minimize(lambda z: soft_ratio_obj(z, 0.002), np.array(best_param),
                   method="Nelder-Mead",
                   options={"xatol": 1e-10, "fatol": 1e-14, "maxiter": 4000})
    if np.isfinite(res.x).all():
        R, h, z0 = res.x
        if R > 0 and h > 0 and z0 > h:
            Pnm = d6_family(R, h, z0)
            Pnm = (Pnm - Pnm.mean(axis=0)) / pair_dists(Pnm).max()
            if ratio(Pnm) > best_r:
                best_r, best = ratio(Pnm), Pnm
            Qc, rc = climb(Pnm)
            if rc > best_r:
                best_r, best = rc, Qc
    if best is None:
        best = polish(d6_family(*best_param))
        best_r = ratio(best)

    # --- jittered restarts to escape any numerical plateau ---
    for k in range(10):
        mag = 0.04 * (0.75 ** k)
        Q = best + mag * rng.normal(size=best.shape)
        Qp = polish(Q)
        rp = ratio(Qp)
        if rp > best_r + 1e-12:
            best_r, best = rp, Qp
        Qc, rc = climb(Qp)
        if rc > best_r + 1e-12:
            best_r, best = rc, Qc
        Qc, rc = climb(Q)
        if rc > best_r + 1e-12:
            best_r, best = rc, Qc

    # --- Stage 5: tiny high-precision polish of the winner ---
    Q = polish(best, taus=(0.001, 0.0003, 0.0001))
    if ratio(Q) > best_r:
        best, best_r = Q, ratio(Q)

    points = np.asarray(best, dtype=float)
    assert points.shape == (N, D) and np.isfinite(points).all()
    assert pair_dists(points).max() > 0
    return points
# EVOLVE-BLOCK-END