# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

N = 14
D = 3
I_ARR, J_ARR = np.triu_indices(N, k=1)
M = len(I_ARR)


def pair_dists(pts):
    return np.linalg.norm(pts[I_ARR] - pts[J_ARR], axis=1)


def true_ratio(pts):
    dd = pair_dists(pts)
    mx = dd.max()
    if mx <= 1e-15:
        return 0.0
    return dd.min() / mx


def soft_min_obj_factory(k):
    """Scale-invariant objective: -softmin(dd/dmax)."""
    def obj(flat):
        pts = flat.reshape(N, D)
        pts = pts - pts.mean(axis=0)
        dd = pair_dists(pts)
        mx = dd.max()
        if mx <= 1e-12:
            return 1e6
        s = dd / mx
        smin = -np.log(np.sum(np.exp(-k * s))) / k
        return -smin
    return obj


def anneal_refine(pts0, ks=(10.0, 30.0, 80.0, 200.0)):
    x = pts0.flatten()
    for k in ks:
        res = minimize(soft_min_obj_factory(k), x, method="L-BFGS-B",
                       options={"maxiter": 300})
        x = res.x
    return x.reshape(N, D)


def build_starts(rng):
    phi = (1 + np.sqrt(5)) / 2
    ico = []
    for a in (-1, 1):
        for b in (-phi, phi):
            ico.append([a, b, 0])
            ico.append([0, a, b])
            ico.append([b, 0, a])
    ico = np.array(ico[:12], dtype=float)
    ico_poles = np.vstack([ico, [[0, 0, 2.0], [0, 0, -2.0]]])

    kk = np.arange(N) + 0.5
    ga = np.pi * (3 - np.sqrt(5))
    fib = np.zeros((N, D))
    fib[:, 2] = 1 - 2 * kk / N
    r_ = np.sqrt(np.maximum(0.0, 1 - fib[:, 2] ** 2))
    fib[:, 0] = r_ * np.cos(ga * kk)
    fib[:, 1] = r_ * np.sin(ga * kk)

    starts = [ico_poles.copy(), fib.copy()]
    for scale in (0.05, 0.2):
        starts.append(ico_poles + scale * rng.standard_normal((N, D)))
        starts.append(fib + scale * rng.standard_normal((N, D)))
    for _ in range(8):
        p = rng.standard_normal((N, D))
        p /= np.linalg.norm(p, axis=1, keepdims=True)
        starts.append(p * (0.5 + rng.random()))
    for _ in range(4):
        starts.append(rng.standard_normal((N, D)))
    return starts


def slsqp_polish(pts):
    """Maximize t s.t. t^2 <= ||pi-pj||^2 <= 1. Per-pair Jacobian fill."""
    P = pts - pts.mean(axis=0)
    mx = pair_dists(P).max()
    if mx <= 1e-12:
        return pts, true_ratio(pts)
    P = P / mx
    t0 = pair_dists(P).min()
    x0 = np.concatenate([P.flatten(), [t0]])
    nv = N * D + 1

    def fobj(x):
        return -x[-1]

    def fjac(x):
        J = np.zeros(nv)
        J[-1] = -1.0
        return J

    def cons(x):
        P2 = x[:N * D].reshape(N, D)
        dv = np.sum((P2[I_ARR] - P2[J_ARR]) ** 2, axis=1)
        return np.concatenate([1.0 - dv, dv - x[-1] ** 2])

    def cjac(x):
        P2 = x[:N * D].reshape(N, D)
        diff = P2[I_ARR] - P2[J_ARR]
        J = np.zeros((2 * M, nv))
        for idx in range(M):
            a = I_ARR[idx]
            b = J_ARR[idx]
            J[idx, 3 * a:3 * a + 3] = -2 * diff[idx]
            J[idx, 3 * b:3 * b + 3] = 2 * diff[idx]
            J[M + idx, 3 * a:3 * a + 3] = 2 * diff[idx]
            J[M + idx, 3 * b:3 * b + 3] = -2 * diff[idx]
            J[M + idx, -1] = -2 * x[-1]
        return J

    res = minimize(fobj, x0, method="SLSQP",
                   constraints=[{"type": "ineq", "fun": cons, "jac": cjac}],
                   options={"maxiter": 500, "ftol": 1e-12})
    P2 = res.x[:N * D].reshape(N, D)
    P2 = P2 - P2.mean(axis=0)
    return P2, true_ratio(P2)


def nelder_polish(pts, rng):
    def obj(flat):
        return -true_ratio(flat.reshape(N, D))
    best = pts
    best_r = true_ratio(pts)
    x = pts.flatten()
    for scale in (0.02, 0.01):
        res = minimize(obj, x, method="Nelder-Mead",
                       options={"maxiter": 6000, "maxfev": 8000,
                                "xatol": 1e-12, "fatol": 1e-14})
        cand = res.x.reshape(N, D)
        r = true_ratio(cand)
        if r > best_r:
            best_r, best = r, cand
        x = (cand + scale * rng.standard_normal((N, D))).flatten()
    return best, best_r


def min_max_dist_dim3_14() -> np.ndarray:
    """Creates 14 points in 3D maximizing the ratio of min to max distance."""
    rng = np.random.default_rng(42)

    # Stage 1: multi-start annealed soft-min refinement
    candidates = []
    for pts0 in build_starts(rng):
        pts = anneal_refine(pts0)
        candidates.append((true_ratio(pts), pts))
    candidates.sort(key=lambda c: -c[0])

    # Stage 2: SLSQP constrained polish on top-3 with perturb restarts
    best_r, best_pts = candidates[0]
    for r0, pts0 in candidates[:3]:
        for sig in (0.0, 0.05, 0.15, 0.3):
            P0 = pts0 if sig == 0 else pts0 + sig * rng.standard_normal((N, D))
            P2, r2 = slsqp_polish(P0)
            if r2 > best_r:
                best_r, best_pts = r2, P2

    # Stage 3: Nelder-Mead polish on the true ratio
    best_pts, best_r = nelder_polish(best_pts, rng)

    # Normalize output: center, scale so dmax = 1
    best_pts = np.asarray(best_pts, dtype=float)
    best_pts = best_pts - best_pts.mean(axis=0)
    dmax = pair_dists(best_pts).max()
    if dmax > 0:
        best_pts = best_pts / dmax
    return best_pts


# EVOLVE-BLOCK-END