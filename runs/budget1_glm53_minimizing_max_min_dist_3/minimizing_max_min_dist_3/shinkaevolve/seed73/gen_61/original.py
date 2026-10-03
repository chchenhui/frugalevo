# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

N, D = 14, 3
H = N // 2  # 7 free antipodal generators
PAIRS_I, PAIRS_J = np.triu_indices(N, k=1)
TARGET_RATIO = 0.4899
TIME_BUDGET = 45.0


def _pair_dists(P):
    return np.linalg.norm(P[PAIRS_I] - P[PAIRS_J], axis=1)


def _ratio(P):
    ds = _pair_dists(P)
    dmax = ds.max()
    if dmax <= 0:
        return -1.0
    return ds.min() / dmax


def _expand(h):
    """7 free vectors -> 14 antipodally symmetric points (centered)."""
    G = h.reshape(H, D)
    return np.vstack([G, -G])


def _safe_input(pts):
    """Pre-SLSQP shape/finiteness guard (closes the Gen-41 crash window)."""
    pts = np.asarray(pts, dtype=float)
    if pts.shape != (N, D) or not np.all(np.isfinite(pts)):
        return np.zeros((N, D)) + 1e-3
    return pts


def _softmin_opt(h0, maxiter=400, beta=40.0):
    """Smoothed max-min via log-sum-exp over pair distances, symmetric param."""
    def obj(h):
        ds = _pair_dists(_expand(h))
        dmax = ds.max()
        if dmax <= 0:
            return 0.0
        dn = ds / dmax
        # -softmin of normalized distances (minimize)
        return (np.log(np.sum(np.exp(-beta * dn)))) / beta

    def con(h):
        ds = _pair_dists(_expand(h))
        return 1.0 - ds.max()

    ds = _pair_dists(_expand(h0))
    dm = ds.max()
    if dm > 0:
        h0 = h0 / dm
    res = minimize(obj, h0.ravel(), method='SLSQP',
                   constraints={'type': 'ineq', 'fun': con},
                   options={'maxiter': maxiter, 'ftol': 1e-14})
    out = res.x
    if not np.all(np.isfinite(out)):
        return _expand(h0)
    return _expand(out)


def _exact_polish(P, maxiter=300):
    """Exact-min SLSQP polish with diameter constraint."""
    P = _safe_input(P)
    ds = _pair_dists(P)
    dm = ds.max()
    if dm > 0:
        P = P / dm

    def obj(flat):
        return -_pair_dists(flat.reshape(N, D)).min()

    def con(flat):
        return 1.0 - _pair_dists(flat.reshape(N, D)).max()

    res = minimize(obj, P.ravel(), method='SLSQP',
                   constraints={'type': 'ineq', 'fun': con},
                   options={'maxiter': maxiter, 'ftol': 1e-16})
    if not np.all(np.isfinite(res.x)):
        return P
    out = res.x.reshape(N, D)
    if _ratio(out) >= _ratio(P):
        return out
    return P


def _icosahedral_seed():
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ], dtype=float)
    ico /= np.linalg.norm(ico, axis=1, keepdims=True)
    # pair the icosahedron antipodally: each vertex with its negation -> 6 gens,
    # plus the two poles collapsed into 1 generator => 7 generators.
    gens = []
    used = np.zeros(12, dtype=bool)
    for i in range(12):
        if used[i]:
            continue
        j = int(np.argmin(np.linalg.norm(ico - (-ico[i]), axis=1)))
        gens.append((ico[i] - ico[j]) / 2.0)
        used[i] = used[j] = True
    gens.append(np.array([0.0, 0.0, 1.0]))
    return np.array(gens)


def _generator_seeds(rng):
    seeds = [_icosahedral_seed()]
    # antipodal pairings of icosahedron with slightly perturbed poles
    base = _icosahedral_seed()
    for _ in range(4):
        s = base.copy()
        s[-1] = s[-1] + 0.05 * rng.standard_normal(3)
        s[-1] /= max(np.linalg.norm(s[-1]), 1e-12)
        seeds.append(s)
    # random generators (antipodal by construction)
    for _ in range(6):
        seeds.append(rng.standard_normal((H, D)))
    return seeds


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3D maximizing min/max pairwise distance.

    Novel pipeline: antipodal-symmetric parametrization (7 generators),
    soft-min smoothing for effective SLSQP navigation of the bottleneck,
    then exact-min polish. Guarded inputs and a time budget keep it fast.
    """
    t0 = time.time()
    rng = np.random.default_rng(42)
    best_pts, best_ratio = None, -1.0

    for gi, g0 in enumerate(_generator_seeds(rng)):
        if time.time() - t0 > TIME_BUDGET * 0.7:
            break
        for beta in (25.0, 60.0):
            P = _softmin_opt(g0, maxiter=350, beta=beta)
            P = _exact_polish(P, maxiter=250)
            r = _ratio(P)
            if r > best_ratio:
                best_ratio, best_pts = r, P.copy()
        if best_ratio >= TARGET_RATIO:
            break

    # Final aggressive polish on the winner while budget remains
    if best_pts is not None:
        cur, cur_r = best_pts, best_ratio
        while time.time() - t0 < TIME_BUDGET:
            nxt = _exact_polish(cur, maxiter=300)
            rn = _ratio(nxt)
            if rn > cur_r + 1e-15:
                cur, cur_r = nxt, rn
            else:
                break

        # one bottleneck kick: separate the closest pair then re-polish
        ds = _pair_dists(cur)
        k = int(np.argmin(ds))
        i, j = PAIRS_I[k], PAIRS_J[k]
        direction = cur[i] - cur[j]
        nrm = np.linalg.norm(direction)
        if nrm > 0:
            direction /= nrm
            for s in (0.01, 0.002, 0.05):
                cand = cur.copy()
                cand[i] += s * direction
                cand[j] -= s * direction
                cand = _exact_polish(cand, maxiter=250)
                rc = _ratio(cand)
                if rc > cur_r:
                    cur, cur_r = cand, rc
        best_pts, best_ratio = cur, cur_r

    if best_pts is None:
        best_pts = rng.standard_normal((N, D))

    best_pts = _safe_input(best_pts)
    return np.asarray(best_pts, dtype=float)


# EVOLVE-BLOCK-END