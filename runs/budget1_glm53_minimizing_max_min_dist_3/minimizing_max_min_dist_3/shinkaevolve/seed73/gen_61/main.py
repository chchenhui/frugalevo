# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

N, D = 14, 3
H = N // 2
PAIRS_I, PAIRS_J = np.triu_indices(N, k=1)
TARGET_RATIO = 0.4895
TIME_BUDGET = 40.0


def _pair_dists(P):
    return np.linalg.norm(P[PAIRS_I] - P[PAIRS_J], axis=1)


def _ratio(P):
    ds = _pair_dists(P)
    dmax = ds.max()
    if dmax <= 0:
        return -1.0
    return ds.min() / dmax


def _expand(h):
    G = h.reshape(H, D)
    return np.vstack([G, -G])


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def _safe_input(pts):
    pts = np.asarray(pts, dtype=float)
    if pts.shape != (N, D) or not np.all(np.isfinite(pts)):
        return np.zeros((N, D)) + 1e-3
    return pts


def _repulsion_stage(pts, iters=300, step=0.03):
    """Cheap pre-optimizer: normalize diameter, repel close pairs."""
    pts = pts.copy()
    for _ in range(iters):
        diff = pts[PAIRS_I] - pts[PAIRS_J]
        dist = np.linalg.norm(diff, axis=1)
        dmax = dist.max()
        if dmax <= 0:
            break
        pts /= dmax
        diff = pts[PAIRS_I] - pts[PAIRS_J]
        dist = np.sqrt(np.maximum((diff * diff).sum(-1), 1e-18))
        w = 1.0 / dist ** 12
        forces = (w[:, None] * diff) / dist[:, None]
        grad = np.zeros_like(pts)
        np.add.at(grad, PAIRS_I, forces)
        np.add.at(grad, PAIRS_J, -forces)
        norm = np.linalg.norm(grad, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        pts += step * grad / norm
        step *= 0.997
    return pts


def _symmetrize(pts):
    """Center and pair points antipodally; return 7 generators."""
    pts = pts - pts.mean(0)
    cost = np.linalg.norm(pts[:, None, :] + pts[None, :, :], axis=2)
    np.fill_diagonal(cost, np.inf)
    partner = np.argmin(cost, axis=1)
    if not np.all(partner[partner] == np.arange(N)):
        # not a valid antipodal involution: fall back to pairing by sorting
        order = np.argsort(pts @ np.array([1.0, 0.5, 0.25]))
        gens = []
        used = np.zeros(N, dtype=bool)
        for i in range(N):
            if used[i]:
                continue
            rem = np.where(~used)[0]
            j = rem[int(np.argmin(np.linalg.norm(pts[rem] + pts[i], axis=1)))]
            gens.append((pts[i] - pts[j]) / 2.0)
            used[i] = used[j] = True
        return np.array(gens)
    return (pts - pts[partner]) / 2.0


def _softmin_opt(h0, maxiter=350, beta=40.0):
    """Smoothed max-min via log-sum-exp, symmetric parametrization."""
    def obj(h):
        ds = _pair_dists(_expand(h))
        dmax = ds.max()
        if dmax <= 0:
            return 0.0
        dn = ds / dmax
        return np.log(np.sum(np.exp(-beta * dn))) / beta

    def con(h):
        return 1.0 - _pair_dists(_expand(h)).max()

    ds = _pair_dists(_expand(h0))
    dm = ds.max()
    if dm > 0:
        h0 = h0 / dm
    res = minimize(obj, h0.ravel(), method='SLSQP',
                   constraints={'type': 'ineq', 'fun': con},
                   options={'maxiter': maxiter, 'ftol': 1e-14})
    if not np.all(np.isfinite(res.x)):
        return _expand(h0)
    return _expand(res.x)


def _slsqp_polish(P, maxiter=300, ftol=1e-14):
    """Exact-min SLSQP polish with diameter constraint."""
    P = _safe_input(P)
    ds = _pair_dists(P)
    dm = ds.max()
    if dm > 0:
        P = P / dm

    def obj(flat):
        return -_pair_dists(flat).min()

    def con(flat):
        return 1.0 - _pair_dists(flat).max()

    res = minimize(obj, P.ravel(), method='SLSQP',
                   constraints={'type': 'ineq', 'fun': con},
                   options={'maxiter': maxiter, 'ftol': ftol})
    if not np.all(np.isfinite(res.x)):
        return P
    out = res.x.reshape(N, D)
    if _ratio(out) >= _ratio(P):
        return out
    return P


def _bottleneck_polish(pts, tries=4, step=0.05, deadline=None):
    """Micro-search: push apart the closest pair, then short SLSQP pass."""
    best, best_r = pts, _ratio(pts)
    for _ in range(tries):
        if best_r >= TARGET_RATIO:
            break
        if deadline is not None and time.time() > deadline:
            break
        ds = _pair_dists(best)
        k = int(np.argmin(ds))
        i, j = PAIRS_I[k], PAIRS_J[k]
        direction = best[i] - best[j]
        nrm = np.linalg.norm(direction)
        if nrm <= 0:
            break
        direction = direction / nrm
        improved = False
        for s in (step, step / 2, step * 2):
            cand = best.copy()
            cand[i] = cand[i] + s * direction
            cand[j] = cand[j] - s * direction
            cand = _slsqp_polish(cand, maxiter=200)
            r = _ratio(cand)
            if r > best_r + 1e-12:
                best, best_r = cand, r
                improved = True
                break
        if not improved:
            break
    return best, best_r


def _icosahedron():
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ], dtype=float)
    return _unit(ico)


def _icosahedral_gens():
    ico = _icosahedron()
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


def _fibonacci_sphere(offset=0.0):
    k = np.arange(N) + 0.5
    ph = np.arccos(1.0 - 2.0 * k / N)
    th = np.pi * (1.0 + 5.0 ** 0.5) * k + offset
    return np.stack([np.cos(th) * np.sin(ph),
                     np.sin(th) * np.sin(ph),
                     np.cos(ph)], axis=1)


def _seed_generator(rng):
    """Lazily yield candidate 14-point configurations, strongest first."""
    ico = _icosahedron()
    # icosahedron + 2 perturbed poles
    for _ in range(3):
        extra = np.array([[0, 0, 1], [0, 0, -1]]) + 0.05 * rng.standard_normal((2, 3))
        yield np.vstack([ico, _unit(extra)])
    # antipodal icosahedral generators (7 gens -> 14 pts)
    yield _expand(_icosahedral_gens())
    # Fibonacci variants
    for off in (0.0, 0.7, 1.9):
        yield _fibonacci_sphere(off)
    # random on sphere and in ball
    for _ in range(5):
        yield _unit(rng.standard_normal((N, D)))
        yield rng.uniform(-1.0, 1.0, (N, D))


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3D maximizing min/max pairwise distance.

    Crossover pipeline: repulsion pre-opt -> antipodal symmetrization ->
    soft-min SLSQP (symmetric param) -> exact-min SLSQP polish ->
    bottleneck micro-polish. Lazy seeds with early exit at target ratio.
    """
    t0 = time.time()
    rng = np.random.default_rng(42)
    best_pts, best_ratio = None, -1.0

    for init in _seed_generator(rng):
        if best_ratio >= TARGET_RATIO:
            break
        if time.time() - t0 > TIME_BUDGET * 0.7:
            break
        pre = _repulsion_stage(init)
        for cand in (pre, init):
            # symmetric soft-min optimization from antipodal generators
            gens = _symmetrize(cand)
            P = _softmin_opt(gens, maxiter=300, beta=40.0)
            P = _slsqp_polish(P, maxiter=250)
            r = _ratio(P)
            if r > best_ratio:
                best_ratio, best_pts = r, P.copy()
            # also direct polish of the full (non-symmetric) candidate
            P2 = _slsqp_polish(cand, maxiter=300)
            r2 = _ratio(P2)
            if r2 > best_ratio:
                best_ratio, best_pts = r2, P2.copy()

    # Final polish + bottleneck kick on the winner while budget remains
    if best_pts is not None:
        cur, cur_r = best_pts, best_ratio
        while time.time() - t0 < TIME_BUDGET:
            nxt = _slsqp_polish(cur, maxiter=300)
            rn = _ratio(nxt)
            if rn > cur_r + 1e-15:
                cur, cur_r = nxt, rn
            else:
                break
        cur, cur_r = _bottleneck_polish(cur, deadline=t0 + TIME_BUDGET)
        best_pts, best_ratio = cur, cur_r

    if best_pts is None:
        best_pts = rng.standard_normal((N, D))

    return np.asarray(_safe_input(best_pts), dtype=float)


# EVOLVE-BLOCK-END