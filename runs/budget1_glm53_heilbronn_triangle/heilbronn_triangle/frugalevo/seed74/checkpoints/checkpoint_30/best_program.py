# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations
from scipy.optimize import minimize
from scipy.special import logsumexp

_V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]])
_H = np.sqrt(3.0) / 2.0
_AREA_NORM = _H / 2.0
_TRIPLES = np.array(list(combinations(range(11), 3)))
_IDX_I = _TRIPLES[:, 0]
_IDX_J = _TRIPLES[:, 1]
_IDX_K = _TRIPLES[:, 2]
# Fixed edge assignment: 4 points on AB, 4 on BC, 3 on CA (cyclic interleave).
# Edge AB = vertices (0,1); BC = (1,2); CA = (2,0).
_EDGE_OF = np.array([0, 0, 1, 0, 1, 1, 2, 0, 2, 1, 2])
_EDGE_VERTS = [(0, 1), (1, 2), (2, 0)]


def _softmax(theta):
    """Row-wise softmax mapping unconstrained (11,3) params to barycentric coords."""
    z = theta - theta.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _areas_from_bary(lam):
    """Barycentric coords (11,3) -> 165 normalized triple areas (vectorized)."""
    pts = lam @ _V
    pi = pts[_IDX_I]
    pj = pts[_IDX_J]
    pk = pts[_IDX_K]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (
        pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    return np.abs(cross) / 2.0 / _AREA_NORM


def _smooth_obj(theta_flat, tau):
    """Negative log-sum-exp smooth minimum of triple areas."""
    a = _areas_from_bary(_softmax(theta_flat.reshape(11, 3)))
    return -tau * logsumexp(-a / tau)


def _edge_bary(u):
    """Edge params u (11,) in logit space -> (11,3) barycentric edge points.

    Each point i sits on edge _EDGE_OF[i] with coordinate t = sigmoid(u[i]),
    exactly on the boundary (third barycentric coord 0)."""
    t = 1.0 / (1.0 + np.exp(-np.clip(u, -20.0, 20.0)))
    t = np.clip(t, 1e-4, 1.0 - 1e-4)
    lam = np.zeros((11, 3))
    for i in range(11):
        a, b = _EDGE_VERTS[_EDGE_OF[i]]
        lam[i, a] = 1.0 - t[i]
        lam[i, b] = t[i]
    return lam


def _edge_obj(u_flat, tau):
    """Smooth min objective in the 11-D edge-constrained subsystem.

    Triples fully within one edge have zero area; the log-sum-exp keeps
    gradients finite and the release phase lifts points off the boundary."""
    a = _areas_from_bary(_edge_bary(u_flat))
    return -tau * logsumexp(-a / tau)


def _edge_seeds():
    """~12 deterministic 11-D edge-parameter seeds (logit space).

    Patterns: equal spacing, midpoints, Chebyshev-like nodes per edge,
    with global shifts to vary the relative phase between edges."""
    seeds = []
    counts = {e: int(np.sum(_EDGE_OF == e)) for e in range(3)}
    for phase in np.linspace(0.0, 0.5, 6):
        for kind in ('uniform', 'cheb'):
            t = np.zeros(11)
            for e in range(3):
                k = counts[e]
                js = np.arange(k)
                if kind == 'uniform':
                    te = (js + 0.5 + phase) / k
                else:
                    te = (1.0 - np.cos(np.pi * (js + 0.5 + phase) / k)) / 2.0
                t[_EDGE_OF == e] = np.clip(te, 1e-4, 1.0 - 1e-4)
            u = np.log(t / (1.0 - t))
            seeds.append(u)
    return seeds


def _free_seeds():
    """Two deterministic interior Dirichlet-style free-space seeds."""
    rng = np.random.RandomState(7)
    seeds = []
    for s in range(2):
        d = rng.dirichlet([1, 1, 1], size=11) * 0.9 + 0.033
        d = d / d.sum(axis=1, keepdims=True)
        seeds.append(np.log(d))
    return seeds


def _edge_to_theta(u):
    """Convert an edge-constrained solution into full 33-param theta,
    nudging points slightly interior (eps toward opposite vertex) so the
    released continuation starts from a strictly feasible smooth point."""
    lam = _edge_bary(u)
    eps = 1e-3
    for i in range(11):
        c = [v for v in range(3) if v not in _EDGE_VERTS[_EDGE_OF[i]]][0]
        lam[i, c] = eps
        lam[i] = lam[i] / lam[i].sum()
    return np.log(lam)


def _run_full_continuation(theta):
    """Full 33-param smooth-min continuation; returns optimized theta."""
    for tau in np.geomspace(0.02, 1e-5, 8):
        res = minimize(_smooth_obj, theta.ravel(), args=(tau,),
                       method='L-BFGS-B',
                       options={'maxiter': 250})
        theta = res.x.reshape(11, 3)
    return theta


def _valid(theta):
    """Feasibility/incumbent guard: finiteness, no duplicates, positive min area."""
    if not np.all(np.isfinite(theta)):
        return False
    lam = _softmax(theta)
    if not np.all(np.isfinite(lam)):
        return False
    pts = lam @ _V
    if np.min(_areas_from_bary(lam)) <= 1e-9:
        return False
    d = pts[:, None, :] - pts[None, :, :]
    if np.min(np.linalg.norm(d, axis=2) + np.eye(11) * 10.0) < 1e-6:
        return False
    return True


def _lcg(state):
    """Deterministic integer LCG (Lehmer, modulus 2^31-1); returns (state, u in [0,1))."""
    state = (state * 48271) % 2147483647
    return state, state / 2147483647.0


def _smooth_min(a, tau):
    """Scalar log-sum-exp smooth minimum of an area vector."""
    return -tau * logsumexp(-a / tau)


def _exact_min(lam):
    """Exact minimum normalized area over all 165 triples."""
    return float(_areas_from_bary(lam).min())


def _valid_lam(lam):
    """Feasibility/incumbent guard on barycentric coordinates."""
    if not np.all(np.isfinite(lam)):
        return False
    if _exact_min(lam) <= 1e-9:
        return False
    pts = lam @ _V
    d = pts[:, None, :] - pts[None, :, :]
    if np.min(np.linalg.norm(d, axis=2) + np.eye(11) * 10.0) < 1e-6:
        return False
    return True


def _d3_key(lam):
    """Canonical D3 key: lexicographic min over the 6 triangle symmetries.

    Each symmetry acts as a permutation of barycentric coordinates applied
    to all points; the key is the sorted flattened coordinate vector, so
    symmetric configurations compare equal and equivalent basins are not
    re-explored."""
    perms = ((0, 1, 2), (1, 2, 0), (2, 0, 1), (0, 2, 1), (2, 1, 0), (1, 0, 2))
    best = None
    for p in perms:
        k = tuple(np.sort(lam[:, p].ravel()))
        if best is None or k < best:
            best = k
    return best


def _anneal(lam0, iters=4000, seed=12345):
    """Deterministic annealing over barycentric coordinates (fixed LCG seed).

    Move set: (a) Gaussian perturbation of one point's coordinates;
    (b) edge projection (zero the smallest coordinate); (c) 3-fold orbit
    insertion — replace two other points with the 120-degree rotated images
    of a chosen point (cyclic barycentric permutation); (d) pairwise
    coordinate swap between two points. Metropolis acceptance on the
    smooth-min surrogate with tau tied to a geometrically decreasing
    temperature; periodic L-BFGS micro-polish; D3 canonical-key dedup of
    visited basins. Returns the best exact-min-area incumbent."""
    rng = seed
    lam = lam0.copy()
    cur = _smooth_min(_areas_from_bary(lam), 1e-3)
    best_lam = lam.copy()
    best_a = _exact_min(lam)
    visited = {}
    tau0, tau1 = 3e-3, 1e-5
    for it in range(iters):
        frac = it / iters
        tau = tau0 * (tau1 / tau0) ** frac
        rng, r1 = _lcg(rng)
        rng, r2 = _lcg(rng)
        rng, r3 = _lcg(rng)
        rng, r4 = _lcg(rng)
        cand = lam.copy()
        mv = int(r1 * 4) % 4
        i = int(r2 * 11) % 11
        if mv == 0:
            rng, u1 = _lcg(rng)
            rng, u2 = _lcg(rng)
            u1 = max(u1, 1e-12)
            z = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
            cand[i] = cand[i] + z * (0.4 * (tau / tau0) + 0.02)
        elif mv == 1:
            c = int(np.argmin(cand[i]))
            cand[i, c] = 0.0
        elif mv == 2:
            j = (i + 1 + int(r3 * 10)) % 11
            k = (j + 1 + int(r4 * 9)) % 11
            if len({i, j, k}) < 3:
                continue
            cand[j] = cand[i][[1, 2, 0]]
            cand[k] = cand[i][[2, 0, 1]]
        else:
            j = (i + 1 + int(r3 * 10)) % 11
            c = int(r4 * 3) % 3
            cand[i, c], cand[j, c] = cand[j, c], cand[i, c]
        cand = np.maximum(cand, 0.0)
        s = cand.sum(axis=1, keepdims=True)
        if np.any(s <= 1e-12):
            continue
        cand = cand / s
        key = _d3_key(cand)
        if visited.get(key, 0) >= 2:
            continue
        visited[key] = visited.get(key, 0) + 1
        s_new = _smooth_min(_areas_from_bary(cand), tau)
        rng, u_acc = _lcg(rng)
        acc = s_new > cur or u_acc < np.exp(
            min(0.0, (s_new - cur) / max(tau, 1e-12)))
        if acc:
            lam = cand
            cur = s_new
            em = _exact_min(cand)
            if em > best_a and _valid_lam(cand):
                best_a = em
                best_lam = cand.copy()
        if (it + 1) % 200 == 0:
            th = np.log(np.maximum(lam, 1e-12))
            res = minimize(_smooth_obj, th.ravel(), args=(tau,),
                           method='L-BFGS-B', options={'maxiter': 60})
            lam_new = _softmax(res.x.reshape(11, 3))
            em = _exact_min(lam_new)
            if np.isfinite(em) and em > _exact_min(lam):
                lam = lam_new
                cur = _smooth_min(_areas_from_bary(lam), tau)
                if em > best_a and _valid_lam(lam):
                    best_a = em
                    best_lam = lam.copy()
    return best_lam


def _area_cons_jac(x):
    """Analytic Jacobian of the 165 area constraints wrt (33 softmax params).

    Chain rule: dA/dpoint (signed cross-product derivatives), then
    dpoint/dbary (vertex matrix V), then dbary/dtheta (softmax Jacobian).
    Fully vectorized; removes SLSQP's finite-difference overhead."""
    lam = _softmax(x[:-1].reshape(11, 3))
    pts = lam @ _V
    pi = pts[_IDX_I]
    pj = pts[_IDX_J]
    pk = pts[_IDX_K]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (
        pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    s = np.sign(cross)
    s[s == 0.0] = 1.0
    scale = s / (2.0 * _AREA_NORM)
    n = cross.shape[0]
    ar = np.arange(n)
    dpts = np.zeros((n, 11, 2))
    dpts[ar, _IDX_I, 0] = pj[:, 1] - pk[:, 1]
    dpts[ar, _IDX_I, 1] = pk[:, 0] - pj[:, 0]
    dpts[ar, _IDX_J, 0] = pk[:, 1] - pi[:, 1]
    dpts[ar, _IDX_J, 1] = pi[:, 0] - pk[:, 0]
    dpts[ar, _IDX_K, 0] = pi[:, 1] - pj[:, 1]
    dpts[ar, _IDX_K, 1] = pj[:, 0] - pi[:, 0]
    dpts *= scale[:, None, None]
    dA_dlam = np.einsum('kic,rc->kir', dpts, _V)
    dlam_dth = lam[:, :, None] * (
        np.eye(3)[None, :, :] - lam[:, None, :])
    J = np.einsum('kir,irs->kis', dA_dlam, dlam_dth)
    return J.reshape(n, 33)


def _exact_polish(lam):
    """Exact max-min SLSQP refinement: variables are 33 softmax params plus
    scalar delta; maximize delta subject to all 165 triple areas >= delta.
    Uses the analytic constraint Jacobian (_area_cons_jac) so the active-set
    method converges quickly and precisely from a good incumbent."""
    th = np.log(np.maximum(lam, 1e-12)).ravel()
    d0 = _exact_min(lam) * 0.999
    x0 = np.concatenate([th, [d0]])
    obj_grad = np.concatenate([np.zeros(33), [-1.0]])

    def cons_fun(x):
        return _areas_from_bary(
            _softmax(x[:-1].reshape(11, 3))) - x[-1]

    def cons_jac(x):
        Jc = _area_cons_jac(x)
        return np.hstack([Jc, -np.ones((Jc.shape[0], 1))])

    try:
        res = minimize(lambda x: -x[-1], x0, jac=lambda x: obj_grad,
                       method='SLSQP',
                       constraints=[{'type': 'ineq',
                                     'fun': cons_fun,
                                     'jac': cons_jac}],
                       options={'maxiter': 400, 'ftol': 1e-14})
        if np.all(np.isfinite(res.x)) and res.x[-1] > d0:
            lam2 = _softmax(res.x[:-1].reshape(11, 3))
            if _valid_lam(lam2) and _exact_min(lam2) > _exact_min(lam):
                return lam2
    except Exception:
        pass
    return lam


def _slsqp_polish(lam):
    """Short SLSQP active-set polish of the smooth min at tiny tau."""
    th0 = np.log(np.maximum(lam, 1e-12)).ravel()
    res = minimize(
        lambda x: -_smooth_min(_areas_from_bary(_softmax(x.reshape(11, 3))), 1e-6),
        th0, method='SLSQP', options={'maxiter': 150, 'ftol': 1e-12})
    lam_p = _softmax(res.x.reshape(11, 3))
    if np.all(np.isfinite(lam_p)) and _exact_min(lam_p) > _exact_min(lam):
        return lam_p
    return lam


def heilbronn_triangle11() -> np.ndarray:
    """Deterministic annealing max-min optimizer for Heilbronn n=11.

    Starts: one edge-subsystem seed (11-D smooth-min continuation, single
    seed) plus one deterministic Dirichlet interior seed. Each start is
    annealed with structured topology-changing moves and an LCG RNG, then
    the best incumbent receives a smooth-min continuation polish followed
    by a short SLSQP active-set polish. Best valid incumbent retained;
    deterministic feasible fallback on any failure."""
    fallback = _free_seeds()[0]
    best_lam = None
    best_a = -1.0
    starts = []
    try:
        for u0 in _edge_seeds()[:3]:
            u = u0.copy()
            for tau in np.geomspace(0.05, 1e-4, 5):
                res = minimize(_edge_obj, u, args=(tau,),
                               method='L-BFGS-B', options={'maxiter': 200})
                u = res.x
            starts.append(_softmax(_edge_to_theta(u)))
    except Exception:
        pass
    for fs in _free_seeds():
        starts.append(fs.copy())

    for s_i, lam0 in enumerate(starts):
        try:
            lam = _anneal(lam0, iters=60000, seed=12345 + 7919 * s_i)
        except Exception:
            continue
        try:
            th = np.log(np.maximum(lam, 1e-12))
            for tau in np.geomspace(2e-3, 1e-7, 8):
                res = minimize(_smooth_obj, th.ravel(), args=(tau,),
                               method='L-BFGS-B', options={'maxiter': 500})
                th = res.x
            lam_p = _softmax(th.reshape(11, 3))
            if _exact_min(lam_p) > _exact_min(lam) and _valid_lam(lam_p):
                lam = lam_p
            lam = _slsqp_polish(lam)
            lam = _exact_polish(lam)
        except Exception:
            pass
        if _valid_lam(lam):
            a = _exact_min(lam)
            if a > best_a:
                best_a = a
                best_lam = lam.copy()

    if best_lam is not None:
        # Basin deepening: low-temperature re-anneal from the best incumbent,
        # then the standard continuation + polish chain. Only replaces the
        # incumbent on a strictly better exact min area.
        try:
            lam_r = _anneal(best_lam, iters=20000, seed=424243)
            th = np.log(np.maximum(lam_r, 1e-12))
            for tau in np.geomspace(2e-3, 1e-7, 8):
                res = minimize(_smooth_obj, th.ravel(), args=(tau,),
                               method='L-BFGS-B', options={'maxiter': 500})
                th = res.x
            lam_rp = _softmax(th.reshape(11, 3))
            if _exact_min(lam_rp) > _exact_min(lam_r) and _valid_lam(lam_rp):
                lam_r = lam_rp
            lam_r = _slsqp_polish(lam_r)
            lam_r = _exact_polish(lam_r)
            if _valid_lam(lam_r) and _exact_min(lam_r) > best_a:
                best_a = _exact_min(lam_r)
                best_lam = lam_r.copy()
        except Exception:
            pass
    if best_lam is None:
        best_lam = fallback
    # Final long exact max-min refinement on the overall best incumbent:
    # SLSQP maximizing delta s.t. all 165 areas >= delta, from the winner.
    try:
        lam_final = _exact_polish(best_lam)
        lam_final = _slsqp_polish(lam_final)
        lam_final = _exact_polish(lam_final)
        if _valid_lam(lam_final) and _exact_min(lam_final) > best_a:
            best_lam = lam_final
    except Exception:
        pass
    return best_lam @ _V


# EVOLVE-BLOCK-END
