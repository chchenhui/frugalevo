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


def _ring_seed(s, t1, t2, u1, u2):
    """Two-ring homothetic-triangle seed with scale parameter s.

    Layout (11 points, all barycentric coords nonnegative => exactly feasible):
      - 6 boundary points: 2 per edge, at along-edge parameters t1 and t2
        (cyclic over the three edges, D3-symmetric boundary ring);
      - 3 inner points on the homothetic ring: cyclic permutations of
        (s/2, s/2, 1-s) — the edge-midpoint orbit of the inner triangle
        at scale s;
      - 2 inner points on medians at parameters u1 and u2.
    The ring scale s coupling interior-ring spacing to boundary spacing is
    the degree of freedom absent from pure median-orbit seeds."""
    lam = np.zeros((11, 3))
    k = 0
    for c0 in range(3):
        for t in (t1, t2):
            lam[k, c0] = 0.0
            lam[k, (c0 + 1) % 3] = 1.0 - t
            lam[k, (c0 + 2) % 3] = t
            k += 1
    em = np.array([0.5 * s, 0.5 * s, 1.0 - s])
    for j in range(3):
        lam[k] = np.roll(em, j)
        k += 1
    for u in (u1, u2):
        lam[k] = np.array([1.0 - 2.0 * u, u, u])
        k += 1
    return lam


def _orbit_seeds():
    """D3-orbit-structured deterministic seeds for the 11-point Heilbronn problem.

    A barycentric point (1-2u, u, u) lies on a median; its 120-degree rotation
    orbit is obtained by cyclic permutation of the barycentric coordinates.
    Seed family (a): three 3-point orbits at median parameters u1<u2<u3 plus
    the centroid and one edge midpoint (11 points, strictly interior except
    the deliberate boundary midpoint). Seed family (b): two 3-point orbits
    plus 5 boundary points at Chebyshev nodes distributed 2/2/1 over the
    three edges. All coordinates are nonnegative by construction, so
    feasibility is exact. A grid over (u1,u2,u3) / (u1,u2) yields ~16
    structured seeds, each scored later by exact min area."""
    seeds = []
    for u1 in (0.08, 0.12, 0.16):
        for u2 in (0.20, 0.28):
            for u3 in (0.35, 0.42):
                lam = np.zeros((11, 3))
                for k, u in enumerate((u1, u2, u3)):
                    base = np.array([1.0 - 2.0 * u, u, u])
                    for j in range(3):
                        lam[3 * k + j] = np.roll(base, j)
                lam[9] = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
                lam[10] = np.array([0.5, 0.5, 0.0])
                seeds.append(lam)
    for u1 in (0.12, 0.18):
        for u2 in (0.30, 0.38):
            lam = np.zeros((11, 3))
            for k, u in enumerate((u1, u2)):
                base = np.array([1.0 - 2.0 * u, u, u])
                for j in range(3):
                    lam[3 * k + j] = np.roll(base, j)
            ts = (1.0 - np.cos(np.pi * (np.arange(5) + 0.5) / 5.0)) / 2.0
            lam[6] = np.array([1.0 - ts[0], ts[0], 0.0])
            lam[7] = np.array([1.0 - ts[3], ts[3], 0.0])
            lam[8] = np.array([0.0, 1.0 - ts[1], ts[1]])
            lam[9] = np.array([0.0, 1.0 - ts[4], ts[4]])
            lam[10] = np.array([ts[2], 0.0, 1.0 - ts[2]])
            seeds.append(lam)
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
        mv = int(r1 * 6) % 6
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
        elif mv == 3:
            j = (i + 1 + int(r3 * 10)) % 11
            c = int(r4 * 3) % 3
            cand[i, c], cand[j, c] = cand[j, c], cand[i, c]
        elif mv == 4:
            # Edge-jump (phase-scheduled): if point i is (near) on some edge,
            # migrate it to a different edge. During the first 60% of the
            # annealing schedule, sample the along-edge position uniformly
            # (parent's original behavior — preserves exploration). In the
            # late, low-temperature regime, place the point at the midpoint
            # of the largest gap between other strictly-on-edge points on
            # the target edge (endpoints included), with jitter proportional
            # to the gap width (20% of gap, capped 0.1), falling back to a
            # uniform sample when the target edge is empty. Feasibility is
            # exact: the candidate is rebuilt as barycentric edge coordinates
            # and renormalized by the existing path below.
            c0 = int(np.argmin(cand[i]))
            if cand[i, c0] > 5e-2:
                continue
            c1 = int(r3 * 3) % 3
            if c1 == c0:
                c1 = (c1 + 1) % 3
            c2 = 3 - c0 - c1
            rng, uu = _lcg(rng)
            t_new = None
            if frac > 0.6:
                edge_mask = (cand[:, c0] < 1e-3)
                edge_mask[i] = False
                idx_on = np.nonzero(edge_mask)[0]
                if idx_on.size >= 1:
                    ts = np.sort(np.clip(
                        cand[idx_on, c2] / np.maximum(
                            cand[idx_on, c1] + cand[idx_on, c2], 1e-12),
                        0.0, 1.0))
                    bounds = np.concatenate([[0.0], ts, [1.0]])
                    gaps = np.diff(bounds)
                    g = int(np.argmax(gaps))
                    gw = gaps[g]
                    rng, u1 = _lcg(rng)
                    rng, u2 = _lcg(rng)
                    z = np.sqrt(-2.0 * np.log(max(u1, 1e-12))) * np.cos(
                        2.0 * np.pi * u2)
                    t_new = min(max(
                        0.5 * (bounds[g] + bounds[g + 1]) + z * min(0.2 * gw, 0.1),
                        0.02), 0.98)
            if t_new is None:
                t_new = 0.05 + 0.9 * uu
            cand[i, :] = 0.0
            cand[i, c1] = 1.0 - t_new
            cand[i, c2] = t_new
        else:
            # Interior-lift: a boundary point moves to a small triangle around
            # the centroid of its nearest neighbors, gaining interior mass.
            c0 = int(np.argmin(cand[i]))
            if cand[i, c0] > 5e-2:
                continue
            d = ((cand[:, None, :] - cand[None, :, :]) ** 2).sum(axis=2)
            d[i, :] = np.inf
            nb = np.argsort(d[i])[:3]
            cen = cand[nb].mean(axis=0)
            cen = cen / cen.sum()
            rng, q1 = _lcg(rng)
            rng, q2 = _lcg(rng)
            r = 0.15
            a1 = (q1 - 0.5) * 2.0 * r
            a2 = (q2 - 0.5) * 2.0 * r
            w = np.array([1.0 - a1 - a2, a1, a2])
            w = np.maximum(w, 0.0)
            w = w / w.sum()
            new = (1.0 - r) * cen + r * w
            cand[i] = new / new.sum()
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
                       options={'maxiter': 800, 'ftol': 1e-16})
        if np.all(np.isfinite(res.x)) and res.x[-1] > d0:
            lam2 = _softmax(res.x[:-1].reshape(11, 3))
            if _valid_lam(lam2) and _exact_min(lam2) > _exact_min(lam):
                return lam2
    except Exception:
        pass
    return lam


def _pairwise_epigraph(lam, passes=3):
    """Pairwise block-coordinate epigraph ascent on the exact max-min problem.

    For each of the 55 point pairs (i, j) in deterministic index order, solve a
    small SLSQP epigraph problem: variables are the 6 softmax params of the two
    points plus scalar delta; maximize delta subject to area(triple) >= delta
    for every triple containing i or j (all other points frozen). The analytic
    constraint Jacobian is derived from dA/dlam restricted to the pair's rows
    and the two points' softmax Jacobians. Each subproblem solution is accepted
    only if the exact min area strictly improves and the configuration stays
    valid; otherwise the pair is left unchanged. Passes stop early when a full
    sweep changes the exact min by < 1e-14. Feasibility is exact via softmax
    barycentric coordinates."""
    lam = lam.copy()
    for _ in range(passes):
        start = _exact_min(lam)
        for i in range(11):
            for j in range(i + 1, 11):
                mask = ((_IDX_I == i) | (_IDX_J == i) | (_IDX_K == i) |
                        (_IDX_I == j) | (_IDX_J == j) | (_IDX_K == j))
                ti = _IDX_I[mask]
                tj = _IDX_J[mask]
                tk = _IDX_K[mask]
                base = lam.copy()

                def _L(x, i=i, j=j, base=base):
                    L = base
                    L[i] = _softmax(x[0:3].reshape(1, 3))[0]
                    L[j] = _softmax(x[3:6].reshape(1, 3))[0]
                    return L

                def cons(x, i=i, j=j, ti=ti, tj=tj, tk=tk):
                    L = _L(x)
                    pi, pj, pk = L[ti], L[tj], L[tk]
                    cross = ((pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) -
                             (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0]))
                    return (np.abs(cross) / (2.0 * _AREA_NORM)) - x[6]

                def cjac(x, i=i, j=j, ti=ti, tj=tj, tk=tk):
                    L = _L(x)
                    pi, pj, pk = L[ti], L[tj], L[tk]
                    cross = ((pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) -
                             (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0]))
                    s = np.sign(cross)
                    s[s == 0.0] = 1.0
                    sc = s / (2.0 * _AREA_NORM)
                    m = cross.shape[0]
                    ar = np.arange(m)
                    dp = np.zeros((m, 11, 2))
                    dp[ar, ti, 0] = pj[:, 1] - pk[:, 1]
                    dp[ar, ti, 1] = pk[:, 0] - pj[:, 0]
                    dp[ar, tj, 0] = pk[:, 1] - pi[:, 1]
                    dp[ar, tj, 1] = pi[:, 0] - pk[:, 0]
                    dp[ar, tk, 0] = pi[:, 1] - pj[:, 1]
                    dp[ar, tk, 1] = pj[:, 0] - pi[:, 0]
                    dp *= sc[:, None, None]
                    dA_dlam = np.einsum('kic,rc->kir', dp, _V)
                    J = np.zeros((m, 7))
                    for col, p in ((0, i), (3, j)):
                        lp = L[p][None, :]
                        dl = lp[:, :, None] * (np.eye(3)[None] - lp[:, None, :])
                        J[:, col:col + 3] = np.einsum(
                            'kir,krs->kis', dA_dlam[:, p, :], dl)[:, 0, :]
                    J[:, 6] = -1.0
                    return J

                x0 = np.concatenate([
                    np.log(np.maximum(lam[i], 1e-12)),
                    np.log(np.maximum(lam[j], 1e-12)),
                    [_exact_min(lam)]])
                try:
                    res = minimize(lambda x: -x[6], x0,
                                   jac=lambda x: np.concatenate(
                                       [np.zeros(6), [-1.0]]),
                                   method='SLSQP',
                                   constraints=[{'type': 'ineq',
                                                 'fun': cons,
                                                 'jac': cjac}],
                                   options={'maxiter': 300, 'ftol': 1e-14})
                    if not np.all(np.isfinite(res.x)):
                        continue
                    L2 = lam.copy()
                    L2[i] = _softmax(res.x[0:3].reshape(1, 3))[0]
                    L2[j] = _softmax(res.x[3:6].reshape(1, 3))[0]
                    if _valid_lam(L2) and _exact_min(L2) > _exact_min(lam):
                        lam = L2
                except Exception:
                    continue
        if _exact_min(lam) - start < 1e-14:
            break
    return lam


def _slsqp_polish(lam):
    """Polish entry point: pairwise block epigraph ascent (replaces the old
    monolithic smooth-min SLSQP polish; same call signature and contract of
    returning a configuration with exact min area >= the input's)."""
    lam_p = _pairwise_epigraph(lam)
    if _valid_lam(lam_p) and _exact_min(lam_p) > _exact_min(lam):
        return lam_p
    return lam


def _final_refine(lam):
    """High-precision final refinement of the winning incumbent.

    Alternates exact max-min SLSQP passes (tight ftol, more iterations)
    with a tiny inward shrink toward the centroid. The shrink moves any
    exactly-on-boundary points a distance ~1e-9 * scale into the interior,
    which lets the smooth/SLSQP machinery move freely in all 33 coordinates
    instead of being pinned by active boundary constraints, and the
    subsequent polish recovers the last ~1e-13 of min area that the
    boundary-pinned parameterization leaves on the table."""
    best = lam.copy()
    best_a = _exact_min(lam)
    cen = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    # Wider shrink ladder: un-pin boundary points at several scales so the
    # KKT/SLSQP polish can move freely in all 33 coordinates; each rung is
    # guarded by the exact min area so the incumbent only improves.
    for shrink in (1e-7, 1e-8, 1e-9, 1e-10, 0.0):
        L = best.copy()
        if shrink > 0.0:
            L = (1.0 - shrink) * L + shrink * cen
            L = L / L.sum(axis=1, keepdims=True)
        for _ in range(4):
            L2 = _exact_polish(L)
            L2 = _slsqp_polish(L2)
            if _exact_min(L2) <= _exact_min(L) + 1e-16:
                L = L2
                break
            L = L2
        a = _exact_min(L)
        if a > best_a and _valid_lam(L):
            best_a = a
            best = L.copy()
    return best


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
        # Combined seed pool: the proven orbit-lattice family plus the new
        # homothetic-ring family (extra DOF: ring scale s coupling the
        # interior ring to boundary spacing). Every seed is scored by exact
        # minimum triple area and only the top 4 are annealed — same start
        # budget as before, no risky pre-solve continuation.
        seeds = _orbit_seeds()
        try:
            for s in (0.30, 0.40, 0.50, 0.60, 0.70):
                for t1 in (0.15, 0.25, 0.35):
                    for t2 in (0.60, 0.75, 0.90):
                        for u1 in (0.12, 0.22):
                            for u2 in (0.36, 0.46):
                                seeds.append(
                                    _ring_seed(s, t1, t2, u1, u2))
        except Exception:
            pass
        scored = sorted(seeds, key=_exact_min, reverse=True)
        starts = [s.copy() for s in scored[:4]]
    except Exception:
        pass

    for s_i, lam0 in enumerate(starts):
        try:
            lam = _anneal(lam0.copy(), iters=60000, seed=12345 + 7919 * s_i)
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
    # Final high-precision refinement on the overall best incumbent:
    # repeated exact max-min SLSQP with tight ftol plus a tiny inward
    # shrink that un-pins boundary points, recovering the residual ~1e-13
    # that keeps the score at exactly 1.0 instead of above it.
    try:
        lam_final = _final_refine(best_lam)
        if _valid_lam(lam_final) and _exact_min(lam_final) > best_a:
            best_a = _exact_min(lam_final)
            best_lam = lam_final.copy()
    except Exception:
        pass
    return best_lam @ _V


# EVOLVE-BLOCK-END
