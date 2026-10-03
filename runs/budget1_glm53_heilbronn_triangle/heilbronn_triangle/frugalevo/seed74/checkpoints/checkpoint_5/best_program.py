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


def heilbronn_triangle11() -> np.ndarray:
    """Edge-subsystem + release max-min optimizer for Heilbronn n=11.

    Phase 1: optimize the 11-D edge-constrained subsystem (each point fixed
    to one triangle edge, one scalar coordinate) with smooth-min continuation
    from ~12 deterministic seeds — a well-conditioned boundary-dominated
    topology. Phase 2: release each incumbent into full barycentric space
    (softmax gives exact simplex feasibility) and run the full 33-param
    continuation. Best valid incumbent retained; deterministic fallback."""
    best_theta = None
    best_area = -1.0
    fallback = _free_seeds()[0]
    tau_edge = np.geomspace(0.05, 1e-4, 5)

    starts = [(_edge_to_theta(u), u) for u in _edge_seeds()]
    for theta0, u0 in starts:
        try:
            # Phase 1: edge-constrained 11-D optimization.
            u = u0.copy()
            for tau in tau_edge:
                res = minimize(_edge_obj, u, args=(tau,),
                               method='L-BFGS-B',
                               options={'maxiter': 200})
                u = res.x
            # Phase 2: release into full barycentric space.
            theta = _run_full_continuation(_edge_to_theta(u))
        except Exception:
            continue
        if _valid(theta):
            a = float(_areas_from_bary(_softmax(theta)).min())
            if a > best_area:
                best_area = a
                best_theta = theta.copy()

    # Two free-space starts as cheap supplementary basins.
    for theta0 in _free_seeds():
        try:
            theta = _run_full_continuation(theta0.copy())
        except Exception:
            continue
        if _valid(theta):
            a = float(_areas_from_bary(_softmax(theta)).min())
            if a > best_area:
                best_area = a
                best_theta = theta.copy()

    if best_theta is None:
        best_theta = fallback
    return _softmax(best_theta) @ _V


# EVOLVE-BLOCK-END
