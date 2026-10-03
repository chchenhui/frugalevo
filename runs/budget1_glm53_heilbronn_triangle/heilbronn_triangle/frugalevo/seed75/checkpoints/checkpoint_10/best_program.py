# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

_A = np.array([0.0, 0.0])
_B = np.array([1.0, 0.0])
_C = np.array([0.5, np.sqrt(3.0) / 2.0])
_TRI = np.vstack([_A, _B, _C])
_CEN = _TRI.mean(axis=0)

# All 165 index triples (i, j, k) for n = 11, precomputed once.
_IJK = np.array([(i, j, k)
                 for i in range(11) for j in range(i + 1, 11)
                 for k in range(j + 1, 11)], dtype=np.intp)


def _areas(P):
    """Exact (unnormalized) areas of all 165 triangles, vectorized."""
    Pi = P[_IJK[:, 0]]
    Pj = P[_IJK[:, 1]]
    Pk = P[_IJK[:, 2]]
    return 0.5 * np.abs(
        (Pj[:, 0] - Pi[:, 0]) * (Pk[:, 1] - Pi[:, 1])
        - (Pj[:, 1] - Pi[:, 1]) * (Pk[:, 0] - Pi[:, 0]))


def _exact_min(P):
    """Minimum triangle area normalized by the container area sqrt(3)/2."""
    return _areas(P).min() / (np.sqrt(3.0) / 2.0)


def _to_xy(z):
    """(22,) raw params in [0,1] -> 11 feasible points via barycentric map.

    a = u, b = v*(1-u), c = 1-a-b >= 0 ensures every point lies in the
    triangle while the optimizer sees a plain box (bounds always match x0).
    """
    z = np.clip(z, 0.0, 1.0)
    a = z[0::2]
    b = z[1::2] * (1.0 - a)
    Bary = np.column_stack([a, b, 1.0 - a - b])
    return Bary @ _TRI


def _to_z(P):
    """Inverse of _to_xy for points already inside the triangle."""
    M = np.vstack([_B - _A, _C - _A]).T
    z = np.empty((len(P), 2))
    for i, p in enumerate(P):
        ab, *_ = np.linalg.lstsq(M, p - _A, rcond=None)
        a = min(max(ab[0], 0.0), 1.0 - 1e-9)
        b = min(max(ab[1], 0.0), 1.0 - a)
        z[i] = [a, b / (1.0 - a)]
    return np.clip(z.ravel(), 0.0, 1.0)


def _to_xy_uv(z):
    """Barycentric map returning the points plus the raw (u, v) parameters."""
    z = np.clip(z, 0.0, 1.0)
    u = z[0::2]
    v = z[1::2]
    a = u
    b = v * (1.0 - u)
    Bary = np.column_stack([a, b, 1.0 - a - b])
    return Bary @ _TRI, u, v


def _areas_grad(z):
    """Areas of all 165 triples and their exact gradient wrt the 22 params."""
    P, u, v = _to_xy_uv(z)
    idx = _IJK
    Pi, Pj, Pk = P[idx[:, 0]], P[idx[:, 1]], P[idx[:, 2]]
    d1 = Pj - Pi
    d2 = Pk - Pi
    s = 0.5 * (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
    areas = np.abs(s)
    sgn = np.where(s >= 0.0, 1.0, -1.0)[:, None]
    rows = np.arange(len(idx))
    gxy = np.zeros((len(idx), 11, 2))
    gxy[rows, idx[:, 1]] = 0.5 * sgn * np.column_stack([-d2[:, 1], d2[:, 0]])
    gxy[rows, idx[:, 2]] = -0.5 * sgn * np.column_stack([-d1[:, 1], d1[:, 0]])
    gxy[rows, idx[:, 0]] = -(gxy[rows, idx[:, 1]] + gxy[rows, idx[:, 2]])
    # chain rule through the barycentric parameterization
    dPdu = _A[None, :] - v[:, None] * _B[None, :] - (1.0 - v)[:, None] * _C[None, :]
    dPdv = (1.0 - u)[:, None] * (_B - _C)[None, :]
    gz = np.empty((len(idx), 22))
    gz[:, 0::2] = np.einsum('tpi,pi->tp', gxy, dPdu)
    gz[:, 1::2] = np.einsum('tpi,pi->tp', gxy, dPdv)
    return areas, gz


def _soft_obj(z, tau):
    """Smooth minimum of the 165 areas (log-sum-exp); minimize this."""
    a = _areas(_to_xy(z))
    m = a.min()
    return m - np.log(np.sum(np.exp(-tau * (a - m)))) / tau


def _soft_obj_grad(z, tau):
    """Exact gradient of _soft_obj: weighted mean of the area gradients."""
    a, gz = _areas_grad(z)
    e = np.exp(-tau * (a - a.min()))
    w = e / e.sum()
    return gz.T @ w


def _epi_obj(w):
    """Epigraph objective: maximize the common lower bound t -> minimize -t."""
    return -w[-1]


def _epi_jac(w):
    g = np.zeros(len(w))
    g[-1] = -1.0
    return g


def _epi_cons(w):
    """Inequalities area_k - t >= 0 for every one of the 165 triples."""
    a, _ = _areas_grad(w[:22])
    return a - w[-1]


def _epi_cons_jac(w):
    _, gz = _areas_grad(w[:22])
    return np.hstack([gz, -np.ones((len(_IJK), 1))])


def _epi_round(z, best_z, best_a, rounds=5):
    """Epigraph SLSQP refinement (max t s.t. area_k >= t) started from z.

    Warm-starts t at the current exact min area and iterates; returns the
    updated (best_z, best_a) incumbent pair.
    """
    bnds = [(0.0, 1.0)] * 22 + [(0.0, None)]
    con = {"type": "ineq", "fun": _epi_cons, "jac": _epi_cons_jac}
    w = np.concatenate([z, [max(_areas(_to_xy(z)).min(), 1e-8)]])
    for _ in range(rounds):
        r = minimize(_epi_obj, w, jac=_epi_jac, method="SLSQP",
                     bounds=bnds, constraints=con,
                     options={"maxiter": 400, "ftol": 1e-14})
        z = np.clip(r.x[:22], 0.0, 1.0)
        a = _exact_min(_to_xy(z))
        if a > best_a:
            best_a, best_z = a, z.copy()
        w = np.concatenate([z, [max(_areas(_to_xy(z)).min(), 1e-8)]])
    return best_z, best_a


def _refine(z0):
    """Shared pipeline: analytic-gradient soft-min continuation (L-BFGS-B)
    followed by exact epigraph SLSQP refinement (max t s.t. area_k >= t),
    plus deterministic structured perturbation restarts around the incumbent
    (symmetry release; no RNG, fixed cosine/sine offsets of decreasing size).

    Returns best feasible point set by exact min-area, never worse than seed.
    """
    z = _to_z(_to_xy(z0))  # project seed into feasible box
    best_z, best_a = z.copy(), _exact_min(_to_xy(z))
    bnds22 = [(0.0, 1.0)] * 22
    for tau in (20.0, 60.0, 200.0, 800.0, 3000.0):
        r = minimize(_soft_obj, z, args=(tau,), jac=_soft_obj_grad,
                     method="L-BFGS-B", bounds=bnds22,
                     options={"maxiter": 400, "ftol": 1e-16})
        z = np.clip(r.x, 0.0, 1.0)
        a = _exact_min(_to_xy(z))
        if a > best_a:
            best_a, best_z = a, z.copy()
    best_z, best_a = _epi_round(z, best_z, best_a)
    # deterministic perturbation restarts: fixed structured offsets (no RNG)
    amp = 0.012
    phases = np.arange(22) * (np.pi / 7.0)
    for trial in range(6):
        zp = np.clip(best_z + amp * np.sin(phases + 1.3 * trial), 0.0, 1.0)
        for tau in (200.0, 3000.0):
            r = minimize(_soft_obj, zp, args=(tau,), jac=_soft_obj_grad,
                         method="L-BFGS-B", bounds=bnds22,
                         options={"maxiter": 300, "ftol": 1e-16})
            zp = np.clip(r.x, 0.0, 1.0)
        best_z, best_a = _epi_round(zp, best_z, best_a, rounds=3)
        amp *= 0.55  # shrink offsets: coarse-to-fine symmetry release
    return _to_xy(best_z), best_a


def _orbit(r, phase, n=3):
    """n points on a circle of radius r around the centroid."""
    return [_CEN + r * np.array([np.cos(phase + 2 * np.pi * k / n),
                                 np.sin(phase + 2 * np.pi * k / n)])
            for k in range(n)]


def _edge_pts(t1, t2):
    """Two points at fractions t1, t2 along each of the three edges."""
    return [t1 * _A + (1 - t1) * _B, t2 * _A + (1 - t2) * _B,
            t1 * _A + (1 - t1) * _C, t2 * _A + (1 - t2) * _C,
            t1 * _B + (1 - t1) * _C, t2 * _B + (1 - t2) * _C]


def _seeds():
    """12 deterministic seeds over three distinct contact topologies.

    Family A: boundary-heavy (6 edge points) + two concentric 3-orbits.
    Family B: interior-hex (centroid + two interleaved 3-orbits + 4 inner).
    Family C: C3-symmetric warm start with an explicit symmetry-breaking
              shift (delta = 0.05) releasing degenerate equal-area ties.
    Variants within a family rotate orbits/scales radii slightly.
    """
    seeds = []
    for vi in range(4):
        d = 0.06 * vi  # deterministic variant offset
        # A: boundary-ring
        P = _edge_pts(1 / 3 + d * 0.3, 2 / 3 - d * 0.3)
        P += _orbit(0.16 + d, np.pi / 2)
        P += _orbit(0.30 + d, np.pi / 2 + np.pi / 3)
        seeds.append(np.array(P[:11]))
        # B: interior-hex
        P = [_CEN] + _orbit(0.20 + d, 0.0) + _orbit(0.36 + d, np.pi / 3)
        P += _orbit(0.09 + d * 0.5, np.pi / 4, n=4)
        seeds.append(np.array(P[:11]))
        # C: symmetry-broken
        P = [_CEN] + _orbit(0.18 + d, np.pi / 2) + _orbit(0.34 + d, np.pi / 2)
        P = [np.array(p) for p in P[:7]]
        P[1:4] = P[1:4] + np.array([0.05, 0.0])  # release C3 symmetry
        P += [(1 - t) * _A + (t / 2) * _B + (t / 2) * _C
              for t in (0.25, 0.40, 0.55, 0.70)]
        seeds.append(np.array(P[:11]))
    return seeds


def heilbronn_triangle11() -> np.ndarray:
    """
     Deterministic multistart Heilbronn n=11 optimizer: 12 seeds spanning
     boundary-ring / interior-hex / symmetry-broken contact topologies, each
     refined by a shared soft-min L-BFGS-B continuation + epigraph SLSQP
    polish with deterministic perturbation restarts, in a box-constrained
    barycentric parameterization (22 vars, 22 matching bounds). Best
    exactly-verified incumbent is returned; valid fallback on any failure.
     """
    try:
        best_P, best_a = None, -np.inf
        for s in _seeds():
            if s.shape != (11, 2):
                continue
            P, a = _refine(s.ravel())
            if np.isfinite(a) and a > best_a:
                best_a, best_P = a, np.array(P, dtype=float)
        if best_P is not None and np.isfinite(best_a) and best_a > 0.0:
            return best_P
    except Exception:
        pass
    # valid deterministic fallback: boundary-ring seed configuration
    return np.array(_seeds()[0][:11], dtype=float)


# EVOLVE-BLOCK-END
