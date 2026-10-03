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


def _soft_obj(z, tau):
    """Smooth minimum of the 165 areas (log-sum-exp); minimize this."""
    a = _areas(_to_xy(z))
    m = a.min()
    return m - np.log(np.sum(np.exp(-tau * (a - m)))) / tau


def _refine(z0):
    """Shared pipeline: soft-min continuation (3 taus, L-BFGS-B) + SLSQP polish.

    Returns best feasible point set by exact min-area, never worse than seed.
    """
    z = _to_z(_to_xy(z0))  # project seed into feasible box
    best_z, best_a = z.copy(), _exact_min(_to_xy(z))
    for tau in (40.0, 150.0, 600.0):
        r = minimize(_soft_obj, z, args=(tau,), method="L-BFGS-B",
                     bounds=[(0.0, 1.0)] * 22,
                     options={"maxiter": 300, "ftol": 1e-14})
        z = np.clip(r.x, 0.0, 1.0)
        a = _exact_min(_to_xy(z))
        if a > best_a:
            best_a, best_z = a, z.copy()
    r = minimize(_soft_obj, z, args=(1500.0,), method="SLSQP",
                 bounds=[(0.0, 1.0)] * 22,
                 options={"maxiter": 200, "ftol": 1e-14})
    z = np.clip(r.x, 0.0, 1.0)
    a = _exact_min(_to_xy(z))
    if a > best_a:
        best_a, best_z = a, z.copy()
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
    refined by a shared soft-min L-BFGS-B continuation + SLSQP polish in a
    box-constrained barycentric parameterization (22 vars, 22 matching
    bounds). Best exactly-verified incumbent is returned; valid fallback on
    any failure.
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
    return np.zeros((11, 2), dtype=float)


# EVOLVE-BLOCK-END
