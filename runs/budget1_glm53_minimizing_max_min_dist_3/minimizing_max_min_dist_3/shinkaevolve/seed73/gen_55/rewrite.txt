# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N, D = 14, 3
H = N // 2  # 7 antipodal generators
PAIRS_I, PAIRS_J = np.triu_indices(N, k=1)
TARGET_RATIO = 0.4899
TIME_BUDGET = 40.0


def _expand(g):
    """7 generator vectors -> 14 antipodally symmetric points."""
    G = g.reshape(H, D)
    return np.vstack([G, -G])


def _pair_dists(P):
    return np.linalg.norm(P[PAIRS_I] - P[PAIRS_J], axis=1)


def _ratio(P):
    ds = _pair_dists(P)
    dmax = ds.max()
    if dmax <= 0:
        return -1.0
    return ds.min() / dmax


def _softmin_obj(g, beta=60.0):
    """Smooth surrogate: minimize soft-min of normalized distances."""
    ds = _pair_dists(_expand(g))
    dmax = ds.max()
    if dmax <= 1e-12:
        return 0.0
    dn = ds / dmax
    return np.log(np.sum(np.exp(-beta * dn))) / beta


def _slsqp_polish(P, maxiter=300):
    """Exact polish: maximize dmin subject to dmax <= 1."""
    P = np.asarray(P, dtype=float)
    if P.shape != (N, D) or not np.all(np.isfinite(P)):
        return np.zeros((N, D)) + 1e-3
    ds = _pair_dists(P)
    if ds.max() > 0:
        P = P / ds.max()
    r0 = _ratio(P)

    def obj(flat):
        return -_pair_dists(flat.reshape(N, D)).min()

    def con(flat):
        return 1.0 - _pair_dists(flat.reshape(N, D)).max()

    res = minimize(obj, P.ravel(), method='SLSQP',
                   constraints={'type': 'ineq', 'fun': con},
                   options={'maxiter': maxiter, 'ftol': 1e-14})
    if not np.all(np.isfinite(res.x)):
        return P
    out = res.x.reshape(N, D)
    return out if _ratio(out) >= r0 else P


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / max(np.linalg.norm(v), 1e-12)


def _ico_seed():
    """Icosahedron (6 antipodal generators) + one pole generator."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ], dtype=float)
    ico /= np.linalg.norm(ico, axis=1, keepdims=True)
    gens = ico[::2]  # every other vertex: 6 generators (antipode is skipped)
    return np.vstack([gens, [0.0, 0.0, 1.0]])


def _ring_seed(theta, jitter=0.0, rng=None):
    """Pole + 6 generators on a ring at polar angle theta."""
    az = np.arange(6) * (np.pi / 3.0)
    if jitter and rng is not None:
        az = az + jitter * rng.standard_normal(6)
    ring = np.stack([np.sin(theta) * np.cos(az),
                     np.sin(theta) * np.sin(az),
                     np.full(6, np.cos(theta))], axis=1)
    return np.vstack([[[0.0, 0.0, 1.0]], ring])


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3D maximizing min/max pairwise distance.

    Approach: antipodal parameterization (7 generators) optimized
    globally with differential evolution on a softmin surrogate,
    followed by exact SLSQP polish. Guarded fallback included.
    """
    t0 = time.time()
    rng = np.random.default_rng(42)
    best_pts, best_ratio = None, -1.0

    # --- Quick deterministic candidates (guaranteed quality) ---
    quick = [_ico_seed()]
    for th in np.linspace(0.5, 1.1, 7):
        quick.append(_ring_seed(th))
    for g in quick:
        if best_ratio >= TARGET_RATIO:
            break
        P = _slsqp_polish(_expand(g), maxiter=400)
        r = _ratio(P)
        if r > best_ratio:
            best_ratio, best_pts = r, P.copy()

    # --- Global search: differential evolution on generators ---
    def de_obj(g):
        return _softmin_obj(g, beta=60.0)

    remaining = TIME_BUDGET - (time.time() - t0)
    if remaining > 8 and best_ratio < TARGET_RATIO:
        try:
            res = differential_evolution(
                de_obj,
                bounds=[(-1.0, 1.0)] * (H * D),
                seed=42,
                maxiter=40,
                popsize=18,
                tol=1e-10,
                polish=False,
                init='sobol',
            )
            if np.all(np.isfinite(res.x)):
                P = _slsqp_polish(_expand(res.x), maxiter=400)
                r = _ratio(P)
                if r > best_ratio:
                    best_ratio, best_pts = r, P.copy()
        except Exception:
            pass

    # --- Random restarts while time remains ---
    while time.time() - t0 < TIME_BUDGET and best_ratio < TARGET_RATIO:
        g = rng.standard_normal((H, D))
        P = _slsqp_polish(_expand(g), maxiter=300)
        r = _ratio(P)
        if r > best_ratio:
            best_ratio, best_pts = r, P.copy()

    if best_pts is None:
        best_pts = rng.standard_normal((N, D))
    out = np.asarray(best_pts, dtype=float)
    if out.shape != (N, D) or not np.all(np.isfinite(out)):
        out = np.zeros((N, D)) + 1e-3
    if _pair_dists(out).max() <= 0:
        out = np.zeros((N, D)) + 1e-3
    return out

# EVOLVE-BLOCK-END