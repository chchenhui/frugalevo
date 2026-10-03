# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

N, D = 14, 3
PAIRS_I, PAIRS_J = np.triu_indices(N, k=1)
TARGET = 0.4903  # just above known-best dmin/dmax for 14 points in 3D


def _pair_dists(P):
    return np.linalg.norm(P[PAIRS_I] - P[PAIRS_J], axis=1)


def _ratio(pts):
    ds = _pair_dists(pts)
    dmax = ds.max()
    if dmax <= 0:
        return -1.0
    return ds.min() / dmax


def _normalize(pts):
    ds = _pair_dists(pts)
    dmax = ds.max()
    return pts / dmax if dmax > 0 else pts


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def _repulsion(pts, iters=300, step=0.03):
    pts = pts.copy()
    for _ in range(iters):
        diff = pts[PAIRS_I] - pts[PAIRS_J]
        dist = np.sqrt(np.maximum((diff * diff).sum(-1), 1e-18))
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
        step *= 0.995
    return pts


def _slsqp(pts, passes=((300, 1e-12), (300, 1e-15), (200, 1e-16))):
    """Escalating SLSQP passes maximizing dmin s.t. dmax<=1. Best kept."""
    pts = _normalize(pts)
    best, best_r = pts.copy(), _ratio(pts)

    def objective(flat):
        return -_pair_dists(flat.reshape(N, D)).min()

    def constraint(flat):
        return 1.0 - _pair_dists(flat.reshape(N, D)).max()

    cur = pts.copy()
    for maxiter, ftol in passes:
        res = minimize(objective, cur.ravel(), method='SLSQP',
                       constraints={'type': 'ineq', 'fun': constraint},
                       options={'maxiter': maxiter, 'ftol': ftol})
        out = res.x.reshape(N, D)
        if np.all(np.isfinite(out)):
            out = _normalize(out)
            r = _ratio(out)
            if r > best_r:
                best_r, best = r, out
            cur = out
    return best, best_r


def _antipodal(pts, passes=((300, 1e-13), (300, 1e-16))):
    """Reparameterize as 7 antipodal pairs; optimize 7 free vectors."""
    pts = _normalize(pts - pts.mean(0))
    cost = np.linalg.norm(pts[:, None, :] + pts[None, :, :], axis=2)
    np.fill_diagonal(cost, np.inf)
    partner = np.argmin(cost, axis=1)
    if not np.all(partner[partner] == np.arange(N)):
        return None, -1.0
    reps, seen = [], set()
    for i in range(N):
        if i not in seen and partner[i] not in seen:
            reps.append(pts[i] - pts[partner[i]])
            seen.add(i); seen.add(partner[i])
    if len(reps) != 7:
        return None, -1.0
    V = np.array(reps)

    def build(v):
        Vm = v.reshape(7, 3)
        return np.vstack([Vm, -Vm])

    def objective(v):
        return -_pair_dists(build(v)).min()

    def constraint(v):
        return 1.0 - _pair_dists(build(v)).max()

    best_v = V.copy()
    best_cfg = build(V.ravel())
    best_r = _ratio(best_cfg)
    cur = V.copy()
    for maxiter, ftol in passes:
        res = minimize(objective, cur.ravel(), method='SLSQP',
                       constraints={'type': 'ineq', 'fun': constraint},
                       options={'maxiter': maxiter, 'ftol': ftol})
        out = res.x.reshape(7, 3)
        if np.all(np.isfinite(out)):
            cfg = _normalize(build(out.ravel()))
            r = _ratio(cfg)
            if r > best_r:
                best_r, best_v, best_cfg = r, out, cfg
            cur = out
    return best_cfg, best_r


def _icosahedron():
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ], dtype=float)
    return _unit(ico)


def _fibonacci_sphere(offset=0.0):
    k = np.arange(N) + 0.5
    ph = np.arccos(1.0 - 2.0 * k / N)
    th = np.pi * (1.0 + 5.0 ** 0.5) * k + offset
    return np.stack([np.cos(th) * np.sin(ph),
                     np.sin(th) * np.sin(ph),
                     np.cos(ph)], axis=1)


def _seeds(rng):
    seeds = []
    ico = _icosahedron()
    poles = np.array([[0, 0, 1], [0, 0, -1]])
    # base ico+poles, perturbed-pole variant, and rotation variants
    base = np.vstack([ico, poles])
    seeds.append(base)
    seeds.append(np.vstack([ico, _unit(poles + 0.1 * rng.standard_normal((2, 3)))]))
    for _ in range(3):
        Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        seeds.append(base @ Q.T)
    # fibonacci + random (trimmed)
    for off in (0.0, 0.7, 1.9):
        seeds.append(_fibonacci_sphere(off))
    for _ in range(2):
        seeds.append(_unit(rng.standard_normal((N, D))))
    return seeds


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of
    minimum to maximum pairwise distance.

    Returns
        points: np.ndarray of shape (14, 3)
    """
    rng = np.random.default_rng(42)
    best_pts, best_ratio = None, -1.0

    for init in _seeds(rng):
        for cand in (init, _repulsion(init)):
            apt_pts, apt_r = _antipodal(cand)
            if apt_pts is not None and apt_r > best_ratio:
                best_ratio, best_pts = apt_r, apt_pts
            pts, r = _slsqp(cand)
            if r > best_ratio:
                best_ratio, best_pts = r, pts
            sym_pts, sym_r = _antipodal(pts, passes=((200, 1e-15),))
            if sym_pts is not None and sym_r > best_ratio:
                best_ratio, best_pts = sym_r, sym_pts
        if best_ratio >= TARGET:
            break

    if best_pts is None:
        np.random.seed(42)
        best_pts = np.random.randn(N, D)

    # final polish on the winner
    pts, r = _slsqp(best_pts, passes=((200, 1e-15), (200, 1e-16)))
    if r > best_ratio:
        best_pts = pts
    apt_pts, apt_r = _antipodal(best_pts)
    if apt_pts is not None and apt_r > best_ratio:
        best_pts = apt_pts

    # normalize into [-1,1]^3 (convenience only)
    best_pts = best_pts - best_pts.mean(0)
    scale = np.abs(best_pts).max()
    if scale > 0:
        best_pts = best_pts / scale

    return np.asarray(best_pts, dtype=float)

# EVOLVE-BLOCK-END