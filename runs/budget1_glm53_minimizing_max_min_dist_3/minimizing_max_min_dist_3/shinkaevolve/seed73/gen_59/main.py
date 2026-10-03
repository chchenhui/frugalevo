# EVOLVE-BLOCK-START
import numpy as np

try:
    from scipy.optimize import minimize
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


# ---------------- Geometry layer (module-level invariants) ----------------

N, D = 14, 3
_IU = np.triu_indices(N, 1)
PI_, PJ = _IU


def _sanitize(pts, rng=None):
    """Return a guaranteed-finite (N, D) point set."""
    pts = np.asarray(pts, dtype=float)
    if pts.shape != (N, D) or not np.all(np.isfinite(pts)):
        if rng is None:
            rng = np.random.default_rng(0)
        pts = rng.standard_normal((N, D))
    return pts


def _true_score(points):
    d2 = np.sum((points[PI_] - points[PJ]) ** 2, axis=1)
    dx2 = d2.max()
    if dx2 <= 0:
        return 0.0
    return d2.min() / dx2


# ---------------- Optimizer layer ----------------

def _neg_obj(flat, beta, sphere):
    pts = flat.reshape(N, D).copy()
    diff = pts[PI_] - pts[PJ]
    dij2 = np.sum(diff * diff, axis=1)

    a = -beta * dij2
    amax = a.max()
    soft_min = -(amax + np.log(np.sum(np.exp(a - amax)))) / beta
    b = beta * dij2
    bmax = b.max()
    soft_max = (bmax + np.log(np.sum(np.exp(b - bmax)))) / beta
    obj = -(soft_min - soft_max)

    w_min = np.exp(a - amax); w_min /= w_min.sum()
    w_max = np.exp(b - bmax); w_max /= w_max.sum()
    g_dij2 = -w_min - w_max
    grad = np.zeros((N, D))
    contrib = 2.0 * g_dij2[:, None] * diff
    np.add.at(grad, PI_, contrib)
    np.add.at(grad, PJ, -contrib)
    if sphere:
        norms = np.maximum(np.linalg.norm(pts, axis=1, keepdims=True), 1e-12)
        u = pts / norms
        grad -= np.sum(grad * u, axis=1, keepdims=True) * u
    return obj, grad.ravel()


def _soft_opt(P0, betas, sphere, maxiter):
    """L-BFGS-B chain with escalating sharpness; fully guarded."""
    P = _sanitize(P0).ravel().copy()
    if not HAVE_SCIPY:
        return P.reshape(N, D)
    for beta in betas:
        try:
            res = minimize(_neg_obj, P, args=(beta, sphere), jac=True,
                           method="L-BFGS-B",
                           options={"maxiter": maxiter, "maxfun": 4 * maxiter})
            if np.all(np.isfinite(res.x)) and res.x.size == N * D:
                P = res.x
        except Exception:
            pass
    return _sanitize(P.reshape(N, D))


def _slsqp_packing(P0, betas=(20.0, 60.0, 200.0, 1000.0), maxiter=400):
    """Maximize soft-min distance s.t. diameter <= 1. Pre/post-guarded."""
    if not HAVE_SCIPY:
        return None
    P = _sanitize(P0)
    P = P - P.mean(axis=0)
    dm = np.linalg.norm(P[PI_] - P[PJ], axis=1).max()
    if not np.isfinite(dm) or dm <= 0:
        return None
    P = P / dm

    def pdists(flat):
        Pm = flat.reshape(N, D)
        return np.linalg.norm(Pm[PI_] - Pm[PJ], axis=1)

    def obj_soft(flat, beta):
        ds = pdists(flat)
        a = -beta * ds
        amax = a.max()
        return (amax + np.log(np.sum(np.exp(a - amax)))) / beta

    for beta in betas:
        try:
            res = minimize(obj_soft, P.ravel(), args=(beta,), method='SLSQP',
                           constraints={'type': 'ineq',
                                        'fun': lambda f: 1.0 - pdists(f).max()},
                           options={'maxiter': maxiter, 'ftol': 1e-14})
        except Exception:
            break
        if res.x.size != N * D or not np.all(np.isfinite(res.x)):
            break
        P = res.x.reshape(N, D)
        dcur = pdists(P.ravel()).max()
        if np.isfinite(dcur) and dcur > 1e-12:
            P = P / dcur

    P = _sanitize(P)
    score = _true_score(P)
    if score <= 0:
        return None
    return P, score


# ---------------- Initial configurations ----------------

def _structured_inits():
    inits = []
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ], dtype=float)
    ico /= np.linalg.norm(ico, axis=1, keepdims=True)
    inits.append(np.vstack([ico, [[0, 0, 1.3], [0, 0, -1.3]]]))
    inits.append(np.vstack([ico, [[0, 0, 1.0], [0, 0, -1.0]]]))
    cube = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], dtype=float)
    cube /= np.linalg.norm(cube, axis=1, keepdims=True)
    octa = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
                     [0, 0, 1], [0, 0, -1]], dtype=float)
    inits.append(np.vstack([cube, 1.05 * octa]))
    inits.append(np.vstack([1.05 * cube, octa]))
    t = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    r1 = np.stack([np.cos(t), np.sin(t), 0.5 * np.ones(7)], axis=1)
    r2 = np.stack([np.cos(t + np.pi / 7), np.sin(t + np.pi / 7), -0.5 * np.ones(7)], axis=1)
    inits.append(np.vstack([r1, r2]))
    k = np.arange(N) + 0.5
    ph = np.arccos(1.0 - 2.0 * k / N)
    th = np.pi * (1.0 + 5.0 ** 0.5) * k
    inits.append(np.stack([np.cos(th) * np.sin(ph),
                           np.sin(th) * np.sin(ph),
                           np.cos(ph)], axis=1))
    return inits


# ---------------- Driver: staged beam search ----------------

def min_max_dist_dim3_14() -> np.ndarray:
    rng = np.random.default_rng(7)
    target = 0.2404
    best_pts, best = None, -1.0

    # Build candidate starts
    starts = [p / np.linalg.norm(p, axis=1, keepdims=True)
              for p in _structured_inits()]
    for seed in range(12):
        v = np.random.RandomState(seed).randn(N, D)
        starts.append(v / np.linalg.norm(v, axis=1, keepdims=True))

    # Stage 1: cheap ranking (sphere-constrained soft optimization)
    ranked = []
    for pts0 in starts:
        pts = _soft_opt(pts0, (20.0, 60.0), True, 80)
        ranked.append((_true_score(pts), pts))
    ranked.sort(key=lambda x: -x[0])

    # Stage 2: exact constrained polish on the top beam
    for s0, pts in ranked[:6]:
        out = _slsqp_packing(pts, maxiter=400)
        if out is not None:
            P, s = out
        else:
            P = _soft_opt(pts, (20.0, 60.0, 150.0, 400.0), True, 300)
            s = _true_score(P)
        if s > best:
            best, best_pts = s, P
        if best >= target:
            break

    # Stage 3: free-space refinement (only if target not yet met)
    if best_pts is not None and best < target:
        cand = best_pts
        for beta in (60.0, 150.0, 400.0):
            cand = _soft_opt(cand, (beta,), False, 300)
            cand = cand - cand.mean(axis=0)
            out = _slsqp_packing(cand, maxiter=500)
            if out is not None:
                cand, s = out
            else:
                s = _true_score(cand)
            if s > best:
                best, best_pts = s, cand

    # Stage 4: incumbent re-polish — only if the target was NOT met by
    # earlier stages (avoids redundant SLSQP chains when already optimal).
    if best_pts is not None and best < target:
        out = _slsqp_packing(best_pts, betas=(200.0, 1000.0), maxiter=500)
        if out is not None:
            P, s = out
            if s > best:
                best_pts = P
    elif best_pts is not None:
        # Target met: one very cheap high-sharpness pass to squeeze any
        # remaining decimal-level slack, at negligible runtime cost.
        out = _slsqp_packing(best_pts, betas=(1000.0,), maxiter=100)
        if out is not None:
            P, s = out
            if s > best:
                best_pts = P

    # Guaranteed-safe return
    if best_pts is None:
        best_pts = rng.standard_normal((N, D))
    best_pts = _sanitize(best_pts, rng)
    best_pts = best_pts - best_pts.mean(axis=0)
    scale = np.max(np.abs(best_pts))
    if scale > 0:
        best_pts = best_pts / scale
    return np.asarray(best_pts, dtype=float)


# EVOLVE-BLOCK-END