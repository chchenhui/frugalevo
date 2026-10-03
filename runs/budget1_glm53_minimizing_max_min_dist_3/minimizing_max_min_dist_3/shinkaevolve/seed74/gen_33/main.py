# EVOLVE-BLOCK-START
import numpy as np

try:
    from scipy.optimize import minimize
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


def _ratio_sq(pts):
    n = len(pts)
    iu = np.triu_indices(n, 1)
    dv = np.linalg.norm(pts[iu[0]] - pts[iu[1]], axis=1)
    dx = dv.max()
    if dx <= 0:
        return 0.0
    return (dv.min() / dx) ** 2


def _grad_soft(flat, n, T, I, J):
    pts = flat.reshape(n, 3)
    diff = pts[I] - pts[J]
    dd = np.sqrt(np.sum(diff * diff, axis=1) + 1e-12)
    a = -dd / T
    amax = a.max()
    w_min = np.exp(a - amax)
    w_min /= w_min.sum()
    b = dd / T
    bmax = b.max()
    w_max = np.exp(b - bmax)
    w_max /= w_max.sum()
    soft_min = -T * (amax + np.log(np.sum(np.exp(a - amax))))
    soft_max = T * (bmax + np.log(np.sum(np.exp(b - bmax))))
    g = -(w_min * soft_max - soft_min * w_max) / (soft_max ** 2)
    grad = np.zeros((n, 3))
    uv = diff / dd[:, None]
    contrib = g[:, None] * uv
    np.add.at(grad, I, contrib)
    np.add.at(grad, J, -contrib)
    return grad.ravel(), -soft_min / soft_max


def _exact_ratio_grad(flat, n, I, J):
    pts = flat.reshape(n, 3)
    diff = pts[I] - pts[J]
    dd = np.sqrt(np.sum(diff * diff, axis=1) + 1e-12)
    im = int(np.argmin(dd))
    iM = int(np.argmax(dd))
    dmin = dd[im]
    dmax = max(dd[iM], 1e-12)
    r = dmin / dmax
    grad = np.zeros((n, 3))

    def add_pair(k, coef):
        aa, bb = I[k], J[k]
        u = diff[k] / dd[k]
        grad[aa] += coef * u
        grad[bb] -= coef * u

    add_pair(im, 2.0 * r / dmax)
    add_pair(iM, -2.0 * r * dmin / (dmax ** 2))
    return -r * r, grad.ravel()


def _normalize(P, I, J):
    P = P - P.mean(axis=0)
    mx = np.linalg.norm(P[I] - P[J], axis=1).max()
    if mx > 1e-12:
        P = P / mx
    return P


def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    rng = np.random.default_rng(42)
    I, J = np.triu_indices(n, k=1)
    m = len(I)

    # ---- structured starts (from prior program) + random starts ----
    phi = (1 + np.sqrt(5)) / 2
    ico = []
    for a in (-1, 1):
        for b in (-phi, phi):
            ico.append([a, b, 0])
            ico.append([0, a, b])
            ico.append([b, 0, a])
    ico = np.array(ico, dtype=float)
    ico_poles = np.vstack([ico, [[0, 0, 2.0], [0, 0, -2.0]]])

    kk = np.arange(n) + 0.5
    ga = np.pi * (3 - np.sqrt(5))
    fib = np.zeros((n, 3))
    fib[:, 2] = 1 - 2 * kk / n
    r_ = np.sqrt(np.maximum(0.0, 1 - fib[:, 2] ** 2))
    fib[:, 0] = r_ * np.cos(ga * kk)
    fib[:, 1] = r_ * np.sin(ga * kk)

    cube = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1)
                     for z in (-1, 1)], dtype=float)
    cube_faces = np.vstack([cube, [[1.5, 0, 0], [-1.5, 0, 0],
                                   [0, 1.5, 0], [0, -1.5, 0],
                                   [0, 0, 1.5], [0, 0, -1.5]]])

    starts = [ico_poles.copy(), fib.copy(), cube_faces.copy()]
    for scale in (0.05, 0.2):
        starts.append(ico_poles + scale * rng.standard_normal((n, d)))
        starts.append(fib + scale * rng.standard_normal((n, d)))
    for _ in range(6):
        p = rng.standard_normal((n, d))
        p = p / np.linalg.norm(p, axis=1, keepdims=True)
        starts.append(p * (0.5 + rng.random()))
    for _ in range(4):
        starts.append(rng.standard_normal((n, d)))

    best_pts = None
    best_val = -1.0

    # ---- soft-annealed L-BFGS then exact-ratio polish (current program) ----
    for s in starts:
        pts = np.asarray(s, dtype=float).copy()
        for T in (0.3, 0.15, 0.08, 0.04, 0.02, 0.01):
            flat = pts.ravel()
            if HAVE_SCIPY:
                res = minimize(lambda x: _grad_soft(x, n, T, I, J),
                               flat, jac=True, method="L-BFGS-B",
                               options={"maxiter": 200})
                flat = res.x
            else:
                g, _ = _grad_soft(flat, n, T, I, J)
                flat = flat + 0.05 * g
            pts = _normalize(flat.reshape(n, 3), I, J)
        if HAVE_SCIPY:
            prev = _ratio_sq(pts)
            try:
                res = minimize(lambda x: _exact_ratio_grad(x, n, I, J),
                               pts.ravel(), jac=True, method="L-BFGS-B",
                               options={"maxiter": 400, "ftol": 1e-16,
                                        "gtol": 1e-12})
                cand = res.x.reshape(n, 3)
                if np.all(np.isfinite(cand)) and _ratio_sq(cand) > prev:
                    pts = _normalize(cand, I, J)
            except Exception:
                pass
        val = _ratio_sq(pts)
        if val > best_val:
            best_val = val
            best_pts = pts.copy()

    if best_pts is None:
        best_pts = rng.standard_normal((n, d))

    # ---- SLSQP epigraph polish (prior program): maximize t ----
    if HAVE_SCIPY:
        def polish(P):
            P = _normalize(P, I, J)
            dv0 = np.linalg.norm(P[I] - P[J], axis=1)
            if dv0.max() <= 1e-12:
                return P, _ratio_sq(P)
            x0 = np.concatenate([P.ravel(), [dv0.min()]])

            def cons(x):
                P2 = x[:42].reshape(n, d)
                dv = np.sum((P2[I] - P2[J]) ** 2, axis=1)
                return np.concatenate([1.0 - dv, dv - x[-1] ** 2])

            def cjac(x):
                P2 = x[:42].reshape(n, d)
                diff = P2[I] - P2[J]
                Jc = np.zeros((2 * m, 43))
                Jc[:m, 3 * I[:, None] + np.arange(3)] = -2 * diff
                Jc[:m, 3 * J[:, None] + np.arange(3)] = 2 * diff
                Jc[m:, 3 * I[:, None] + np.arange(3)] = 2 * diff
                Jc[m:, 3 * J[:, None] + np.arange(3)] = -2 * diff
                Jc[m:, -1] = -2 * x[-1]
                return Jc

            try:
                res = minimize(lambda x: -x[-1], x0, method="SLSQP",
                               constraints=[{"type": "ineq", "fun": cons,
                                             "jac": cjac}],
                               options={"maxiter": 500, "ftol": 1e-12})
                P2 = res.x[:42].reshape(n, d)
                P2 = P2 - P2.mean(axis=0)
                if np.all(np.isfinite(P2)):
                    return P2, _ratio_sq(P2)
            except Exception:
                pass
            return P, _ratio_sq(P)

        P2, r2 = polish(best_pts)
        if r2 > best_val:
            best_val = r2
            best_pts = P2
        for _ in range(4):
            Pp = best_pts + 0.02 * rng.standard_normal((n, d))
            P2, r2 = polish(Pp)
            if r2 > best_val:
                best_val = r2
                best_pts = P2

    if not np.all(np.isfinite(best_pts)):
        best_pts = np.zeros((n, d))
        best_pts[:, 0] = np.arange(n)

    # center and scale so dmax = 1
    best_pts = np.asarray(best_pts, dtype=float)
    best_pts = best_pts - best_pts.mean(axis=0)
    dmax = np.linalg.norm(best_pts[I] - best_pts[J], axis=1).max()
    if dmax > 0:
        best_pts = best_pts / dmax
    return best_pts


# EVOLVE-BLOCK-END