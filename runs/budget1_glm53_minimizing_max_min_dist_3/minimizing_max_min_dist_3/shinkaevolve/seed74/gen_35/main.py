# EVOLVE-BLOCK-START
import numpy as np

try:
    from scipy.optimize import minimize
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


def _ratio_sq(pts):
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    iu = np.triu_indices(len(pts), 1)
    dm = d[iu].min()
    dx = d[iu].max()
    if dx <= 0:
        return 0.0
    return (dm / dx) ** 2


def _grad_soft(flat, n, T):
    pts = flat.reshape(n, 3)
    diff = pts[:, None, :] - pts[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-12)
    i, j = np.triu_indices(n, 1)
    dv = d[i, j]
    a = -dv / T
    amax = a.max()
    w_min = np.exp(a - amax)
    w_min /= w_min.sum()
    b = dv / T
    bmax = b.max()
    w_max = np.exp(b - bmax)
    w_max /= w_max.sum()
    soft_min = -T * (amax + np.log(np.sum(np.exp(a - amax))))
    soft_max = T * (bmax + np.log(np.sum(np.exp(b - bmax))))
    g = -(w_min * soft_max - soft_min * w_max) / (soft_max ** 2)
    grad = np.zeros((n, 3))
    uv = diff[i, j] / d[i, j][:, None]
    contrib = g[:, None] * uv
    np.add.at(grad, i, contrib)
    np.add.at(grad, j, -contrib)
    return grad.ravel(), -soft_min / soft_max


def _exact_ratio_grad(flat, n):
    pts = flat.reshape(n, 3)
    diff = pts[:, None, :] - pts[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-12)
    i, j = np.triu_indices(n, 1)
    dv = d[i, j]
    im = int(np.argmin(dv))
    iM = int(np.argmax(dv))
    dmin = dv[im]
    dmax = max(dv[iM], 1e-12)
    r = dmin / dmax
    grad = np.zeros((n, 3))

    def add_pair(k, coef):
        a, b = i[k], j[k]
        u = (pts[a] - pts[b]) / dv[k]
        grad[a] += coef * u
        grad[b] -= coef * u

    add_pair(im, 2.0 * r / dmax)
    add_pair(iM, -2.0 * r * dmin / (dmax ** 2))
    return -r * r, grad.ravel()


def _recentre(pts):
    pts = pts - pts.mean(axis=0)
    sc = np.linalg.norm(pts, axis=1).max()
    if sc > 0:
        pts = pts / sc
    return pts


def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    rng = np.random.default_rng(42)

    starts = []
    phi = (1 + np.sqrt(5)) / 2

    # Icosahedron vertices (unit norm)
    ico = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ], dtype=float) / np.sqrt(1 + phi ** 2)

    # Swept-pole icosahedron variants: poles along x, y, z with pz grid
    for axis in range(3):
        for pz in (1.0, 1.2, 1.5):
            p1 = np.zeros(3); p1[axis] = pz
            p2 = np.zeros(3); p2[axis] = -pz
            starts.append(_recentre(np.vstack([ico, p1, p2])))

    # Cuboctahedron vertices + 2 poles
    cubo = []
    for x, y in ((-1, 1),):
        pass
    cubo = np.array([[x, y, 0] for x in (-1, 1) for y in (-1, 1)] +
                    [[x, 0, z] for x in (-1, 1) for z in (-1, 1)] +
                    [[0, y, z] for y in (-1, 1) for z in (-1, 1)],
                    dtype=float)
    starts.append(_recentre(np.vstack([cubo, [[0, 0, 1.3], [0, 0, -1.3]]])))

    # 14-point Fibonacci spiral
    kk = np.arange(n) + 0.5
    ga = np.pi * (3 - np.sqrt(5))
    fib = np.zeros((n, 3))
    fib[:, 2] = 1 - 2 * kk / n
    r_ = np.sqrt(np.maximum(0.0, 1 - fib[:, 2] ** 2))
    fib[:, 0] = r_ * np.cos(ga * kk)
    fib[:, 1] = r_ * np.sin(ga * kk)
    starts.append(fib.copy())

    # Jittered copies of structured seeds
    for base in (ico_with_poles := [np.vstack([ico, [0, 0, 1.2], [0, 0, -1.2]])]):
        pass
    jitter_bases = [np.vstack([ico, [0, 0, 1.2], [0, 0, -1.2]]), fib,
                    np.vstack([cubo, [[0, 0, 1.3], [0, 0, -1.3]]])]
    for base in jitter_bases:
        for scale in (0.03, 0.12):
            starts.append(_recentre(base + scale * rng.standard_normal((n, 3))))

    # Cube vertices + face pushes
    c = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)
    starts.append(np.vstack([c, [[1.5, 0, 0], [-1.5, 0, 0], [0, 1.5, 0],
                                 [0, -1.5, 0], [0, 0, 1.5], [0, 0, -1.5]]]))

    # Random starts
    for _ in range(10):
        starts.append(rng.standard_normal((n, 3)))

    best_pts = None
    best_val = -1.0

    def refine(pts):
        for T in (0.3, 0.15, 0.08, 0.04, 0.02, 0.01, 0.005):
            flat = pts.ravel()
            if HAVE_SCIPY:
                def fg(x, T=T):
                    g, v = _grad_soft(x, n, T)
                    return v, g
                res = minimize(fg, flat, jac=True, method="L-BFGS-B",
                               options={"maxiter": 250})
                flat = res.x
            else:
                for _ in range(100):
                    g, v = _grad_soft(flat, n, T)
                    flat = flat + 0.02 * g
            pts = _recentre(flat.reshape(n, 3))
        if HAVE_SCIPY:
            prev = _ratio_sq(pts)
            try:
                res = minimize(lambda x: _exact_ratio_grad(x, n), pts.ravel(),
                               jac=True, method="L-BFGS-B",
                               options={"maxiter": 500, "ftol": 1e-16,
                                        "gtol": 1e-12})
                cand = res.x.reshape(n, 3)
                if np.all(np.isfinite(cand)) and _ratio_sq(cand) > prev:
                    pts = _recentre(cand)
            except Exception:
                pass
        return pts

    for s in starts:
        pts = refine(s.copy())
        val = _ratio_sq(pts)
        if val > best_val:
            best_val = val
            best_pts = pts.copy()

    # Perturbation restarts around the best solution
    if best_pts is not None:
        for scale in (0.02, 0.05, 0.10):
            for _ in range(2):
                pts = refine(_recentre(best_pts + scale * rng.standard_normal((n, 3))))
                val = _ratio_sq(pts)
                if val > best_val:
                    best_val = val
                    best_pts = pts.copy()

    if best_pts is None:
        best_pts = rng.standard_normal((n, 3))
    if not np.all(np.isfinite(best_pts)):
        best_pts = np.zeros((n, 3))
        best_pts[:, 0] = np.arange(n)
    return np.asarray(best_pts, dtype=float)


# EVOLVE-BLOCK-END