# EVOLVE-BLOCK-START
import numpy as np

try:
    from scipy.optimize import minimize
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

N = 14
D = 3
I_ARR, J_ARR = np.triu_indices(N, k=1)
M = len(I_ARR)
NV = N * D + 1

# Precomputed index arrays for fully vectorized Jacobian construction
ROWS = np.repeat(np.arange(M)[:, None], 3, axis=1)          # (M, 3)
IDX_A = 3 * I_ARR[:, None] + np.arange(3)                   # (M, 3)
IDX_B = 3 * J_ARR[:, None] + np.arange(3)                   # (M, 3)


def pair_dists(pts):
    return np.linalg.norm(pts[I_ARR] - pts[J_ARR], axis=1)


def true_ratio(pts):
    dd = pair_dists(pts)
    mx = dd.max()
    if mx <= 1e-15:
        return 0.0
    return dd.min() / mx


def ratio_sq(pts):
    dd = pair_dists(pts)
    mx = dd.max()
    if mx <= 0:
        return 0.0
    return (dd.min() / mx) ** 2


def recentre(pts):
    pts = pts - pts.mean(axis=0)
    sc = np.linalg.norm(pts, axis=1).max()
    if sc > 0:
        pts = pts / sc
    return pts


def grad_soft(flat, T):
    """Value and analytic gradient of -softmin(d)/softmax(d) (to minimize)."""
    pts = flat.reshape(N, 3)
    diff = pts[:, None, :] - pts[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-12)
    dv = d[I_ARR, J_ARR]
    a = -dv / T
    amax = a.max()
    e_min = np.exp(a - amax)
    w_min = e_min / e_min.sum()
    b = dv / T
    bmax = b.max()
    e_max = np.exp(b - bmax)
    w_max = e_max / e_max.sum()
    soft_min = -T * (amax + np.log(e_min.sum()))
    soft_max = T * (bmax + np.log(e_max.sum()))
    g = -(w_min * soft_max - soft_min * w_max) / (soft_max ** 2)
    grad = np.zeros((N, 3))
    uv = diff[I_ARR, J_ARR] / dv[:, None]
    contrib = g[:, None] * uv
    np.add.at(grad, I_ARR, contrib)
    np.add.at(grad, J_ARR, -contrib)
    return grad.ravel(), -soft_min / soft_max


def exact_ratio_grad(flat):
    """Value and analytic gradient of -(dmin/dmax)^2 (to minimize)."""
    pts = flat.reshape(N, 3)
    diff = pts[:, None, :] - pts[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-12)
    dv = d[I_ARR, J_ARR]
    im = int(np.argmin(dv))
    iM = int(np.argmax(dv))
    dmin = dv[im]
    dmax = max(dv[iM], 1e-12)
    r = dmin / dmax
    grad = np.zeros((N, 3))
    for k, coef in ((im, 2.0 * r / dmax),
                    (iM, -2.0 * r * dmin / (dmax ** 2))):
        aa, bb = I_ARR[k], J_ARR[k]
        u = (pts[aa] - pts[bb]) / dv[k]
        grad[aa] += coef * u
        grad[bb] -= coef * u
    return -r * r, grad.ravel()


def anneal_refine(pts):
    """Fast gradient-based annealing of the soft ratio + exact-ratio polish."""
    for T in (0.30, 0.12, 0.05, 0.02, 0.01, 0.005):
        flat = pts.ravel()
        if HAVE_SCIPY:
            def fg(x, T=T):
                g, v = grad_soft(x, T)
                return v, g
            res = minimize(fg, flat, jac=True, method="L-BFGS-B",
                           options={"maxiter": 200})
            flat = res.x
        else:
            for _ in range(150):
                g, v = grad_soft(flat, T)
                flat = flat + 0.02 * g
        pts = recentre(flat.reshape(N, 3))
    if HAVE_SCIPY:
        prev = ratio_sq(pts)
        try:
            res = minimize(exact_ratio_grad, pts.ravel(), jac=True,
                           method="L-BFGS-B",
                           options={"maxiter": 400, "ftol": 1e-16,
                                    "gtol": 1e-12})
            cand = res.x.reshape(N, 3)
            if np.all(np.isfinite(cand)) and ratio_sq(cand) > prev:
                pts = recentre(cand)
        except Exception:
            pass
    return pts


def build_starts(rng):
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ], dtype=float) / np.sqrt(1 + phi ** 2)

    starts = []
    # Swept-pole icosahedron variants
    for axis in range(3):
        for pz in (1.0, 1.2, 1.5):
            p1 = np.zeros(3); p1[axis] = pz
            p2 = np.zeros(3); p2[axis] = -pz
            starts.append(recentre(np.vstack([ico, p1, p2])))

    # Cuboctahedron + 2 poles
    cubo = np.array([[x, y, 0] for x in (-1, 1) for y in (-1, 1)] +
                    [[x, 0, z] for x in (-1, 1) for z in (-1, 1)] +
                    [[0, y, z] for y in (-1, 1) for z in (-1, 1)],
                    dtype=float)
    cubo_poles = np.vstack([cubo, [[0, 0, 1.3], [0, 0, -1.3]]])
    starts.append(recentre(cubo_poles))

    # 14-point Fibonacci spiral
    kk = np.arange(N) + 0.5
    ga = np.pi * (3 - np.sqrt(5))
    fib = np.zeros((N, 3))
    fib[:, 2] = 1 - 2 * kk / N
    r_ = np.sqrt(np.maximum(0.0, 1 - fib[:, 2] ** 2))
    fib[:, 0] = r_ * np.cos(ga * kk)
    fib[:, 1] = r_ * np.sin(ga * kk)
    starts.append(fib.copy())

    # Cube vertices + face pushes
    cube = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1)
                     for z in (-1, 1)], dtype=float)
    starts.append(np.vstack([cube, [[1.5, 0, 0], [-1.5, 0, 0], [0, 1.5, 0],
                                     [0, -1.5, 0], [0, 0, 1.5], [0, 0, -1.5]]]))

    # Jittered structured seeds
    ico_poles = np.vstack([ico, [0, 0, 1.2], [0, 0, -1.2]])
    for base in (ico_poles, fib, cubo_poles):
        for scale in (0.03, 0.12):
            starts.append(recentre(base + scale * rng.standard_normal((N, 3))))

    # Random starts
    for _ in range(6):
        starts.append(rng.standard_normal((N, 3)))
    return starts


def slsqp_polish(pts):
    """Maximize t s.t. t^2 <= ||pi-pj||^2 <= 1 (vectorized Jacobians)."""
    P = pts - pts.mean(axis=0)
    mx = pair_dists(P).max()
    if mx <= 1e-12:
        return pts, true_ratio(pts)
    P = P / mx
    t0 = pair_dists(P).min()
    x0 = np.concatenate([P.ravel(), [t0]])
    nd = N * D

    def fobj(x):
        return -x[-1]

    def fjac(x):
        J = np.zeros(NV)
        J[-1] = -1.0
        return J

    def cons(x):
        P2 = x[:nd].reshape(N, D)
        dv = np.sum((P2[I_ARR] - P2[J_ARR]) ** 2, axis=1)
        return np.concatenate([1.0 - dv, dv - x[-1] ** 2])

    def cjac(x):
        P2 = x[:nd].reshape(N, D)
        diff = P2[I_ARR] - P2[J_ARR]
        Ju = np.zeros((M, nd))
        Ju[ROWS, IDX_A] = -2.0 * diff
        Ju[ROWS, IDX_B] = 2.0 * diff
        Jl = np.zeros((M, nd))
        Jl[ROWS, IDX_A] = 2.0 * diff
        Jl[ROWS, IDX_B] = -2.0 * diff
        J = np.zeros((2 * M, NV))
        J[:M, :nd] = Ju
        J[M:, :nd] = Jl
        J[M:, -1] = -2.0 * x[-1]
        return J

    res = minimize(fobj, x0, method="SLSQP", jac=fjac,
                   constraints=[{"type": "ineq", "fun": cons, "jac": cjac}],
                   options={"maxiter": 500, "ftol": 1e-12})
    P2 = res.x[:nd].reshape(N, D)
    P2 = P2 - P2.mean(axis=0)
    return P2, true_ratio(P2)


def nelder_polish(pts, rng):
    """Nelder-Mead on the exact ratio as a final safeguard polish."""
    best = pts
    best_r = true_ratio(pts)
    if not HAVE_SCIPY:
        return best, best_r

    def obj(flat):
        return -true_ratio(flat.reshape(N, D))

    x = pts.ravel()
    for scale in (0.02, 0.01):
        res = minimize(obj, x, method="Nelder-Mead",
                       options={"maxiter": 4000, "maxfev": 5000,
                                "xatol": 1e-12, "fatol": 1e-14})
        cand = res.x.reshape(N, D)
        r = true_ratio(cand)
        if r > best_r:
            best_r, best = r, cand
        x = (cand + scale * rng.standard_normal((N, D))).ravel()
    return best, best_r


def min_max_dist_dim3_14() -> np.ndarray:
    """Creates 14 points in 3D maximizing the ratio of min to max distance."""
    rng = np.random.default_rng(42)

    # Stage 1: fast multi-start gradient-annealed soft-ratio refinement
    candidates = []
    for pts0 in build_starts(rng):
        pts = anneal_refine(pts0.copy())
        candidates.append((ratio_sq(pts), pts))
    candidates.sort(key=lambda c: -c[0])
    best_val, best_pts = candidates[0]

    # Perturbation restarts around the global best (cheap gradient stage)
    for scale in (0.05, 0.10):
        pts = anneal_refine(recentre(best_pts + scale * rng.standard_normal((N, 3))))
        val = ratio_sq(pts)
        if val > best_val:
            best_val, best_pts = val, pts.copy()

    # Stage 2: SLSQP constrained polish on top candidates
    if HAVE_SCIPY:
        for r0, pts0 in candidates[:3]:
            for sig in (0.0, 0.05, 0.15):
                P0 = pts0 if sig == 0 else pts0 + sig * rng.standard_normal((N, D))
                try:
                    P2, r2 = slsqp_polish(P0)
                except Exception:
                    continue
                val = ratio_sq(P2)
                if val > best_val:
                    best_val, best_pts = val, P2.copy()
        # Extra SLSQP restarts around current best
        for sig in (0.02, 0.08):
            try:
                P2, _ = slsqp_polish(best_pts + sig * rng.standard_normal((N, D)))
            except Exception:
                continue
            val = ratio_sq(P2)
            if val > best_val:
                best_val, best_pts = val, P2.copy()

    # Stage 3: Nelder-Mead polish on the true ratio
    best_pts, best_r = nelder_polish(best_pts, rng)
    if ratio_sq(best_pts) < best_val:
        # keep whichever is better (nelder works on ratio, not sq)
        pass

    # Normalize output: center, scale so dmax = 1
    best_pts = np.asarray(best_pts, dtype=float)
    if not np.all(np.isfinite(best_pts)):
        best_pts = np.zeros((N, 3))
        best_pts[:, 0] = np.arange(N)
    best_pts = best_pts - best_pts.mean(axis=0)
    dmax = pair_dists(best_pts).max()
    if dmax > 0:
        best_pts = best_pts / dmax
    return best_pts


# EVOLVE-BLOCK-END