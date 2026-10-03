# EVOLVE-BLOCK-START
import numpy as np

try:
    from scipy.optimize import minimize
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


def _true_score(P):
    n = len(P)
    d2 = np.sum((P[:, None, :] - P[None, :, :]) ** 2, axis=-1)
    iu = np.triu_indices(n, 1)
    dm2 = d2[iu].min()
    dx2 = d2[iu].max()
    if dx2 <= 0:
        return 0.0
    return dm2 / dx2


def _neg_obj(flat, n, beta, sphere):
    pts = flat.reshape(n, 3).copy()
    diff = pts[:, None, :] - pts[None, :, :]
    d2 = np.sum(diff * diff, axis=-1)
    iu = np.triu_indices(n, 1)
    dij2 = d2[iu]
    dij = diff[iu]

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
    grad_pts = np.zeros((n, 3))
    idx_i, idx_j = iu
    contrib = 2.0 * g_dij2[:, None] * dij
    np.add.at(grad_pts, idx_i, contrib)
    np.add.at(grad_pts, idx_j, -contrib)

    if sphere:
        norms = np.maximum(np.linalg.norm(pts, axis=1, keepdims=True), 1e-12)
        u = pts / norms
        grad_pts = grad_pts - np.sum(grad_pts * u, axis=1, keepdims=True) * u

    return obj, grad_pts.ravel()


def _opt(P, n, beta, sphere, maxiter):
    if HAVE_SCIPY:
        res = minimize(_neg_obj, P.ravel().copy(), args=(n, beta, sphere),
                       jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "maxfun": 4 * maxiter})
        return res.x.reshape(n, 3)
    return P


def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    best_pts = None
    best = -1.0
    target = 0.2404

    betas = [20.0, 60.0, 150.0, 400.0]

    # --- seed loop: sphere-projected starts with beta annealing ---
    for seed in range(24):
        rng = np.random.RandomState(seed)
        if seed == 0:
            cube = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)
            cube = cube / np.linalg.norm(cube[0])
            P = np.vstack([cube[:7], -cube[:7]])
        else:
            v = rng.randn(n, 3)
            P = v / np.linalg.norm(v, axis=1, keepdims=True)
        for beta in betas:
            P = _opt(P, n, beta, True, 300)
        s = _true_score(P)
        if s > best:
            best = s
            best_pts = P
        if best >= target:
            break

    # --- candidate bookkeeping for bottleneck micro-polish ---
    cands = []

    def note(P):
        s = _true_score(P)
        cands.append((s, P))
        nonlocal best, best_pts
        if s > best:
            best = s
            best_pts = P
        return s

    # --- free-space refinement with increasing sharpness ---
    if best_pts is not None:
        cand = best_pts
        for beta in betas:
            cand = _opt(cand, n, beta, False, 400)
            cand = cand - cand.mean(axis=0)
            note(cand)

    # --- greedy local polish: perturb-and-reoptimize ---
    if best_pts is not None:
        rng = np.random.RandomState(123)
        for _ in range(8):
            cand = best_pts + 0.02 * rng.randn(n, 3)
            for beta in [150.0, 400.0]:
                cand = _opt(cand, n, beta, False, 250)
                cand = cand - cand.mean(axis=0)
            note(cand)

    # --- bottleneck-pair-targeted micro-polish on top-2 candidates ---
    if HAVE_SCIPY and cands:
        cands.sort(key=lambda t: -t[0])
        for s0, P0 in cands[:2]:
            d2 = np.sum((P0[:, None, :] - P0[None, :, :]) ** 2, axis=-1)
            iu = np.triu_indices(n, 1)
            k = np.argmin(d2[iu])
            i, j = iu[0][k], iu[1][k]
            u = P0[i] - P0[j]
            nu = np.linalg.norm(u)
            if nu < 1e-12:
                continue
            u = u / nu
            cand = P0.copy()
            cand[i] += 1e-4 * u
            cand[j] -= 1e-4 * u
            cand = _opt(cand, n, 400.0, False, 200)
            cand = cand - cand.mean(axis=0)
            note(cand)  # fallback check guarantees no regression

    if best_pts is None:
        np.random.seed(42)
        best_pts = np.random.randn(n, 3)

    best_pts = best_pts - best_pts.mean(axis=0)
    scale = np.max(np.abs(best_pts))
    if scale > 0:
        best_pts = best_pts / scale
    return np.asarray(best_pts, dtype=float)


# EVOLVE-BLOCK-END