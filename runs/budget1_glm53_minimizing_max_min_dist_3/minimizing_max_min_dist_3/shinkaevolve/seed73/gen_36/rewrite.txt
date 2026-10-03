# EVOLVE-BLOCK-START
import numpy as np

try:
    from scipy.optimize import minimize
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

N = 14
D = 3


def _pairwise_sq(P):
    diff = P[:, None, :] - P[None, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    return d2, diff


def _true_score(P):
    P = np.asarray(P, dtype=float)
    if not np.all(np.isfinite(P)) or P.shape != (N, D):
        return -1.0
    d2, _ = _pairwise_sq(P)
    iu = np.triu_indices(N, 1)
    lo = d2[iu].min()
    hi = d2[iu].max()
    if hi <= 0:
        return -1.0
    return lo / hi  # squared-distance ratio == (dmin/dmax)^2


def _neg_obj(flat, beta, sphere):
    P = flat.reshape(N, D)
    d2, diff = _pairwise_sq(P)
    iu = np.triu_indices(N, 1)
    dij2 = d2[iu]
    dij = diff[iu]
    ii, jj = iu

    # log-sum-exp softmin / softmax of squared distances
    a = -beta * dij2
    amax = a.max()
    smin = -(amax + np.log(np.sum(np.exp(a - amax)))) / beta
    b = beta * dij2
    bmax = b.max()
    smax = (bmax + np.log(np.sum(np.exp(b - bmax)))) / beta
    F = smin - smax

    w_min = np.exp(a - amax); w_min /= w_min.sum()
    w_max = np.exp(b - bmax); w_max /= w_max.sum()
    g = -w_min - w_max  # dF/d(dij2)

    grad = np.zeros((N, D))
    contrib = 2.0 * g[:, None] * dij
    np.add.at(grad, ii, contrib)
    np.add.at(grad, jj, -contrib)

    if sphere:
        nrm = np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)
        u = P / nrm
        grad = grad - np.sum(grad * u, axis=1, keepdims=True) * u

    return -F, grad.ravel()


def _sanitize(P):
    P = np.asarray(P, dtype=float).reshape(N, D).copy()
    if not np.all(np.isfinite(P)):
        P = np.zeros((N, D))
        P[:, 0] = np.linspace(-1, 1, N)
    return P


def _opt(P, beta, sphere, maxiter):
    # guard before any optimizer call
    P = _sanitize(P)
    if HAVE_SCIPY:
        try:
            res = minimize(_neg_obj, P.ravel(), args=(beta, sphere),
                           jac=True, method="L-BFGS-B",
                           options={"maxiter": maxiter, "maxfun": 4 * maxiter})
            out = res.x.reshape(N, D)
        except Exception:
            out = P
    else:
        out = P
        step = 0.02
        cur = P.ravel().copy()
        for _ in range(maxiter):
            f, g = _neg_obj(cur, beta, sphere)
            cur = cur - step * g
            if sphere:
                q = cur.reshape(N, D)
                q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
                cur = q.ravel()
        out = cur.reshape(N, D)
    out = _sanitize(out)
    if not np.all(np.isfinite(out)) or out.shape != (N, D):
        out = _sanitize(P)
    return out


def _structured_seeds():
    seeds = []
    # cube (8) + scaled octahedron (6): natural 14-point compound
    cube = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)
    # quick 1-D search over octahedron scale
    best_r, best_s = 1.0, -1.0
    for r in np.linspace(0.8, 2.6, 181):
        octa = np.array([[r, 0, 0], [-r, 0, 0], [0, r, 0],
                         [0, -r, 0], [0, 0, r], [0, 0, -r]], dtype=float)
        P = np.vstack([cube, octa])
        s = _true_score(P)
        if s > best_s:
            best_s, best_r = s, r
    octa = np.array([[best_r, 0, 0], [-best_r, 0, 0], [0, best_r, 0],
                     [0, -best_r, 0], [0, 0, best_r], [0, 0, -best_r]], dtype=float)
    seeds.append(np.vstack([cube, octa]))
    # antipodal cube start
    seeds.append(np.vstack([cube[:7], -cube[:7]]))
    return seeds


def min_max_dist_dim3_14() -> np.ndarray:
    best_P, best_s = None, -1.0

    cands = _structured_seeds()
    rng = np.random.RandomState(7)
    for _ in range(12):
        v = rng.randn(N, D)
        cands.append(v / np.linalg.norm(v, axis=1, keepdims=True))

    for P0 in cands:
        P = P0
        for beta in (20.0, 60.0, 150.0):
            P = _opt(P, beta, True, 250)
        # free-space refinement
        for beta in (60.0, 150.0, 400.0):
            P = _opt(P, beta, False, 300)
            P = P - P.mean(axis=0)
        s = _true_score(P)
        if s > best_s:
            best_s, best_P = s, P
        if best_s >= 0.2400:
            break

    # greedy polish from the best configuration
    if best_P is not None:
        rng2 = np.random.RandomState(123)
        for _ in range(8):
            Q = best_P + 0.03 * rng2.randn(N, D)
            for beta in (150.0, 400.0):
                Q = _opt(Q, beta, False, 250)
                Q = Q - Q.mean(axis=0)
            s = _true_score(Q)
            if s > best_s:
                best_s, best_P = s, Q

    if best_P is None:
        np.random.seed(42)
        best_P = np.random.randn(N, D)

    best_P = _sanitize(best_P)
    best_P = best_P - best_P.mean(axis=0)
    scale = np.max(np.abs(best_P))
    if scale > 0:
        best_P = best_P / scale
    if not np.all(np.isfinite(best_P)):
        best_P = np.zeros((N, D))
    return np.asarray(best_P, dtype=float)
# EVOLVE-BLOCK-END