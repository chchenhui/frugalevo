# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

N = 14
PAIRS = np.array([(i, j) for i in range(N) for j in range(i + 1, N)], dtype=int)
NP_ = PAIRS.shape[0]


def _ratio(pts):
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    iu = np.triu_indices(N, 1)
    dmax = d[iu].max()
    if dmax <= 0:
        return 0.0
    return d[iu].min() / dmax


def _normalize(pts):
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    iu = np.triu_indices(N, 1)
    dmax = d[iu].max()
    if dmax <= 0:
        return pts
    return pts / dmax


def _unpack(x):
    return x[:3 * N].reshape(N, 3), x[3 * N]


def _c_fun(x):
    p, t = _unpack(x)
    diff = p[:, None, :] - p[None, :, :]
    d2 = (diff ** 2).sum(-1)
    return d2[PAIRS[:, 0], PAIRS[:, 1]] - t * t


def _c_jac(x):
    p, t = _unpack(x)
    J = np.zeros((NP_, 3 * N + 1))
    for k, (i, j) in enumerate(PAIRS):
        g = 2.0 * (p[i] - p[j])
        J[k, 3 * i:3 * i + 3] = g
        J[k, 3 * j:3 * j + 3] = -g
        J[k, -1] = -2.0 * t
    return J


def _obj(x):
    return -x[-1]


def _obj_grad(x):
    g = np.zeros(3 * N + 1)
    g[-1] = -1.0
    return g


def _refine(pts, rounds=3, maxiter=150):
    pts = _normalize(np.asarray(pts, dtype=float).copy())
    best = pts.copy()
    best_r = _ratio(pts)
    cons = ({'type': 'ineq', 'fun': _c_fun, 'jac': _c_jac},)
    bnds = [(None, None)] * (3 * N) + [(1e-9, 1.0)]
    for _ in range(rounds):
        pts = _normalize(pts)
        m = _ratio(pts) * 0.999
        x0 = np.concatenate([pts.ravel(), [max(m, 1e-6)]])
        try:
            res = minimize(_obj, x0, jac=_obj_grad, method='SLSQP',
                           bounds=bnds, constraints=cons,
                           options={'maxiter': maxiter, 'ftol': 1e-10})
            cand = _normalize(_unpack(res.x)[0])
        except Exception:
            break
        r = _ratio(cand)
        if r > best_r:
            best_r, best = r, cand.copy()
            pts = cand
        else:
            pts = cand
            if r < best_r - 1e-4:
                break
    return best, best_r


def min_max_dist_dim3_14() -> np.ndarray:
    phi = (1 + 5 ** 0.5) / 2
    ico = []
    for a in (-1, 1):
        for b in (-phi, phi):
            ico.append((0, a, b))
            ico.append((a, b, 0))
            ico.append((b, 0, a))
    ico = np.array(ico[:12], dtype=float)

    seeds = []

    # Fibonacci spherical spiral (14 near-uniform sphere points)
    k = np.arange(N) + 0.5
    z = 1.0 - 2.0 * k / N
    r = np.sqrt(np.maximum(0.0, 1.0 - z ** 2))
    ang = np.pi * (1.0 + 5 ** 0.5) * k
    seeds.append(np.stack([r * np.cos(ang), r * np.sin(ang), z], axis=1))

    # cuboctahedron (12) + 2 poles, swept radius
    cubo = []
    for x in (-1, 1):
        for y in (-1, 1):
            cubo.append((x, y, 0))
            cubo.append((x, 0, y))
            cubo.append((0, x, y))
    cubo = np.array(cubo[:12], dtype=float)
    for pr in (0.6, 0.9, 1.2, 1.6, 2.2):
        seeds.append(np.vstack([cubo, [[0, 0, pr], [0, 0, -pr]]]))

    # icosahedron (12) + 2 poles, swept radius, z-axis and x-axis variants
    for pr in (0.6, 0.9, 1.2, 1.6, 2.2):
        seeds.append(np.vstack([ico, [[0, 0, pr], [0, 0, -pr]]]))
    for pr in (0.9, 1.5):
        seeds.append(np.vstack([ico, [[pr, 0, 0], [-pr, 0, 0]]]))

    # staggered heptagon rings (2x7)
    ring = []
    for kk in range(7):
        a1 = 2 * np.pi * kk / 7
        a2 = a1 + np.pi / 7
        ring.append(np.array([np.cos(a1), np.sin(a1), 0.7]))
        ring.append(np.array([np.cos(a2), np.sin(a2), -0.7]))
    seeds.append(np.array(ring))

    # rhombic dodecahedron vertices
    rhomb = np.array(
        [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)] +
        [[s * 2, 0, 0] for s in (-1, 1)] +
        [[0, s * 2, 0] for s in (-1, 1)] +
        [[0, 0, s * 2] for s in (-1, 1)], dtype=float)
    seeds.append(rhomb)

    # fixed-seed random configs
    rng = np.random.default_rng(2024)
    for _ in range(6):
        seeds.append(rng.normal(0, 1, (N, 3)))

    best_pts, best_val = None, -1.0
    for s, pts in enumerate(seeds):
        pts = np.asarray(pts, dtype=float)
        if pts.shape != (N, 3):
            continue
        p, v = _refine(pts, rounds=3, maxiter=150)
        if v > best_val:
            best_val, best_pts = v, p

    # extra polish rounds on the winner
    p, v = _refine(best_pts, rounds=4, maxiter=250)
    if v > best_val:
        best_val, best_pts = v, p

    return _normalize(best_pts)


# EVOLVE-BLOCK-END