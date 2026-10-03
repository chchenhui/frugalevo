# EVOLVE-BLOCK-START
import numpy as np


def _normalize(points):
    d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    i, j = np.unravel_index(np.argmax(d), d.shape)
    dmax = d[i, j]
    if dmax <= 0:
        return points, d, 0.0
    return points / dmax, d / dmax, dmax


def _min_dist(points):
    n = points.shape[0]
    diff = points[:, None, :] - points[None, :, :]
    d = np.linalg.norm(diff, axis=-1)
    iu = np.triu_indices(n, 1)
    return d[iu].min(), d


def _ratio_sq(points):
    n = points.shape[0]
    iu = np.triu_indices(n, 1)
    diff = points[:, None, :] - points[None, :, :]
    d2 = (diff ** 2).sum(-1)
    u = d2[iu]
    mx = u.max()
    if mx <= 0:
        return 0.0
    return u.min() / mx


def _local_search(points, iters=3000, seed=0, step0=0.05):
    """Gradient-style annealing for the spread ratio dmin/dmax.

    Pushes apart pairs near the current minimum distance, mildly pulls
    together pairs near the diameter, adds decaying jitter, then rescales
    the configuration so that dmax = 1.
    """
    rng = np.random.default_rng(seed)
    pts, _, _ = _normalize(np.asarray(points, dtype=float).copy())
    n = pts.shape[0]
    iu = np.triu_indices(n, 1)
    best = pts.copy()
    best_m = _min_dist(pts)[0]
    for it in range(iters):
        f = 1.0 - it / iters
        step = step0 * f + 2e-4
        T = 0.03 * f
        diff = pts[:, None, :] - pts[None, :, :]
        d = np.sqrt((diff ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        m = d[iu].min()
        close = (d < m + 0.12 * f + 0.005).astype(float)
        far = (d > 1.0 - 0.02).astype(float)
        inv = diff / d[..., None]
        gc = (inv * close[..., None]).sum(axis=1)
        gf = (inv * far[..., None]).sum(axis=1)
        disp = step * gc - 0.6 * step * gf + rng.normal(0, T, pts.shape)
        nd = np.linalg.norm(disp, axis=1, keepdims=True)
        cap = 3.0 * step + 0.1
        disp = np.where(nd > cap, disp / np.maximum(nd, 1e-12) * cap, disp)
        pts, _, _ = _normalize(pts + disp)
        mm = _min_dist(pts)[0]
        if mm > best_m:
            best_m = mm
            best = pts.copy()
    return best, best_m


def min_max_dist_dim3_14() -> np.ndarray:
    n, dim = 14, 3
    seeds = []

    # structured seed: icosahedron vertices + center-ish cluster
    phi = (1 + 5 ** 0.5) / 2
    ico = []
    for a in (-1, 1):
        for b in (-phi, phi):
            ico.append((0, a, b))
            ico.append((a, b, 0))
            ico.append((b, 0, a))
    ico = np.array(ico, dtype=float)
    # two-layer: two staggered heptagons
    for k in range(7):
        ang1 = 2 * np.pi * k / 7
        ang2 = ang1 + np.pi / 7
        seeds.append(np.vstack([
            np.array([np.cos(ang1), np.sin(ang1), 0.8]),
            np.array([np.cos(ang2), np.sin(ang2), -0.8]),
        ]))
    seeds.append(ico[:14])

    rng = np.random.default_rng(12345)
    for _ in range(8):
        seeds.append(rng.normal(0, 1, (n, dim)))

    # --- diversified structured starts to escape the 0.24 basin ---

    # Fibonacci spherical spiral: 14 near-uniform points on the unit sphere
    k = np.arange(14) + 0.5
    z = 1.0 - 2.0 * k / 14.0
    r_sq = np.maximum(0.0, 1.0 - z ** 2)
    ang = np.pi * (1.0 + 5.0 ** 0.5) * k
    fib = np.stack([np.sqrt(r_sq) * np.cos(ang), np.sqrt(r_sq) * np.sin(ang), z], axis=1)
    seeds.append(fib)

    # cuboctahedron vertices (12) + 2 poles, sweeping pole radius
    cubo = []
    for x in (-1, 1):
        for y in (-1, 1):
            cubo.append((x, y, 0))
            cubo.append((x, 0, y))
            cubo.append((0, x, y))
    cubo = np.array(cubo[:12], dtype=float)
    for pr in (0.5, 0.8, 1.0, 1.3, 1.7, 2.2, 3.0):
        seeds.append(np.vstack([cubo, [[0, 0, pr], [0, 0, -pr]]]))

    # rhombic dodecahedron vertices: 8 cube corners + 6 octahedron vertices
    rhomb = np.array(
        [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)] +
        [[s * 2, 0, 0] for s in (-1, 1)] +
        [[0, s * 2, 0] for s in (-1, 1)] +
        [[0, 0, s * 2] for s in (-1, 1)], dtype=float)
    seeds.append(rhomb)

    # icosahedron + poles along z, sweeping pole radius (0.5 .. 3.0)
    # and one variant with poles along the x-axis to break symmetry
    for pr in (0.5, 0.8, 1.1, 1.5, 2.0, 2.6, 3.0):
        seeds.append(np.vstack([ico, [[0, 0, pr], [0, 0, -pr]]]))
    for pr in (1.0, 1.6, 2.4):
        seeds.append(np.vstack([ico, [[pr, 0, 0], [-pr, 0, 0]]]))

    # keep the top candidates (not just the single best) for a
    # constrained polish stage
    cands = []  # list of (ratio_sq, pts)
    total = len(seeds)
    for s, pts in enumerate(seeds):
        if pts.shape != (n, dim):
            continue
        # spread iteration budget so runtime stays bounded
        it = 1600 if s < total - 20 else 1000
        p, v = _local_search(pts, iters=it, seed=s)
        cands.append((v * v, p))
    # short polish of the current best, then re-merge
    cands.sort(key=lambda c: -c[0])
    p, v = _local_search(cands[0][1], iters=1500, seed=999)
    cands.append((v * v, p))
    cands.sort(key=lambda c: -c[0])

    # -------- constrained SLSQP polish on the top-3 candidates --------
    # maximize t = dmin^2 subject to d^2_ij - t >= 0 and 1 - d^2_ij >= 0
    from scipy.optimize import minimize as _minimize

    nvar = 3 * n + 1

    def _obj(z):
        return -z[-1]

    def _obj_grad(z):
        g = np.zeros(nvar)
        g[-1] = -1.0
        return g

    _cons = []
    for i in range(n):
        for j in range(i + 1, n):
            def fmin(z, i=i, j=j):
                return (np.sum((z[3*i:3*i+3] - z[3*j:3*j+3])**2)
                        - z[-1])

            def gmin(z, i=i, j=j):
                g = np.zeros(nvar)
                di = 2.0 * (z[3*i:3*i+3] - z[3*j:3*j+3])
                g[3*i:3*i+3] = di
                g[3*j:3*j+3] = -di
                g[-1] = -1.0
                return g

            def fmax(z, i=i, j=j):
                return 1.0 - np.sum((z[3*i:3*i+3] - z[3*j:3*j+3])**2)

            def gmax(z, i=i, j=j):
                g = np.zeros(nvar)
                di = 2.0 * (z[3*i:3*i+3] - z[3*j:3*j+3])
                g[3*i:3*i+3] = -di
                g[3*j:3*j+3] = di
                return g

            _cons.append({'type': 'ineq', 'fun': fmin, 'jac': gmin})
            _cons.append({'type': 'ineq', 'fun': fmax, 'jac': gmax})

    def _slsqp(pts):
        pts, _, _ = _normalize(pts)
        d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
        iu = np.triu_indices(n, 1)
        t0 = d2[iu].min()
        z0 = np.concatenate([pts.ravel(), [t0]])
        try:
            res = _minimize(_obj, z0, jac=_obj_grad, constraints=_cons,
                            method='SLSQP',
                            options={'maxiter': 300, 'ftol': 1e-12})
            p2 = res.x[:-1].reshape(n, dim)
            r2 = _ratio_sq(p2)
            if not np.isfinite(r2):
                return None, 0.0
            return p2, r2
        except Exception:
            return None, 0.0

    rng2 = np.random.default_rng(7)
    best_pts, best_val = cands[0][1], cands[0][0]
    for s0, p0 in cands[:3]:
        for sigma in (0.0, 0.05, 0.15, 0.3):
            q = p0 if sigma == 0.0 else p0 + sigma * rng2.standard_normal(p0.shape)
            p2, r2 = _slsqp(q)
            if p2 is not None and r2 > best_val:
                best_val, best_pts = r2, p2.copy()
            # short annealing jiggle from the perturbed start, then SLSQP
            if sigma > 0.0:
                qa, va = _local_search(q, iters=600, seed=1000 + int(sigma * 100))
                p3, r3 = _slsqp(qa)
                if p3 is not None and r3 > best_val:
                    best_val, best_pts = r3, p3.copy()

    best_pts, _, _ = _normalize(best_pts)
    return best_pts


# EVOLVE-BLOCK-END