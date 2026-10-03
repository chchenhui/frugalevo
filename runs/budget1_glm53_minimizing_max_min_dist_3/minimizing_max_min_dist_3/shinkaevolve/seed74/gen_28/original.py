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
    for _ in range(12):
        seeds.append(rng.normal(0, 1, (n, dim)))

    best_pts, best_val = None, -1.0
    for s, pts in enumerate(seeds):
        if pts.shape != (n, dim):
            continue
        p, v = _local_search(pts, iters=2500, seed=s)
        if v > best_val:
            best_val, best_pts = v, p

    # final polish with small steps
    p, v = _local_search(best_pts, iters=4000, seed=999)
    if v > best_val:
        best_val, best_pts = v, p

    best_pts, _, _ = _normalize(best_pts)
    return best_pts


# EVOLVE-BLOCK-END