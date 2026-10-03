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


def _local_search(points, iters=3000, seed=0):
    rng = np.random.default_rng(seed)
    points, d, _ = _normalize(points.copy())
    best = points.copy()
    m, dm = _min_dist(points)
    best_m = m
    T0 = 0.35
    for it in range(iters):
        T = T0 * (1.0 - it / iters) ** 2 + 1e-6
        cand = points.copy()
        # targeted move: perturb one endpoint of closest pair strongly
        if rng.random() < 0.5 and best_m < 1.0:
            iu = np.triu_indices(points.shape[0], 1)
            dd = dm[iu]
            k = iu[0][np.argmin(dd)], iu[1][np.argmin(dd)]
            idx = k[rng.integers(2)]
            cand[idx] += rng.normal(0, T, 3) * 2.0
        else:
            idx = rng.integers(points.shape[0])
            cand[idx] += rng.normal(0, T, 3)
        # occasional pull-together of farthest pair
        if rng.random() < 0.25:
            fi, fj = np.unravel_index(np.argmax(dm), dm.shape)
            cand[fi] += (cand[fj] - cand[fi]) * 0.15 * rng.random()
        cand, _, _ = _normalize(cand)
        m, dm2 = _min_dist(cand)
        if m > best_m or rng.random() < np.exp((m - best_m) / max(T, 1e-9)) * 0.1:
            points, dm = cand, dm2
            if m > best_m:
                best_m = m
                best = cand.copy()
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
