# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    n, d = 14, 3

    def ratio(pts):
        dm = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        iu = np.triu_indices(n, 1)
        dists = dm[iu]
        return dists.min() / dists.max()

    def optimize(seed, iters=4000, lr=0.05):
        rng = np.random.default_rng(seed)
        pts = rng.normal(size=(n, d))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        best_pts = pts.copy()
        best_r = ratio(pts)
        temp = 0.5
        for it in range(iters):
            dm = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
            iu = np.triu_indices(n, 1)
            target = dm[iu].max()
            # soft-min gradient: penalize distances below target
            diffs = pts[:, None, :] - pts[None, :, :]
            with np.errstate(divide="ignore", invalid="ignore"):
                inv = np.where(dm > 1e-12, 1.0 / np.maximum(dm, 1e-12), 0.0)
            w = np.where((dm < target) & (dm > 1e-12), (target - dm) * inv, 0.0)
            # force pushing pairs apart along their connecting axis
            force = np.einsum("ijk,ij->ik", diffs, w) / (n - 1)
            # keep points on sphere via tangential projection of force
            rad = pts / np.maximum(np.linalg.norm(pts, axis=1, keepdims=True), 1e-12)
            force_t = force - rad * np.sum(force * rad, axis=1, keepdims=True)
            step = lr * force_t * (1.0 + temp * rng.standard_normal((n, d)))
            stepn = np.linalg.norm(step, axis=1, keepdims=True)
            mx = stepn.max()
            if mx > 0.3:
                step *= 0.3 / mx
            pts_new = pts + step
            pts_new /= np.linalg.norm(pts_new, axis=1, keepdims=True)
            r = ratio(pts_new)
            if r >= best_r - 0.001:  # annealing acceptance
                pts = pts_new
                if r > best_r:
                    best_r = r
                    best_pts = pts.copy()
            temp *= 0.999
        return best_pts, best_r

    best_pts, best_r = None, -1.0
    for seed in range(6):
        p, r = optimize(seed)
        if r > best_r:
            best_r = r
            best_pts = p

    # rescale so max pairwise distance is 1
    dm = np.linalg.norm(best_pts[:, None, :] - best_pts[None, :, :], axis=-1)
    scale = dm[np.triu_indices(n, 1)].max()
    best_pts = best_pts / scale
    return np.asarray(best_pts, dtype=float)


# EVOLVE-BLOCK-END