# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y) coordinates of the 14 points.

    """

    n = 14
    d = 3
    rng = np.random.default_rng(42)
    PI, PJ = np.triu_indices(n, k=1)

    from scipy.optimize import minimize

    def pair_dists(P):
        return np.linalg.norm(P[PI] - P[PJ], axis=1)

    def ratio(pts):
        ds = pair_dists(pts)
        dmax = ds.max()
        if dmax <= 0:
            return 0.0
        return ds.min() / dmax

    def normalize(pts):
        pts = pts - pts.mean(0)
        dmax = pair_dists(pts).max() if len(pts) == n else \
            np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1)).max()
        if dmax <= 0:
            return pts
        return pts / dmax

    def repulsion(pts, iters=300, step=0.03):
        pts = normalize(np.array(pts, dtype=float))
        for _ in range(iters):
            diff = pts[PI] - pts[PJ]
            dist = np.linalg.norm(diff, axis=1)
            w = 1.0 / np.maximum(dist, 1e-6) ** 12
            forces = (w[:, None] * diff) / np.maximum(dist, 1e-6)[:, None]
            grad = np.zeros_like(pts)
            np.add.at(grad, PI, forces)
            np.add.at(grad, PJ, -forces)
            nrm = np.linalg.norm(grad, axis=1, keepdims=True)
            nrm[nrm == 0] = 1.0
            pts += step * grad / nrm
            pts = normalize(pts)
            step *= 0.997
        return pts

    def slsqp(pts, beta=80.0):
        pts = normalize(pts)
        def obj(flat):
            ds = pair_dists(flat.reshape(n, d))
            # smooth softmin surrogate for -min(ds)
            m = ds.min()
            return -(m + np.log(np.sum(np.exp(-beta * (ds - m)))) / beta)
        def con(flat):
            return 1.0 - pair_dists(flat.reshape(n, d)).max()
        res = minimize(obj, pts.ravel(), method='SLSQP',
                       constraints={'type': 'ineq', 'fun': con},
                       options={'maxiter': 300, 'ftol': 1e-12})
        return res.x.reshape(n, d)

    def bottleneck_escape(pts):
        # Push the point in the min-distance pair away from its nearest
        # neighbor, then renormalize; keep any strict improvement.
        best = normalize(pts)
        best_r = ratio(best)
        ds = pair_dists(best)
        k = int(np.argmin(ds))
        i, j = PI[k], PJ[k]
        for step in (0.02, 0.01, 0.005, 0.002, 0.001):
            for pt, other in ((i, j), (j, i)):
                cand = best.copy()
                dirv = cand[pt] - cand[other]
                nrm = np.linalg.norm(dirv)
                if nrm <= 0:
                    continue
                cand[pt] = cand[pt] + step * dirv / nrm
                cand = normalize(cand)
                r = ratio(cand)
                if r > best_r + 1e-14:
                    best_r, best = r, cand
        return best, best_r

    def polish(pts, rounds=4):
        best = pts
        best_r = ratio(pts)
        cur = pts
        for _ in range(rounds):
            cur = slsqp(cur)
            r = ratio(cur)
            if r > best_r + 1e-12:
                best_r, best = r, cur
            else:
                break
        # Targeted micro-search on the bottleneck pair + one more SLSQP pass
        cur, r = bottleneck_escape(best)
        if r > best_r + 1e-14:
            cur = slsqp(cur)
            r = ratio(cur)
            if r > best_r:
                best_r, best = r, cur
        return best, best_r

    phi = (1 + np.sqrt(5)) / 2
    ico12 = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ], dtype=float)
    ico12 /= np.linalg.norm(ico12, axis=1, keepdims=True)

    def unit(v):
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)

    seeds = []
    # Icosahedron + 2 perturbed poles (several variants)
    for _ in range(3):
        extra = np.array([[0, 0, 1], [0, 0, -1]]) + 0.05 * rng.standard_normal((2, d))
        seeds.append(np.vstack([ico12, unit(extra)]))
    # Fibonacci spheres with different offsets
    k = np.arange(n) + 0.5
    zeta = 1.0 - 2.0 * k / n
    for off in (0.0, 0.7, 1.9, 3.3):
        th = np.pi * (1 + 5 ** 0.5) * k + off
        seeds.append(np.stack([np.cos(th) * np.sqrt(1 - zeta ** 2),
                               np.sin(th) * np.sqrt(1 - zeta ** 2), zeta], axis=1))
    # Random on sphere
    for _ in range(5):
        seeds.append(unit(rng.standard_normal((n, d))))

    best_pts, best_val = None, -1.0
    for init in seeds:
        pre = repulsion(init)
        for cand in (init, pre):
            pts, r = polish(cand)
            if r > best_val:
                best_val, best_pts = r, pts.copy()

    if best_pts is None:
        best_pts = rng.standard_normal((n, d))

    return np.asarray(best_pts, dtype=float)


# EVOLVE-BLOCK-END