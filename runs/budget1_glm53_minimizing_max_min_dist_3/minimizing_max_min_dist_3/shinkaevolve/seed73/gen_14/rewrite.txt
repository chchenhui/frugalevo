# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of
    minimum to maximum pairwise distance.

    Strategy:
      1. Fast repulsion-based optimization (with diameter normalization)
         explores many inits cheaply.
      2. Best candidates are polished with SLSQP maximizing the minimum
         distance subject to diameter <= 1.

    Returns
        points: np.ndarray of shape (14,3)
    """
    n, d = 14, 3
    idx_i, idx_j = np.triu_indices(n, k=1)
    rng = np.random.default_rng(42)

    def repulse(pts, iters=800, step=0.03):
        pts = pts.copy()
        for _ in range(iters):
            diff = pts[idx_i] - pts[idx_j]
            dist = np.linalg.norm(diff, axis=1)
            dmax = dist.max()
            if dmax <= 0:
                break
            pts /= dmax
            diff = pts[idx_i] - pts[idx_j]
            dist = np.linalg.norm(diff, axis=1)
            w = 1.0 / dist ** 12
            forces = (w[:, None] * diff) / dist[:, None]
            grad = np.zeros_like(pts)
            np.add.at(grad, idx_i, forces)
            np.add.at(grad, idx_j, -forces)
            norm = np.linalg.norm(grad, axis=1, keepdims=True)
            norm[norm == 0] = 1.0
            pts = pts + step * grad / norm
            step *= 0.997
        dist = np.linalg.norm(pts[idx_i] - pts[idx_j], axis=1)
        return pts, dist.min() / dist.max()

    def pdists(flat):
        P = flat.reshape(n, d)
        return np.linalg.norm(P[idx_i] - P[idx_j], axis=1)

    # Structured init: icosahedron + 2 perturbed polar points
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ], dtype=float)
    ico /= np.linalg.norm(ico, axis=1, keepdims=True)

    candidates = []
    for seed in range(12):
        if seed == 0:
            extra = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
            extra[0] += 0.05 * rng.standard_normal(3)
            extra[1] += 0.05 * rng.standard_normal(3)
            extra /= np.linalg.norm(extra, axis=1, keepdims=True)
            init = np.vstack([ico, extra])
        else:
            init = rng.standard_normal((n, d))
            init /= np.linalg.norm(init, axis=1, keepdims=True)
        pts, ratio = repulse(init)
        candidates.append((ratio, pts))

    # Polish the best few candidates with SLSQP
    candidates.sort(key=lambda c: -c[0])
    best_pts = None
    best_ratio = -1.0
    for ratio0, p0 in candidates[:4]:
        res = minimize(
            lambda f: -pdists(f).min(),
            p0.ravel(),
            method='SLSQP',
            constraints={'type': 'ineq', 'fun': lambda f: 1.0 - pdists(f).max()},
            options={'maxiter': 400, 'ftol': 1e-12},
        )
        ds = pdists(res.x)
        dmax = ds.max()
        if dmax <= 0:
            continue
        r = ds.min() / dmax
        if r > best_ratio:
            best_ratio = r
            best_pts = res.x.reshape(n, d)

    if best_pts is None:
        best_pts = candidates[0][1]

    return best_pts


# EVOLVE-BLOCK-END