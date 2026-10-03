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
    idx_i, idx_j = np.triu_indices(n, k=1)

    def optimize(pts, iters=1500, step=0.02):
        """Repulsion optimization that directly maximizes dmin/dmax:
        each iteration rescales so the diameter equals 1, then applies
        strong short-range repulsion to push the closest pairs apart."""
        pts = pts.copy()
        for it in range(iters):
            diff = pts[idx_i] - pts[idx_j]
            dist = np.linalg.norm(diff, axis=1)
            dmax = dist.max()
            if dmax <= 0:
                break
            pts /= dmax
            diff = pts[idx_i] - pts[idx_j]
            dist = np.linalg.norm(diff, axis=1)
            ratio = dist.min()  # dmax == 1 here
            # Strongly repel only the near pairs (drives dmin up)
            w = 1.0 / dist ** 12
            forces = (w[:, None] * diff) / dist[:, None]
            grad = np.zeros_like(pts)
            np.add.at(grad, idx_i, forces)
            np.add.at(grad, idx_j, -forces)
            norm = np.linalg.norm(grad, axis=1, keepdims=True)
            norm[norm == 0] = 1.0
            pts = pts + step * grad / norm
            step *= 0.998
        diff = pts[idx_i] - pts[idx_j]
        dist = np.linalg.norm(diff, axis=1)
        return pts, dist.min() / dist.max()

    # Structured init: icosahedron + 2 perturbed polar points
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
        [ 0, -1, phi], [ 0,  1, phi], [ 0, -1, -phi], [ 0,  1, -phi],
        [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1],
    ], dtype=float)
    ico /= np.linalg.norm(ico, axis=1, keepdims=True)

    best_pts = None
    best_ratio = -1.0
    for seed in range(10):
        np.random.seed(seed)
        if seed == 0:
            extra = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
            extra[0] += 0.05 * np.random.randn(3)
            extra[1] += 0.05 * np.random.randn(3)
            extra /= np.linalg.norm(extra, axis=1, keepdims=True)
            init = np.vstack([ico, extra])
        else:
            init = np.random.randn(n, d)
            init /= np.linalg.norm(init, axis=1, keepdims=True)
        pts, ratio = optimize(init)
        if ratio > best_ratio:
            best_ratio = ratio
            best_pts = pts

    points = best_pts
    return points


# EVOLVE-BLOCK-END