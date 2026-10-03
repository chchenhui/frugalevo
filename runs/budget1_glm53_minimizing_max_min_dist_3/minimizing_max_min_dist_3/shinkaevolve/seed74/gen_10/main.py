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

    # Initialize on the unit sphere (packing problems on the sphere are near-optimal here)
    pts = rng.normal(size=(n, d))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)

    def pdist(P):
        diff = P[:, None, :] - P[None, :, :]
        return np.sqrt(np.sum(diff * diff, axis=-1) + 1e-18)

    def optimize(P, iters, lr, temp):
        for _ in range(iters):
            D = pdist(P)
            iu = np.triu_indices(n, k=1)
            dist = D[iu]
            # softmin of distances: -temp * logsumexp(-dist/temp)
            w = np.exp(-(dist - dist.min()) / temp)
            w /= w.sum()
            # weights on pairwise gradients: pull together pairs with small distance
            W = np.zeros((n, n))
            W[iu] = w
            W = W + W.T
            # gradient of softmin w.r.t. points (push apart close pairs)
            diff = P[:, None, :] - P[None, :, :]
            grad = np.einsum('ij,ijk->ik', W - W.T, diff) / dist[:, :, None] * (-temp)
            # normalize gradient magnitude for stability
            gn = np.linalg.norm(grad)
            if gn > 0:
                grad *= lr / gn
            P = P + grad
            # rescale so that dmax == 1 (ratio is scale invariant)
            D = pdist(P)
            P = P / D.max()
        return P

    # Coarse optimization then fine-tuning with small temperature
    pts = optimize(pts, iters=800, lr=0.05, temp=0.1)
    pts = optimize(pts, iters=800, lr=0.02, temp=0.01)

    return pts


# EVOLVE-BLOCK-END