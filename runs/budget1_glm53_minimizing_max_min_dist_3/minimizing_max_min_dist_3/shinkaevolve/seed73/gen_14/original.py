# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations
from scipy.optimize import minimize


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of
    minimum to maximum pairwise distance.

    Strategy: directly optimize the packing objective
        maximize  min pairwise distance
        subject to max pairwise distance <= 1
    so that dmin/dmax equals the optimized minimum distance.

    Returns
        points: np.ndarray of shape (14,3)
    """
    n, d = 14, 3
    rng = np.random.default_rng(42)

    pairs = list(combinations(range(n), 2))
    ii = np.array([a for a, b in pairs])
    jj = np.array([b for a, b in pairs])

    def pdists(flat):
        P = flat.reshape(n, d)
        return np.linalg.norm(P[ii] - P[jj], axis=1)

    def objective(flat):
        # maximize the minimum pairwise distance
        return -pdists(flat).min()

    def diameter_constraint(flat):
        # max pairwise distance <= 1
        return 1.0 - pdists(flat).max()

    best_pts = None
    best_ratio = -1.0

    for start in range(10):
        if start == 0:
            # deterministic structured start: Fibonacci sphere
            k = np.arange(n) + 0.5
            phi = np.arccos(1.0 - 2.0 * k / n)
            theta = np.pi * (1.0 + 5.0 ** 0.5) * k
            P0 = np.stack([np.cos(theta) * np.sin(phi),
                           np.sin(theta) * np.sin(phi),
                           np.cos(phi)], axis=1)
        else:
            P0 = rng.standard_normal((n, d))
            P0 /= np.linalg.norm(P0, axis=1, keepdims=True)

        res = minimize(
            objective,
            P0.ravel(),
            method='SLSQP',
            constraints={'type': 'ineq', 'fun': diameter_constraint},
            options={'maxiter': 300, 'ftol': 1e-12},
        )

        ds = pdists(res.x)
        dmax = ds.max()
        if dmax <= 0:
            continue
        ratio = ds.min() / dmax
        if ratio > best_ratio:
            best_ratio = ratio
            best_pts = res.x.reshape(n, d)

    if best_pts is None:
        np.random.seed(42)
        best_pts = np.random.randn(n, d)

    return best_pts


# EVOLVE-BLOCK-END