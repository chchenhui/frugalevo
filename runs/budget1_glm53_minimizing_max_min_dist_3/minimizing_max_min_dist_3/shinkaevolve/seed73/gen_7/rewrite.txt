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
    Uses multiple structured starts (icosahedron + poles, Fibonacci sphere)
    plus seeded random starts for robustness.

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
        return -pdists(flat).min()

    def diameter_constraint(flat):
        return 1.0 - pdists(flat).max()

    # Structured start 1: icosahedron (12) + 2 slightly perturbed polar points
    phi_gold = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [-1,  phi_gold, 0], [ 1,  phi_gold, 0],
        [-1, -phi_gold, 0], [ 1, -phi_gold, 0],
        [ 0, -1, phi_gold], [ 0,  1, phi_gold],
        [ 0, -1, -phi_gold], [ 0,  1, -phi_gold],
        [ phi_gold, 0, -1], [ phi_gold, 0,  1],
        [-phi_gold, 0, -1], [-phi_gold, 0,  1],
    ], dtype=float)
    ico /= np.linalg.norm(ico, axis=1, keepdims=True)
    pol = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    pol += 0.05 * np.random.default_rng(0).standard_normal((2, 3))
    pol /= np.linalg.norm(pol, axis=1, keepdims=True)
    icosa_start = np.vstack([ico, pol])

    # Structured start 2: Fibonacci sphere
    k = np.arange(n) + 0.5
    ph = np.arccos(1.0 - 2.0 * k / n)
    th = np.pi * (1.0 + 5.0 ** 0.5) * k
    fib_start = np.stack([np.cos(th) * np.sin(ph),
                          np.sin(th) * np.sin(ph),
                          np.cos(ph)], axis=1)

    starts = [icosa_start, fib_start]
    for _ in range(4):
        P0 = rng.standard_normal((n, d))
        P0 /= np.linalg.norm(P0, axis=1, keepdims=True)
        starts.append(P0)

    best_pts = None
    best_ratio = -1.0

    for P0 in starts:
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