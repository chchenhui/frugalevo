# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """Optimize a diameter-one 14-point packing using exact-Jacobian SLSQP.

    Two fixed endpoints remove translation, rotation, and scale freedom.
    Twelve staggered-ring seed points are refined over deterministic
    multistarts, maximizing a common squared-distance lower bound while
    constraining every pairwise squared distance to be at most one.
    """
    from scipy.optimize import minimize

    beta = 2.0 * (1.0 - np.cos(np.pi / 6.0))
    h = np.sqrt((1.0 - beta) / (5.0 - beta))
    r = np.sqrt(1.0 - h * h)
    a = np.arange(6, dtype=float) * (np.pi / 3.0)

    rings = np.vstack((
        np.column_stack((r * np.cos(a), r * np.sin(a), np.full(6, h))),
        np.column_stack((r * np.cos(a + np.pi / 6.0),
                         r * np.sin(a + np.pi / 6.0), np.full(6, -h))),
    ))
    seed = np.vstack((
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        np.column_stack(((rings[:, 2] + 1.0) / 2.0,
                         rings[:, 0] / 2.0, rings[:, 1] / 2.0)),
    ))

    ii, jj = np.triu_indices(14, 1)

    def unpack(x):
        return np.vstack((
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            x[:-1].reshape(12, 3),
        ))

    def squared_distances(x):
        p = unpack(x)
        d = p[ii] - p[jj]
        return np.einsum("ij,ij->i", d, d)

    def constraints(x):
        d2 = squared_distances(x)
        return np.concatenate((d2 - x[-1], 1.0 - d2))

    def constraint_jacobian(x):
        """Exact derivative of lower- and upper-distance constraints."""
        p = unpack(x)
        delta = p[ii] - p[jj]
        m = len(ii)
        grad = np.zeros((m, 36))
        rows = np.arange(m)
        left = ii >= 2
        right = jj >= 2
        for axis in range(3):
            grad[rows[left], 3 * (ii[left] - 2) + axis] = (
                2.0 * delta[left, axis]
            )
            grad[rows[right], 3 * (jj[right] - 2) + axis] = (
                -2.0 * delta[right, axis]
            )
        return np.vstack((
            np.column_stack((grad, -np.ones(m))),
            np.column_stack((-grad, np.zeros(m))),
        ))

    x0 = np.r_[seed[2:].ravel(), 0.0]
    x0[-1] = squared_distances(x0).min()
    best = seed.copy()
    best_ratio = x0[-1]

    rng = np.random.default_rng(140314)
    starts = [x0]
    for scale, count in ((0.004, 8), (0.012, 12), (0.030, 20),
                         (0.065, 28), (0.12, 24), (0.18, 16)):
        for _ in range(count):
            starts.append(np.r_[
                x0[:-1] + rng.normal(scale=scale, size=36), x0[-1]
            ])

    for start in starts:
        result = minimize(
            lambda x: -x[-1],
            start,
            method="SLSQP",
            jac=lambda x: np.r_[np.zeros(36), -1.0],
            bounds=[(None, None)] * 36 + [(0.0, 1.0)],
            constraints={
                "type": "ineq",
                "fun": constraints,
                "jac": constraint_jacobian,
            },
            options={"maxiter": 1800, "ftol": 2e-14, "disp": False},
        )
        candidate = unpack(result.x)
        d = candidate[ii] - candidate[jj]
        d2 = np.einsum("ij,ij->i", d, d)
        ratio = d2.min() / d2.max()
        if np.isfinite(ratio) and ratio > best_ratio:
            best_ratio = ratio
            best = candidate

    return best.astype(float)


# EVOLVE-BLOCK-END
