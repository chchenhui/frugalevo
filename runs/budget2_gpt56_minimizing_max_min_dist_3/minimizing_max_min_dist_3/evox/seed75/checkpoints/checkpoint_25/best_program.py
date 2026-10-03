# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """Optimize a diameter-one 14-point code from staggered hexagonal rings."""
    from scipy.optimize import minimize

    # A symmetric, already feasible seed: two poles and two staggered
    # latitude rings.  The affine map puts the poles at distance one.
    beta = 2.0 * (1.0 - np.cos(np.pi / 6.0))
    h = np.sqrt((1.0 - beta) / (5.0 - beta))
    r = np.sqrt(1.0 - h * h)
    a = np.arange(6, dtype=float) * (np.pi / 3.0)
    rings = np.vstack((
        np.column_stack((r * np.cos(a), r * np.sin(a), np.full(6, h))),
        np.column_stack((
            r * np.cos(a + np.pi / 6.0),
            r * np.sin(a + np.pi / 6.0),
            np.full(6, -h),
        )),
    ))
    seed = np.vstack((
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        np.column_stack(((rings[:, 2] + 1.0) / 2.0,
                         rings[:, 0] / 2.0, rings[:, 1] / 2.0)),
    ))
    ii, jj = np.triu_indices(14, 1)

    def unpack(x):
        return np.vstack(([0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                          x[:-1].reshape(12, 3)))

    def distances2(x):
        p = unpack(x)
        d = p[ii] - p[jj]
        return np.einsum("ij,ij->i", d, d)

    # x[-1] is the lower squared-distance bound.  Since every squared
    # distance is constrained below one, its optimum is the target ratio.
    def constraints(x):
        d2 = distances2(x)
        return np.concatenate((d2 - x[-1], 1.0 - d2))

    def constraint_jacobian(x):
        """Exact derivatives of all lower- and upper-distance constraints."""
        p = unpack(x)
        d = p[ii] - p[jj]
        m = len(ii)
        g = np.zeros((m, 36))
        rows = np.arange(m)
        left = ii >= 2
        right = jj >= 2

        for axis in range(3):
            g[rows[left], 3 * (ii[left] - 2) + axis] = 2.0 * d[left, axis]
            g[rows[right], 3 * (jj[right] - 2) + axis] = -2.0 * d[right, axis]

        return np.vstack((
            np.column_stack((g, -np.ones(m))),
            np.column_stack((-g, np.zeros(m))),
        ))

    x0 = np.r_[seed[2:].ravel(), 0.0]
    x0[-1] = distances2(x0).min()
    best, best_ratio = seed.copy(), x0[-1]

    rng = np.random.default_rng(140314)
    starts = [x0]
    # More starts at medium and large scales are useful because the best
    # known configuration is slightly asymmetric relative to the ring seed.
    # Analytic constraint derivatives below keep this larger search cheap.
    for scale, count in ((0.008, 8), (0.02, 16), (0.045, 32),
                         (0.08, 48), (0.13, 48), (0.19, 32)):
        for _ in range(count):
            start = np.r_[x0[:-1] + rng.normal(scale=scale, size=36), 0.0]
            start[-1] = distances2(start).min()
            starts.append(start)

    for start in starts:
        result = minimize(
            lambda x: -x[-1],
            start,
            method="SLSQP",
            bounds=[(None, None)] * 36 + [(0.0, 1.0)],
            constraints={
                "type": "ineq",
                "fun": constraints,
                "jac": constraint_jacobian,
            },
            options={"maxiter": 1800, "ftol": 1e-13, "disp": False},
        )
        candidate = unpack(result.x)
        d2 = distances2(result.x)
        ratio = d2.min() / d2.max()
        if np.isfinite(ratio) and ratio > best_ratio:
            best, best_ratio = candidate, ratio

    return best.astype(float)


# EVOLVE-BLOCK-END
