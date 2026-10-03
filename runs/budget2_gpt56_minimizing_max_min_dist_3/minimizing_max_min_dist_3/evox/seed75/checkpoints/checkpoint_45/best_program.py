# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """Optimize a fixed-diameter 14-point packing with feasible multistart SLSQP."""
    from scipy.optimize import minimize

    # The previous two-ring construction is an effective feasible starting
    # point.  Rotate and translate it so that its poles are (0,0,0) and
    # (1,0,0), fixing translation, rotation, and scale degeneracies.
    beta = 2.0 * (1.0 - np.cos(np.pi / 6.0))
    h2 = (1.0 - beta) / (5.0 - beta)
    h = np.sqrt(h2)
    r = np.sqrt(1.0 - h2)
    a = np.arange(6, dtype=float) * (np.pi / 3.0)
    seed = np.vstack((
        np.column_stack((r * np.cos(a), r * np.sin(a), np.full(6, h))),
        np.column_stack((r * np.cos(a + np.pi / 6.0),
                         r * np.sin(a + np.pi / 6.0), np.full(6, -h))),
        [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
    ))
    seed = np.vstack((
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        np.column_stack(((seed[:12, 2] + 1.0) / 2.0,
                         seed[:12, 0] / 2.0, seed[:12, 1] / 2.0)),
    ))

    ii, jj = np.triu_indices(14, 1)

    def unpack(x):
        return np.vstack(([0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                          x[:-1].reshape(12, 3)))

    def pair_data(x):
        p = unpack(x)
        d = p[ii] - p[jj]
        return np.einsum("ij,ij->i", d, d), d

    def distances2(x):
        return pair_data(x)[0]

    def constraints(x):
        d2, _ = pair_data(x)
        return np.concatenate((d2 - x[-1], 1.0 - d2))

    def constraint_jacobian(x):
        """Differentiate all lower- and upper-squared-distance constraints."""
        _, d = pair_data(x)
        m = len(ii)
        grad = np.zeros((m, 36))
        rows = np.arange(m)
        left = ii >= 2
        right = jj >= 2
        for axis in range(3):
            grad[rows[left], 3 * (ii[left] - 2) + axis] = 2.0 * d[left, axis]
            grad[rows[right], 3 * (jj[right] - 2) + axis] = -2.0 * d[right, axis]
        return np.vstack((
            np.column_stack((grad, -np.ones(m))),
            np.column_stack((-grad, np.zeros(m))),
        ))

    d0 = distances2(np.r_[seed[2:].ravel(), 0.0])
    x0 = np.r_[seed[2:].ravel(), d0.min()]
    best = seed.copy()
    best_ratio = d0.min() / d0.max()

    # Give each perturbed geometry its actual minimum separation as its
    # lower-bound variable.  Reusing x0[-1] makes nearly every perturbation
    # infeasible for the lower constraints and wastes SLSQP iterations before
    # the optimizer can search the asymmetric packing basin.
    def make_start(coordinates):
        probe = np.r_[coordinates, 0.0]
        return np.r_[coordinates, distances2(probe).min()]

    rng = np.random.default_rng(140314)
    starts = [x0]
    for scale, count in ((0.006, 12), (0.012, 24), (0.025, 40),
                         (0.045, 56), (0.075, 72), (0.12, 48),
                         (0.18, 24)):
        for _ in range(count):
            starts.append(make_start(
                x0[:-1] + rng.normal(scale=scale, size=36)
            ))

    for start in starts:
        result = minimize(
            lambda x: -x[-1], start, method="SLSQP",
            bounds=[(None, None)] * 36 + [(0.0, 1.0)],
            constraints={
                "type": "ineq",
                "fun": constraints,
                "jac": constraint_jacobian,
            },
            options={"maxiter": 1600, "ftol": 3e-14, "disp": False},
        )
        candidate = unpack(result.x)
        delta = candidate[ii] - candidate[jj]
        d2 = np.einsum("ij,ij->i", delta, delta)
        ratio = d2.min() / d2.max()
        if np.isfinite(ratio) and ratio > best_ratio:
            best_ratio = ratio
            best = candidate

    return best.astype(float)


# EVOLVE-BLOCK-END
