# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """Use exact-Jacobian multistart SLSQP to maximize 14-point separation.

    Two points are fixed one unit apart, all pairwise squared distances are
    constrained to be at most one, and optimization maximizes their common
    lower squared-distance bound from deterministic staggered-ring starts.
    """
    from scipy.optimize import minimize

    # Start from the symmetric two-ring code, mapped so fixed points are
    # (0,0,0) and (1,0,0).  These fixed endpoints remove Euclidean and
    # scale degeneracies while ensuring the diameter is at least one.
    beta = 2.0 * (1.0 - np.cos(np.pi / 6.0))
    h2 = (1.0 - beta) / (5.0 - beta)
    h = np.sqrt(h2)
    r = np.sqrt(1.0 - h2)
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
        return np.vstack(([0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                          x[:-1].reshape(12, 3)))

    def pair_data(x):
        p = unpack(x)
        delta = p[ii] - p[jj]
        return np.einsum("ij,ij->i", delta, delta), delta

    def distances2(x):
        return pair_data(x)[0]

    def constraints(x):
        d2 = distances2(x)
        return np.concatenate((d2 - x[-1], 1.0 - d2))

    def constraint_jacobian(x):
        """Analytic derivatives of lower-separation and diameter bounds."""
        _, delta = pair_data(x)
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
    x0[-1] = distances2(x0).min()
    best = seed.copy()
    best_ratio = x0[-1]
    rng = np.random.default_rng(140314)

    def make_start(coordinates):
        """Set the separation variable to this start's feasible value."""
        probe = np.r_[coordinates, 0.0]
        return np.r_[coordinates, distances2(probe).min()]

    starts = [x0]
    # The best code is a small asymmetric deformation of the highly
    # symmetric ring construction.  Concentrate additional deterministic
    # starts near that deformation while retaining broader basin coverage.
    for scale, count in ((0.0015, 20), (0.004, 28), (0.010, 36),
                         (0.024, 42), (0.050, 42), (0.090, 36),
                         (0.145, 24)):
        for _ in range(count):
            starts.append(make_start(
                x0[:-1] + rng.normal(scale=scale, size=36)
            ))

    for start in starts:
        result = minimize(
            lambda x: -x[-1],
            start,
            jac=lambda x: np.r_[np.zeros(36), -1.0],
            method="SLSQP",
            bounds=[(None, None)] * 36 + [(0.0, 1.0)],
            constraints={
                "type": "ineq",
                "fun": constraints,
                "jac": constraint_jacobian,
            },
            options={"maxiter": 3000, "ftol": 5e-15, "disp": False},
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
