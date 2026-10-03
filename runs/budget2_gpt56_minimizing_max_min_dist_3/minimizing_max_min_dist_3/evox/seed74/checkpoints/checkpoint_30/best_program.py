# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """Construct 14 points by minimizing diameter with unit separation.

    Pairwise squared distances are constrained to be at least one, while a
    scalar bounds all squared distances through its square.  Minimizing this
    scalar directly maximizes the evaluator's squared minimum/maximum ratio.
    The cube--octahedron compound supplies a deterministic initial packing;
    fixed-seed perturbations allow escape from its symmetric local optimum.
    """
    s = np.sqrt(3.0)
    base = np.array([
        [-1., -1., -1.], [-1., -1.,  1.], [-1.,  1., -1.],
        [-1.,  1.,  1.], [ 1., -1., -1.], [ 1., -1.,  1.],
        [ 1.,  1., -1.], [ 1.,  1.,  1.],
        [ s, 0., 0.], [-s, 0., 0.], [0., s, 0.],
        [0., -s, 0.], [0., 0., s], [0., 0., -s],
    ])
    ii, jj = np.triu_indices(14, 1)

    def d2(p):
        return np.sum((p[ii] - p[jj]) ** 2, axis=1)

    def normalize(p):
        p = p - p.mean(axis=0)
        return p / np.sqrt(d2(p).min())

    def quality(p):
        ds = d2(p)
        return ds.min() / ds.max()

    best = normalize(base)
    best_value = quality(best)

    try:
        from scipy.optimize import minimize
    except ImportError:
        return best

    # There are 91 lower-distance and 91 upper-distance constraints.
    # Supplying their exact derivatives is substantially more reliable than
    # finite differences near the highly active optimum.
    def constraints(z):
        p = z[:-1].reshape(14, 3)
        diameter = z[-1]
        ds = d2(p)
        return np.concatenate((ds - 1.0, diameter * diameter - ds))

    def constraint_jacobian(z):
        p = z[:-1].reshape(14, 3)
        diameter = z[-1]
        diff = p[ii] - p[jj]
        jac = np.zeros((182, 43))
        rows = np.arange(91)

        for axis in range(3):
            grad = 2.0 * diff[:, axis]
            jac[rows, 3 * ii + axis] = grad
            jac[rows, 3 * jj + axis] = -grad

        jac[91:, :42] = -jac[:91, :42]
        jac[91:, 42] = 2.0 * diameter
        return jac

    def objective_jacobian(z):
        grad = np.zeros(43)
        grad[-1] = 1.0
        return grad

    rng = np.random.default_rng(20260912)
    scales = (0.025, 0.05, 0.09, 0.14, 0.20, 0.30)

    # Small perturbations refine the symmetric incumbent; larger ones provide
    # deterministic escapes to packings with a different active contact graph.
    for trial in range(30):
        if trial == 0:
            start = best.copy()
        else:
            seed = best if trial % 3 == 0 else base
            start = normalize(seed + rng.normal(
                scale=scales[(trial - 1) % len(scales)], size=(14, 3)
            ))

        result = minimize(
            lambda z: z[-1],
            np.r_[start.ravel(), np.sqrt(d2(start).max())],
            jac=objective_jacobian,
            method="SLSQP",
            bounds=[(None, None)] * 42 + [(1.0, None)],
            constraints={
                "type": "ineq",
                "fun": constraints,
                "jac": constraint_jacobian,
            },
            options={"maxiter": 2200, "ftol": 2e-13, "disp": False},
        )
        candidate = result.x[:-1].reshape(14, 3)
        value = quality(candidate)
        if np.isfinite(value) and value > best_value:
            best, best_value = candidate, value

    return best


# EVOLVE-BLOCK-END
