# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """Minimize diameter at unit separation using SLSQP with exact derivatives.

    A rhombic-dodecahedral start is gauge-fixed by pinning one contact pair.
    Every remaining squared distance is constrained to lie between one and
    the squared diameter, and analytic constraint Jacobians are used for a
    high-accuracy final polish.
    """
    axes = np.array(
        [[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.],
         [0., -1., 0.], [0., 0., 1.], [0., 0., -1.]]
    )
    signs = np.array(
        [[-1., -1., -1.], [-1., -1., 1.], [-1., 1., -1.],
         [-1., 1., 1.], [1., -1., -1.], [1., -1., 1.],
         [1., 1., -1.], [1., 1., 1.]]
    )
    base = np.vstack((axes, signs / np.sqrt(3.0)))

    # Put an initially closest axis/cube pair into the fixed gauge.
    order = [0, 6] + [i for i in range(14) if i not in (0, 6)]
    base = base[order]
    v = base[1] - base[0]
    d0 = np.linalg.norm(v)
    e1 = v / d0
    ref = np.array([0., 1., 0.])
    if abs(ref @ e1) > .9:
        ref = np.array([0., 0., 1.])
    e2 = ref - (ref @ e1) * e1
    e2 /= np.linalg.norm(e2)
    e3 = np.cross(e1, e2)
    start = (base - base[0]) @ np.column_stack((e1, e2, e3)) / d0

    ii, jj = np.triu_indices(14, 1)

    def distances(p):
        return np.sum((p[ii] - p[jj]) ** 2, axis=1)

    initial_d2 = distances(start)
    initial_ratio = initial_d2.min() / initial_d2.max()

    # Optimizing D rather than D^2 was empirically substantially more robust
    # for this degenerate contact configuration.
    x0 = np.r_[start[2:].ravel(), np.sqrt(initial_d2.max())]

    try:
        from scipy.optimize import minimize

        def unpack(x):
            return np.vstack((np.zeros(3), [1., 0., 0.], x[:-1].reshape(12, 3)))

        def constraints(x):
            d2 = distances(unpack(x))
            return np.r_[d2 - 1.0, x[-1] ** 2 - d2]

        def constraint_jacobian(x):
            """Exact Jacobian of lower and upper squared-distance bounds."""
            p = unpack(x)
            g = np.zeros((len(ii), 37))
            for k, (a, b) in enumerate(zip(ii, jj)):
                delta = 2.0 * (p[a] - p[b])
                if a >= 2:
                    g[k, 3 * (a - 2):3 * (a - 1)] = delta
                if b >= 2:
                    g[k, 3 * (b - 2):3 * (b - 1)] = -delta
            upper = -g.copy()
            upper[:, -1] = 2.0 * x[-1]
            return np.vstack((g, upper))

        constraint = {
            "type": "ineq",
            "fun": constraints,
            "jac": constraint_jacobian,
        }
        objective_jacobian = lambda x: np.r_[np.zeros(36), 1.0]

        # The first solve identifies the active contact graph.  Re-solving
        # from it with a tighter tolerance is inexpensive and improves the
        # diameter/contact balance that determines the reported ratio.
        result = minimize(
            lambda x: x[-1],
            x0,
            jac=objective_jacobian,
            method="SLSQP",
            bounds=[(None, None)] * 36 + [(1.0, 10.0)],
            constraints=constraint,
            options={"maxiter": 4000, "ftol": 1e-12, "disp": False},
        )
        result = minimize(
            lambda x: x[-1],
            result.x,
            jac=objective_jacobian,
            method="SLSQP",
            bounds=[(None, None)] * 36 + [(1.0, 10.0)],
            constraints=constraint,
            options={"maxiter": 4000, "ftol": 1e-14, "disp": False},
        )
        candidate = unpack(result.x)
        cd2 = distances(candidate)
        candidate_ratio = cd2.min() / cd2.max()

        # Preserve the feasible symmetric construction if SLSQP fails to
        # produce a genuine ratio improvement.
        if np.isfinite(candidate_ratio) and candidate_ratio > initial_ratio:
            return candidate
    except Exception:
        pass

    return start


# EVOLVE-BLOCK-END
