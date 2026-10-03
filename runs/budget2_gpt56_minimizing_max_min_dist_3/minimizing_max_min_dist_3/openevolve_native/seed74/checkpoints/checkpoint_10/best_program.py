# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """Optimize 14 points directly with minimum distance fixed to one.

    A rhombic-dodecahedron configuration supplies a feasible deterministic
    start.  Translation and rotation are removed by fixing one shortest
    pair at (0,0,0) and (1,0,0).  SLSQP then minimizes the allowed diameter
    subject to every pair distance lying in [1, diameter].  The original
    construction is retained if numerical optimization does not improve it.
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
    x0 = np.r_[start[2:].ravel(), np.sqrt(initial_d2.max())]

    try:
        from scipy.optimize import minimize

        def unpack(x):
            return np.vstack((np.zeros(3), [1., 0., 0.], x[:-1].reshape(12, 3)))

        def constraints(x):
            d2 = distances(unpack(x))
            return np.r_[d2 - 1.0, x[-1] ** 2 - d2]

        result = minimize(
            lambda x: x[-1],
            x0,
            method="SLSQP",
            bounds=[(None, None)] * 36 + [(1.0, 10.0)],
            constraints={"type": "ineq", "fun": constraints},
            options={"maxiter": 3000, "ftol": 1e-11, "disp": False},
        )
        candidate = unpack(result.x)
        cd2 = distances(candidate)
        candidate_ratio = cd2.min() / cd2.max()

        # This also protects against an unsuccessful/inaccurate optimizer.
        if np.isfinite(candidate_ratio) and candidate_ratio > initial_ratio:
            return candidate
    except Exception:
        pass

    return start


# EVOLVE-BLOCK-END
