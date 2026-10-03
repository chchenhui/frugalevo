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
            """Recover the 14-point configuration from the fixed gauge."""
            return np.vstack((np.zeros(3), [1., 0., 0.], x[:-1].reshape(12, 3)))

        def constraints(x):
            """Enforce squared separations between one and diameter squared."""
            d2 = distances(unpack(x))
            return np.r_[d2 - 1.0, x[-1] ** 2 - d2]

        def constraint_jacobian(x):
            """Return the exact Jacobian of every lower and upper distance bound."""
            p = unpack(x)
            delta = p[ii] - p[jj]
            m = len(ii)
            jac = np.zeros((2 * m, 37))
            rows = np.arange(m)

            # Points zero and one are fixed by translation, scale, and rotation.
            mask_i = ii >= 2
            cols_i = 3 * (ii[mask_i] - 2)
            for k in range(3):
                jac[rows[mask_i], cols_i + k] += 2.0 * delta[mask_i, k]

            mask_j = jj >= 2
            cols_j = 3 * (jj[mask_j] - 2)
            for k in range(3):
                jac[rows[mask_j], cols_j + k] -= 2.0 * delta[mask_j, k]

            jac[m:, :36] = -jac[:m, :36]
            jac[m:, 36] = 2.0 * x[-1]
            return jac

        best = start
        best_ratio = initial_ratio
        rng = np.random.default_rng(20260912)

        # Small perturbations break the high symmetry of the initial code.
        # The fixed closest pair remains exactly unit length in every restart.
        starts = [x0]
        for noise, count in ((0.002, 2), (0.008, 3), (0.025, 3)):
            for _ in range(count):
                trial = x0.copy()
                trial[:-1] += noise * rng.normal(size=36)
                trial[-1] = np.sqrt(distances(unpack(trial)).max())
                starts.append(trial)

        for trial in starts:
            result = minimize(
                lambda x: x[-1],
                trial,
                jac=lambda x: np.r_[np.zeros(36), 1.0],
                method="SLSQP",
                bounds=[(None, None)] * 36 + [(1.0, 10.0)],
                constraints={
                    "type": "ineq",
                    "fun": constraints,
                    "jac": constraint_jacobian,
                },
                options={"maxiter": 5000, "ftol": 2e-13, "disp": False},
            )

            if result.x.shape != x0.shape or not np.all(np.isfinite(result.x)):
                continue

            candidate = unpack(result.x)
            cd2 = distances(candidate)
            candidate_ratio = float(cd2.min() / cd2.max())
            if np.isfinite(candidate_ratio) and candidate_ratio > best_ratio:
                best = candidate
                best_ratio = candidate_ratio

        if best_ratio > initial_ratio:
            return best
    except Exception:
        pass

    return start


# EVOLVE-BLOCK-END
