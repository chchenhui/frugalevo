# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """Minimize diameter subject to all pairwise distances being at least one.

    A gauge-fixed rhombic-dodecahedron configuration initializes SLSQP.
    The optimizer moves twelve points while retaining one shortest pair at
    unit distance, and minimizes an explicit diameter variable.
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
    # Use diameter rather than squared diameter: its upper constraints have
    # better scaling for SLSQP near the active distance constraints.
    x0 = np.r_[start[2:].ravel(), np.sqrt(initial_d2.max())]

    try:
        from scipy.optimize import minimize

        def unpack(x):
            """Recover points after fixing translation, scale, and rotation."""
            return np.vstack((np.zeros(3), [1., 0., 0.], x[:-1].reshape(12, 3)))

        def constraints(x):
            """Require every squared distance to lie between 1 and D squared."""
            d2 = distances(unpack(x))
            return np.r_[d2 - 1.0, x[-1] ** 2 - d2]

        def constraint_jacobian(x):
            """Analytic Jacobian avoids finite-difference noise at active contacts."""
            p = unpack(x)
            jac = np.zeros((2 * len(ii), 37))
            for q, (a, b) in enumerate(zip(ii, jj)):
                delta = 2.0 * (p[a] - p[b])
                if a >= 2:
                    ka = 3 * (a - 2)
                    jac[q, ka:ka + 3] += delta
                if b >= 2:
                    kb = 3 * (b - 2)
                    jac[q, kb:kb + 3] -= delta
                jac[len(ii) + q, :-1] = -jac[q, :-1]
                jac[len(ii) + q, -1] = 2.0 * x[-1]
            return jac

        def objective_jacobian(x):
            g = np.zeros_like(x)
            g[-1] = 1.0
            return g

        best = start
        best_ratio = initial_ratio
        rng = np.random.default_rng(140314)
        starts = [x0]

        # Explore several symmetry-breaking radii.  Each perturbed start gets
        # a diameter coordinate larger than its actual diameter, avoiding an
        # initially violated upper-distance constraint.
        for scale in (0.002, 0.006, 0.012, 0.025, 0.050):
            for _ in range(2):
                z = x0.copy()
                z[:-1] += rng.normal(scale=scale, size=36)
                z[-1] = np.sqrt(distances(unpack(z)).max()) + 0.03
                starts.append(z)

        for z0 in starts:
            result = minimize(
                lambda x: x[-1],
                z0,
                jac=objective_jacobian,
                method="SLSQP",
                bounds=[(None, None)] * 36 + [(1.0, 10.0)],
                constraints={
                    "type": "ineq",
                    "fun": constraints,
                    "jac": constraint_jacobian,
                },
                options={"maxiter": 7000, "ftol": 5e-14, "disp": False},
            )
            candidate = unpack(result.x)
            cd2 = distances(candidate)
            candidate_ratio = cd2.min() / cd2.max()

            # Select by the evaluator's scale-invariant quantity, rather than
            # trusting a solver status that can be conservative near contacts.
            if np.isfinite(candidate_ratio) and candidate_ratio > best_ratio:
                best = candidate
                best_ratio = candidate_ratio

        if best_ratio > initial_ratio:
            return best
    except Exception:
        pass

    return start


# EVOLVE-BLOCK-END
