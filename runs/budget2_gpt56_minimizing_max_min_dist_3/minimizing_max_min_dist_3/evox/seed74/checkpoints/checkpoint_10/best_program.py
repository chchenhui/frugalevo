# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """Directly optimize a 14-point diameter packing in three dimensions.

    The minimum pairwise distance is fixed to one through inequality
    constraints, while a separate variable bounds every pairwise distance.
    SLSQP minimizes that diameter from several deterministic perturbations
    of the cube--octahedron compound.  The best returned candidate is chosen
    using the evaluator's actual squared min-distance/max-distance ratio.
    """
    # The symmetric cube--octahedron compound is a reliable incumbent.
    s = np.sqrt(3.0)
    base = np.array([
        [-1., -1., -1.], [-1., -1.,  1.], [-1.,  1., -1.],
        [-1.,  1.,  1.], [ 1., -1., -1.], [ 1., -1.,  1.],
        [ 1.,  1., -1.], [ 1.,  1.,  1.],
        [ s, 0., 0.], [-s, 0., 0.], [0., s, 0.],
        [0., -s, 0.], [0., 0., s], [0., 0., -s],
    ])
    ii, jj = np.triu_indices(14, 1)

    def distances(p):
        return np.sum((p[ii] - p[jj]) ** 2, axis=1)

    def ratio(p):
        ds = distances(p)
        return float(ds.min() / ds.max())

    # Scale every initial point set so its shortest distance is exactly one.
    def unit_minimum(p):
        p = p - p.mean(axis=0)
        ds = distances(p)
        return p / np.sqrt(ds.min())

    best = unit_minimum(base)
    best_ratio = ratio(best)

    try:
        from scipy.optimize import minimize
    except ImportError:
        return best

    rng = np.random.default_rng(20260912)

    def constraints(z):
        p = z[:-1].reshape(14, 3)
        diameter = z[-1]
        ds = distances(p)
        # Both sets are nonnegative at feasibility:
        # d_ij^2 >= 1 and d_ij^2 <= diameter^2.
        return np.concatenate((ds - 1.0, diameter * diameter - ds))

    # Include the exact symmetric arrangement, plus reproducible symmetry-
    # breaking starts.  Accepting candidates by their true ratio protects
    # against an incomplete or marginally feasible optimizer termination.
    for trial in range(14):
        if trial == 0:
            start = best.copy()
        else:
            start = unit_minimum(base + rng.normal(scale=0.10, size=(14, 3)))

        d0 = np.sqrt(distances(start).max())
        result = minimize(
            lambda z: z[-1],
            np.concatenate((start.ravel(), [d0])),
            method="SLSQP",
            bounds=[(None, None)] * 42 + [(1.0, None)],
            constraints={"type": "ineq", "fun": constraints},
            options={"maxiter": 1400, "ftol": 1e-11, "disp": False},
        )

        candidate = result.x[:-1].reshape(14, 3)
        value = ratio(candidate)
        if np.isfinite(value) and value > best_ratio:
            best, best_ratio = candidate, value

    return best
    try:
        from scipy.optimize import differential_evolution, minimize
    except ImportError:
        # Deterministic, already-good antipodal fallback: four cube diagonals
        # and the three coordinate axes.
        s = 1.0 / np.sqrt(3.0)
        lines = np.array([
            [ s,  s,  s],
            [ s,  s, -s],
            [ s, -s,  s],
            [-s,  s,  s],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        return np.vstack((lines, -lines))

    upper = np.triu_indices(7, 1)

    def lines_from_angles(x):
        # Rotational freedom fixes line 0 to z and line 1 to the x-z plane.
        theta = np.empty(7)
        phi = np.zeros(7)
        theta[0] = 0.0
        theta[1:] = x[:6]
        phi[2:] = x[6:]

        st = np.sin(theta)
        return np.column_stack((
            st * np.cos(phi),
            st * np.sin(phi),
            np.cos(theta),
        ))

    def coherence(x):
        q = lines_from_angles(x)
        return np.max(np.abs((q @ q.T)[upper]))

    def smooth_coherence(x, power):
        q = lines_from_angles(x)
        c = np.abs((q @ q.T)[upper])
        # A high p-norm is smooth but closely tracks the largest correlation.
        return np.sum(c ** power) ** (1.0 / power)

    bounds = [(0.0, np.pi)] * 6 + [(0.0, 2.0 * np.pi)] * 5

    # Fixed-seed Sobol initialization makes the construction reproducible.
    global_result = differential_evolution(
        lambda x: smooth_coherence(x, 24),
        bounds,
        seed=20260912,
        init="sobol",
        popsize=18,
        maxiter=900,
        tol=1e-9,
        polish=False,
        updating="immediate",
    )

    # First smooth the max objective aggressively, then directly polish the
    # actual maximum absolute inner product.
    refined = minimize(
        lambda x: smooth_coherence(x, 64),
        global_result.x,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 8000, "ftol": 1e-15, "gtol": 1e-10},
    )
    polished = minimize(
        coherence,
        refined.x,
        method="Powell",
        bounds=bounds,
        options={"maxiter": 5000, "xtol": 1e-12, "ftol": 1e-12},
    )

    candidates = [global_result.x, refined.x, polished.x]
    best = min(candidates, key=coherence)
    lines = lines_from_angles(best)

    # Seven antipodal pairs have maximum pairwise distance exactly 2.
    return np.vstack((lines, -lines))


# EVOLVE-BLOCK-END
