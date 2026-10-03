# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """Use smooth simplex-parameterized differential evolution and SLSQP max-min polishing."""
    try:
        from scipy.optimize import differential_evolution, minimize
    except Exception:
        # Valid deterministic fallback if scipy is unavailable.
        uv = np.array([
            [0.00, 0.00], [1.00, 0.00], [0.00, 1.00],
            [0.23, 0.03], [0.55, 0.04], [0.82, 0.06],
            [0.08, 0.28], [0.42, 0.25], [0.73, 0.23],
            [0.20, 0.61], [0.51, 0.47],
        ])
        return np.column_stack((uv[:, 0] + 0.5 * uv[:, 1],
                                (np.sqrt(3.0) / 2.0) * uv[:, 1]))

    n = 11
    triples = np.array(
        [(i, j, k) for i in range(n) for j in range(i + 1, n)
         for k in range(j + 1, n)],
        dtype=np.intp,
    )

    def fold_to_simplex(z):
        """Reflect unit-square coordinates into the reference simplex."""
        p = np.asarray(z, dtype=float).reshape(n, 2).copy()
        outside = p.sum(axis=1) > 1.0
        p[outside] = 1.0 - p[outside]
        return p

    def determinants(p):
        """Return signed normalized areas for all 165 triples."""
        a = p[triples[:, 0]]
        b = p[triples[:, 1]]
        c = p[triples[:, 2]]
        return ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))

    def global_objective(z):
        p = fold_to_simplex(z)
        return -float(np.min(np.abs(determinants(p))))

    # Explore several reproducible DE populations.  The nonsmooth max-min
    # objective has many locally competitive order types; independent seeded
    # runs are more useful here than spending all evaluations in one cell.
    initial = None
    best_area = -np.inf
    for seed in (194711, 631421, 918273):
        try:
            result = differential_evolution(
                global_objective,
                bounds=[(0.0, 1.0)] * (2 * n),
                seed=seed,
                strategy="best1bin",
                popsize=18,
                maxiter=1100,
                tol=2e-7,
                mutation=(0.45, 0.95),
                recombination=0.82,
                polish=False,
                updating="immediate",
                workers=1,
            )
            candidate = fold_to_simplex(result.x)
            candidate_area = float(np.min(np.abs(determinants(candidate))))
            if candidate_area > best_area:
                initial, best_area = candidate, candidate_area
        except Exception:
            pass

    # This should only be reached if every optimizer invocation failed.
    if initial is None:
        initial = np.array([
            [0.00, 0.00], [1.00, 0.00], [0.00, 1.00],
            [0.23, 0.03], [0.55, 0.04], [0.82, 0.06],
            [0.08, 0.28], [0.42, 0.25], [0.73, 0.23],
            [0.20, 0.61], [0.51, 0.47],
        ], dtype=float)
    signs = np.sign(determinants(initial))
    signs[signs == 0.0] = 1.0

    # Optimize (u_i, v_i, t), with signs fixed to the promising oriented matroid.
    # Signed determinant >= t is smooth within this cell.
    def local_objective(q):
        return -q[-1]

    def constraints(q):
        p = q[:-1].reshape(n, 2)
        return signs * determinants(p) - q[-1]

    def constraint_jacobian(q):
        p = q[:-1].reshape(n, 2)
        jac = np.zeros((len(triples), 2 * n + 1))
        for row, (ia, ib, ic) in enumerate(triples):
            ua, va = p[ia]
            ub, vb = p[ib]
            uc, vc = p[ic]
            s = signs[row]
            # d det((b-a), (c-a)) / d(a_u, a_v)
            # = (v_b-v_c, u_c-u_b).
            jac[row, 2 * ia] = s * (vb - vc)
            jac[row, 2 * ia + 1] = s * (uc - ub)
            jac[row, 2 * ib] = s * (vc - va)
            jac[row, 2 * ib + 1] = s * (ua - uc)
            jac[row, 2 * ic] = s * (va - vb)
            jac[row, 2 * ic + 1] = s * (ub - ua)
            jac[row, -1] = -1.0
        return jac

    q0 = np.concatenate((initial.ravel(),
                         [np.min(np.abs(determinants(initial))) * 0.999]))
    # Supply the sparse block-structured simplex Jacobian explicitly.
    def simplex_jacobian(q):
        j = np.zeros((n, 2 * n + 1))
        for i in range(n):
            j[i, 2 * i:2 * i + 2] = -1.0
        return j

    local = minimize(
        local_objective,
        q0,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * (2 * n) + [(0.0, 1.0)],
        constraints=[
            {"type": "ineq", "fun": constraints, "jac": constraint_jacobian},
            {
                "type": "ineq",
                "fun": lambda q: 1.0 - q[:-1].reshape(n, 2).sum(axis=1),
                "jac": simplex_jacobian,
            },
        ],
        options={"maxiter": 2500, "ftol": 1e-12, "disp": False},
    )

    # SLSQP occasionally reports an iteration-limit status despite returning a
    # feasible point better than its starting point; retain such a point rather
    # than needlessly reverting to the global-search result.
    candidate = local.x[:-1].reshape(n, 2)
    candidate_ok = (
        np.all(np.isfinite(candidate))
        and np.all(candidate >= -1e-10)
        and np.all(candidate.sum(axis=1) <= 1.0 + 1e-10)
    )
    if candidate_ok and np.min(np.abs(determinants(candidate))) >= np.min(np.abs(determinants(initial))):
        uv = candidate
    else:
        uv = initial
    return np.column_stack((
        uv[:, 0] + 0.5 * uv[:, 1],
        (np.sqrt(3.0) / 2.0) * uv[:, 1],
    ))


# EVOLVE-BLOCK-END
