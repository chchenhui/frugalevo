# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """Construct 11 simplex points by deterministic multi-start global search.

    Points are represented in reference-simplex coordinates, so each signed
    determinant equals normalized triangle area.  Several fixed-seed differential
    evolution runs explore different oriented-matroid cells; the best result is
    then polished by the existing signed max-min SLSQP formulation.
    """
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
        """Fold independent unit-square variables into the reference simplex."""
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

    # The objective has many disconnected order-type basins.  A single
    # best1bin population tends to collapse into the same basin repeatedly, so
    # use a small deterministic portfolio and retain its genuinely best member.
    # Only the retained member is sent to the comparatively expensive local solve.
    search_specs = (
        (194711, "best1bin"),
        (837291, "rand1bin"),
        (527119, "currenttobest1bin"),
    )
    initial = None
    initial_area = -np.inf
    for seed, strategy in search_specs:
        try:
            result = differential_evolution(
                global_objective,
                bounds=[(0.0, 1.0)] * (2 * n),
                seed=seed,
                strategy=strategy,
                popsize=20,
                maxiter=1400,
                tol=1e-8,
                mutation=(0.45, 0.95),
                recombination=0.82,
                polish=False,
                updating="immediate",
                workers=1,
            )
            candidate = fold_to_simplex(result.x)
            candidate_area = float(np.min(np.abs(determinants(candidate))))
            if candidate_area > initial_area:
                initial, initial_area = candidate, candidate_area
        except Exception:
            # Continue with other deterministic starts if one optimizer run fails.
            pass

    if initial is None:
        # This is reached only after unexpected optimizer failures; the existing
        # simplex-valid fallback avoids returning an infeasible construction.
        initial = np.array([
            [0.00, 0.00], [1.00, 0.00], [0.00, 1.00],
            [0.23, 0.03], [0.55, 0.04], [0.82, 0.06],
            [0.08, 0.28], [0.42, 0.25], [0.73, 0.23],
            [0.20, 0.61], [0.51, 0.47],
        ], dtype=float)
        initial_area = float(np.min(np.abs(determinants(initial))))
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

    initial_area = float(np.min(np.abs(determinants(initial))))
    q0 = np.concatenate((initial.ravel(), [0.999 * initial_area]))

    # Explicit Jacobian avoids fragile broadcasting and is evaluated cheaply.
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

    # SLSQP can occasionally declare success at a numerically degraded point;
    # only retain its result when it remains simplex-feasible and preserves the
    # genuine, unsigned max-min objective found by the global stage.
    uv = initial
    if local.success:
        candidate = local.x[:-1].reshape(n, 2)
        candidate_area = float(np.min(np.abs(determinants(candidate))))
        if (np.all(np.isfinite(candidate))
                and np.min(candidate) >= -1e-9
                and np.max(candidate.sum(axis=1)) <= 1.0 + 1e-9
                and candidate_area >= 0.995 * initial_area):
            uv = candidate
    return np.column_stack((
        uv[:, 0] + 0.5 * uv[:, 1],
        (np.sqrt(3.0) / 2.0) * uv[:, 1],
    ))


# EVOLVE-BLOCK-END
