# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations


def heilbronn_triangle11() -> np.ndarray:
    """
    Deterministically search for an 11 point maximin configuration.

    The search is carried out in coordinates (u, v) of the reference simplex
    u >= 0, v >= 0, u + v <= 1.  A determinant in these coordinates is already
    the area normalized by the area of the containing equilateral triangle.
    """
    triples = np.asarray(list(combinations(range(11), 3)), dtype=int)
    root3 = np.sqrt(3.0)

    # Keeping the three extreme vertices is both geometrically natural and
    # reduces the expensive global search from 22 to 16 dimensions.
    fixed = np.array(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))

    def unpack(z: np.ndarray) -> np.ndarray:
        # (a, b) in [0,1]^2 maps exactly onto the reference simplex.
        tail = np.empty((8, 2))
        tail[:, 0] = z[0::2]
        tail[:, 1] = (1.0 - tail[:, 0]) * z[1::2]
        return np.vstack((fixed, tail))

    def determinants(uv: np.ndarray) -> np.ndarray:
        q = uv[triples]
        return ((q[:, 1, 0] - q[:, 0, 0]) * (q[:, 2, 1] - q[:, 0, 1])
                - (q[:, 2, 0] - q[:, 0, 0]) * (q[:, 1, 1] - q[:, 0, 1]))

    def objective(z: np.ndarray) -> float:
        return -float(np.min(np.abs(determinants(unpack(z)))))

    # A valid, deterministic fallback is also a useful initial population
    # member when optional optimization dependencies are absent.
    fallback_uv = np.array((
        (0.0, 0.0), (1.0, 0.0), (0.0, 1.0),
        (0.22, 0.08), (0.51, 0.07), (0.78, 0.08),
        (0.10, 0.31), (0.38, 0.29), (0.66, 0.22),
        (0.19, 0.57), (0.47, 0.45),
    ))
    fallback_z = np.empty(16)
    fallback_z[0::2] = fallback_uv[3:, 0]
    fallback_z[1::2] = fallback_uv[3:, 1] / (1.0 - fallback_uv[3:, 0])

    best_z = fallback_z
    try:
        from scipy.optimize import differential_evolution, minimize

        # The objective is nonsmooth; DE supplies a reliable basin, while the
        # second stage solves the locally fixed-orientation maximin problem.
        global_result = differential_evolution(
            objective,
            bounds=[(0.0, 1.0)] * 16,
            seed=11031987,
            popsize=20,
            maxiter=900,
            tol=1e-8,
            atol=1e-10,
            polish=False,
            updating="immediate",
            workers=1,
        )
        if global_result.success or objective(global_result.x) < objective(best_z):
            best_z = global_result.x

        start_uv = unpack(best_z)
        signs = np.sign(determinants(start_uv))
        signs[signs == 0.0] = 1.0

        # Maximize t subject to every presently oriented determinant being at
        # least t.  This is the exact local formulation of the abs-area goal.
        def local_objective(w: np.ndarray) -> float:
            return -w[-1]

        def local_constraints(w: np.ndarray) -> np.ndarray:
            uv = np.vstack((fixed, w[:-1].reshape(8, 2)))
            return np.concatenate((
                signs * determinants(uv) - w[-1],
                w[:-1].reshape(8, 2).ravel(),
                1.0 - np.sum(w[:-1].reshape(8, 2), axis=1),
            ))

        local_start = np.concatenate((start_uv[3:].ravel(),
                                      [np.min(np.abs(determinants(start_uv)))]))
        polished = minimize(
            local_objective,
            local_start,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * 16 + [(0.0, 1.0)],
            constraints={"type": "ineq", "fun": local_constraints},
            options={"maxiter": 3000, "ftol": 1e-12, "disp": False},
        )
        if polished.success:
            candidate_uv = np.vstack((fixed, polished.x[:-1].reshape(8, 2)))
            if np.min(np.abs(determinants(candidate_uv))) >= -objective(best_z):
                best_z = np.empty(16)
                best_z[0::2] = candidate_uv[3:, 0]
                best_z[1::2] = candidate_uv[3:, 1] / (1.0 - candidate_uv[3:, 0])
    except Exception:
        # Returning a feasible configuration is preferable to propagating a
        # missing-optional-dependency or numerical-optimizer failure.
        pass

    uv = unpack(best_z)
    return np.column_stack((uv[:, 0] + 0.5 * uv[:, 1], 0.5 * root3 * uv[:, 1]))


# EVOLVE-BLOCK-END