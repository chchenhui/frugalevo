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
        # Start DE in perturbations of a useful triangular boundary-support
        # pattern, rather than requiring a random population to rediscover it.
        # The parameterization still guarantees that every member is feasible.
        lattice = np.array((
            (0.24, 0.015), (0.51, 0.020), (0.76, 0.015),
            (0.015, 0.255), (0.315, 0.245), (0.615, 0.235),
            (0.020, 0.525), (0.310, 0.500),
        ))
        init_rng = np.random.default_rng(719381)
        initial_uv = np.empty((320, 8, 2))
        scales = np.linspace(0.018, 0.155, len(initial_uv))
        for i, scale in enumerate(scales):
            trial = np.maximum(
                lattice + init_rng.normal(0.0, scale, size=(8, 2)),
                0.0,
            )
            total = np.sum(trial, axis=1)
            outside = total > 1.0
            trial[outside] /= total[outside, None]
            initial_uv[i] = trial

        initial_z = np.empty((len(initial_uv), 16))
        initial_z[:, 0::2] = initial_uv[:, :, 0]
        initial_z[:, 1::2] = initial_uv[:, :, 1] / np.maximum(
            1.0 - initial_uv[:, :, 0], 1.0e-14
        )
        initial_z[0] = fallback_z

        global_result = differential_evolution(
            objective,
            bounds=[(0.0, 1.0)] * 16,
            seed=11031987,
            init=initial_z,
            maxiter=700,
            tol=1e-8,
            atol=1e-10,
            polish=False,
            updating="immediate",
            workers=1,
        )
        if global_result.success or objective(global_result.x) < objective(best_z):
            best_z = global_result.x

        # A solve with frozen determinant signs is smooth, but a better point
        # can lie just across a sign-cell boundary.  Re-freezing signs after
        # each genuine improvement provides a short deterministic homotopy.
        best_uv = unpack(best_z)
        best_value = float(np.min(np.abs(determinants(best_uv))))

        for phase in range(4):
            signs = np.sign(determinants(best_uv))
            signs[signs == 0.0] = 1.0

            def local_objective(w: np.ndarray) -> float:
                return -w[-1]

            def local_constraints(w: np.ndarray) -> np.ndarray:
                tail = w[:-1].reshape(8, 2)
                uv = np.vstack((fixed, tail))
                return np.concatenate((
                    signs * determinants(uv) - w[-1],
                    tail.ravel(),
                    1.0 - np.sum(tail, axis=1),
                ))

            local_start = np.concatenate((best_uv[3:].ravel(), [best_value]))
            polished = minimize(
                local_objective,
                local_start,
                method="SLSQP",
                bounds=[(0.0, 1.0)] * 16 + [(0.0, 1.0)],
                constraints={"type": "ineq", "fun": local_constraints},
                options={
                    "maxiter": 1800 + 400 * phase,
                    "ftol": 2e-13,
                    "disp": False,
                },
            )
            if not polished.success or not np.all(np.isfinite(polished.x)):
                break

            candidate_uv = np.vstack((fixed, polished.x[:-1].reshape(8, 2)))
            candidate_value = float(np.min(np.abs(determinants(candidate_uv))))
            if candidate_value > best_value + 1.0e-10:
                best_uv = candidate_uv
                best_value = candidate_value
            else:
                break

        best_z = np.empty(16)
        best_z[0::2] = best_uv[3:, 0]
        best_z[1::2] = best_uv[3:, 1] / np.maximum(
            1.0 - best_uv[3:, 0], 1.0e-14
        )
    except Exception:
        # Returning a feasible configuration is preferable to propagating a
        # missing-optional-dependency or numerical-optimizer failure.
        pass

    uv = unpack(best_z)
    return np.column_stack((uv[:, 0] + 0.5 * uv[:, 1], 0.5 * root3 * uv[:, 1]))


# EVOLVE-BLOCK-END