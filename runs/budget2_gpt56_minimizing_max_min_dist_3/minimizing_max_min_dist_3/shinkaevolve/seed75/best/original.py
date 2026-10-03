# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Construct 14 points in R^3 with a large minimum pairwise distance
    relative to their diameter.

    Returns
    -------
    np.ndarray
        Finite array of shape (14, 3).
    """

    n = 14
    pair_i, pair_j = np.triu_indices(n, 1)
    m = len(pair_i)

    def pairwise_squared(points: np.ndarray) -> np.ndarray:
        delta = points[pair_i] - points[pair_j]
        return np.einsum("ij,ij->i", delta, delta)

    def normalized(points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        points = points - points[0]
        d2 = pairwise_squared(points)
        diameter = np.sqrt(np.max(d2))
        if not np.isfinite(diameter) or diameter <= 0.0:
            return points
        return points / diameter

    # A symmetric, already fairly good starting configuration:
    # eight cube vertices and six coordinate-axis vertices.
    cube = np.array(
        [[sx, sy, sz]
         for sx in (-1.0, 1.0)
         for sy in (-1.0, 1.0)
         for sz in (-1.0, 1.0)],
        dtype=float,
    )
    axes = np.array(
        [
            [np.sqrt(3.0), 0.0, 0.0],
            [-np.sqrt(3.0), 0.0, 0.0],
            [0.0, np.sqrt(3.0), 0.0],
            [0.0, -np.sqrt(3.0), 0.0],
            [0.0, 0.0, np.sqrt(3.0)],
            [0.0, 0.0, -np.sqrt(3.0)],
        ],
        dtype=float,
    )
    base = normalized(np.vstack((cube, axes)))

    best = base.copy()
    best_ratio2 = float(np.min(pairwise_squared(best)) / np.max(pairwise_squared(best)))

    try:
        from scipy.optimize import minimize

        # Point zero is fixed at the origin.  This removes the translation
        # invariance and makes the constrained optimization better conditioned.
        def unpack(x: np.ndarray) -> np.ndarray:
            return np.vstack((np.zeros((1, 3)), x[: 3 * (n - 1)].reshape(n - 1, 3)))

        def constraint_values(x: np.ndarray) -> np.ndarray:
            points = unpack(x)
            d2 = pairwise_squared(points)
            t = x[-1]
            # d_ij^2 >= t and d_ij^2 <= 1.
            return np.concatenate((d2 - t, 1.0 - d2))

        def constraint_jacobian(x: np.ndarray) -> np.ndarray:
            points = unpack(x)
            jac = np.zeros((2 * m, 3 * (n - 1) + 1), dtype=float)

            for k, (a, b) in enumerate(zip(pair_i, pair_j)):
                diff = 2.0 * (points[a] - points[b])

                if a != 0:
                    aa = 3 * (a - 1)
                    jac[k, aa:aa + 3] += diff
                    jac[m + k, aa:aa + 3] -= diff

                if b != 0:
                    bb = 3 * (b - 1)
                    jac[k, bb:bb + 3] -= diff
                    jac[m + k, bb:bb + 3] += diff

                jac[k, -1] = -1.0

            return jac

        # Fixed perturbations avoid dependence on ambient random state while
        # helping the optimizer escape the highly symmetric initial point.
        rng = np.random.default_rng(20240517)
        starts = [base]
        for scale in (0.015, 0.035, 0.070):
            perturbation = rng.standard_normal((n, 3))
            perturbation[0] = 0.0
            starts.append(normalized(base + scale * perturbation))

        bounds = [(-1.25, 1.25)] * (3 * (n - 1)) + [(0.0, 1.0)]
        constraints = {
            "type": "ineq",
            "fun": constraint_values,
            "jac": constraint_jacobian,
        }

        for start in starts:
            start = normalized(start)
            initial_d2 = pairwise_squared(start)
            x0 = np.concatenate(
                (
                    start[1:].reshape(-1),
                    [0.995 * float(np.min(initial_d2))],
                )
            )

            result = minimize(
                fun=lambda x: -x[-1],
                x0=x0,
                jac=lambda x: np.r_[np.zeros(3 * (n - 1)), -1.0],
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={
                    "maxiter": 450,
                    "ftol": 1.0e-11,
                    "disp": False,
                },
            )

            if result.x is None or not np.all(np.isfinite(result.x)):
                continue

            candidate = normalized(unpack(result.x))
            d2 = pairwise_squared(candidate)
            if np.min(d2) <= 0.0 or np.max(d2) <= 0.0:
                continue

            ratio2 = float(np.min(d2) / np.max(d2))
            if ratio2 > best_ratio2:
                best = candidate
                best_ratio2 = ratio2

    except Exception:
        # The structured normalized configuration remains a valid deterministic
        # fallback when SciPy is not installed or an optimizer is unavailable.
        pass

    return np.asarray(normalized(best), dtype=float)


# EVOLVE-BLOCK-END
