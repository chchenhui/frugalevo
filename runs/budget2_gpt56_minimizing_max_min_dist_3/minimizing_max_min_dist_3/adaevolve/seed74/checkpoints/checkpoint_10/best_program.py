# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def min_max_dist_dim3_14() -> np.ndarray:
    """Use spherical repulsion for a seed, then diameter-constrained SLSQP
    with an archive of nearby local optima to explore multiple contact graphs."""
    n = 14
    rng = np.random.default_rng(20260912)
    ii, jj = np.triu_indices(n, 1)

    def ratio(p):
        q = p[ii] - p[jj]
        d = np.sqrt(np.sum(q * q, axis=1))
        return np.min(d) / np.max(d)

    # A deterministic spherical code is a useful, well-separated starting
    # point, but the final constrained optimization is not restricted to it.
    best = None
    best_ratio = -1.0
    for _ in range(8):
        p = rng.normal(size=(n, 3))
        p /= np.linalg.norm(p, axis=1)[:, None]
        p += 0.015 * rng.normal(size=(n, 3))
        p /= np.linalg.norm(p, axis=1)[:, None]

        for power in (4, 8, 16, 28):
            for it in range(1500):
                q = p[:, None, :] - p[None, :, :]
                d2 = np.sum(q * q, axis=2)
                np.fill_diagonal(d2, 1.0)
                w = d2 ** (-(power + 2.0) / 2.0)
                np.fill_diagonal(w, 0.0)
                f = np.sum(w[:, :, None] * q, axis=1)
                f -= np.sum(f * p, axis=1)[:, None] * p
                s = np.max(np.linalg.norm(f, axis=1))
                if s:
                    p += (0.055 * (1.0 - 0.82 * it / 1499.0) / s) * f
                    p /= np.linalg.norm(p, axis=1)[:, None]

        r = ratio(p)
        if r > best_ratio:
            best, best_ratio = p.copy(), r

    # Set diameter to one.  Fixing point zero removes only translation, which
    # is an exact invariance of the problem.
    d = np.sqrt(np.sum((best[ii] - best[jj]) ** 2, axis=1))
    p0 = (best - best[0]) / np.max(d)
    t0 = ratio(p0) * 0.999999
    z0 = np.r_[p0[1:].ravel(), t0]

    def unpack(z):
        p = np.zeros((n, 3))
        p[1:] = z[:-1].reshape(n - 1, 3)
        return p, z[-1]

    def constraints(z):
        p, t = unpack(z)
        q = p[ii] - p[jj]
        d2 = np.sum(q * q, axis=1)
        return np.r_[d2 - t * t, 1.0 - d2]

    def constraint_jacobian(z):
        """Analytic Jacobian for all lower- and upper-distance constraints."""
        p, t = unpack(z)
        q = p[ii] - p[jj]
        m = len(ii)
        jac = np.zeros((2 * m, 3 * (n - 1) + 1))
        rows = np.arange(m)
        xyz = np.arange(3)

        # Point zero is fixed, so only coordinates of endpoints 1,...,13
        # occur in the optimization vector.  Advanced indexing fills all
        # pair derivatives at once instead of running a Python loop.
        has_i = ii != 0
        ri = rows[has_i]
        ci = 3 * (ii[has_i, None] - 1) + xyz
        jac[ri[:, None], ci] = 2.0 * q[has_i]

        has_j = jj != 0
        rj = rows[has_j]
        cj = 3 * (jj[has_j, None] - 1) + xyz
        jac[rj[:, None], cj] = -2.0 * q[has_j]

        # Derivatives of 1 - d^2 are the negatives of those of d^2.
        jac[m:, :-1] = -jac[:m, :-1]
        jac[:m, -1] = -2.0 * t
        return jac

    result = minimize(
        lambda z: -z[-1],
        z0,
        jac=lambda z: np.r_[np.zeros(3 * (n - 1)), -1.0],
        constraints={"type": "ineq", "fun": constraints, "jac": constraint_jacobian},
        method="SLSQP",
        options={"maxiter": 2500, "ftol": 1e-12, "disp": False},
    )

    # The maximin constraints have several nearby local optima.  Starting
    # additional SLSQP runs from small feasible perturbations of the polished
    # result is substantially more targeted than generating many more random
    # spherical codes.  Keep the best actual geometric ratio, not merely the
    # optimizer's auxiliary t value.
    incumbent = result.x.copy()
    incumbent_points, _ = unpack(incumbent)
    incumbent_ratio = ratio(incumbent_points)

    # Keep several nearly optimal local configurations.  A better packing can
    # lie beyond a contact-graph transition for which every direct perturbation
    # of the current champion returns to the same basin.
    archive = [(incumbent_points.copy(), incumbent_ratio)]

    for scale in (0.003, 0.006, 0.012, 0.02, 0.03, 0.045, 0.065,
                  0.09, 0.12, 0.16):
        for _ in range(18):
            source, _ = archive[rng.integers(len(archive))]
            trial = source.copy()
            trial[1:] += scale * rng.normal(size=(n - 1, 3))

            # Translation is fixed by point zero.  Rescaling makes all upper
            # distance constraints feasible before invoking SLSQP.
            trial /= np.max(np.sqrt(np.sum(
                (trial[ii] - trial[jj]) ** 2, axis=1
            )))
            trial_z = np.r_[trial[1:].ravel(), ratio(trial) * 0.99999]

            local = minimize(
                lambda z: -z[-1],
                trial_z,
                jac=lambda z: np.r_[np.zeros(3 * (n - 1)), -1.0],
                constraints={
                    "type": "ineq",
                    "fun": constraints,
                    "jac": constraint_jacobian,
                },
                method="SLSQP",
                options={"maxiter": 1800, "ftol": 1e-12, "disp": False},
            )
            candidate, _ = unpack(local.x)
            candidate_ratio = ratio(candidate)

            if np.all(np.isfinite(candidate)):
                if candidate_ratio > incumbent_ratio:
                    incumbent = local.x.copy()
                    incumbent_points = candidate.copy()
                    incumbent_ratio = candidate_ratio

                # Near-best alternatives are useful stepping stones into
                # different basins, but discard clearly inferior solutions.
                if candidate_ratio >= 0.992 * incumbent_ratio:
                    archive.append((candidate.copy(), candidate_ratio))
                    archive.sort(key=lambda item: item[1], reverse=True)
                    del archive[12:]

    if np.all(np.isfinite(incumbent_points)) and incumbent_ratio > best_ratio:
        return incumbent_points
    return best


# EVOLVE-BLOCK-END
