# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """Use deterministic multi-start simulated annealing in barycentric coordinates, followed by constrained refinement."""
    # In (u,v) barycentric coordinates, the containing triangle is
    # u >= 0, v >= 0, u + v <= 1.  The absolute determinant of three
    # such coordinate pairs is exactly the corresponding normalized area.
    rng = np.random.default_rng(110731)
    triples = np.array(
        [(i, j, k) for i in range(11) for j in range(i + 1, 11)
         for k in range(j + 1, 11)],
        dtype=np.intp,
    )

    def areas(q):
        a = q[triples[:, 0]]
        b = q[triples[:, 1]]
        c = q[triples[:, 2]]
        return np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                      - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))

    def soft_min(q, tau):
        ar = areas(q)
        lo = ar.min()
        # A smooth lower approximation to min(area), numerically stable.
        return lo - tau * np.log(np.exp(-(ar - lo) / tau).sum())

    def project(p):
        p = np.maximum(p, 0.0)
        s = p.sum()
        if s > 1.0:
            p /= s
        return p

    best = None
    best_value = -1.0

    # The three corners are retained: this removes affine degeneracy and
    # leaves sixteen well-scaled continuous variables to optimize.
    for restart in range(8):
        q = np.empty((11, 2), dtype=float)
        q[:3] = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))

        # Uniform points in the barycentric simplex, with a different
        # reproducible start on every restart.
        r = rng.random((8, 3))
        r /= r.sum(axis=1, keepdims=True)
        q[3:] = r[:, 1:]

        current = soft_min(q, 0.008)
        for it in range(50000):
            frac = it / 49999.0
            tau = 0.008 * (1.0 - frac) ** 2 + 0.00012
            step = 0.085 * (1.0 - frac) ** 1.7 + 0.0012
            idx = int(rng.integers(3, 11))
            old = q[idx].copy()
            q[idx] = project(old + rng.normal(scale=step, size=2))
            candidate = soft_min(q, tau)

            # A small Metropolis allowance early on avoids poor local
            # maxima caused by a single temporarily limiting triangle.
            temperature = 0.0015 * (1.0 - frac) ** 2 + 1.0e-7
            if candidate >= current or rng.random() < np.exp(
                np.clip((candidate - current) / temperature, -60.0, 0.0)
            ):
                current = candidate
            else:
                q[idx] = old

        # Finish each annealing run with strict maximin coordinate moves.
        for scale in (0.008, 0.004, 0.002, 0.0008):
            improved = True
            while improved:
                improved = False
                order = rng.permutation(np.arange(3, 11))
                for idx in order:
                    old = q[idx].copy()
                    old_value = areas(q).min()
                    for direction in ((1, 0), (-1, 0), (0, 1), (0, -1),
                                      (1, 1), (-1, -1), (1, -1), (-1, 1)):
                        q[idx] = project(old + scale * np.asarray(direction))
                        value = areas(q).min()
                        if value > old_value + 1.0e-12:
                            old = q[idx].copy()
                            old_value = value
                            improved = True
                    q[idx] = old

        value = areas(q).min()
        if value > best_value:
            best_value = value
            best = q.copy()

    # SLSQP is optional: the stochastic construction above is a complete
    # fallback, while this final stage accurately balances active triangles.
    try:
        from scipy.optimize import minimize

        signs = np.sign(
            (best[triples[:, 1], 0] - best[triples[:, 0], 0])
            * (best[triples[:, 2], 1] - best[triples[:, 0], 1])
            - (best[triples[:, 1], 1] - best[triples[:, 0], 1])
            * (best[triples[:, 2], 0] - best[triples[:, 0], 0])
        )
        signs[signs == 0.0] = 1.0

        def signed_triangle_constraints(x):
            q = np.empty((11, 2))
            q[:3] = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
            q[3:] = x[:16].reshape(8, 2)
            a, b, c = q[triples[:, 0]], q[triples[:, 1]], q[triples[:, 2]]
            det = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - \
                  (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
            return signs * det - x[16]

        x0 = np.r_[best[3:].ravel(), best_value]
        constraints = (
            {"type": "ineq", "fun": signed_triangle_constraints},
            {"type": "ineq", "fun": lambda x: 1.0 - x[:16].reshape(8, 2).sum(axis=1)},
        )
        result = minimize(
            lambda x: -x[16], x0, method="SLSQP",
            bounds=[(0.0, 1.0)] * 16 + [(0.0, 1.0)],
            constraints=constraints,
            options={"maxiter": 700, "ftol": 1e-12, "disp": False},
        )
        if result.success:
            refined = np.empty((11, 2))
            refined[:3] = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
            refined[3:] = result.x[:16].reshape(8, 2)
            if areas(refined).min() > best_value:
                best = refined
    except Exception:
        # Returning the deterministic annealed solution is preferable to
        # failing when SciPy is unavailable or a numerical refinement fails.
        pass

    # Convert barycentric coordinates back to Cartesian coordinates.
    return np.column_stack((best[:, 0] + 0.5 * best[:, 1],
                            (np.sqrt(3.0) / 2.0) * best[:, 1]))


# EVOLVE-BLOCK-END
