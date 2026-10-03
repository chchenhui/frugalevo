# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """Use deterministic multistart simplex annealing, exact simplex projection, and signed maximin SLSQP refinement."""
    # In coordinates (u, v), u >= 0, v >= 0, u + v <= 1 defines the
    # reference triangle.  Triangle determinants in these coordinates are
    # precisely their areas normalized by the containing triangle's area.
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
        return np.abs(
            (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
            - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        )

    def soft_min(q, tau):
        values = areas(q)
        lo = values.min()
        return lo - tau * np.log(np.exp(-(values - lo) / tau).sum())

    def project(p):
        """Euclidean projection onto {(u,v): u>=0, v>=0, u+v<=1}."""
        p = np.maximum(np.asarray(p, dtype=float), 0.0)
        total = p.sum()
        if total <= 1.0:
            return p
        # Radial normalization biases proposals toward simplex corners and is
        # not the nearest feasible point.  The two-variable simplex
        # projection preserves tangential boundary motion, which is important
        # because good Heilbronn configurations commonly have edge points.
        shift = 0.5 * (total - 1.0)
        p -= shift
        if p[0] < 0.0:
            return np.array((0.0, 1.0))
        if p[1] < 0.0:
            return np.array((1.0, 0.0))
        return p

    best = None
    best_value = -1.0

    # The objective has many separated orientation cells.  Extra independent
    # deterministic starts are more valuable than an excessively long single
    # cooling trajectory, while remaining comfortably below the time limit.
    for restart in range(24):
        q = np.empty((11, 2), dtype=float)
        q[:3] = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))

        # Normalized exponential variables are Dirichlet(1,1,1), i.e.
        # genuinely uniform over the simplex, including useful edge starts.
        r = rng.exponential(size=(8, 3))
        r /= r.sum(axis=1, keepdims=True)
        q[3:] = r[:, 1:]

        current = soft_min(q, 0.008)
        for iteration in range(60000):
            fraction = iteration / 59999.0
            tau = 0.008 * (1.0 - fraction) ** 2 + 0.00012
            step = 0.085 * (1.0 - fraction) ** 1.7 + 0.0012
            # The corners are only an initial seed, not forced members of
            # the final configuration.
            index = int(rng.integers(0, 11))
            old = q[index].copy()
            q[index] = project(old + rng.normal(scale=step, size=2))
            candidate = soft_min(q, tau)

            temperature = 0.0015 * (1.0 - fraction) ** 2 + 1.0e-7
            if candidate >= current or rng.random() < np.exp(
                np.clip((candidate - current) / temperature, -60.0, 0.0)
            ):
                current = candidate
            else:
                q[index] = old

        # Strict maximin local moves recover sharp nonsmooth improvements
        # that the soft-min annealing objective may not fully capture.
        for scale in (0.008, 0.004, 0.002, 0.0008):
            improved = True
            while improved:
                improved = False
                for index in rng.permutation(np.arange(11)):
                    old = q[index].copy()
                    old_value = areas(q).min()
                    for direction in (
                        (1, 0), (-1, 0), (0, 1), (0, -1),
                        (1, 1), (-1, -1), (1, -1), (-1, 1),
                    ):
                        q[index] = project(old + scale * np.asarray(direction))
                        value = areas(q).min()
                        if value > old_value + 1.0e-12:
                            old = q[index].copy()
                            old_value = value
                            improved = True
                    q[index] = old

        value = areas(q).min()
        if value > best_value:
            best_value = value
            best = q.copy()

    # Refine all eleven points.  The signs define the orientation cell found
    # by annealing, so every signed determinant is smooth in this local
    # maximin problem.  Retain the annealed arrangement on any solver issue.
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
            q = x[:22].reshape(11, 2)
            a = q[triples[:, 0]]
            b = q[triples[:, 1]]
            c = q[triples[:, 2]]
            determinant = (
                (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
            )
            return signs * determinant - x[22]

        x0 = np.r_[best.ravel(), best_value]
        result = minimize(
            lambda x: -x[22],
            x0,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * 22 + [(0.0, 1.0)],
            constraints=(
                {"type": "ineq", "fun": signed_triangle_constraints},
                {
                    "type": "ineq",
                    "fun": lambda x: 1.0 - x[:22].reshape(11, 2).sum(axis=1),
                },
            ),
            options={"maxiter": 1200, "ftol": 1e-12, "disp": False},
        )
        if result.success:
            refined = result.x[:22].reshape(11, 2)
            if areas(refined).min() > best_value + 1e-12:
                best = refined
    except Exception:
        pass

    # Map (u, v) back to Cartesian coordinates in the requested equilateral
    # triangle: (u + v/2, sqrt(3) v / 2).
    return np.column_stack((
        best[:, 0] + 0.5 * best[:, 1],
        (np.sqrt(3.0) / 2.0) * best[:, 1],
    ))


# EVOLVE-BLOCK-END
