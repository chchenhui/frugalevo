# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, Bounds, dual_annealing
from scipy.spatial import ConvexHull


def heilbronn_convex13() -> np.ndarray:
    """
    Build a deterministic triangular incumbent, then perform one unrestricted
    affine-gauged dual-annealing search of the exact hull-normalized objective.
    """
    # Fixed seed makes the multi-start stochastic search reproducible.
    rng = np.random.default_rng(20260912)
    hull = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    triples = np.array(
        [(i, j, k) for i in range(13) for j in range(i + 1, 13)
         for k in range(j + 1, 13)], dtype=np.intp
    )

    def project_triangle(p):
        """Project candidate interior points into the fixed right-triangle hull."""
        p = np.maximum(p, 0.001)
        return p / np.maximum(1.0, p.sum(axis=-1, keepdims=True) / 0.998)

    def areas(configs):
        a, b, c = (configs[:, triples[:, q]] for q in range(3))
        return np.abs(
            (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1])
            - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])
        )

    def select(values):
        """Select lexicographically strong candidates using active triangle areas."""
        # Retaining more of the active constraint set produces configurations
        # whose bottleneck is less likely to collapse during final refinement.
        small = np.partition(values, 7, axis=1)[:, :8]
        small.sort(axis=1)
        key = (small[:, 0] * 1.0e6 + small[:, 1] * 1.0e3
               + small[:, 2] * 1.0e1 + small[:, 3:].mean(axis=1))
        return int(np.argmax(key)), small[:, 0]

    best = None
    best_min = -np.inf
    batch = 24

    # Sixteen longer trajectories gave a better exploration/refinement balance
    # than many shorter independent trajectories.
    for restart in range(16):
        u = rng.random((10, 2))
        u[u.sum(axis=1) > 1.0] = 1.0 - u[u.sum(axis=1) > 1.0]
        current = np.vstack((hull, project_triangle(u)))

        # A near-lattice start covers a basin which pure uniform starts seldom
        # enter, while the fixed jitter avoids exact collinear triples.
        if restart == 0:
            current[3:] = project_triangle(np.array([
                [.16, .12], [.42, .10], [.73, .10], [.10, .38],
                [.34, .34], [.59, .29], [.10, .68], [.27, .56],
                [.48, .47], [.20, .23]
            ]) + rng.normal(0.0, 0.018, (10, 2)))

        for iteration in range(2600):
            t = iteration / 2599.0
            sigma = 0.105 * (1.0 - t) ** 1.7 + 0.0012
            candidates = np.repeat(current[None, :, :], batch, axis=0)

            for q in range(1, batch):
                # Draw directly in the valid movable-index interval [3, 13).
                # Independent draws also allow a repeated index in a rare
                # two-point proposal, yielding a useful larger one-point move
                # without ever producing the invalid index 13.
                changed = [int(rng.integers(3, 13))]
                if rng.random() < 0.22:
                    changed.append(int(rng.integers(3, 13)))
                candidates[q, changed] += rng.normal(
                    0.0, sigma, size=(len(changed), 2)
                )

            candidates[:, 3:] = project_triangle(candidates[:, 3:])
            chosen, minima = select(areas(candidates))
            current = candidates[chosen]

            value = float(minima[chosen])
            if value > best_min:
                best_min = value
                best = current.copy()

    # A wider final batch is valuable because only a few nearly equal active
    # triangles remain at this point.  Fine moves equalize those constraints.
    current = best.copy()
    polish_batch = 48
    for iteration in range(1800):
        sigma = 0.006 * (1.0 - iteration / 1799.0) ** 1.8 + 0.00008
        candidates = np.repeat(current[None, :, :], polish_batch, axis=0)
        for q in range(1, polish_batch):
            count = 1 if q < 35 else 2
            changed = rng.choice(10, count, replace=False) + 3
            candidates[q, changed] += rng.normal(0.0, sigma, (count, 2))
        candidates[:, 3:] = project_triangle(candidates[:, 3:])
        chosen, minima = select(areas(candidates))
        current = candidates[chosen]
        if minima[chosen] > best_min:
            best_min = float(minima[chosen])
            best = current.copy()

    # Direct epigraph maximin refinement.  A fixed determinant sign turns every
    # absolute-area condition into a smooth inequality:
    #     sign(det_seed) * det(point_i, point_j, point_k) >= t.
    # Since the fixed outer triangle has area 1/2, determinant t is exactly
    # the evaluator's normalized triangle-area objective.
    if best is not None and np.isfinite(best).all():
        seed = best.copy()
        seed_values = areas(seed[None, :, :])[0]
        signs = np.where(seed_values > 0.0, 1.0, 1.0)

        # Recover oriented rather than absolute determinants for the seed.
        aa, bb, cc = (seed[triples[:, q]] for q in range(3))
        oriented_seed = ((bb[:, 0] - aa[:, 0]) * (cc[:, 1] - aa[:, 1])
                         - (bb[:, 1] - aa[:, 1]) * (cc[:, 0] - aa[:, 0]))
        signs = np.where(oriented_seed >= 0.0, 1.0, -1.0)

        # Variables are ten movable (x,y) pairs followed by epigraph t.
        z0 = np.empty(21, dtype=float)
        z0[:20] = seed[3:].ravel()
        z0[20] = best_min * (1.0 - 1.0e-10)

        def unpack(z):
            return np.vstack((hull, z[:20].reshape(10, 2)))

        def determinant_data(z):
            p = unpack(z)
            a, b, c = (p[triples[:, q]] for q in range(3))
            det = ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                   - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
            return p, a, b, c, det

        def epigraph_constraints(z):
            p, a, b, c, det = determinant_data(z)
            # First ten constraints retain every movable point in the hull.
            boundary = 1.0 - p[3:, 0] - p[3:, 1]
            return np.concatenate((boundary, signs * det - z[20]))

        def epigraph_jacobian(z):
            p, a, b, c, det = determinant_data(z)
            jac = np.zeros((296, 21), dtype=float)

            # Derivatives of the ten x+y<=1 constraints.
            for q in range(10):
                jac[q, 2 * q] = -1.0
                jac[q, 2 * q + 1] = -1.0

            # For det(a,b,c), gradients are:
            # da=(by-cy, cx-bx), db=(cy-ay, ax-cx), dc=(ay-by, bx-ax).
            gradients = (
                np.column_stack((b[:, 1] - c[:, 1], c[:, 0] - b[:, 0])),
                np.column_stack((c[:, 1] - a[:, 1], a[:, 0] - c[:, 0])),
                np.column_stack((a[:, 1] - b[:, 1], b[:, 0] - a[:, 0])),
            )
            for vertex_slot, gradient in enumerate(gradients):
                indices = triples[:, vertex_slot]
                movable = indices >= 3
                rows = np.nonzero(movable)[0] + 10
                cols = 2 * (indices[movable] - 3)
                jac[rows, cols] += signs[movable] * gradient[movable, 0]
                jac[rows, cols + 1] += signs[movable] * gradient[movable, 1]

            jac[10:, 20] = -1.0
            return jac

        objective_jac = np.zeros(21, dtype=float)
        objective_jac[20] = -1.0
        result = minimize(
            lambda z: -z[20],
            z0,
            method="SLSQP",
            jac=lambda z: objective_jac,
            bounds=Bounds(np.zeros(21), np.ones(21)),
            constraints={"type": "ineq", "fun": epigraph_constraints,
                         "jac": epigraph_jacobian},
            options={"maxiter": 2500, "ftol": 1.0e-12, "disp": False},
        )

        # Solver success alone is insufficient: independently validate the
        # original absolute-determinant objective and reject sign violations.
        if result.success and np.isfinite(result.x).all():
            candidate = unpack(result.x)
            candidate_oriented = determinant_data(result.x)[4]
            candidate_min = float(np.min(np.abs(candidate_oriented)))
            inside = np.all(candidate[3:] >= -1.0e-10)
            inside &= np.all(candidate[3:].sum(axis=1) <= 1.0 + 1.0e-10)
            same_branch = np.all(signs * candidate_oriented > 0.0)
            feasible = np.min(epigraph_constraints(result.x)) >= -2.0e-8
            if inside and same_branch and feasible and candidate_min >= best_min:
                best = candidate
                best_min = candidate_min

    if best is None or not np.isfinite(best).all():
        best = np.vstack((hull, np.array([
            [.12, .12], [.38, .10], [.68, .10], [.10, .38], [.34, .32],
            [.58, .25], [.10, .66], [.25, .55], [.46, .43], [.20, .22],
        ])))

    # Search the evaluator's actual affine-invariant objective.  Points zero
    # and one fix translation, rotation, reflection, and scale; all remaining
    # eleven points, including the former third triangular gauge point, are
    # free to become hull vertices.  Thus this explores arbitrary convex-hull
    # combinatorics rather than only configurations contained in `hull`.
    incumbent = best.copy()
    record = {"ratio": -np.inf, "points": incumbent.copy()}

    def exact_loss(flat):
        """Return negative exact min-triangle-area / convex-hull-area ratio."""
        flat = np.asarray(flat, dtype=float)
        if flat.shape != (22,) or not np.isfinite(flat).all():
            return 1.0e6

        points = np.vstack((
            np.array([[0.0, 0.0], [1.0, 0.0]]),
            flat.reshape(11, 2),
        ))
        try:
            hull_area = float(ConvexHull(points).volume)
        except Exception:
            return 1.0e6
        if not np.isfinite(hull_area) or hull_area <= 1.0e-11:
            return 1.0e6

        a, b, c = (points[triples[:, q]] for q in range(3))
        determinants = np.abs(
            (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) -
            (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        )
        smallest = float(np.min(determinants))
        ratio = 0.5 * smallest / hull_area
        if not np.isfinite(ratio) or ratio <= 0.0:
            return 1.0e6

        # Retain the best iterate seen during annealing: dual_annealing's
        # terminal state need not be its best valid state after reannealing.
        if ratio > record["ratio"]:
            record["ratio"] = ratio
            record["points"] = points.copy()
        return -ratio

    # Register the SLSQP incumbent before the unrestricted search, ensuring
    # that a marginal annealing run cannot lower the delivered quality.
    exact_loss(incumbent[2:].ravel())

    try:
        dual_annealing(
            exact_loss,
            bounds=[(-1.5, 2.5)] * 22,
            x0=incumbent[2:].ravel(),
            seed=20260912,
            maxfun=70000,
            initial_temp=5230.0,
            restart_temp_ratio=2.0e-5,
            visit=2.75,
            accept=-6.0,
            no_local_search=True,
        )
    except Exception:
        # Preserve the validated incumbent if a platform-specific Qhull or
        # annealing failure occurs.
        pass

    candidate = record["points"]
    if (candidate.shape == (13, 2) and np.isfinite(candidate).all() and
            record["ratio"] > 0.0):
        try:
            area = float(ConvexHull(candidate).volume)
            a, b, c = (candidate[triples[:, q]] for q in range(3))
            det = np.abs(
                (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) -
                (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
            )
            if area > 1.0e-11 and float(det.min()) > 1.0e-12:
                best = candidate
        except Exception:
            pass
    return best


# EVOLVE-BLOCK-END
