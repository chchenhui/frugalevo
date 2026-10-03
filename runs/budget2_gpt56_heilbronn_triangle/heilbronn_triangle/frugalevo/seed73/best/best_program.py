# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def heilbronn_triangle11() -> np.ndarray:
    """Optimize 11 points by deterministic multi-start simulated annealing.

    Points are searched in affine coordinates (u, v) in the simplex
    u >= 0, v >= 0, u + v <= 1, then mapped to the requested equilateral
    triangle by (x, y) = (u + v/2, sqrt(3)*v/2).  The objective is the
    smallest absolute determinant over all 165 triples.
    """
    n = 11
    rng = np.random.default_rng(918273)

    # Precompute the 165 unordered triples once.
    triples = np.array(
        [(i, j, k) for i in range(n) for j in range(i + 1, n)
         for k in range(j + 1, n)],
        dtype=np.int64,
    )

    def project_simplex(z: np.ndarray) -> np.ndarray:
        """Project rows onto the triangle u >= 0, v >= 0, u + v <= 1."""
        z = np.maximum(z, 0.0)
        excess = z[:, 0] + z[:, 1] - 1.0
        mask = excess > 0.0
        if np.any(mask):
            # Projection onto the hypotenuse of the 2D simplex.
            z[mask] -= (excess[mask, None] * 0.5)
            z[mask] = np.maximum(z[mask], 0.0)
            # Handle the rare case where the previous projection reaches
            # an axis and numerical clipping is needed.
            sums = z[mask].sum(axis=1)
            over = sums > 1.0
            if np.any(over):
                zmask = z[mask]
                zmask[over] /= sums[over, None]
                z[mask] = zmask
        return z

    def objective(p: np.ndarray) -> float:
        a = p[triples[:, 0]]
        b = p[triples[:, 1]]
        c = p[triples[:, 2]]
        det = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
        det -= (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        return float(np.min(np.abs(det)))

    best_points = None
    best_value = -1.0

    # Use 30 deterministic interlaced five-point radial-shell seeds and 48
    # unrestricted uniform-simplex seeds.  The shell construction supplies a
    # different order type: a displaced center, an inner pentagonal shell,
    # and a phase-shifted outer pentagonal shell.
    for restart in range(78):
        if restart < 30:
            center = np.array([1.0 / 3.0, 1.0 / 3.0],
                              dtype=np.float64)
            phi = (7 * restart % 30) / 150.0
            psi = (11 * restart % 30) / 150.0
            inner_radius = 0.145 + 0.004 * (restart % 5)
            outer_radius = 0.285 + 0.006 * ((3 * restart) % 5)

            q = np.empty((n, 2), dtype=np.float64)
            angle = 2.0 * np.pi * restart / 30.0
            q[0] = center + 0.008 * np.array(
                [np.cos(angle), np.sin(angle)], dtype=np.float64
            )

            shell_index = 1
            for k in range(5):
                theta = 2.0 * np.pi * (k / 5.0 + phi)
                displacement = inner_radius * np.array(
                    [np.cos(theta), np.sin(theta)], dtype=np.float64
                )
                raw = center + displacement
                limits = [1.0]
                if displacement[0] < 0.0:
                    limits.append(center[0] / (-displacement[0]))
                if displacement[1] < 0.0:
                    limits.append(center[1] / (-displacement[1]))
                if displacement.sum() > 0.0:
                    limits.append(
                        (1.0 - center.sum()) / displacement.sum()
                    )
                q[shell_index] = center + displacement * (
                    0.985 * min(limits)
                )
                shell_index += 1

            for k in range(5):
                theta = 2.0 * np.pi * ((k + 0.5) / 5.0 + psi)
                displacement = outer_radius * np.array(
                    [np.cos(theta), np.sin(theta)], dtype=np.float64
                )
                raw = center + displacement
                limits = [1.0]
                if displacement[0] < 0.0:
                    limits.append(center[0] / (-displacement[0]))
                if displacement[1] < 0.0:
                    limits.append(center[1] / (-displacement[1]))
                if displacement.sum() > 0.0:
                    limits.append(
                        (1.0 - center.sum()) / displacement.sum()
                    )
                q[shell_index] = center + displacement * (
                    0.985 * min(limits)
                )
                shell_index += 1

            q = project_simplex(q)
        else:
            bary = rng.dirichlet((1.0, 1.0, 1.0), size=n)
            q = bary[:, :2].copy()

        current_value = objective(q)

        for iteration in range(5000 if restart < 30 else 8500):
            fraction = iteration / 8500.0
            late = max(0.0, 1.0 - fraction)
            scale = 0.085 * late ** 1.45 + 0.00020
            temperature = 0.010 * late ** 2.4 + 2.0e-8

            candidate = q.copy()
            if rng.random() < 0.30:
                indices = rng.choice(n, size=2, replace=False)
                candidate[indices] += rng.normal(0.0, scale, (2, 2))
            else:
                index = int(rng.integers(n))
                candidate[index] += rng.normal(0.0, scale, 2)

            candidate = project_simplex(candidate)
            candidate_value = objective(candidate)
            delta = candidate_value - current_value

            if delta >= 0.0 or rng.random() < np.exp(delta / temperature):
                q = candidate
                current_value = candidate_value

            if current_value > best_value:
                best_value = current_value
                best_points = q.copy()

        # Deterministic local polishing using both single-point and coordinated
        # two-point moves.  Coupled moves are important when several active
        # minimum-area triples share vertices.
        for iteration in range(900 if restart < 30 else 1700):
            fraction = iteration / 1700.0
            scale = 0.0020 * (1.0 - fraction) ** 1.3 + 1.0e-5
            candidate = q.copy()

            if rng.random() < 0.45:
                indices = rng.choice(n, size=2, replace=False)
                candidate[indices] += rng.normal(0.0, scale, (2, 2))
            else:
                index = int(rng.integers(n))
                candidate[index] += rng.normal(0.0, scale, 2)

            candidate = project_simplex(candidate)
            candidate_value = objective(candidate)
            if candidate_value >= current_value:
                q = candidate
                current_value = candidate_value
                if current_value > best_value:
                    best_value = current_value
                    best_points = q.copy()

    """Polish the incumbent with a signed determinant epigraph SLSQP solve.

    The locally fixed orientation of every nonzero determinant is used to
    maximize a shared lower bound across all 165 signed triangle areas, while
    explicit simplex constraints preserve feasibility.
    """
    q = best_points.copy()
    incumbent_value = objective(q)

    def determinant_values(p: np.ndarray) -> np.ndarray:
        a = p[triples[:, 0]]
        b = p[triples[:, 1]]
        c = p[triples[:, 2]]
        return ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))

    def determinant_jacobian(p: np.ndarray) -> np.ndarray:
        """Return the analytic Jacobian of all signed determinants."""
        jac = np.zeros((len(triples), 2 * n), dtype=np.float64)
        a = p[triples[:, 0]]
        b = p[triples[:, 1]]
        c = p[triples[:, 2]]

        ia = 2 * triples[:, 0]
        ib = 2 * triples[:, 1]
        ic = 2 * triples[:, 2]

        jac[np.arange(len(triples)), ia] = b[:, 1] - c[:, 1]
        jac[np.arange(len(triples)), ia + 1] = c[:, 0] - b[:, 0]
        jac[np.arange(len(triples)), ib] = c[:, 1] - a[:, 1]
        jac[np.arange(len(triples)), ib + 1] = a[:, 0] - c[:, 0]
        jac[np.arange(len(triples)), ic] = a[:, 1] - b[:, 1]
        jac[np.arange(len(triples)), ic + 1] = b[:, 0] - a[:, 0]
        return jac

    # Fix orientations only at the incumbent.  Exact zero signs are assigned
    # consistently; the unsigned objective remains the final acceptance test.
    signs = np.sign(determinant_values(q))
    signs[signs == 0.0] = 1.0

    def epigraph_constraints(x: np.ndarray) -> np.ndarray:
        """Return signed determinant and simplex inequality residuals."""
        p = x[:-1].reshape(n, 2)
        signed = signs * determinant_values(p) - x[-1]
        simplex = 1.0 - p[:, 0] - p[:, 1]
        return np.concatenate((signed, p[:, 0], p[:, 1], simplex))

    def epigraph_jacobian(x: np.ndarray) -> np.ndarray:
        """Return the analytic Jacobian of all epigraph inequalities."""
        p = x[:-1].reshape(n, 2)
        rows = len(triples) + 3 * n
        jac = np.zeros((rows, 2 * n + 1), dtype=np.float64)
        jac[:len(triples), :-1] = signs[:, None] * determinant_jacobian(p)
        jac[:len(triples), -1] = -1.0

        offset = len(triples)
        for i in range(n):
            jac[offset + i, 2 * i] = 1.0
            jac[offset + n + i, 2 * i + 1] = 1.0
            jac[offset + 2 * n + i, 2 * i] = -1.0
            jac[offset + 2 * n + i, 2 * i + 1] = -1.0
        return jac

    def epigraph_objective(x: np.ndarray) -> float:
        """Minimize the negative common signed-area lower bound."""
        return -float(x[-1])

    def epigraph_objective_jac(x: np.ndarray) -> np.ndarray:
        """Return the constant analytic objective gradient."""
        gradient = np.zeros(2 * n + 1, dtype=np.float64)
        gradient[-1] = -1.0
        return gradient

    """Improve the incumbent with soft-min homotopy and bounded epigraph polishing.

    Eight deterministic starts are continued through five smooth log-sum-exp
    objectives using analytic determinant gradients; each resulting basin then
    receives one signed-determinant SLSQP maximin solve.
    """
    def soft_value_and_gradient(x: np.ndarray, tau: float):
        """Return a smooth absolute-determinant minimum and its gradient."""
        p = x.reshape(n, 2)
        det = determinant_values(p)
        smooth_abs = np.sqrt(det * det + 1.0e-20)
        weights = np.exp(-(smooth_abs - np.min(smooth_abs)) / tau)
        weights /= np.sum(weights)
        value = float(np.min(smooth_abs) - tau * np.log(
            np.sum(np.exp(-(smooth_abs - np.min(smooth_abs)) / tau))
        ))
        jac = determinant_jacobian(p)
        signed_factor = det / smooth_abs
        gradient = -np.sum(
            (weights * signed_factor)[:, None] * jac, axis=0
        )
        violation = np.maximum(-p, 0.0)
        violation += np.maximum(p[:, 0] + p[:, 1] - 1.0, 0.0)[:, None]
        value -= 20.0 * float(np.sum(violation * violation))
        gradient = gradient.reshape(n, 2)
        gradient += 40.0 * violation
        over = np.maximum(p[:, 0] + p[:, 1] - 1.0, 0.0)
        gradient += (40.0 * over)[:, None]
        return -value, -gradient.ravel()

    """Build active-incidence nullspace branches and polish them epigraphically.

    The 18 smallest incumbent determinants are signed at the incumbent and
    divided into tight, intermediate, and near-active groups.  An SVD of the
    tight signed Jacobian supplies first-order nullspace escape directions.
    Three such directions are oriented toward the near-active constraints and
    tested at deterministic step sizes before bounded SLSQP polishing.
    """
    """Refine the active vertices in a feasible barycentric chart.

    The 24 smallest incumbent triples define an active vertex subset.  Up to
    five active points are optimized in unconstrained barycentric coordinates,
    so every Nelder-Mead trial remains inside the simplex without projection
    distortion.  Six deterministic starts are followed by acceptance based
    on the exact nonsmooth minimum determinant.
    """
    incumbent_dets = determinant_values(q)
    incumbent_value = objective(q)

    active_order = np.argsort(np.abs(incumbent_dets))[:24]
    active_vertices, active_counts = np.unique(
        triples[active_order].ravel(), return_counts=True
    )
    if active_vertices.size > 5:
        ranking = np.argsort(-active_counts, kind="stable")[:5]
        active_vertices = active_vertices[ranking]
    active_vertices = np.asarray(active_vertices, dtype=np.int64)

    def chart_to_simplex(z: np.ndarray) -> np.ndarray:
        """Map unconstrained point variables to strict simplex coordinates."""
        result = q.copy()
        for local, vertex in enumerate(active_vertices):
            a = float(np.clip(z[2 * local], -40.0, 40.0))
            b = float(np.clip(z[2 * local + 1], -40.0, 40.0))
            ea = np.exp(a)
            eb = np.exp(b)
            denominator = 1.0 + ea + eb
            result[vertex, 0] = ea / denominator
            result[vertex, 1] = eb / denominator
        return result

    if active_vertices.size:
        chart0 = np.empty(2 * active_vertices.size, dtype=np.float64)
        for local, vertex in enumerate(active_vertices):
            u = float(np.clip(q[vertex, 0], 1.0e-9, 1.0 - 2.0e-9))
            v = float(np.clip(q[vertex, 1], 1.0e-9, 1.0 - u - 1.0e-9))
            chart0[2 * local] = np.log(u / (1.0 - u - v))
            chart0[2 * local + 1] = np.log(v / (1.0 - u - v))

        starts = [chart0.copy()]
        hadamard = np.array(
            [[1.0 if ((row >> col) & 1) == 0 else -1.0
              for col in range(chart0.size)]
             for row in range(1, 6)],
            dtype=np.float64,
        )
        for row in hadamard:
            starts.append(chart0 + 0.025 * row)

        direct_best = q.copy()
        direct_value = incumbent_value

        for start in starts[:6]:
            result = minimize(
                lambda z: -objective(chart_to_simplex(z)),
                start,
                method="Nelder-Mead",
                options={
                    "maxiter": 900,
                    "xatol": 2.0e-7,
                    "fatol": 2.0e-10,
                    "disp": False,
                },
            )
            if result.x.shape == start.shape and np.all(np.isfinite(result.x)):
                candidate = chart_to_simplex(result.x)
                candidate_value = objective(candidate)
                if candidate_value > direct_value:
                    direct_value = candidate_value
                    direct_best = candidate.copy()

        if direct_value > incumbent_value:
            q = direct_best
            incumbent_dets = determinant_values(q)
            incumbent_value = direct_value
    incumbent_abs = np.abs(incumbent_dets)
    incumbent_value = objective(q)
    ordered = np.argsort(incumbent_abs)[:18]

    ordered_signs = np.sign(incumbent_dets[ordered])
    ordered_signs[ordered_signs == 0.0] = 1.0
    local_jacobian = determinant_jacobian(q)[ordered]
    signed_jacobian = ordered_signs[:, None] * local_jacobian
    signed_values = ordered_signs * incumbent_dets[ordered]

    tight = incumbent_abs[ordered] <= 1.05 * incumbent_value
    intermediate = (
        (incumbent_abs[ordered] > 1.05 * incumbent_value)
        & (incumbent_abs[ordered] <= 1.35 * incumbent_value)
    )
    near = ~(tight | intermediate)

    tight_jacobian = signed_jacobian[tight]
    near_jacobian = signed_jacobian[near]
    near_values = signed_values[near]

    """Pivot from a weakly supported active contact using epigraph continuation.

    A projected nonnegative least-squares fit estimates active multipliers from
    the gradients of near-minimal determinants.  The least-supported active
    triple is relaxed through four decreasing continuation slacks, with
    determinant orientations recomputed after every accepted pivot.
    """
    incumbent_dets = determinant_values(q)
    incumbent_value = objective(q)

    active_mask = np.abs(incumbent_dets) <= 1.12 * incumbent_value
    active_indices = np.flatnonzero(active_mask)
    if active_indices.size:
        active_jacobian = determinant_jacobian(q)[active_indices]
        signed_active = (
            np.sign(incumbent_dets[active_indices])[:, None]
            * active_jacobian
        )
        target = np.sum(signed_active, axis=0)

        # Deterministic projected-gradient NNLS for approximate
        # complementarity multipliers.
        multipliers = np.full(active_indices.size, 1.0, dtype=np.float64)
        gram = signed_active @ signed_active.T
        lipschitz = float(np.linalg.norm(gram, 2))
        step = 1.0 / max(lipschitz, 1.0e-12)
        for _ in range(180):
            gradient = gram @ multipliers - signed_active @ target
            multipliers = np.maximum(
                0.0, multipliers - step * gradient
            )

        weak_local = int(np.argmin(multipliers))
        relaxed_index = int(active_indices[weak_local])
    else:
        relaxed_index = int(np.argmin(np.abs(incumbent_dets)))

    pivot_point = q.copy()
    pivot_value = incumbent_value

    for slack_factor in (0.30, 0.18, 0.10, 0.05, 0.02, 0.0):
        pivot_dets = determinant_values(pivot_point)
        pivot_signs = np.sign(pivot_dets)
        pivot_signs[pivot_signs == 0.0] = 1.0
        slack = slack_factor * max(pivot_value, 1.0e-12)

        def pivot_constraints(x: np.ndarray) -> np.ndarray:
            """Return relaxed signed-area and simplex feasibility residuals."""
            p = x[:-1].reshape(n, 2)
            signed = pivot_signs * determinant_values(p) - x[-1]
            signed[relaxed_index] += slack
            simplex = 1.0 - p[:, 0] - p[:, 1]
            return np.concatenate((signed, p[:, 0], p[:, 1], simplex))

        def pivot_jacobian(x: np.ndarray) -> np.ndarray:
            """Return the analytic Jacobian of pivot constraints."""
            p = x[:-1].reshape(n, 2)
            rows = len(triples) + 3 * n
            jac = np.zeros((rows, 2 * n + 1), dtype=np.float64)
            jac[:len(triples), :-1] = (
                pivot_signs[:, None] * determinant_jacobian(p)
            )
            jac[:len(triples), -1] = -1.0
            offset = len(triples)
            for i in range(n):
                jac[offset + i, 2 * i] = 1.0
                jac[offset + n + i, 2 * i + 1] = 1.0
                jac[offset + 2 * n + i, 2 * i] = -1.0
                jac[offset + 2 * n + i, 2 * i + 1] = -1.0
            return jac

        signed_start = pivot_signs * pivot_dets
        x0 = np.concatenate((
            pivot_point.ravel(),
            [max(0.0, float(np.min(signed_start)))],
        ))
        result = minimize(
            epigraph_objective,
            x0,
            jac=epigraph_objective_jac,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * (2 * n) + [(0.0, 1.0)],
            constraints={
                "type": "ineq",
                "fun": pivot_constraints,
                "jac": pivot_jacobian,
            },
            options={"maxiter": 450, "ftol": 1.0e-11, "disp": False},
        )

        if result.x.shape == x0.shape and np.all(np.isfinite(result.x)):
            trial = project_simplex(
                result.x[:-1].reshape(n, 2).copy()
            )
            trial_dets = determinant_values(trial)
            trial_signs = np.sign(trial_dets)
            trial_signs[trial_signs == 0.0] = 1.0

            # Reject orientation-crossing pivots; continuation is only valid
            # while every determinant retains its incumbent-side orientation.
            orientation_ok = np.all(
                trial_signs == pivot_signs
            )
            trial_value = objective(trial)
            if orientation_ok and trial_value >= pivot_value:
                pivot_point = trial
                pivot_value = trial_value
                if trial_value > best_value:
                    best_value = trial_value
                    best_points = trial.copy()

    if pivot_value > incumbent_value:
        q = pivot_point.copy()
        incumbent_value = pivot_value

    # Disable the superseded unweighted-nullspace branch portfolio.  The
    # continuation pivot above is the sole post-incumbent branch mechanism.
    branches = []
    null_vectors = np.empty((0, 2 * n), dtype=np.float64)

    if tight_jacobian.shape[0] > 0:
        _, singular_values, vh = np.linalg.svd(
            tight_jacobian, full_matrices=True
        )
        if singular_values.size:
            tolerance = (
                max(tight_jacobian.shape)
                * float(singular_values[0])
                * 1.0e-9
            )
            rank = int(np.sum(singular_values > tolerance))
        else:
            rank = 0

        null_vectors = vh[rank:rank + 3].copy()

    for null_vector in null_vectors:
        direction = null_vector.reshape(n, 2).copy()

        # Orient toward increasing signed values of the near-active blockers.
        if near_jacobian.shape[0] > 0:
            directional_gain = near_jacobian @ null_vector
            weighted_gain = float(np.dot(
                directional_gain,
                np.maximum(0.0, 1.35 * incumbent_value - near_values),
            ))
            if weighted_gain < 0.0:
                direction *= -1.0

        point_norm = np.linalg.norm(direction, axis=1)
        maximum_norm = float(np.max(point_norm))
        if maximum_norm <= 1.0e-14:
            continue
        direction *= 0.003 / maximum_norm

        for fraction in (0.25, 0.5, 1.0):
            branches.append(project_simplex(
                (q + fraction * direction).astype(np.float64, copy=True)
            ))
            if len(branches) >= 10:
                break
        if len(branches) >= 10:
            break

    # Every branch is sent directly to the exact signed epigraph solve.  This
    # preserves the nullspace perturbation instead of smoothing it away first.
    for start in branches[:10]:
        candidate = start.copy()
        candidate_dets = determinant_values(candidate)
        signs = np.sign(candidate_dets)
        signs[signs == 0.0] = 1.0
        signed_start = signs * candidate_dets

        x0 = np.concatenate((
            candidate.ravel(),
            [max(0.0, float(np.min(signed_start)))],
        ))
        bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 1.0)]

        result = minimize(
            epigraph_objective,
            x0,
            jac=epigraph_objective_jac,
            method="SLSQP",
            bounds=bounds,
            constraints={
                "type": "ineq",
                "fun": epigraph_constraints,
                "jac": epigraph_jacobian,
            },
            options={"maxiter": 700, "ftol": 1.0e-11, "disp": False},
        )
        if result.x.shape[0] == 2 * n + 1 and np.all(np.isfinite(result.x)):
            polished = project_simplex(
                result.x[:-1].reshape(n, 2).copy()
            )
            polished_value = objective(polished)
            if polished_value > best_value:
                best_value = polished_value
                best_points = polished.copy()

    # Always retain the original incumbent if numerical optimization fails or
    # the locally signed epigraph does not improve the unsigned objective.
    if best_points is None or objective(best_points) < incumbent_value:
        best_points = q.copy()
        best_value = incumbent_value

    # Convert affine simplex coordinates to Cartesian equilateral-triangle
    # coordinates.  The slight clipping guards against floating-point drift.
    u = best_points[:, 0]
    v = best_points[:, 1]
    points = np.column_stack((u + 0.5 * v, (np.sqrt(3.0) / 2.0) * v))
    return np.clip(points, 0.0, 1.0)


# EVOLVE-BLOCK-END
