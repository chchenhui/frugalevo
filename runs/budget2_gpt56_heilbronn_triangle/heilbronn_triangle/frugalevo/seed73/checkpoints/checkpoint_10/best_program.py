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

    # Use many independent uniform-simplex starts.  The minimum-triangle
    # objective is nonsmooth, so additional basins are more valuable than
    # making any one annealing trajectory excessively long.  Two-point moves
    # help repair multiple simultaneously active near-collinear triples.
    # Explore substantially more independent basins.  For this nonsmooth
    # objective, additional restarts are generally more valuable than making
    # any single annealing trajectory much longer.
    for restart in range(120):
        bary = rng.dirichlet((1.0, 1.0, 1.0), size=n)
        q = bary[:, :2].copy()
        current_value = objective(q)

        for iteration in range(8500):
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
        for iteration in range(1700):
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

    starts = [q.copy()]
    for _ in range(3):
        start = project_simplex(
            q + rng.normal(0.0, 0.0015, size=(n, 2))
        )
        starts.append(start)

    for start in starts:
        signed_start = signs * determinant_values(start)
        x0 = np.concatenate((start.ravel(), [max(0.0, float(np.min(
            signed_start
        )))]))
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
            options={
                "maxiter": 900,
                "ftol": 1.0e-11,
                "disp": False,
            },
        )
        if result.x.shape[0] == 2 * n + 1 and np.all(np.isfinite(result.x)):
            candidate = project_simplex(result.x[:-1].reshape(n, 2))
            candidate_value = objective(candidate)
            if candidate_value > best_value:
                best_value = candidate_value
                best_points = candidate.copy()

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
