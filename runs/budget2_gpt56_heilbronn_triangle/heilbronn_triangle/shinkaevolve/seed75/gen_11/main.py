# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations


def heilbronn_triangle11() -> np.ndarray:
    """
    Deterministically search for a high-minimum-area configuration of eleven
    points in the unit equilateral triangle.
    """
    n = 11
    height = np.sqrt(3.0) / 2.0
    triples = np.asarray(list(combinations(range(n), 3)), dtype=int)
    rng = np.random.default_rng(11031987)

    def project_to_triangle(point: np.ndarray) -> np.ndarray:
        """Project a point into x >= 0, y >= 0, y <= sqrt(3) min(x, 1-x)."""
        x, y = np.clip(point, 0.0, 1.0)
        y = min(y, height)
        if y > np.sqrt(3.0) * min(x, 1.0 - x):
            # Orthogonal projection onto the nearest sloping edge.
            if x <= 0.5:
                x = (x + np.sqrt(3.0) * y) / 4.0
                y = np.sqrt(3.0) * x
            else:
                x = (x + np.sqrt(3.0) * (1.0 - y) + 3.0) / 4.0
                y = np.sqrt(3.0) * (1.0 - x)
        return np.array((np.clip(x, 0.0, 1.0), np.clip(y, 0.0, height)))

    def areas(points: np.ndarray) -> np.ndarray:
        a = points[triples[:, 0]]
        b = points[triples[:, 1]]
        c = points[triples[:, 2]]
        return 0.5 * np.abs(
            (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
            - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        )

    def objective(points: np.ndarray) -> tuple[float, float]:
        values = np.sort(areas(points))
        # Rewarding several active constraints avoids brittle one-triangle moves.
        return values[0], values[0] + 0.18 * values[:12].mean()

    vertices = np.array(((0.0, 0.0), (1.0, 0.0), (0.5, height)))
    best_points = None
    best_minimum = -1.0

    # Independent starts are valuable because the maximin landscape is highly
    # non-convex.  The total work is still small: 8 * 18000 * 165 areas.
    for restart in range(8):
        points = np.empty((n, 2))
        points[:3] = vertices
        barycentric = rng.dirichlet((1.0, 1.0, 1.0), size=n - 3)
        points[3:, 0] = barycentric[:, 1] + 0.5 * barycentric[:, 2]
        points[3:, 1] = height * barycentric[:, 2]

        current_minimum, current_score = objective(points)
        local_best = points.copy()
        local_best_minimum = current_minimum

        for iteration in range(18000):
            fraction = iteration / 17999.0
            step = 0.105 * (1.0 - fraction) ** 1.7 + 0.0015
            temperature = 0.0018 * (1.0 - fraction) ** 2 + 0.000015

            index = int(rng.integers(3, n))
            candidate = points.copy()
            candidate[index] = project_to_triangle(
                candidate[index] + rng.normal(0.0, step, size=2)
            )
            candidate_minimum, candidate_score = objective(candidate)

            change = candidate_score - current_score
            if change >= 0.0 or rng.random() < np.exp(change / temperature):
                points = candidate
                current_minimum = candidate_minimum
                current_score = candidate_score

            if candidate_minimum > local_best_minimum:
                local_best_minimum = candidate_minimum
                local_best = candidate.copy()

        if local_best_minimum > best_minimum:
            best_minimum = local_best_minimum
            best_points = local_best

    # The annealer is effective at finding the correct combinatorial region,
    # but its final small random moves are comparatively inaccurate.  Polish
    # that region directly in reference-simplex coordinates.  A determinant in
    # (u, v), where x=u+v/2 and y=height*v, is area normalized by the area of
    # the containing equilateral triangle.
    try:
        from scipy.optimize import minimize

        def cartesian_to_simplex(p: np.ndarray) -> np.ndarray:
            v = p[:, 1] / height
            return np.column_stack((p[:, 0] - 0.5 * v, v))

        def simplex_determinants(uv: np.ndarray) -> np.ndarray:
            q = uv[triples]
            return (
                (q[:, 1, 0] - q[:, 0, 0]) * (q[:, 2, 1] - q[:, 0, 1])
                - (q[:, 2, 0] - q[:, 0, 0]) * (q[:, 1, 1] - q[:, 0, 1])
            )

        start_uv = cartesian_to_simplex(best_points)
        start_det = simplex_determinants(start_uv)
        signs = np.sign(start_det)
        signs[signs == 0.0] = 1.0
        start_t = float(np.min(np.abs(start_det)))

        # Keep the first three simplex vertices fixed.  Within the orientation
        # cell found above, these are smooth explicit maximin constraints.
        def polish_constraints(w: np.ndarray) -> np.ndarray:
            uv = np.vstack((np.array(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
                            w[:-1].reshape(8, 2)))
            tail = w[:-1].reshape(8, 2)
            return np.concatenate((
                signs * simplex_determinants(uv) - w[-1],
                tail.ravel(),
                1.0 - np.sum(tail, axis=1),
            ))

        polished = minimize(
            lambda w: -w[-1],
            np.concatenate((start_uv[3:].ravel(), [start_t])),
            method="SLSQP",
            bounds=[(0.0, 1.0)] * 16 + [(0.0, 1.0)],
            constraints={"type": "ineq", "fun": polish_constraints},
            options={"maxiter": 2500, "ftol": 1.0e-13, "disp": False},
        )

        if polished.success:
            polished_uv = np.vstack((
                np.array(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
                polished.x[:-1].reshape(8, 2),
            ))
            polished_minimum = float(
                np.min(np.abs(simplex_determinants(polished_uv)))
            )
            # Convert normalized determinant back to Cartesian area for the
            # comparison used by the annealing stage.
            if polished_minimum * (height * 0.5) > best_minimum + 1.0e-12:
                best_points = np.column_stack((
                    polished_uv[:, 0] + 0.5 * polished_uv[:, 1],
                    height * polished_uv[:, 1],
                ))
    except Exception:
        # The annealed configuration is valid even without optional SciPy.
        pass

    return best_points


# EVOLVE-BLOCK-END