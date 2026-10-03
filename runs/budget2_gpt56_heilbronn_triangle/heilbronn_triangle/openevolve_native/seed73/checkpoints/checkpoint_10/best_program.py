# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """Explore many deterministic order types, then SLSQP-polish max-min determinants."""
    try:
        from scipy.optimize import minimize
    except Exception:
        minimize = None

    rng = np.random.default_rng(11031989)
    n = 11
    root3 = np.sqrt(3.0)
    height = root3 / 2.0
    triples = np.array(
        [(a, b, c) for a in range(n) for b in range(a + 1, n)
         for c in range(b + 1, n)],
        dtype=np.intp,
    )
    ia, ib, ic = triples.T

    def project_to_triangle(p):
        """Project Cartesian coordinates into the closed equilateral triangle."""
        bary = np.empty(3)
        bary[2] = p[1] / height
        bary[1] = p[0] - 0.5 * bary[2]
        bary[0] = 1.0 - bary[1] - bary[2]
        bary = np.maximum(bary, 0.0)
        total = bary.sum()
        if total < 1.0e-14:
            bary[:] = (1.0, 0.0, 0.0)
        else:
            bary /= total
        return np.array((bary[1] + 0.5 * bary[2], height * bary[2]))

    def random_points():
        """Uniformly sample points using simplex barycentric coordinates."""
        u = rng.random((n, 2))
        swap = u[:, 0] + u[:, 1] > 1.0
        u[swap] = 1.0 - u[swap]
        return np.column_stack((u[:, 0] + 0.5 * u[:, 1], height * u[:, 1]))

    def determinants(p):
        q1 = p[ib] - p[ia]
        q2 = p[ic] - p[ia]
        return q1[:, 0] * q2[:, 1] - q1[:, 1] * q2[:, 0]

    def score(p):
        # Determinants are twice ordinary Cartesian areas; normalization in the
        # evaluator is invariant under this fixed scale during optimization.
        return float(np.min(np.abs(determinants(p))))

    candidates = []
    # Distinct starts matter more than extending one trajectory: SLSQP cannot
    # cross a zero-determinant boundary and therefore remains in one order type.
    for restart in range(64):
        current = random_points()
        current_score = score(current)
        local_best = current.copy()
        local_score = current_score

        steps = 45000
        for step in range(steps):
            fraction = step / (steps - 1)
            sigma = 0.125 * (1.0 - fraction) ** 1.9 + 0.00030
            trial = current.copy()

            # Direct moves toward vertices participating in near-bottleneck
            # triangles, while retaining enough random moves to change order type.
            absdet = np.abs(determinants(current))
            near = triples[
                absdet <= current_score + max(0.0015, 0.18 * current_score)
            ]
            if len(near) and rng.random() < 0.72:
                index = near[rng.integers(len(near)), rng.integers(3)]
            else:
                index = rng.integers(n)

            trial[index] = project_to_triangle(
                trial[index] + rng.normal(0.0, sigma, size=2)
            )
            trial_score = score(trial)
            temperature = 0.018 * (1.0 - fraction) ** 2.35 + 0.000010

            if (trial_score >= current_score or
                    rng.random() < np.exp(
                        (trial_score - current_score) / temperature
                    )):
                current = trial
                current_score = trial_score

            if current_score > local_score:
                local_score = current_score
                local_best = current.copy()

        candidates.append((local_score, local_best))

    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, best = candidates[0]

    if minimize is None:
        return best

    # Within a fixed oriented-matroid cell, every determinant has a known sign.
    # This makes the lower bound on all absolute determinants a smooth constrained
    # max-min problem, which SLSQP can polish effectively.
    # Exact derivatives avoid the expensive and occasionally inaccurate finite
    # difference approximation of the 165 oriented-area constraints.
    # Preserve a broad set of high-quality sign cells; annealing rank is only an
    # imperfect predictor of the optimum obtained after continuous polishing.
    for _, start in candidates[:32]:
        signs = np.sign(determinants(start))
        signs[signs == 0.0] = 1.0
        z0 = np.concatenate((start.ravel(), [score(start) * 0.995]))

        def constraints(z, orientation=signs):
            p = z[:-1].reshape(n, 2)
            t = z[-1]
            det = determinants(p)
            x = p[:, 0]
            y = p[:, 1]
            inside = np.concatenate((
                y,
                x - y / root3,
                1.0 - x - y / root3,
            ))
            return np.concatenate((orientation * det - t, inside, [t]))

        def constraints_jacobian(z, orientation=signs):
            p = z[:-1].reshape(n, 2)
            jac = np.zeros((len(triples) + 3 * n + 1, 2 * n + 1))
            a, b, c = ia, ib, ic
            ax, ay = p[a, 0], p[a, 1]
            bx, by = p[b, 0], p[b, 1]
            cx, cy = p[c, 0], p[c, 1]
            rows = np.arange(len(triples))

            # d cross(b-a,c-a) / d(a,b,c), multiplied by fixed signs.
            jac[rows, 2 * a] = orientation * (by - cy)
            jac[rows, 2 * a + 1] = orientation * (cx - bx)
            jac[rows, 2 * b] = orientation * (cy - ay)
            jac[rows, 2 * b + 1] = orientation * (ax - cx)
            jac[rows, 2 * c] = orientation * (ay - by)
            jac[rows, 2 * c + 1] = orientation * (bx - ax)
            jac[rows, -1] = -1.0

            offset = len(triples)
            point_rows = offset + np.arange(n)
            jac[point_rows, 2 * np.arange(n) + 1] = 1.0
            point_rows += n
            jac[point_rows, 2 * np.arange(n)] = 1.0
            jac[point_rows, 2 * np.arange(n) + 1] = -1.0 / root3
            point_rows += n
            jac[point_rows, 2 * np.arange(n)] = -1.0
            jac[point_rows, 2 * np.arange(n) + 1] = -1.0 / root3
            jac[-1, -1] = 1.0
            return jac

        try:
            result = minimize(
                lambda z: -z[-1],
                z0,
                jac=lambda z: np.r_[np.zeros(2 * n), -1.0],
                method="SLSQP",
                constraints={
                    "type": "ineq",
                    "fun": constraints,
                    "jac": constraints_jacobian,
                },
                options={"maxiter": 3500, "ftol": 1.0e-12, "disp": False},
            )
            # SLSQP may report a line-search or iteration status after finding a
            # valid better iterate, so score every finite returned configuration.
            if np.all(np.isfinite(result.x)):
                candidate = result.x[:-1].reshape(n, 2)
                candidate = np.array(
                    [project_to_triangle(point) for point in candidate]
                )
                candidate_score = score(candidate)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best = candidate
        except Exception:
            # The best feasible annealed arrangement is always retained.
            pass

    return best


# EVOLVE-BLOCK-END
