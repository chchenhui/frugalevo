# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """Deterministically search many order types with bottleneck-directed annealing,
    then maximize the signed-determinant lower bound by SLSQP polishing."""
    try:
        from scipy.optimize import minimize
    except Exception:
        minimize = None

    rng = np.random.default_rng(11031989)
    n = 11
    root3 = np.sqrt(3.0)
    height = root3 / 2.0

    ii, jj, kk = np.triu_indices(n, 1)[0], None, None
    triples = np.array(
        [(a, b, c) for a in range(n) for b in range(a + 1, n)
         for c in range(b + 1, n)],
        dtype=np.intp,
    )
    ia, ib, ic = triples.T

    def project_to_triangle(p):
        """Project via clipped barycentric coordinates, preserving feasibility."""
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
        """Sample uniformly from the equilateral triangle."""
        u = rng.random((n, 2))
        swap = u[:, 0] + u[:, 1] > 1.0
        u[swap] = 1.0 - u[swap]
        return np.column_stack((u[:, 0] + 0.5 * u[:, 1], height * u[:, 1]))

    def determinants(p):
        q1 = p[ib] - p[ia]
        q2 = p[ic] - p[ia]
        return q1[:, 0] * q2[:, 1] - q1[:, 1] * q2[:, 0]

    def score(p):
        return float(np.min(np.abs(determinants(p))))

    # A short multi-start anneal is substantially more reliable than optimizing one
    # random order type.  Determinants are twice the ordinary Cartesian areas.
    candidates = []
    # The problem has many locally optimal determinant-sign order types.  More
    # independent, deterministic starts are much more useful than a single very
    # long trajectory because SLSQP can only improve within one such order type.
    for restart in range(32):
        current = random_points()
        current_score = score(current)
        local_best = current.copy()
        local_score = current_score

        steps = 32000
        for step in range(steps):
            fraction = step / (steps - 1)
            # Large moves find a good order type; fine moves improve active triples.
            sigma = 0.125 * (1.0 - fraction) ** 1.9 + 0.00030
            trial = current.copy()

            # Most useful moves involve a vertex of a nearly active triple.
            # Retaining random moves prevents premature trapping and permits
            # changes to the determinant-sign order type.
            absdet = np.abs(determinants(current))
            near = triples[absdet <= current_score + max(0.0015, 0.18 * current_score)]
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
                    rng.random() < np.exp((trial_score - current_score) / temperature)):
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

    # For a fixed order type, signs of all determinants are fixed.  This converts
    # abs(det) >= t into smooth inequality constraints suitable for SLSQP.
    # Polish a wider collection of distinct high-quality order types; the best
    # annealed candidate is not always the order type with the best NLP optimum.
    for _, start in candidates[:12]:
        signs = np.sign(determinants(start))
        signs[signs == 0.0] = 1.0
        initial_t = score(start) * 0.995
        z0 = np.concatenate((start.ravel(), [initial_t]))

        def constraints(z, orientation=signs):
            p = z[:-1].reshape(n, 2)
            t = z[-1]
            det = determinants(p)

            # Barycentric coordinates of each point must be nonnegative.
            x = p[:, 0]
            y = p[:, 1]
            inside = np.concatenate((
                y,
                x - y / root3,
                1.0 - x - y / root3,
            ))
            return np.concatenate((orientation * det - t, inside, [t]))

        try:
            result = minimize(
                lambda z: -z[-1],
                z0,
                method="SLSQP",
                constraints={"type": "ineq", "fun": constraints},
                options={"maxiter": 1600, "ftol": 5.0e-12, "disp": False},
            )
            if result.success:
                candidate = result.x[:-1].reshape(n, 2)
                candidate = np.array([project_to_triangle(p) for p in candidate])
                candidate_score = score(candidate)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best = candidate
        except Exception:
            # A valid annealing solution remains available if a local NLP fails.
            pass

    return best


# EVOLVE-BLOCK-END
