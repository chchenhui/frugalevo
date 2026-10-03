# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """Explore 288 deterministic order types with incremental annealing, then SLSQP-polish their feasible sign cells."""
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

    def random_points(anchor_vertices=False):
        """Sample a feasible start, optionally retaining all three triangle vertices."""
        count = n - 3 if anchor_vertices else n
        u = rng.random((count, 2))
        swap = u[:, 0] + u[:, 1] > 1.0
        u[swap] = 1.0 - u[swap]
        interior = np.column_stack((u[:, 0] + 0.5 * u[:, 1], height * u[:, 1]))
        if not anchor_vertices:
            return interior
        return np.vstack(((0.0, 0.0), (1.0, 0.0), (0.5, height), interior))

    def determinants(p):
        q1 = p[ib] - p[ia]
        q2 = p[ic] - p[ia]
        return q1[:, 0] * q2[:, 1] - q1[:, 1] * q2[:, 0]

    def score(p):
        # Determinants are twice ordinary Cartesian areas; normalization in the
        # evaluator is invariant under this fixed scale during optimization.
        return float(np.min(np.abs(determinants(p))))

    candidates = []
    # A single-point proposal affects only triples containing that point:
    # maintaining determinants incrementally avoids recomputing all 165 areas
    # for every annealing proposal.
    incident = [
        np.flatnonzero((ia == point) | (ib == point) | (ic == point))
        for point in range(n)
    ]
    nonincident = [
        np.flatnonzero((ia != point) & (ib != point) & (ic != point))
        for point in range(n)
    ]

    # Precompute the deterministic annealing schedules: this avoids repeated
    # scalar exponentiation over millions of proposal evaluations.
    steps = 38000
    fraction = np.linspace(0.0, 1.0, steps)
    proposal_scale = 0.125 * (1.0 - fraction) ** 1.9 + 0.00030
    temperature_schedule = 0.018 * (1.0 - fraction) ** 2.35 + 0.000010

    # SLSQP cannot cross a zero-determinant boundary, so broad independent
    # sampling of oriented-matroid cells is more useful than hull restrictions.
    # Independent restarts explore distinct oriented-matroid cells.  This is
    # preferable to substantially lengthening one anneal, since a single run
    # rarely changes its useful determinant-sign pattern late in the schedule.
    # Better layouts are usually found by entering a rare determinant-sign cell;
    # use a larger deterministic portfolio rather than extending one trajectory.
    for restart in range(288):
        current = random_points()
        current_det = determinants(current)
        current_score = float(np.min(np.abs(current_det)))
        local_best = current.copy()
        local_score = current_score

        for step in range(steps):
            sigma = proposal_scale[step]
            absdet = np.abs(current_det)
            near = np.flatnonzero(
                absdet <= current_score + max(0.0015, 0.18 * current_score)
            )
            if len(near) and rng.random() < 0.72:
                index = triples[near[rng.integers(len(near))], rng.integers(3)]
            else:
                index = rng.integers(n)

            trial_point = project_to_triangle(
                current[index] + rng.normal(0.0, sigma, size=2)
            )
            affected = incident[index]

            # Only incident triples change.  Compare their new minimum with the
            # pre-existing complementary minimum, avoiding a full vector copy.
            pa = current[ia[affected]].copy()
            pb = current[ib[affected]].copy()
            pc = current[ic[affected]].copy()
            pa[ia[affected] == index] = trial_point
            pb[ib[affected] == index] = trial_point
            pc[ic[affected] == index] = trial_point
            q1 = pb - pa
            q2 = pc - pa
            changed_det = q1[:, 0] * q2[:, 1] - q1[:, 1] * q2[:, 0]
            trial_score = float(min(
                np.min(np.abs(changed_det)),
                np.min(np.abs(current_det[nonincident[index]])),
            ))
            temperature = temperature_schedule[step]

            if (trial_score >= current_score or
                    rng.random() < np.exp(
                        (trial_score - current_score) / temperature
                    )):
                current[index] = trial_point
                current_det[affected] = changed_det
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
    # Annealing rank is noisy: polish a larger portfolio of feasible sign cells
    # rather than trusting only the very best discrete candidates.
    # Annealing score is only a coarse proxy for the optimum in a sign cell.
    # Preserve more distinct cells for exact constrained max-min polishing.
    for _, start in candidates[:112]:
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

        # SLSQP's final reported point can be inferior to an earlier accepted
        # iterate. Retain only iterates that explicitly satisfy containment.
        polished_best_score = best_score
        polished_best = None

        def retain_feasible_iterate(z):
            nonlocal polished_best_score, polished_best
            if not np.all(np.isfinite(z)):
                return
            p = z[:-1].reshape(n, 2)
            x, y = p[:, 0], p[:, 1]
            if (np.min(y) < -2.0e-10 or
                    np.min(x - y / root3) < -2.0e-10 or
                    np.min(1.0 - x - y / root3) < -2.0e-10):
                return
            value = score(p)
            if value > polished_best_score:
                polished_best_score = value
                polished_best = p.copy()

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
                callback=retain_feasible_iterate,
                options={"maxiter": 3500, "ftol": 1.0e-12, "disp": False},
            )
            retain_feasible_iterate(result.x)
            if polished_best is not None:
                best_score = polished_best_score
                best = polished_best
        except Exception:
            # The best feasible annealed arrangement is always retained.
            pass

    return best


# EVOLVE-BLOCK-END
