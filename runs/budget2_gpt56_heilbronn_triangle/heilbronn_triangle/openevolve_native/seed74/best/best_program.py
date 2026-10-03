# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """Run 48 seeded cached-update anneals, then strictly max-min polish the best simplex layout."""
    rng = np.random.default_rng(11031991)
    n = 11

    # In coordinates (u,v), the reference triangle is
    # u >= 0, v >= 0, u + v <= 1.  Its triangle-area ratio is simply
    # abs(det([p_j-p_i, p_k-p_i])).
    triples = np.array(
        [(i, j, k) for i in range(n) for j in range(i + 1, n)
         for k in range(j + 1, n)],
        dtype=np.intp,
    )
    # Moving point i changes only these triples.  Caching the remaining
    # determinants avoids evaluating all 165 triangles at every proposal.
    affected = [np.flatnonzero(np.any(triples == i, axis=1)) for i in range(n)]

    def random_simplex_points(count: int) -> np.ndarray:
        """Draw uniformly distributed points in the reference simplex."""
        z = rng.random((count, 2))
        mask = z.sum(axis=1) > 1.0
        z[mask] = 1.0 - z[mask]
        return z

    def area_values(p: np.ndarray, subset: np.ndarray = None) -> np.ndarray:
        """Return normalized determinant areas, optionally for a triple subset."""
        t = triples if subset is None else triples[subset]
        a = p[t[:, 1]] - p[t[:, 0]]
        b = p[t[:, 2]] - p[t[:, 0]]
        return np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])

    # Anneal with moves concentrated on vertices of the currently smallest
    # triangle.  This is substantially more effective than selecting every
    # point uniformly for the nonsmooth max-min objective.
    best = None
    best_value = -1.0

    # More independent basins are substantially more valuable here than
    # lengthening a single annealing trajectory on this nonsmooth objective.
    for _ in range(48):
        p = np.empty((n, 2), dtype=float)
        p[:3] = np.array([[0.5, 0.0], [0.0, 0.5], [0.5, 0.5]])
        p[3:] = random_simplex_points(8)
        areas = area_values(p)
        current = float(np.min(areas))
        steps = 55000

        for step in range(steps):
            t = step / (steps - 1)
            sigma = 0.115 * (1.0 - t) ** 1.65 + 0.00018
            temperature = 0.0090 * (1.0 - t) ** 2.15 + 0.000001

            limiting = triples[int(np.argmin(areas))]
            if rng.random() < 0.68:
                idx = int(limiting[rng.integers(0, 3)])
            else:
                idx = int(rng.integers(n))
            old = p[idx].copy()

            candidate = None
            if rng.random() < 0.30 * (1.0 - t) ** 1.5 + 0.015:
                s = float(np.clip(rng.normal(0.5, 0.28), 0.015, 0.985))
                edge = int(rng.integers(3))
                if edge == 0:
                    candidate = np.array([s, 0.0])
                elif edge == 1:
                    candidate = np.array([0.0, s])
                else:
                    candidate = np.array([s, 1.0 - s])
            else:
                # Occasionally move directly away from the opposite side of
                # the limiting triangle, increasing its determinant.
                if idx in limiting and rng.random() < 0.24:
                    other = limiting[limiting != idx]
                    q, r = p[other[0]], p[other[1]]
                    determinant = ((q[0] - old[0]) * (r[1] - old[1])
                                   - (q[1] - old[1]) * (r[0] - old[0]))
                    direction = np.array([q[1] - r[1], r[0] - q[0]])
                    direction *= 1.0 if determinant >= 0.0 else -1.0
                    direction /= max(np.linalg.norm(direction), 1e-15)
                    trial = old + direction * sigma * (0.25 + rng.random())
                    if trial[0] >= 0.0 and trial[1] >= 0.0 and trial.sum() <= 1.0:
                        candidate = trial

                if candidate is None:
                    for _ in range(10):
                        trial = old + rng.normal(0.0, sigma, size=2)
                        if trial[0] >= 0.0 and trial[1] >= 0.0 and trial.sum() <= 1.0:
                            candidate = trial
                            break
                if candidate is None:
                    candidate = random_simplex_points(1)[0]

            p[idx] = candidate
            changed = affected[idx]
            old_changed = areas[changed].copy()
            areas[changed] = area_values(p, changed)
            value = float(np.min(areas))

            if value >= current or rng.random() < np.exp((value - current) / temperature):
                current = value
            else:
                p[idx] = old
                areas[changed] = old_changed

            if current > best_value:
                best_value = current
                best = p.copy()

    # Deterministic coordinate-pattern search polishes the best annealed
    # configuration without accepting any decrease in the true objective.
    directions = np.vstack((
        np.column_stack((
            np.cos(np.arange(16) * np.pi / 8.0),
            np.sin(np.arange(16) * np.pi / 8.0),
        )),
        np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float),
    ))

    def full_minimum(points: np.ndarray) -> float:
        """Return the exact smallest normalized determinant."""
        return float(np.min(area_values(points)))

    polished_value = full_minimum(best)
    for scale in 0.006 * 0.5 ** np.arange(13):
        improved = True
        while improved:
            improved = False
            for idx in range(n):
                original = best[idx].copy()
                trial_value = polished_value
                trial_point = original

                for direction in directions:
                    candidate = original + scale * direction
                    if (candidate[0] < 0.0 or candidate[1] < 0.0
                            or candidate[0] + candidate[1] > 1.0):
                        continue
                    best[idx] = candidate
                    value = full_minimum(best)
                    if value > trial_value + 1e-14:
                        trial_value = value
                        trial_point = candidate.copy()

                best[idx] = trial_point
                if trial_value > polished_value + 1e-14:
                    polished_value = trial_value
                    improved = True
                else:
                    best[idx] = original

    # The pattern search is robust but moves one point at a time.  In the
    # final basin, fix every determinant's orientation and solve the resulting
    # smooth max-min nonlinear program jointly.  This optional refinement is
    # guarded so environments without SciPy retain the annealed result.
    try:
        from scipy.optimize import minimize

        signed = area_values(best)
        signs = np.where(
            ((best[triples[:, 1], 0] - best[triples[:, 0], 0])
             * (best[triples[:, 2], 1] - best[triples[:, 0], 1])
             - (best[triples[:, 1], 1] - best[triples[:, 0], 1])
             * (best[triples[:, 2], 0] - best[triples[:, 0], 0])) >= 0.0,
            1.0, -1.0,
        )

        def smooth_constraints(x: np.ndarray) -> np.ndarray:
            """Signed triangle margins and reference-simplex margins."""
            p = x[:-1].reshape(n, 2)
            a = p[triples[:, 1]] - p[triples[:, 0]]
            b = p[triples[:, 2]] - p[triples[:, 0]]
            determinants = signs * (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
            return np.concatenate((
                determinants - x[-1], p[:, 0], p[:, 1],
                1.0 - p[:, 0] - p[:, 1],
            ))

        start = np.concatenate((best.ravel(), [polished_value]))
        refined = minimize(
            lambda x: -x[-1], start, method="SLSQP",
            bounds=[(0.0, 1.0)] * (2 * n) + [(0.0, 1.0)],
            constraints={"type": "ineq", "fun": smooth_constraints},
            options={"maxiter": 600, "ftol": 1e-12, "disp": False},
        )
        if refined.x.shape == start.shape:
            candidate = refined.x[:-1].reshape(n, 2)
            candidate_value = full_minimum(candidate)
            if (np.all(candidate >= -1e-12)
                    and np.all(candidate.sum(axis=1) <= 1.0 + 1e-12)
                    and candidate_value > polished_value + 1e-12):
                best = candidate
                polished_value = candidate_value
    except Exception:
        pass

    # Convert barycentric reference coordinates to the requested Cartesian
    # equilateral triangle: (x,y) = (u + v/2, sqrt(3)*v/2).
    result = np.empty_like(best)
    result[:, 0] = best[:, 0] + 0.5 * best[:, 1]
    result[:, 1] = (np.sqrt(3.0) * 0.5) * best[:, 1]
    return result


# EVOLVE-BLOCK-END
