# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """
    Deterministic hybrid annealing and coordinate-LP construction for the
    eleven-point Heilbronn problem in an equilateral triangle.

    Internal coordinates are (u, v), with u >= 0, v >= 0, u + v <= 1.
    Their determinant areas equal Cartesian triangle areas normalized by the
    enclosing equilateral triangle area.
    """
    n = 11
    rng = np.random.default_rng(7711031987)
    sqrt3_over_2 = np.sqrt(3.0) * 0.5

    triples = np.array(
        [(i, j, k) for i in range(n - 2)
         for j in range(i + 1, n - 1)
         for k in range(j + 1, n)],
        dtype=np.intp,
    )
    vertices = np.array(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=float)

    def project(x: np.ndarray) -> np.ndarray:
        x = np.maximum(np.asarray(x, dtype=float), 0.0)
        s = float(x[0] + x[1])
        return x / s if s > 1.0 else x

    def determinants(p: np.ndarray) -> np.ndarray:
        a = p[triples[:, 0]]
        b = p[triples[:, 1]]
        c = p[triples[:, 2]]
        return ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))

    def values(p: np.ndarray) -> np.ndarray:
        return np.abs(determinants(p))

    def score_and_merit(p: np.ndarray) -> tuple[np.ndarray, float, float]:
        z = values(p)
        low = np.partition(z, 17)[:18]
        # The mean lower tail breaks ties and gives annealing useful guidance.
        return z, float(z.min()), float(z.min() + 0.055 * low.mean())

    def make_start(restart: int) -> np.ndarray:
        p = np.empty((n, 2), dtype=float)
        p[:3] = vertices

        # Alternate edge-seeking, uniform/interior, and structured starts.
        mode = restart % 4
        if mode == 0:
            bary = rng.dirichlet((0.42, 0.42, 0.42), size=8)
            p[3:] = bary[:, 1:]
        elif mode == 1:
            bary = rng.dirichlet((0.82, 0.82, 0.82), size=8)
            p[3:] = bary[:, 1:]
        elif mode == 2:
            bary = rng.dirichlet((1.55, 1.55, 1.55), size=8)
            p[3:] = bary[:, 1:]
        else:
            # A staggered triangular template, with fixed reproducible jitter.
            template = np.array(
                ((0.16, 0.07), (0.43, 0.055), (0.76, 0.07),
                 (0.075, 0.38), (0.30, 0.27), (0.58, 0.25),
                 (0.075, 0.72), (0.31, 0.55)),
                dtype=float,
            )
            p[3:] = np.array([project(q + rng.normal(0.0, 0.035, 2))
                              for q in template])
        return p

    best = np.vstack((vertices, np.full((8, 2), 1.0 / 3.0)))
    _, best_score, best_merit = score_and_merit(best)
    elite = []

    # Diverse global phase.  The lower iteration count is offset by more
    # independent cells and occasional joint moves.
    restarts = 14
    iterations = 14000
    for restart in range(restarts):
        current = make_start(restart)
        current_values, current_score, current_merit = score_and_merit(current)

        for it in range(iterations):
            q = it / float(iterations - 1)
            step = 0.125 * (1.0 - q) ** 1.48 + 0.00065
            scale = 0.012 * (1.0 - q) ** 1.15 + 0.00038
            temperature = 0.0028 * (1.0 - q) ** 2.15 + 1.5e-6

            # Bias selection toward points occurring in dangerous triangles.
            if rng.random() < 0.88:
                cutoff = current_score + max(2.4 * scale, 0.0012)
                active = triples[current_values <= cutoff]
                chosen = active[int(rng.integers(len(active)))]
                movable = chosen[chosen >= 3]
                index = (int(rng.choice(movable)) if len(movable)
                         else int(rng.integers(3, n)))
            else:
                index = int(rng.integers(3, n))

            candidate = current.copy()
            candidate[index] = project(
                candidate[index] + rng.normal(0.0, step, size=2)
            )

            # Rare paired displacement helps reorganize a bad local cell.
            if rng.random() < 0.075 * (1.0 - q):
                other = int(rng.integers(3, n - 1))
                if other >= index:
                    other += 1
                candidate[other] = project(
                    candidate[other] + rng.normal(0.0, 0.72 * step, size=2)
                )

            cand_values, cand_score, cand_merit = score_and_merit(candidate)
            delta = cand_merit - current_merit
            if delta >= 0.0 or rng.random() < np.exp(max(-700.0, delta / temperature)):
                current = candidate
                current_values = cand_values
                current_score = cand_score
                current_merit = cand_merit

            if (current_score > best_score + 1e-14 or
                    (abs(current_score - best_score) <= 1e-14 and
                     current_merit > best_merit)):
                best = current.copy()
                best_score = current_score
                best_merit = current_merit

        elite.append((current_score, current_merit, current.copy()))

    elite.append((best_score, best_merit, best.copy()))
    elite.sort(key=lambda x: (x[0], x[1]), reverse=True)
    elite = elite[:5]

    def clip_halfplane(poly: np.ndarray, aa: float, bb: float,
                       rhs: float) -> np.ndarray:
        """Clip convex polygon by aa*x + bb*y >= rhs."""
        if len(poly) == 0:
            return poly
        result = []
        prev = poly[-1]
        pv = aa * prev[0] + bb * prev[1] - rhs
        for cur in poly:
            cv = aa * cur[0] + bb * cur[1] - rhs
            pin = pv >= -2e-13
            cin = cv >= -2e-13
            if pin != cin:
                den = pv - cv
                if abs(den) > 1e-18:
                    t = pv / den
                    result.append(prev + t * (cur - prev))
            if cin:
                result.append(cur)
            prev, pv = cur, cv
        return np.asarray(result, dtype=float)

    def coordinate_lp(p: np.ndarray, index: int) -> np.ndarray:
        """
        Maximize the minimum determinant by moving one point, retaining signs
        of all its current oriented triples.  Feasibility at a fixed target is
        a convex polygon, solved exactly by half-plane clipping.
        """
        det = determinants(p)
        involved = np.flatnonzero(np.any(triples == index, axis=1))
        outside = np.flatnonzero(~np.any(triples == index, axis=1))
        fixed_limit = float(np.min(np.abs(det[outside]))) if len(outside) else 1.0

        constraints = []
        base = p[index].copy()
        for row in involved:
            tri = triples[row]
            sign = 1.0 if det[row] >= 0.0 else -1.0

            def oriented_at(x: np.ndarray) -> float:
                trial = p.copy()
                trial[index] = x
                ii, jj, kk = tri
                a, b, c = trial[ii], trial[jj], trial[kk]
                return ((b[0] - a[0]) * (c[1] - a[1])
                        - (b[1] - a[1]) * (c[0] - a[0]))

            f0 = oriented_at(np.array((0.0, 0.0)))
            f1 = oriented_at(np.array((1.0, 0.0)))
            f2 = oriented_at(np.array((0.0, 1.0)))
            # sign * (A*u + B*v + C) >= target
            constraints.append((sign * (f1 - f0),
                                sign * (f2 - f0),
                                sign * f0))

        old_score = float(np.min(np.abs(det)))
        lo, hi = old_score, fixed_limit
        best_poly = None
        for _ in range(30):
            target = 0.5 * (lo + hi)
            poly = np.array(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
            for aa, bb, cc in constraints:
                poly = clip_halfplane(poly, aa, bb, target - cc)
                if len(poly) == 0:
                    break
            if len(poly):
                lo = target
                best_poly = poly
            else:
                hi = target

        if best_poly is None or not np.all(np.isfinite(best_poly)):
            return p[index]
        # Centroid is safely in the feasible cell and avoids gratuitously
        # selecting a vertex with poor lower-tail slack.
        return project(np.mean(best_poly, axis=0))

    # Exact coordinate maximin passes on several elite annealing basins.
    for _, _, seed in elite:
        candidate = seed.copy()
        for sweep in range(22):
            before = float(values(candidate).min())
            order = rng.permutation(np.arange(3, n))
            for idx in order:
                proposal = coordinate_lp(candidate, int(idx))
                trial = candidate.copy()
                trial[idx] = proposal
                trial_values, trial_score, trial_merit = score_and_merit(trial)
                old_values, old_score, old_merit = score_and_merit(candidate)
                if (trial_score > old_score + 1e-12 or
                        (abs(trial_score - old_score) <= 1e-12 and
                         trial_merit > old_merit + 1e-13)):
                    candidate = trial
            if float(values(candidate).min()) <= before + 2e-11:
                break

        _, cand_score, cand_merit = score_and_merit(candidate)
        if (cand_score > best_score + 1e-14 or
                (abs(cand_score - best_score) <= 1e-14 and
                 cand_merit > best_merit)):
            best, best_score, best_merit = candidate, cand_score, cand_merit

    points = np.empty((n, 2), dtype=float)
    points[:, 0] = best[:, 0] + 0.5 * best[:, 1]
    points[:, 1] = sqrt3_over_2 * best[:, 1]
    return points


# EVOLVE-BLOCK-END