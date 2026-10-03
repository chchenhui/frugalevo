# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """
    Deterministic hybrid construction for the eleven-point Heilbronn problem.

    Points are optimized in barycentric-like coordinates (u, v):
        u >= 0, v >= 0, u + v <= 1.
    In these coordinates, determinant triangle areas are already normalized
    by the area of the enclosing equilateral triangle.
    """
    n = 11
    rng = np.random.default_rng(7711031987)
    sqrt3_over_2 = np.sqrt(3.0) * 0.5

    triples = np.array(
        [(i, j, k)
         for i in range(n - 2)
         for j in range(i + 1, n - 1)
         for k in range(j + 1, n)],
        dtype=np.intp,
    )
    vertices = np.array(
        ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
        dtype=float,
    )

    def project(x: np.ndarray) -> np.ndarray:
        """Project a point onto u>=0, v>=0, u+v<=1."""
        y = np.maximum(np.asarray(x, dtype=float), 0.0)
        total = float(y[0] + y[1])
        if total > 1.0:
            y = y / total
        return y

    def determinants(p: np.ndarray) -> np.ndarray:
        a = p[triples[:, 0]]
        b = p[triples[:, 1]]
        c = p[triples[:, 2]]
        return ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))

    def values(p: np.ndarray) -> np.ndarray:
        return np.abs(determinants(p))

    def score_and_merit(p: np.ndarray):
        z = values(p)
        low = np.partition(z, 23)[:24]
        lowest = np.partition(z, 11)[:12]
        score = float(z.min())
        # Small-area tail guidance makes it easier to leave weak arrangements
        # without replacing the actual maximin objective.
        merit = float(score + 0.070 * low.mean() + 0.030 * lowest.mean())
        return z, score, merit

    def signature(p: np.ndarray) -> np.ndarray:
        """Label-independent geometry signature for archive diversity."""
        d = p[:, None, :] - p[None, :, :]
        dist = np.sqrt(np.sum(d * d, axis=2))
        return np.sort(dist[np.triu_indices(n, 1)])

    archive = []

    def add_archive(p: np.ndarray, score=None, merit=None) -> None:
        nonlocal archive
        if score is None:
            _, score, merit = score_and_merit(p)
        sig = signature(p)
        replacement = None
        for q in archive:
            if np.max(np.abs(sig - q[3])) < 0.012:
                replacement = q
                break

        record = (float(score), float(merit), p.copy(), sig)
        if replacement is not None:
            if (score, merit) > (replacement[0], replacement[1]):
                archive.remove(replacement)
                archive.append(record)
        else:
            archive.append(record)

        archive.sort(key=lambda x: (x[0], x[1]), reverse=True)
        if len(archive) > 10:
            archive = archive[:10]

    def make_start(restart: int) -> np.ndarray:
        p = np.empty((n, 2), dtype=float)
        p[:3] = vertices
        mode = restart % 6

        if mode == 0:
            bary = rng.dirichlet((0.38, 0.38, 0.38), size=8)
            p[3:] = bary[:, 1:]
        elif mode == 1:
            bary = rng.dirichlet((0.70, 0.70, 0.70), size=8)
            p[3:] = bary[:, 1:]
        elif mode == 2:
            bary = rng.dirichlet((1.20, 1.20, 1.20), size=8)
            p[3:] = bary[:, 1:]
        elif mode == 3:
            bary = rng.dirichlet((2.10, 2.10, 2.10), size=8)
            p[3:] = bary[:, 1:]
        else:
            template = np.array(
                ((0.125, 0.055), (0.395, 0.060), (0.710, 0.070),
                 (0.065, 0.335), (0.280, 0.275), (0.555, 0.255),
                 (0.070, 0.675), (0.305, 0.520)),
                dtype=float,
            )
            jitter = 0.026 if mode == 4 else 0.046
            p[3:] = np.array(
                [project(q + rng.normal(0.0, jitter, 2)) for q in template]
            )
        return p

    best = np.vstack((vertices, np.full((8, 2), 1.0 / 3.0)))
    _, best_score, best_merit = score_and_merit(best)
    add_archive(best, best_score, best_merit)

    # Independent annealing cells. Archive snapshots are retained during,
    # rather than only after, each run because good basins may later drift.
    restarts = 15
    iterations = 10800

    for restart in range(restarts):
        current = make_start(restart)
        current_values, current_score, current_merit = score_and_merit(current)
        local_best = current.copy()
        local_score, local_merit = current_score, current_merit

        for it in range(iterations):
            progress = it / float(iterations - 1)
            step = 0.145 * (1.0 - progress) ** 1.52 + 0.00042
            tail_scale = 0.011 * (1.0 - progress) ** 1.20 + 0.00030
            temperature = 0.0030 * (1.0 - progress) ** 2.10 + 1.0e-6

            if rng.random() < 0.90:
                cutoff = current_score + max(2.8 * tail_scale, 0.0010)
                active = triples[current_values <= cutoff]
                tri = active[int(rng.integers(len(active)))]
                movable = tri[tri >= 3]
                if len(movable):
                    index = int(rng.choice(movable))
                else:
                    index = int(rng.integers(3, n))
            else:
                index = int(rng.integers(3, n))

            candidate = current.copy()
            candidate[index] = project(
                candidate[index] + rng.normal(0.0, step, size=2)
            )

            # Occasional correlated moves reorganize congested local cells.
            if rng.random() < 0.10 * (1.0 - progress):
                other = int(rng.integers(3, n - 1))
                if other >= index:
                    other += 1
                candidate[other] = project(
                    candidate[other] + rng.normal(0.0, 0.68 * step, size=2)
                )

            cand_values, cand_score, cand_merit = score_and_merit(candidate)
            delta = cand_merit - current_merit
            if delta >= 0.0 or rng.random() < np.exp(
                max(-700.0, delta / temperature)
            ):
                current = candidate
                current_values = cand_values
                current_score = cand_score
                current_merit = cand_merit

            if (current_score > local_score + 1e-14 or
                    (abs(current_score - local_score) <= 1e-14 and
                     current_merit > local_merit)):
                local_best = current.copy()
                local_score = current_score
                local_merit = current_merit

            if it > 0 and it % 1800 == 0:
                add_archive(local_best, local_score, local_merit)

        add_archive(local_best, local_score, local_merit)
        if (local_score > best_score + 1e-14 or
                (abs(local_score - best_score) <= 1e-14 and
                 local_merit > best_merit)):
            best, best_score, best_merit = local_best, local_score, local_merit

    def clip_halfplane(poly: np.ndarray, aa: float, bb: float,
                       rhs: float) -> np.ndarray:
        """Clip a convex polygon by aa*x + bb*y >= rhs."""
        if len(poly) == 0:
            return poly
        output = []
        prev = poly[-1]
        prev_value = aa * prev[0] + bb * prev[1] - rhs

        for cur in poly:
            cur_value = aa * cur[0] + bb * cur[1] - rhs
            prev_inside = prev_value >= -2e-13
            cur_inside = cur_value >= -2e-13

            if prev_inside != cur_inside:
                denominator = prev_value - cur_value
                if abs(denominator) > 1e-18:
                    t = prev_value / denominator
                    output.append(prev + t * (cur - prev))
            if cur_inside:
                output.append(cur)

            prev = cur
            prev_value = cur_value

        return np.asarray(output, dtype=float)

    def coordinate_polygon(p: np.ndarray, index: int) -> np.ndarray:
        """
        Compute a coordinate-wise maximin feasible polygon while preserving
        all current determinant signs. Coefficients are formed analytically.
        """
        det = determinants(p)
        involved_mask = np.any(triples == index, axis=1)
        involved = np.flatnonzero(involved_mask)
        outside = np.flatnonzero(~involved_mask)

        old_score = float(np.min(np.abs(det)))
        fixed_limit = (float(np.min(np.abs(det[outside])))
                       if len(outside) else 1.0)

        constraints = []
        for row in involved:
            i, j, k = triples[row]
            sign = 1.0 if det[row] >= 0.0 else -1.0
            a, b, c = p[i], p[j], p[k]

            if index == i:
                # det(x,b,c) = cross(b,c) + (by-cy)x + (cx-bx)y
                aa = b[1] - c[1]
                bb = c[0] - b[0]
                cc = b[0] * c[1] - b[1] * c[0]
            elif index == j:
                # det(a,x,c)
                aa = c[1] - a[1]
                bb = a[0] - c[0]
                cc = a[1] * c[0] - a[0] * c[1]
            else:
                # det(a,b,x)
                aa = a[1] - b[1]
                bb = b[0] - a[0]
                cc = a[0] * b[1] - a[1] * b[0]

            constraints.append((sign * aa, sign * bb, sign * cc))

        lo, hi = old_score, fixed_limit
        best_poly = None
        for _ in range(34):
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
            return np.empty((0, 2), dtype=float)
        return best_poly

    def coordinate_proposals(poly: np.ndarray) -> np.ndarray:
        """Return interior and boundary candidates from a feasible LP cell."""
        if len(poly) == 0:
            return poly
        candidates = [np.mean(poly, axis=0)]
        candidates.extend(poly)
        if len(poly) > 1:
            candidates.extend(
                0.5 * (poly + np.roll(poly, -1, axis=0))
            )
        return np.asarray([project(x) for x in candidates], dtype=float)

    # Polish the strongest diverse archive members. The coordinate LP raises
    # active constraints exactly, while testing several points preserves tail
    # slack needed by subsequent coordinate passes.
    archive.sort(key=lambda x: (x[0], x[1]), reverse=True)
    polish_seeds = archive[:8]

    for _, _, seed, _ in polish_seeds:
        candidate = seed.copy()
        _, candidate_score, candidate_merit = score_and_merit(candidate)

        for sweep in range(26):
            before = candidate_score
            improved = False

            # Deterministic but different orders among basins.
            order = rng.permutation(np.arange(3, n))
            for idx in order:
                poly = coordinate_polygon(candidate, int(idx))
                proposals = coordinate_proposals(poly)

                best_local = candidate
                best_local_score = candidate_score
                best_local_merit = candidate_merit

                for proposal in proposals:
                    trial = candidate.copy()
                    trial[idx] = proposal
                    _, trial_score, trial_merit = score_and_merit(trial)
                    if (trial_score > best_local_score + 1e-12 or
                            (abs(trial_score - best_local_score) <= 1e-12 and
                             trial_merit > best_local_merit + 1e-13)):
                        best_local = trial
                        best_local_score = trial_score
                        best_local_merit = trial_merit

                if best_local is not candidate:
                    candidate = best_local
                    candidate_score = best_local_score
                    candidate_merit = best_local_merit
                    improved = True

            if not improved or candidate_score <= before + 2e-11:
                break

        add_archive(candidate, candidate_score, candidate_merit)
        if (candidate_score > best_score + 1e-14 or
                (abs(candidate_score - best_score) <= 1e-14 and
                 candidate_merit > best_merit)):
            best = candidate
            best_score = candidate_score
            best_merit = candidate_merit

    points = np.empty((n, 2), dtype=float)
    points[:, 0] = best[:, 0] + 0.5 * best[:, 1]
    points[:, 1] = sqrt3_over_2 * best[:, 1]
    return points


# EVOLVE-BLOCK-END