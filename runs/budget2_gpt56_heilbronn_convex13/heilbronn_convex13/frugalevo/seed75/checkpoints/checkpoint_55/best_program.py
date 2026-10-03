# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_convex13() -> np.ndarray:
    """Optimize six symmetric starts, then resynthesize four five-point bottleneck cavities."""
    n = 13
    rng = np.random.default_rng(1729)
    triples = np.array(
        [(i, j, k) for i in range(n) for j in range(i + 1, n)
         for k in range(j + 1, n)], dtype=np.int64)

    def hull_area(p):
        order = np.lexsort((p[:, 1], p[:, 0]))
        q = p[order]

        def cross(a, b, c):
            return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

        lo = []
        for x in q:
            while len(lo) >= 2 and cross(lo[-2], lo[-1], x) <= 0:
                lo.pop()
            lo.append(x)
        hi = []
        for x in q[::-1]:
            while len(hi) >= 2 and cross(hi[-2], hi[-1], x) <= 0:
                hi.pop()
            hi.append(x)
        h = np.asarray(lo[:-1] + hi[:-1])
        if len(h) < 3:
            return 0.0
        return 0.5 * abs(np.sum(h[:, 0] * np.roll(h[:, 1], -1)
                                - h[:, 1] * np.roll(h[:, 0], -1)))

    def values(p):
        a, b, c = p[triples[:, 0]], p[triples[:, 1]], p[triples[:, 2]]
        areas = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                             - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
        ha = hull_area(p)
        if ha < 1e-10:
            return -1.0, -1.0
        z = areas / ha
        return float(z.min()), float(0.65 * z.min() + 0.35 * np.partition(z, 8)[:9].mean())

    def orbit(x):
        r = x[:4]
        th = x[4:]
        p = [np.array([0.5, 0.5], dtype=float)]
        for radius, angle in zip(r, th):
            for k in range(3):
                a = angle + 2.0 * np.pi * k / 3.0
                p.append([0.5 + radius * np.cos(a), 0.5 + radius * np.sin(a)])
        return np.asarray(p, dtype=float)

    best_points = None
    best_min = -1.0

    for restart in range(18):
        if restart < 6:
            x = np.r_[np.array([0.14, 0.245, 0.345, 0.445]),
                      np.array([0.03, 0.29, 0.63, 0.94]) +
                      restart * np.pi / 18.0]
            x[:4] = np.clip(x[:4] + rng.normal(0, .012, 4), .06, .47)
            x[4:] %= (2.0 * np.pi / 3.0)
            points = orbit(x)
            cur_min, cur_score = values(points)
            local = points.copy()
            local_min, local_score = cur_min, cur_score

            for iteration in range(3500):
                t = 1.0 - iteration / 3500.0
                y = x.copy()
                fraction = iteration / 3500.0
                decay = 0.03 ** fraction
                radius_step = .018 * decay
                angle_step = .035 * decay

                # Coupled orbit moves adjust one radius and its phase together.
                # The entire three-point orbit is then regenerated, preserving
                # the useful rotational contact structure during annealing.
                if rng.random() < 0.62:
                    orbit_id = int(rng.integers(4))
                    y[orbit_id] += rng.normal(0.0, radius_step)
                    y[4 + orbit_id] += rng.normal(0.0, angle_step)
                else:
                    # Occasional full-manifold moves retain global exploration.
                    y[:4] += rng.normal(0.0, radius_step, 4)
                    y[4:] += rng.normal(0.0, angle_step, 4)

                y[:4] = np.clip(y[:4], .06, .47)
                y[4:] %= (2.0 * np.pi / 3.0)
                qmin, qscore = values(orbit(y))
                temp = .0012 * t + 1e-7
                if qscore > cur_score or rng.random() < np.exp((qscore - cur_score) / temp):
                    x, cur_min, cur_score = y, qmin, qscore
                if cur_min > local_min or (cur_min == local_min and cur_score > local_score):
                    local, local_min, local_score = orbit(x), cur_min, cur_score

            points = local.copy()
            cur_min, cur_score = values(points)
            for iteration in range(3500):
                t = 1.0 - iteration / 3500.0
                y = points.copy()
                idx = rng.integers(n)
                y[idx] += rng.normal(0, .075 * (0.025 ** (iteration / 3500)), 2)
                y[idx] = np.clip(y[idx], .002, .998)
                qmin, qscore = values(y)
                temp = .0012 * t + 1e-7
                if qscore > cur_score or rng.random() < np.exp((qscore - cur_score) / temp):
                    points, cur_min, cur_score = y, qmin, qscore
                if cur_min > local_min or (cur_min == local_min and cur_score > local_score):
                    local, local_min, local_score = points.copy(), cur_min, cur_score
            points, cur_min = local, local_min
        else:
            # Replace the long unrestricted grid anneals with one bounded
            # five-point cavity resynthesis pass on the best released orbit.
            if restart == 6 and False:
                points = np.asarray(best_points, dtype=float).copy()
                cur_min, cur_score = values(points)

                a, b, c = (
                    points[triples[:, 0]],
                    points[triples[:, 1]],
                    points[triples[:, 2]],
                )
                raw = 0.5 * np.abs(
                    (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                    - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
                )
                # Build weighted lower-tail hypergraph cavities.  Each cavity
                # grows across its current boundary using weighted incidence,
                # then receives a separator score rewarding contained contacts
                # and penalizing contacts cut by the separator.
                zraw = raw / (hull_area(points) + 1e-12)
                seeds = np.argsort(zraw, kind="stable")[:32]
                weights = 1.0 / (zraw[seeds] + 1e-12)
                candidates = []
                seen = set()

                for local_seed, seed_index in enumerate(seeds):
                    cavity_set = set(map(int, triples[seed_index]))
                    while len(cavity_set) < 6:
                        incidence = np.zeros(n, dtype=float)
                        for tid, weight in zip(seeds, weights):
                            tri_set = set(map(int, triples[tid]))
                            if tri_set & cavity_set:
                                for vertex in tri_set - cavity_set:
                                    incidence[vertex] += weight
                        for vertex in cavity_set:
                            incidence[vertex] = -1.0
                        vertex = int(np.argmax(incidence))
                        if incidence[vertex] <= 0.0:
                            remaining = [v for v in range(n)
                                         if v not in cavity_set]
                            vertex = int(remaining[0])
                        cavity_set.add(vertex)

                    cavity_key = tuple(sorted(cavity_set))
                    if cavity_key in seen:
                        continue
                    seen.add(cavity_key)

                    contained = 0.0
                    cut = 0.0
                    for tid, weight in zip(seeds, weights):
                        tri_set = set(map(int, triples[tid]))
                        inside = len(tri_set & cavity_set)
                        if inside == 3:
                            contained += weight
                        elif inside > 0:
                            cut += weight
                    separator_score = contained - 0.35 * cut
                    candidates.append((separator_score, cavity_key))

                candidates.sort(key=lambda item: (-item[0], item[1]))
                chosen = []
                for _, cavity_key in candidates[:24]:
                    cavity_set = set(cavity_key)
                    if all(len(cavity_set & set(old)) <= 3 for old in chosen):
                        chosen.append(cavity_key)
                    if len(chosen) == 3:
                        break

                for seed in chosen:
                    cavity = np.asarray(seed, dtype=np.int64)
                    base = np.asarray(points, dtype=float).copy()
                    dim = 12
                    population = np.repeat(
                        base[cavity].reshape(1, dim), 18, axis=0
                    )
                    population[1:] = np.clip(
                        population[1:] + rng.normal(0.0, .05, (17, dim)),
                        .002, .998
                    )

                    fit = np.full(18, -1.0, dtype=float)
                    rankfit = np.full(18, -1.0, dtype=float)
                    incumbent = float(cur_min)
                    for q in range(18):
                        candidate = base.copy()
                        candidate[cavity] = population[q].reshape(6, 2)
                        qmin, _ = values(candidate)
                        if np.isfinite(qmin):
                            a, b, c = (
                                candidate[triples[:, 0]],
                                candidate[triples[:, 1]],
                                candidate[triples[:, 2]],
                            )
                            az = .5 * np.abs(
                                (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                                - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
                            ) / (hull_area(candidate) + 1e-12)
                            fit[q] = qmin
                            rankfit[q] = np.mean(np.partition(az, 11)[:12])

                    for generation in range(90):
                        for q in range(18):
                            ids = [v for v in range(18) if v != q]
                            ia, ib, ic = rng.choice(ids, 3, replace=False)
                            trial = population[q].copy()
                            mask = rng.random(dim) < .8
                            mask[rng.integers(dim)] = True
                            trial[mask] = (
                                population[ia, mask] +
                                .72 * (population[ib, mask] -
                                       population[ic, mask])
                            )
                            trial = np.clip(trial, .002, .998)
                            candidate = base.copy()
                            candidate[cavity] = trial.reshape(6, 2)
                            qmin, _ = values(candidate)
                            if not np.isfinite(qmin) or qmin < incumbent:
                                continue
                            aa, bb, cc = (
                                candidate[triples[:, 0]],
                                candidate[triples[:, 1]],
                                candidate[triples[:, 2]],
                            )
                            az = .5 * np.abs(
                                (bb[:, 0] - aa[:, 0]) * (cc[:, 1] - aa[:, 1])
                                - (bb[:, 1] - aa[:, 1]) * (cc[:, 0] - aa[:, 0])
                            ) / (hull_area(candidate) + 1e-12)
                            rf = float(np.mean(np.partition(az, 11)[:12]))
                            if qmin > fit[q] or (
                                    qmin >= fit[q] and rf > rankfit[q]):
                                population[q], fit[q], rankfit[q] = (
                                    trial, qmin, rf
                                )

                    winner = int(np.argmax(fit))
                    candidate = base.copy()
                    candidate[cavity] = population[winner].reshape(6, 2)
                    step = .012
                    for _ in range(100):
                        improved = False
                        current = values(candidate)[0]
                        for d in range(dim):
                            for sign in (-1.0, 1.0):
                                trial = candidate.copy()
                                flat = trial[cavity].reshape(-1).copy()
                                flat[d] = np.clip(flat[d] + sign * step,
                                                  .002, .998)
                                trial[cavity] = flat.reshape(6, 2)
                                qmin = values(trial)[0]
                                if np.isfinite(qmin) and qmin > current:
                                    candidate, current = trial, qmin
                                    improved = True
                        if not improved:
                            step *= .55
                            if step < 2e-5:
                                break
                    qmin, qscore = values(candidate)
                    if (candidate.shape == (13, 2) and
                            np.all(np.isfinite(candidate)) and
                            qmin > cur_min):
                        points, cur_min, cur_score = (
                            candidate.copy(), qmin, qscore
                        )
                local, local_min, local_score = (
                    points.copy(), cur_min, cur_score
                )
            else:
                if restart == 6:
                    # Deterministic full-configuration Nelder–Mead polish.
                    # Point zero is held fixed as the affine gauge anchor;
                    # the remaining 24 coordinates are optimized jointly.
                    incumbent = np.asarray(
                        best_points, dtype=np.float64
                    ).copy()
                    incumbent_min, incumbent_score = values(incumbent)
                    base_vector = incumbent[1:].reshape(-1).copy()
                    dimension = int(base_vector.size)

                    def simplex_objective(vector):
                        """Evaluate an exact normalized minimum-area objective."""
                        candidate = np.empty((13, 2), dtype=np.float64)
                        candidate[0] = incumbent[0]
                        candidate[1:] = np.asarray(
                            vector, dtype=np.float64
                        ).reshape(12, 2)
                        candidate[1:] = np.clip(
                            candidate[1:], .002, .998
                        )
                        qmin, qscore = values(candidate)
                        if not np.isfinite(qmin):
                            return (1e6, 1e6, candidate)
                        return (-qmin, -qscore, candidate)

                    for start_id, scale in enumerate(
                            (0.0015, 0.004, 0.009)):
                        vertices = np.empty(
                            (dimension + 1, dimension),
                            dtype=np.float64
                        )
                        vertices[0] = base_vector
                        for vertex in range(1, dimension + 1):
                            vertices[vertex] = base_vector.copy()
                            coordinate = vertex - 1
                            direction = 1.0 if (
                                (coordinate + start_id) % 2 == 0
                            ) else -1.0
                            vertices[vertex, coordinate] += (
                                direction * scale
                            )

                        evaluations = [
                            simplex_objective(vertex)
                            for vertex in vertices
                        ]

                        for _ in range(1400):
                            order = np.argsort(
                                np.asarray(
                                    [(item[0], item[1])
                                     for item in evaluations],
                                    dtype=[
                                        ("primary", np.float64),
                                        ("secondary", np.float64),
                                    ],
                                ),
                                order=("primary", "secondary"),
                            )
                            vertices = vertices[order]
                            evaluations = [
                                evaluations[int(index)] for index in order
                            ]

                            if abs(
                                    evaluations[-1][0] -
                                    evaluations[0][0]
                            ) < 1e-12:
                                break

                            centroid = np.mean(vertices[:-1], axis=0)
                            reflected = centroid + (
                                centroid - vertices[-1]
                            )
                            reflected_eval = simplex_objective(reflected)

                            if reflected_eval[:2] < evaluations[0][:2]:
                                expanded = centroid + 2.0 * (
                                    reflected - centroid
                                )
                                expanded_eval = simplex_objective(expanded)
                                if expanded_eval[:2] < reflected_eval[:2]:
                                    vertices[-1] = expanded
                                    evaluations[-1] = expanded_eval
                                else:
                                    vertices[-1] = reflected
                                    evaluations[-1] = reflected_eval
                            elif reflected_eval[:2] < evaluations[-2][:2]:
                                vertices[-1] = reflected
                                evaluations[-1] = reflected_eval
                            else:
                                contracted = centroid + .5 * (
                                    vertices[-1] - centroid
                                )
                                contracted_eval = simplex_objective(
                                    contracted
                                )
                                if contracted_eval[:2] < evaluations[-1][:2]:
                                    vertices[-1] = contracted
                                    evaluations[-1] = contracted_eval
                                else:
                                    anchor = vertices[0].copy()
                                    for vertex in range(1, dimension + 1):
                                        vertices[vertex] = anchor + .5 * (
                                            vertices[vertex] - anchor
                                        )
                                        evaluations[vertex] = (
                                            simplex_objective(
                                                vertices[vertex]
                                            )
                                        )

                        order = np.argsort(
                            np.asarray(
                                [(item[0], item[1])
                                 for item in evaluations],
                                dtype=[
                                    ("primary", np.float64),
                                    ("secondary", np.float64),
                                ],
                            ),
                            order=("primary", "secondary"),
                        )
                        winner = evaluations[int(order[0])][2]
                        polished = np.asarray(
                            winner, dtype=np.float64
                        ).copy()
                        polished_min, polished_score = values(polished)
                        if (
                                np.isfinite(polished_min) and
                                polished_min > incumbent_min
                        ):
                            incumbent = polished
                            incumbent_min = polished_min
                            incumbent_score = polished_score
                            base_vector = incumbent[1:].reshape(-1).copy()

                    points = incumbent.copy()
                    cur_min, cur_score = values(points)
                else:
                    points = np.asarray(
                        best_points, dtype=np.float64
                    ).copy()
                    cur_min, cur_score = values(points)
        if cur_min > best_min:
            best_min, best_points = cur_min, points.copy()

    return np.asarray(best_points, dtype=float)


# EVOLVE-BLOCK-END
