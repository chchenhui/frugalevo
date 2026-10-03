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
            if restart == 6:
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
                # Select overlapping six-vertex patches from the lower tail.
                # The extra three vertices are chosen by co-occurrence in the
                # lowest normalized triangles, allowing adjacent bottlenecks
                # to move together rather than freezing one contact.
                zraw = raw / (hull_area(points) + 1e-12)
                seeds = np.argsort(zraw)[:24]
                chosen = []
                for s in seeds:
                    tri = np.asarray(triples[s], dtype=np.int64)
                    if not any(len(set(map(int, tri)) &
                                   set(map(int, old))) >= 2 for old in chosen):
                        chosen.append(tri)
                    if len(chosen) == 4:
                        break

                for seed in chosen:
                    seed = np.asarray(seed, dtype=np.int64)
                    seed_set = set(map(int, seed))
                    cooccur = np.zeros(n, dtype=float)
                    for tid in seeds:
                        tri = triples[tid]
                        if len(seed_set.intersection(map(int, tri))) > 0:
                            weight = 1.0 / (zraw[tid] + 1e-12)
                            for vertex in tri:
                                iv = int(vertex)
                                if iv not in seed_set:
                                    cooccur[iv] += weight
                    outside = np.argsort(-cooccur, kind="stable")
                    cavity_list = [int(v) for v in seed]
                    cavity_list.extend(
                        int(v) for v in outside
                        if int(v) not in seed_set and cooccur[int(v)] > 0.0
                    )
                    cavity_list = cavity_list[:6]
                    if len(cavity_list) != 6 or len(set(cavity_list)) != 6:
                        continue

                    cavity = np.asarray(cavity_list, dtype=np.int64)
                    base = np.asarray(points, dtype=float).copy()
                    dim = 12
                    population = np.repeat(
                        base[cavity].reshape(1, dim), 20, axis=0
                    )
                    population[1:] = np.clip(
                        population[1:] + rng.normal(0.0, .05, (19, dim)),
                        .002, .998
                    )

                    fit = np.full(20, -1.0, dtype=float)
                    rankfit = np.full(20, -1.0, dtype=float)
                    incumbent = float(cur_min)
                    for q in range(20):
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

                    for generation in range(100):
                        for q in range(20):
                            ids = [v for v in range(20) if v != q]
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
                points = np.asarray(best_points, dtype=float).copy()
                cur_min, cur_score = values(points)
        if cur_min > best_min:
            best_min, best_points = cur_min, points.copy()

    return np.asarray(best_points, dtype=float)


# EVOLVE-BLOCK-END
