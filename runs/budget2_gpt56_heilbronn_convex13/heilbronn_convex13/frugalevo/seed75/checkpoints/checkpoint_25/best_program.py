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
                seeds = np.argsort(raw)[:16]
                chosen = []
                for s in seeds:
                    tri = triples[s]
                    if not any(len(set(tri) & set(old)) >= 2 for old in chosen):
                        chosen.append(tri.copy())
                    if len(chosen) == 4:
                        break

                # Incidence is used as a lightweight Delaunay-cavity proxy:
                # points repeatedly participating near a seed are included.
                for seed in chosen:
                    incidence = np.zeros(n, dtype=float)
                    for tid, tri in enumerate(triples):
                        overlap = len(set(seed) & set(tri))
                        if overlap:
                            incidence[tri] += (overlap / (raw[tid] + 1e-12))
                    order = np.argsort(-incidence)
                    cavity = list(dict.fromkeys(
                        [int(v) for v in seed] +
                        [int(v) for v in order if v not in seed]
                    ))[:5]
                    if len(cavity) != 5:
                        continue
                    cavity = np.asarray(cavity, dtype=np.int64)
                    frozen = np.ones(n, dtype=bool)
                    frozen[cavity] = False
                    base = points.copy()
                    dim = 10
                    population = np.repeat(
                        base[cavity].reshape(1, dim), 18, axis=0
                    )
                    population[1:] = np.clip(
                        population[1:] + rng.normal(0.0, .055, (17, dim)),
                        .002, .998
                    )
                    fit = np.empty(18, dtype=float)
                    for q in range(18):
                        candidate = base.copy()
                        candidate[cavity] = population[q].reshape(5, 2)
                        fit[q] = values(candidate)[0]
                    for generation in range(90):
                        for q in range(18):
                            ids = [v for v in range(18) if v != q]
                            ia, ib, ic = rng.choice(ids, 3, replace=False)
                            trial = population[q].copy()
                            mask = rng.random(dim) < .82
                            mask[rng.integers(dim)] = True
                            trial[mask] = (
                                population[ia, mask] +
                                .72 * (population[ib, mask] -
                                       population[ic, mask])
                            )
                            trial = np.clip(trial, .002, .998)
                            candidate = base.copy()
                            candidate[cavity] = trial.reshape(5, 2)
                            score = values(candidate)[0]
                            if score > fit[q]:
                                population[q], fit[q] = trial, score
                    winner = int(np.argmax(fit))
                    candidate = base.copy()
                    candidate[cavity] = population[winner].reshape(5, 2)
                    # Bounded Powell-like coordinate polishing.
                    step = .012
                    for _ in range(120):
                        improved = False
                        for d in range(dim):
                            for sign in (-1.0, 1.0):
                                trial = candidate.copy()
                                flat = trial[cavity].reshape(-1)
                                flat[d] = np.clip(flat[d] + sign * step,
                                                  .002, .998)
                                trial[cavity] = flat.reshape(5, 2)
                                score = values(trial)[0]
                                if score > values(candidate)[0]:
                                    candidate = trial
                                    improved = True
                        if not improved:
                            step *= .55
                            if step < 2e-5:
                                break
                    qmin, qscore = values(candidate)
                    if qmin > cur_min:
                        points, cur_min, cur_score = (
                            candidate, qmin, qscore
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
