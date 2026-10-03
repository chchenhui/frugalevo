# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_convex13() -> np.ndarray:
    """Search six threefold-symmetric starts, release them, then use twelve ordinary starts."""
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
                y[:4] += rng.normal(0, .018 * (0.03 ** (iteration / 3500)), 4)
                y[4:] += rng.normal(0, .035 * (0.03 ** (iteration / 3500)), 4)
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
            base = np.array([(i / 3, j / 3) for i in range(4) for j in range(4)], float)
            points = base[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 15]].copy()
            rng.shuffle(points)
            points = np.clip(points + rng.normal(0, .045, points.shape), .015, .985)
            cur_min, cur_score = values(points)
            local, local_min, local_score = points.copy(), cur_min, cur_score
            for iteration in range(7000):
                t = 1.0 - iteration / 7000.0
                y = points.copy()
                if rng.random() < .12:
                    idx = rng.choice(n, 2, replace=False)
                    y[idx] += rng.normal(0, .075 * (0.025 ** (iteration / 7000)), (2, 2))
                else:
                    idx = rng.integers(n)
                    y[idx] += rng.normal(0, .075 * (0.025 ** (iteration / 7000)), 2)
                y = np.clip(y, .002, .998)
                qmin, qscore = values(y)
                if qscore > cur_score or rng.random() < np.exp((qscore - cur_score) / (.0012 * t + 1e-7)):
                    points, cur_min, cur_score = y, qmin, qscore
                if cur_min > local_min or (cur_min == local_min and cur_score > local_score):
                    local, local_min, local_score = points.copy(), cur_min, cur_score
            points, cur_min = local, local_min
        if cur_min > best_min:
            best_min, best_points = cur_min, points.copy()

    return np.asarray(best_points, dtype=float)


# EVOLVE-BLOCK-END
