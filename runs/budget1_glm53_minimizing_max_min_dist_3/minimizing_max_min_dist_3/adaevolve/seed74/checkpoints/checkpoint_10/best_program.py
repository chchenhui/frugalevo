# EVOLVE-BLOCK-START
import time
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Constructs 14 points in 3D maximizing (dmin/dmax)^2.

    Approach: feasibility homotopy. A configuration is feasible for t if
    every pairwise distance lies in [t, 1]; feasibility certifies
    dmin/dmax >= t. For each seed (sphere-repulsion configs plus
    structured seeds: icosahedron+2 poles, cuboctahedron+2 poles, rhombic
    dodecahedron, double hexagon rings) we bisect on t: at each trial t
    the smooth penalty sum(relu(t-d)^2) + sum(relu(d-1)^2) is minimized
    with L-BFGS (analytic gradient; Adam fallback if scipy is missing)
    with random restarts to test feasibility. The largest feasible t per
    seed lower-bounds its ratio; the best config across seeds gets a
    second homotopy pass and a targeted hill-climb on the exact ratio.
    Output is normalized to unit diameter. Wall-clock guards keep the
    run well inside the evaluation timeout.
    """

    n = 14
    rng = np.random.default_rng(2024)
    T0 = time.time()
    BUDGET_HOM = 220.0
    BUDGET_TOT = 300.0

    try:
        from scipy.optimize import minimize as _sci_min
        HAVE_SCIPY = True
    except Exception:
        HAVE_SCIPY = False

    ii, jj = np.triu_indices(n, 1)

    def pdist(p):
        return np.linalg.norm(p[ii] - p[jj], axis=1)

    def ratio(p):
        d = pdist(p)
        return d.min() / d.max()

    def penalty(p, t):
        d = pdist(p)
        low = np.maximum(t - d, 0.0)
        high = np.maximum(d - 1.0, 0.0)
        P = float((low * low).sum() + (high * high).sum())
        c = (2.0 * high - 2.0 * low)[:, None]
        u = (p[ii] - p[jj]) / np.maximum(d, 1e-12)[:, None]
        g = np.zeros((n, 3))
        np.add.at(g, ii, c * u)
        np.add.at(g, jj, -(c * u))
        return P, g

    def solve(p, t):
        p = p / pdist(p).max()
        if HAVE_SCIPY:
            def fg(x):
                P, g = penalty(x.reshape(n, 3), t)
                return P, g.ravel()
            res = _sci_min(fg, p.ravel(), jac=True, method='L-BFGS-B',
                           options={'maxiter': 600, 'ftol': 1e-16,
                                    'gtol': 1e-12})
            q = res.x.reshape(n, 3)
            return q, penalty(q, t)[0] < 1e-10
        x = p.copy()
        m = np.zeros((n, 3))
        v = np.zeros((n, 3))
        for k in range(1, 2001):
            P, g = penalty(x, t)
            if P < 1e-12:
                break
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g * g
            x = x - 0.02 * (m / (1 - 0.9 ** k)) / (
                np.sqrt(v / (1 - 0.999 ** k)) + 1e-8)
        return x, penalty(x, t)[0] < 1e-9

    def homotopy(p):
        p = p / pdist(p).max()
        lo, hi = ratio(p), 1.0
        while hi - lo > 2e-5 and time.time() - T0 < BUDGET_HOM:
            t = 0.5 * (lo + hi)
            q, ok = solve(p, t)
            if not ok:
                for _ in range(3):
                    if time.time() - T0 > BUDGET_HOM:
                        break
                    q2 = q + 0.03 * rng.standard_normal((n, 3))
                    q2, ok = solve(q2, t)
                    if ok:
                        q = q2
                        break
            if ok:
                lo, p = t, q
            else:
                hi = t
        return p, lo

    def repel(p, steps=350, lr=0.02):
        p = p / np.linalg.norm(p, axis=1, keepdims=True)
        for s in range(steps):
            step = lr * (1 - s / steps) + 1e-4
            diff = p[:, None, :] - p[None, :, :]
            d = np.linalg.norm(diff, axis=-1)
            np.fill_diagonal(d, np.inf)
            inv = 1.0 / np.maximum(d, 1e-6) ** 2
            f = (diff * inv[:, :, None]).sum(axis=1)
            fn = np.linalg.norm(f, axis=1, keepdims=True)
            p = p + f / np.maximum(fn, 1e-12) * step
            p /= np.linalg.norm(p, axis=1, keepdims=True)
        return p

    seeds = [repel(rng.standard_normal((n, 3))) for _ in range(14)]
    phi = 0.5 * (1 + np.sqrt(5))
    ico = np.array([[0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
                    [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
                    [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]],
                   float)
    for c in (0.4, 0.9, 1.4):
        seeds.append(np.vstack([ico, [[0, 0, c], [0, 0, -c]]])
                     + 0.01 * rng.standard_normal((n, 3)))
    co = np.array([[a, b, 0] for a in (1, -1) for b in (1, -1)] +
                  [[a, 0, b] for a in (1, -1) for b in (1, -1)] +
                  [[0, a, b] for a in (1, -1) for b in (1, -1)], float)
    for a in (0.6, 1.0, 1.5):
        seeds.append(np.vstack([co, [[0, 0, a], [0, 0, -a]]])
                     + 0.01 * rng.standard_normal((n, 3)))
    rd = np.vstack([np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
                              [0, 0, 1], [0, 0, -1]], float),
                    0.5 * np.array([[a, b, c] for a in (1, -1) for b in (1, -1)
                                     for c in (1, -1)], float)])
    seeds.append(rd + 0.01 * rng.standard_normal((n, 3)))
    ang = np.arange(6) * np.pi / 3
    for (r, h, c, tw) in ((1.0, 0.5, 1.2, 0.5), (0.9, 0.7, 1.0, np.pi / 6),
                          (1.1, 0.3, 1.4, np.pi / 12)):
        ring1 = np.stack([r * np.cos(ang + tw), r * np.sin(ang + tw),
                          np.zeros(6)], axis=1)
        ring2 = np.stack([r * np.cos(ang), r * np.sin(ang), np.zeros(6)],
                         axis=1)
        seeds.append(np.vstack([ring1 + [0, 0, h], ring2 - [0, 0, h],
                                [[0, 0, c], [0, 0, -c]]])
                     + 0.01 * rng.standard_normal((n, 3)))

    seeds.sort(key=lambda q: -ratio(q))
    best_p, best_r = seeds[0], ratio(seeds[0])
    for q in seeds:
        if time.time() - T0 > BUDGET_HOM:
            break
        p2, _ = homotopy(q)
        r = ratio(p2)
        if r > best_r:
            best_r, best_p = r, p2.copy()

    # second pass from the best configuration found so far
    if time.time() - T0 < BUDGET_HOM:
        p2, _ = homotopy(best_p)
        if ratio(p2) > best_r:
            best_r, best_p = ratio(p2), p2.copy()

    p = best_p / pdist(best_p).max()
    cur = ratio(p)
    for it in range(1, 20001):
        if time.time() - T0 > BUDGET_TOT:
            break
        d = pdist(p)
        if rng.random() < 0.6:
            k, is_min = int(np.argmin(d)), True
        else:
            k, is_min = int(np.argmax(d)), False
        i, j = int(ii[k]), int(jj[k])
        m = i if rng.random() < 0.5 else j
        o = j if m == i else i
        scale = 0.01 * (1 - it / 20000) + 1e-5
        cand = p.copy()
        if rng.random() < 0.5:
            u = cand[m] - cand[o]
            nu = np.linalg.norm(u)
            if nu > 1e-12:
                cand[m] = cand[m] + (1.0 if is_min else -1.0) * u / nu * scale
        else:
            cand[m] = cand[m] + rng.standard_normal(3) * scale
        s = ratio(cand)
        if s >= cur:
            p, cur = cand, s

    return p / pdist(p).max()


# EVOLVE-BLOCK-END
