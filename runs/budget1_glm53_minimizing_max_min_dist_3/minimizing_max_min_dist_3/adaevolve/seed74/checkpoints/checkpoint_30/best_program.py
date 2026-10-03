# EVOLVE-BLOCK-START
import time
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Constructs 14 points in 3D maximizing (dmin/dmax)^2.

    Approach: contact-graph topology search plus feasibility homotopy.
    At the optimum every shortest AND longest contact equals 1 after
    normalization, so candidate optimal configurations can be described
    by a contact graph. For each hypothesized topology (double hexagonal
    rings + 2 poles, icosahedron skeleton + 2 poles on opposite faces,
    cuboctahedron + 2 poles on opposite faces, full rhombic dodecahedron
    graph) we embed the graph spectrally (Laplacian eigenvectors), then
    run a force-directed stress majorization driving every graph edge to
    unit length (the first-order equilibrium condition of the optimum).
    These graph seeds are then refined together with sphere-repulsion
    and structured seeds by a feasibility homotopy: bisect on t, where a
    config is feasible if all pairwise distances lie in [t, 1], solved by
    minimizing the smooth penalty sum(relu(t-d)^2)+sum(relu(d-1)^2) with
    L-BFGS and random restarts. The best config gets a second homotopy
    pass and targeted hill-climbs on the exact ratio. Output is
    normalized to unit diameter. Wall-clock guards apply throughout.
    """

    n = 14
    rng = np.random.default_rng(2024)
    T0 = time.time()
    BUDGET_HOM = 280.0
    BUDGET_TOT = 345.0

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
        while hi - lo > 3e-6 and time.time() - T0 < BUDGET_HOM:
            t = 0.5 * (lo + hi)
            q, ok = solve(p, t)
            if not ok:
                for _ in range(5):
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

    seeds = [repel(rng.standard_normal((n, 3))) for _ in range(16)]
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

    # ---- BREAKTHROUGH: contact-graph topology seeds ----
    # Hypothesize contact topologies; embed spectrally via the graph
    # Laplacian eigenvectors, then stress-majorize so every graph edge
    # tends to unit length (equilibrium condition of the optimum).
    def graph_embed(A, steps=350):
        A = ((A + A.T) > 0).astype(float)
        np.fill_diagonal(A, 0.0)
        L = np.diag(A.sum(1)) - A
        w, V = np.linalg.eigh(L)
        X = V[:, 1:4] @ np.diag(1.0 + 2.0 * np.abs(w[1:4]))
        ei, ej = np.triu_indices(n, 1)
        msk = A[ei, ej] > 0
        ei, ej = ei[msk], ej[msk]
        for s in range(steps):
            u = X[ei] - X[ej]
            d = np.linalg.norm(u, axis=1)
            f = ((d - 1.0) / np.maximum(d, 1e-9))[:, None] * u
            F = np.zeros((n, 3))
            np.add.at(F, ei, f)
            np.add.at(F, ej, -f)
            F -= 0.05 * X  # mild radial spread prevents collapse
            X += (0.2 * (1 - s / steps) + 0.02) * F
            X -= X.mean(0)
        return X / max(pdist(X).max(), 1e-9)

    tops = []
    # (a) two staggered hexagonal rings + two poles (6+6+2)
    A = np.zeros((n, n))
    for r in (0, 6):
        for k in range(6):
            A[r + k, r + (k + 1) % 6] = 1.0  # short ring contacts
            A[r + k, r + (k + 3) % 6] = 1.0  # long ring contacts
    for k in (0, 2, 4):  # poles cap alternating vertices of both rings
        A[12, k] = A[12, 6 + k + 1] = 1.0
        A[13, 6 + k] = A[13, k + 1] = 1.0
    A[12, 13] = 1.0
    tops.append(A)
    # (b) icosahedron skeleton + poles over opposite faces
    Di = np.sqrt(((ico[:, None] - ico[None, :, :]) ** 2).sum(-1))
    Ai = (np.abs(Di - Di[Di > 0].min()) < 1e-9).astype(float)
    np.fill_diagonal(Ai, 0.0)
    tri = None
    for a in range(12):
        nb = np.where(Ai[a] > 0)[0]
        for b in nb:
            for c in nb[nb > b]:
                if Ai[b, c]:
                    tri = (int(a), int(b), int(c))
                    break
            if tri:
                break
        if tri:
            break
    neg = [int(np.argmin(np.linalg.norm(ico + ico[t], axis=1)))
           for t in tri]
    A = np.zeros((n, n))
    A[:12, :12] = Ai
    for t in tri:
        A[12, t] = 1.0
    for t in neg:
        A[13, t] = 1.0
    A[12, 13] = 1.0
    tops.append(A)
    # (c) cuboctahedron skeleton + poles over opposite triangular faces
    Dc = np.sqrt(((co[:, None] - co[None, :, :]) ** 2).sum(-1))
    Ac = (np.abs(Dc - Dc[Dc > 0].min()) < 1e-9).astype(float)
    np.fill_diagonal(Ac, 0.0)
    A = np.zeros((n, n))
    A[:12, :12] = Ac
    for t in (0, 4, 8):  # face (1,1,0),(1,0,1),(0,1,1)
        A[12, t] = 1.0
    for t in (3, 7, 11):  # opposite face, negated vertices
        A[13, t] = 1.0
    A[12, 13] = 1.0
    tops.append(A)
    # (d) rhombic dodecahedron: 8 cube + 6 octa vertices (exactly 14)
    A = np.zeros((n, n))
    cubv = 0.5 * np.array([[a, b, c] for a in (1, -1) for b in (1, -1)
                           for c in (1, -1)], float)
    for ci in range(8):
        sx, sy, sz = cubv[ci] * 2
        A[ci, 6 if sx > 0 else 7] = 1.0
        A[ci, 8 if sy > 0 else 9] = 1.0
        A[ci, 10 if sz > 0 else 11] = 1.0
    A[6, 7] = A[8, 9] = A[10, 11] = 1.0  # diameter contacts
    tops.append(A)

    for A in tops:
        try:
            g = graph_embed(A)
            if np.isfinite(g).all() and pdist(g).max() > 1e-6:
                seeds.append(g + 0.005 * rng.standard_normal((n, 3)))
                seeds.append(repel(g.copy()))
        except Exception:
            pass

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
    for it in range(1, 60001):
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
        scale = 0.01 * (1 - it / 60000) + 1e-5
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

    # final re-polish: one more homotopy pass from the hill-climbed
    # configuration (it may have drifted off the feasibility boundary),
    # then a short fine-scale hill-climb on the shortest edges.
    if time.time() - T0 < BUDGET_TOT - 20:
        p2, _ = homotopy(p)
        if ratio(p2) > cur:
            p, cur = p2, ratio(p2)
        for it in range(1, 8001):
            if time.time() - T0 > BUDGET_TOT:
                break
            d = pdist(p)
            k = int(np.argmin(d))
            i, j = int(ii[k]), int(jj[k])
            scale = 0.003 * (1 - it / 8000) + 1e-6
            best_c, best_s = None, cur
            for m in (i, j):
                for _ in range(4):
                    cand = p.copy()
                    cand[m] = cand[m] + rng.standard_normal(3) * scale
                    s = ratio(cand)
                    if s > best_s:
                        best_s, best_c = s, cand
            if best_c is not None:
                p, cur = best_c, best_s

    return p / pdist(p).max()


# EVOLVE-BLOCK-END
