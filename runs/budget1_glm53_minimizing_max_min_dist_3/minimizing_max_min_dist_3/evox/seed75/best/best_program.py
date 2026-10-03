# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Construct 14 points in 3D maximizing dmin/dmax.

    Approach: two-stage multi-start optimization.
    Stage 1: maximize the smooth, scale-invariant surrogate
        sum_{i<j} log d_ij - m * log dmax   (m = #pairs = 91)
    with analytic gradients using L-BFGS-B. This "geometric mean of
    normalized distances" objective is smooth and drives all pairwise
    distances up together, avoiding the flat/kinked landscape of the
    raw min/max ratio.
    Stage 2: SLSQP polish of the exact problem: maximize t subject to
        d_ij^2 >= t^2 (all pairs) and dmax^2 <= 1,
    which recovers the precise optimal ratio.
    Stage 3: shrinking-step basin hopping around the incumbent.
    Seeds: rhombic-dodecahedron-like shells, staggered hexagonal prisms,
    Fibonacci spheres, icosahedron-based shells, and random starts.
    Best configuration is returned normalized so dmax = 1.
    """

    n, d = 14, 3
    iu = np.triu_indices(n, 1)
    ii, jj = iu
    m = len(ii)
    rng = np.random.default_rng(42)

    def ratio(P):
        P = np.asarray(P, dtype=float).reshape(n, d)
        dv = P[ii] - P[jj]
        d2 = (dv * dv).sum(1)
        mx = d2.max()
        if mx <= 1e-18:
            return 0.0
        return np.sqrt(d2.min() / mx)

    def neg_surrogate(x):
        """-(sum log d_ij - m log dmax) with analytic gradient."""
        P = x.reshape(n, d)
        dv = P[ii] - P[jj]
        d2 = (dv * dv).sum(1)
        mx = d2.max()
        mx = max(mx, 1e-18)
        f = -0.5 * np.log(d2).sum() + 0.5 * m * np.log(mx)
        g = np.zeros((n, d))
        # gradient of -0.5*sum(log d2): weight 2*(p_i-p_j)/d2 at i, neg at j
        w = (-0.5 / d2)[:, None] * 2.0 * dv
        np.add.at(g, ii, w)
        np.add.at(g, jj, -w)
        # gradient of +0.5*m*log(mx): only the argmax pair
        k = int(np.argmax(d2))
        gw = (m / mx) * dv[k]
        g[ii[k]] += gw
        g[jj[k]] -= gw
        return f, g.ravel()

    def polish(P):
        """SLSQP: maximize t s.t. d_ij^2 - t^2 >= 0, 1 - dmax^2 >= 0."""
        P = P - P.mean(0)
        dv = P[ii] - P[jj]
        P = P / np.sqrt((dv * dv).sum(1).max())
        dv = P[ii] - P[jj]
        t0 = np.sqrt((dv * dv).sum(1).min())
        x0 = np.concatenate([P.ravel(), [t0]])

        def c_pairs(v):
            Q = v[:n * d].reshape(n, d)
            t = v[-1]
            dvv = Q[ii] - Q[jj]
            return (dvv * dvv).sum(1) - t * t

        def c_diam(v):
            Q = v[:n * d].reshape(n, d)
            dvv = Q[ii] - Q[jj]
            return 1.0 - (dvv * dvv).sum(1).max()

        def negt(v):
            return -v[-1]

        try:
            res = minimize(negt, x0, method="SLSQP",
                           constraints=[{"type": "ineq", "fun": c_pairs},
                                        {"type": "ineq", "fun": c_diam}],
                           options={"maxiter": 400, "ftol": 1e-14})
            Q = res.x[:n * d].reshape(n, d)
            if np.all(np.isfinite(Q)):
                return Q, ratio(Q)
        except Exception:
            pass
        return P, ratio(P)

    seeds = []

    # cube vertices (8) + 6 face centers (rhombic dodecahedron vertices)
    cube = np.array([[x, y, z] for x in (-1, 1)
                     for y in (-1, 1) for z in (-1, 1)], dtype=float)
    fc = np.array([[1.5, 0, 0], [-1.5, 0, 0], [0, 1.5, 0],
                   [0, -1.5, 0], [0, 0, 1.5], [0, 0, -1.5]], dtype=float)
    seeds.append(np.vstack([cube, fc]))

    # two staggered hexagons on parallel planes
    ang = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    for h in (0.5, 0.7, 0.9, 1.1, 1.3):
        h1 = np.stack([np.cos(ang), np.sin(ang), np.full(6, h)], axis=1)
        h2 = np.stack([np.cos(ang + np.pi / 6), np.sin(ang + np.pi / 6),
                       np.full(6, -h)], axis=1)
        seeds.append(np.vstack([h1, h2]))

    # Fibonacci sphere
    k = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * k / n)
    theta = np.pi * (1 + 5 ** 0.5) * k
    fib = np.stack([np.cos(theta) * np.sin(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(phi)], axis=1)
    seeds.append(fib)

    # icosahedron vertices (12) + 2 poles
    a = (1 + np.sqrt(5)) / 2
    ico = np.array([[-1, a, 0], [1, a, 0], [-1, -a, 0], [1, -a, 0],
                    [0, -1, a], [0, 1, a], [0, -1, -a], [0, 1, -a],
                    [a, 0, -1], [a, 0, 1], [-a, 0, -1], [-a, 0, 1]],
                   dtype=float)
    seeds.append(np.vstack([ico, [[0, 0, 1.8], [0, 0, -1.8]]]))

    # twisted double heptagons (7+7) on parallel planes — a family
    # structurally distinct from the 6+6 prisms above
    ang7 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    for h in (0.3, 0.6, 0.9):
        for tw in (0.0, np.pi / 7, 2 * np.pi / 7):
            r1 = np.stack([np.cos(ang7), np.sin(ang7), np.full(7, h)], 1)
            r2 = np.stack([np.cos(ang7 + tw), np.sin(ang7 + tw),
                           np.full(7, -h)], 1)
            seeds.append(np.vstack([r1, r2]))

    # 4+6+4 layered icosahedral stack: two staggered triangles around a
    # hexagonal belt — a distinct topology from all ring-based seeds
    ang3 = np.linspace(0, 2 * np.pi, 3, endpoint=False)
    ang6 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    for h in (0.45, 0.7, 0.95):
        top = np.stack([np.cos(ang3), np.sin(ang3), np.full(3, 2 * h)], 1)
        bot = np.stack([np.cos(ang3 + np.pi / 3), np.sin(ang3 + np.pi / 3),
                        np.full(3, -2 * h)], 1)
        mid = np.stack([np.cos(ang6 + np.pi / 6), np.sin(ang6 + np.pi / 6),
                        np.zeros(6)], 1)
        seeds.append(np.vstack([top, mid, bot]))

    # hexagon + twisted octagon hybrid (6+8): another topology where the
    # diameter pairs differ qualitatively from the 7+7 basins
    ang8 = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    for tw in (0.0, np.pi / 8):
        for h in (0.5, 0.8):
            r6 = np.stack([np.cos(ang6), np.sin(ang6), np.full(6, h)], 1)
            r8 = 0.8 * np.stack([np.cos(ang8 + tw), np.sin(ang8 + tw),
                                 np.full(8, -h)], 1)
            seeds.append(np.vstack([r6, r8]))

    # random seeds (fewer — structured seeds above cover more distinct
    # basins per unit of Stage-1 cost)
    for _ in range(10):
        seeds.append(rng.normal(size=(n, d)))

    best_P, best_r = None, -1.0
    for s in seeds:
        x0 = s.ravel()
        for _ in range(2):
            try:
                res = minimize(neg_surrogate, x0, method="L-BFGS-B",
                               jac=True,
                               options={"maxiter": 2000, "maxfun": 4000,
                                        "ftol": 1e-14, "gtol": 1e-10})
                P1 = res.x.reshape(n, d)
                P2, r2 = polish(P1)
                if r2 > best_r:
                    best_r, best_P = r2, P2
            except Exception:
                pass
            x0 = (best_P if best_P is not None else s).ravel() + \
                0.1 * rng.normal(size=n * d)

    if best_P is None:
        best_P = rng.normal(size=(n, d))

    # Stage 3: basin-hopping style refinement around the incumbent.
    # Repeatedly perturb the best configuration (with shrinking step
    # sizes) and re-polish with SLSQP; keep any improvement. For the
    # larger step sizes also re-run the smooth surrogate optimization
    # from the perturbed point, which lets the search hop between
    # qualitatively different basins before the exact polish.
    for sc in (0.15, 0.05, 0.02, 0.005):
        for _ in range(8):
            Q0 = best_P + sc * rng.normal(size=(n, d))
            if sc >= 0.05:
                try:
                    res = minimize(neg_surrogate, Q0.ravel(),
                                   method="L-BFGS-B", jac=True,
                                   options={"maxiter": 800, "maxfun": 1600,
                                            "ftol": 1e-14, "gtol": 1e-10})
                    Qc = res.x.reshape(n, d)
                    if np.all(np.isfinite(Qc)):
                        Q0 = Qc
                except Exception:
                    pass
            Q1, r1 = polish(Q0)
            if r1 > best_r:
                best_r, best_P = r1, Q1

    # Stage 4: iterated self-re-polish — restarting SLSQP from its own
    # output lets the active constraint set change between iterations,
    # squeezing out extra ratio that a single pass misses. Stop when it
    # converges.
    for _ in range(30):
        Q1, r1 = polish(best_P)
        if r1 > best_r + 1e-15:
            best_r, best_P = r1, Q1
        else:
            break

    # Stage 5 (replaces soft-min surrogate): cheap targeted refinement.
    # Perturb only the points involved in the currently-shortest pairs
    # and the diameter pair, then re-polish. This focuses effort on the
    # active constraints (which determine the ratio) instead of paying
    # for full-configuration smooth surrogates that Stage 4 already
    # effectively converges. Much faster than the soft-min stage.
    for sc in (0.01, 0.003, 0.001):
        for _ in range(10):
            dv = best_P[ii] - best_P[jj]
            d2 = (dv * dv).sum(1)
            short = np.argsort(d2)[:6]          # 6 tightest pairs
            diam = int(np.argmax(d2))
            idx = set(ii[short]) | set(jj[short])
            idx |= {ii[diam], jj[diam]}
            Q0 = best_P.copy()
            for i in idx:
                Q0[i] += sc * rng.normal(size=d)
            Q1, r1 = polish(Q0)
            if r1 > best_r:
                best_r, best_P = r1, Q1

    # normalize: center and scale so dmax = 1
    P = best_P - best_P.mean(axis=0)
    dv = P[ii] - P[jj]
    mx = np.sqrt((dv * dv).sum(1).max())
    if mx > 0:
        P = P / mx
    return P


# EVOLVE-BLOCK-END
