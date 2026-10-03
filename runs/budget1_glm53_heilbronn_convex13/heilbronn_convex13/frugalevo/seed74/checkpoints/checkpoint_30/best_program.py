# EVOLVE-BLOCK-START
import numpy as np

try:
    from scipy.optimize import minimize
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def _triplets(n=13):
    """All C(n,3) index triplets, precomputed once."""
    return np.array([(i, j, k) for i in range(n) for j in range(i + 1, n)
                     for k in range(j + 1, n)])


def _areas(P, idx):
    """Areas of all triangles of point set P (vectorized cross products)."""
    a, b, c = P[idx[:, 0]], P[idx[:, 1]], P[idx[:, 2]]
    return 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                        - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))


def _softmin_obj(x, tau, idx):
    """Stable negative log-sum-exp soft-min of triangle areas; smooth
    max-min surrogate that becomes exact as tau -> 0."""
    P = x.reshape(-1, 2)
    ar = _areas(P, idx) + 1e-12
    m = ar.min()
    return m - tau * np.log(np.sum(np.exp(-(ar - m) / tau)))


def _softmin_obj_jac(x, tau, idx):
    """Soft-min value AND its exact analytic gradient (softmax-weighted
    sum of per-triangle area gradients), enabling fast SLSQP hops."""
    P = x.reshape(-1, 2)
    a, b, c = P[idx[:, 0]], P[idx[:, 1]], P[idx[:, 2]]
    cross = ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
             - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    ar = 0.5 * np.abs(cross) + 1e-12
    m = ar.min()
    e = np.exp(-(ar - m) / tau)
    s = e.sum()
    val = m - tau * np.log(s)
    sgn = np.where(cross >= 0, 0.5, -0.5) * (e / s)
    g = np.zeros((P.shape[0], 2))
    np.add.at(g, idx[:, 0], sgn[:, None] * np.stack([b[:, 1] - c[:, 1],
                                                     b[:, 0] - c[:, 0]], 1))
    np.add.at(g, idx[:, 1], sgn[:, None] * np.stack([c[:, 1] - a[:, 1],
                                                     a[:, 0] - c[:, 0]], 1))
    np.add.at(g, idx[:, 2], sgn[:, None] * np.stack([a[:, 1] - b[:, 1],
                                                     b[:, 0] - a[:, 0]], 1))
    return val, g.ravel()


def _hull_area(P):
    """Convex hull area via monotone-chain + shoelace formula."""
    pts = P[np.lexsort((P[:, 1], P[:, 0]))]
    def half(pts):
        h = []
        for p in pts:
            while len(h) >= 2 and ((h[-1][0] - h[-2][0]) * (p[1] - h[-2][1])
                                   - (h[-1][1] - h[-2][1]) * (p[0] - h[-2][0])) <= 0:
                h.pop()
            h.append(p)
        return h
    lower = half(pts)
    upper = half(pts[::-1])
    hull = np.array(lower[:-1] + upper[:-1])
    x, y = hull[:, 0], hull[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _score(P, idx):
    """Normalized objective: min triangle area / convex hull area."""
    return _areas(P, idx).min() / max(_hull_area(P), 1e-12)


def _coord_polish(P, idx, rounds=120, step0=0.05):
    """Deterministic greedy ascent on the TRUE min-area objective.

    Line-searches each point along coordinate axes AND along a 45-degree
    rotated basis (4 directions total), which escapes stalls that pure
    axis-aligned moves cannot. Adaptive step: grows on success, halves
    on a full round of failure. Accepts only strict improvements.
    """
    P = P.copy()
    best = _areas(P, idx).min()
    step = step0
    inv = 0.5 ** 0.5
    dirs = ((1.0, 0.0), (0.0, 1.0), (inv, inv), (inv, -inv))
    for _ in range(rounds):
        improved = False
        for i in range(P.shape[0]):
            for dx, dy in dirs:
                for s in (+1.0, -1.0):
                    ox, oy = P[i, 0], P[i, 1]
                    P[i, 0] = ox + s * dx * step
                    P[i, 1] = oy + s * dy * step
                    v = _areas(P, idx).min()
                    if v > best + 1e-15:
                        best = v
                        improved = True
                    else:
                        P[i, 0], P[i, 1] = ox, oy
        if improved:
            step = min(step * 1.3, 0.1)
        else:
            step *= 0.5
            if step < 1e-8:
                break
    return P


def _lp_polish(P, idx, iters=150):
    """Exact nonsmooth steepest-ascent finisher via small LPs.

    At a max-min optimum several triangle areas tie; the incumbent's
    coordinate ascent (8 fixed directions) and single-active-gradient
    SLSQP stall there. Each iteration solves  max t  s.t.
    g_k · d >= t  for the tightly-active (near-minimal) triangle
    gradients, ||d||_inf <= 1, via scipy.optimize.linprog (HiGHS), then
    backtracks from step 0.02 on the TRUE normalized score. Terminates
    on KKT (t* <= 0), line-search failure, or the iteration cap, always
    returning a valid configuration (unmoved P on any failure).
    """
    P = np.array(P, dtype=float, copy=True)
    try:
        from scipy.optimize import linprog
    except Exception:
        return P
    m = P.shape[0]
    for _ in range(int(iters)):
        ar = _areas(P, idx)
        jmin = ar.min()
        rows = idx[ar <= jmin + 1e-12]
        k = rows.shape[0]
        if k == 0:
            break
        a, b, c = P[rows[:, 0]], P[rows[:, 1]], P[rows[:, 2]]
        cross = ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                 - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
        s = np.where(cross >= 0, 0.5, -0.5)[:, None]
        g0 = s * np.stack([b[:, 1] - c[:, 1], c[:, 0] - b[:, 0]], axis=1)
        g1 = s * np.stack([c[:, 1] - a[:, 1], a[:, 0] - c[:, 0]], axis=1)
        g2 = s * np.stack([a[:, 1] - b[:, 1], b[:, 0] - a[:, 0]], axis=1)
        G = np.zeros((k, 2 * m))
        rk = np.arange(k)
        G[rk, 2 * rows[:, 0]] = g0[:, 0]
        G[rk, 2 * rows[:, 0] + 1] = g0[:, 1]
        G[rk, 2 * rows[:, 1]] = g1[:, 0]
        G[rk, 2 * rows[:, 1] + 1] = g1[:, 1]
        G[rk, 2 * rows[:, 2]] = g2[:, 0]
        G[rk, 2 * rows[:, 2] + 1] = g2[:, 1]
        c_vec = np.zeros(2 * m + 1)
        c_vec[-1] = -1.0
        try:
            res = linprog(c_vec,
                          A_ub=np.hstack([-G, np.ones((k, 1))]),
                          b_ub=np.zeros(k),
                          bounds=[(-1.0, 1.0)] * (2 * m) + [(None, None)],
                          method="highs")
        except Exception:
            break
        if not res.success or not np.all(np.isfinite(res.x)):
            break
        d = res.x[:2 * m].reshape(m, 2)
        t = float(res.x[2 * m])
        if not np.isfinite(t) or t <= 1e-12:
            break  # KKT point: coordinate ascent already at local optimum
        base = _score(P, idx)
        step = 0.02
        improved = False
        for _ls in range(30):
            Q = P + step * d
            if _hull_area(Q) > 1e-12 and _score(Q, idx) > base + 1e-15:
                P = Q
                improved = True
                break
            step *= 0.5
        if not improved:
            break
    return P


def heilbronn_convex13() -> np.ndarray:
    """
    Multistart max-min optimization of the minimum triangle area for 13
    points, normalized by convex hull area. Uses a family of two-ring
    structured starts (varying ring sizes and radii, with a LARGE inner
    ring, since tiny interior points create many small triangles),
    annealed SLSQP soft-min with a fine temperature schedule, an exact
    active-set gradient polish, and a final deterministic greedy
    coordinate-ascent polish on the true objective. Best valid incumbent
    is always retained and returned as a finite (13, 2) array.
    """
    n = 13
    idx = _triplets(n)
    best, best_val = None, -np.inf

    def _normalize(Q):
        """Affine-normalize a seed set: centroid at origin, max-norm 1."""
        Q = np.asarray(Q, dtype=float)
        Q = Q - Q.mean(axis=0)
        s = np.abs(Q).max()
        return Q / s if s > 0 else Q

    starts = []
    # --- Core ring seeds FIRST (retain reachability of the incumbent's
    # best basin; attempt 1 showed dropping these loses the 0.0296
    # optimum). Running them before the time guard can trim anything
    # guarantees the winning basin is always visited.
    for k, rin, phase in ((8, 0.26, 0.3), (9, 0.22, 0.0), (7, 0.30, 0.45)):
        a_out = 2 * np.pi * np.arange(k) / k
        outer = 0.5 + 0.45 * np.stack([np.cos(a_out), np.sin(a_out)], axis=1)
        m = n - k
        a_in = 2 * np.pi * np.arange(m) / m + phase
        inner = 0.5 + rin * np.stack([np.cos(a_in), np.sin(a_in)], axis=1)
        base = np.vstack([outer, inner])
        starts.append(base)
        rng = np.random.default_rng(seed=500 + k)
        starts.append(base + 0.03 * (rng.random((n, 2)) - 0.5))
    # --- Family A: Vogel / golden-angle phyllotaxis spirals. Point k at
    # radius c*sqrt(k), angle k*2*pi*(1 - 1/phi): quasirandom, well
    # spread, no ring degeneracies. Sweep radial scale + jitter.
    golden = 2.0 * np.pi * (1.0 - (np.sqrt(5.0) - 1.0) / 2.0)
    ks = np.arange(n, dtype=float)
    base_dirs = np.stack([np.cos(golden * ks), np.sin(golden * ks)], axis=1)
    for c in (0.125, 0.13, 0.135, 0.14, 0.145, 0.15, 0.155, 0.16):
        starts.append(_normalize(c * np.sqrt(ks + 0.6)[:, None] * base_dirs))
    # Refinement: rotation offsets and elliptical stretch of the spiral
    # reach affine-conjugate basins the axis-aligned spiral cannot.
    for rot in (np.pi / 13.0, np.pi / 7.0, np.pi / 3.5, np.pi / 5.0):
        R = np.array([[np.cos(rot), -np.sin(rot)],
                      [np.sin(rot), np.cos(rot)]])
        starts.append(_normalize(
            0.14 * (base_dirs @ R.T) * np.sqrt(ks + 0.6)[:, None]))
    for ecc in (0.85, 1.2):
        Q = 0.14 * np.sqrt(ks + 0.6)[:, None] * base_dirs
        Q[:, 1] *= ecc
        starts.append(_normalize(Q))
    for s in range(6):
        rng = np.random.default_rng(seed=400 + 13 * s)
        c = 0.13 + 0.0035 * (s % 6)
        starts.append(_normalize(
            c * np.sqrt(ks + 0.6)[:, None] * base_dirs
            + 0.02 * (rng.random((n, 2)) - 0.5)))
    # --- Family B: classical Heilbronn parabola construction (t, t^2)
    # with 13 nearly equally spaced parameters; sweep affine stretch.
    for g1, g2 in ((0.0, 0.0), (0.1, 0.0), (0.0, 0.1), (0.1, 0.1),
                   (0.2, 0.05), (0.05, 0.2)):
        t = (np.arange(n, dtype=float) + g1) / (n - 1.0)
        t = t + g2 * np.sin(3.0 * np.pi * t)
        Q = np.stack([2.0 * (t - 0.5), 3.0 * (t ** 2 - t.mean() ** 2)], axis=1)
        starts.append(_normalize(Q))
    for s in range(6):
        rng = np.random.default_rng(seed=900 + 29 * s)
        t = (np.arange(n, dtype=float) + 0.05 * rng.random()) / (n - 1.0)
        Q = np.stack([2.0 * (t - 0.5), 3.0 * (t ** 2 - 0.3)], axis=1) \
            + 0.02 * (rng.random((n, 2)) - 0.5)
        starts.append(_normalize(Q))
    # --- Family C: hexagonal lattice interior + hexagon boundary.
    ax = np.stack([np.cos(2.0 * np.pi * np.arange(6) / 6.0),
                   np.sin(2.0 * np.pi * np.arange(6) / 6.0)], axis=1)
    hexdirs = np.vstack([ax[0], ax[1], ax[1] - ax[0], -ax[0], -ax[1],
                         ax[0] - ax[1]])
    for a in (0.18, 0.20, 0.22, 0.24, 0.26):
        lat = np.vstack([np.zeros(2), a * hexdirs])
        starts.append(_normalize(np.vstack([1.0 * ax, lat])))
    for s in range(5):
        rng = np.random.default_rng(seed=1600 + 37 * s)
        a = 0.18 + 0.02 * (s % 5)
        lat = np.vstack([np.zeros(2), a * hexdirs])
        starts.append(_normalize(
            np.vstack([1.0 * ax, lat]) + 0.02 * (rng.random((n, 2)) - 0.5)))
    # --- A few deterministic random starts (diversity safety net).
    for s in range(4):
        rng = np.random.default_rng(seed=1000 + s)
        starts.append(rng.random((n, 2)))

    # Time guard: keep well inside the 360s budget. Eval time is ~172 s
    # with a 120 s hop budget, so extending the hop budget to 165 s and
    # the global guard to 330 s spends idle headroom on the stage that
    # produces the incumbent's final polish.
    import time
    t_end = time.time() + 330.0

    if not _HAS_SCIPY:
        for P in starts:
            v = _score(P, idx)
            if v > best_val:
                best_val, best = v, P.copy()
        return np.asarray(best, dtype=float)

    for P0 in starts:
        if time.time() > t_end:
            break
        x = np.clip(np.asarray(P0, dtype=float), 1e-3, 1 - 1e-3).ravel().copy()
        if x.size != 2 * n:
            continue
        try:
            # Anneal soft-min temperature toward the exact max-min problem.
            for tau in (0.02, 0.006, 0.002, 6e-4, 2e-4):
                res = minimize(_softmin_obj, x, args=(tau, idx),
                               method="SLSQP",
                               options={"maxiter": 250, "ftol": 1e-14})
                if np.all(np.isfinite(res.x)):
                    x = res.x

            # Exact max-min polish with analytic gradient of the active
            # (minimum) triangle's area.
            def exact_obj(xx):
                Pp = xx.reshape(-1, 2)
                ar = _areas(Pp, idx)
                j = int(np.argmin(ar))
                i0, i1, i2 = idx[j]
                cross = ((Pp[i1, 0] - Pp[i0, 0]) * (Pp[i2, 1] - Pp[i0, 1])
                         - (Pp[i1, 1] - Pp[i0, 1]) * (Pp[i2, 0] - Pp[i0, 0]))
                sgn = 1.0 if cross >= 0 else -1.0
                g = np.zeros((len(Pp), 2))
                a3, b3, c3 = Pp[i0], Pp[i1], Pp[i2]
                g[i0] = sgn * 0.5 * np.array([b3[1] - c3[1], c3[0] - b3[0]])
                g[i1] = sgn * 0.5 * np.array([c3[1] - a3[1], a3[0] - c3[0]])
                g[i2] = sgn * 0.5 * np.array([a3[1] - b3[1], b3[0] - a3[0]])
                return ar[j], g.ravel()

            res = minimize(exact_obj, x, method="SLSQP", jac=True,
                           options={"maxiter": 250, "ftol": 1e-14})
            if np.all(np.isfinite(res.x)) and \
                    exact_obj(res.x)[0] >= exact_obj(x)[0] - 1e-15:
                x = res.x
        except Exception:
            pass  # keep incumbent x

        P = x.reshape(n, 2)
        if not np.all(np.isfinite(P)):
            continue
        # Deterministic coordinate-ascent polish on the true objective.
        try:
            P = _coord_polish(P, idx, rounds=30)
        except Exception:
            pass
        v = _score(P, idx)
        if v > best_val:
            best_val, best = v, P.copy()

    # Deterministic basin hopping around the incumbent: seeded Gaussian
    # jitter applied to all (or half of the) coordinates with sigma
    # decaying geometrically (0.01 -> 1e-5), then a FAST analytic-gradient
    # soft-min SLSQP re-anneal (tau 0.004 -> 1.5e-4) followed by the
    # exact-objective coordinate polish. The cheap per-hop cost (vs. the
    # former 26-dim Powell descent) permits many more ridge-crossing hops
    # inside the same 150 s budget; acceptance remains strict improvement
    # of the true normalized score, so the incumbent is always retained.
    if best is not None:
        try:
            rng = np.random.default_rng(seed=1234567)
            t_bh = time.time() + min(165.0, max(60.0, t_end - time.time()))
            sig0, sig1 = 3e-2, 1e-5
            h, hops = 0, 4000
            while h < hops:
                if time.time() > t_bh or time.time() > t_end:
                    break
                sig = sig0 * (sig1 / sig0) ** (h / 3999.0)
                cand = best.copy()
                if h % 2 == 0:
                    # Full-configuration jitter: cross active-set ridges.
                    cand = cand + sig * rng.standard_normal((n, 2))
                else:
                    # Partial jitter: perturb a seeded random half of the
                    # points, preserving part of the incumbent basin.
                    mask = rng.random(n) < 0.5
                    if not mask.any():
                        mask[rng.integers(n)] = True
                    cand[mask] += sig * rng.standard_normal((int(mask.sum()), 2))
                h += 1
                if not np.all(np.isfinite(cand)):
                    continue
                if _hull_area(cand) <= 1e-12:
                    continue
                xh = cand.ravel().copy()
                try:
                    for tau_h in (0.003, 5e-4):
                        res = minimize(_softmin_obj, xh, jac=_softmin_obj_jac,
                                       args=(tau_h, idx), method="SLSQP",
                                       options={"maxiter": 80, "ftol": 1e-14})
                        if np.all(np.isfinite(res.x)):
                            xh = res.x
                except Exception:
                    pass
                cand = xh.reshape(n, 2)
                if not np.all(np.isfinite(cand)) or _hull_area(cand) <= 1e-12:
                    continue
                cand = _coord_polish(cand, idx, rounds=25)
                vc = _score(cand, idx)
                if np.isfinite(vc) and vc > best_val + 1e-15:
                    best_val, best = vc, cand.copy()
        except Exception:
            pass  # incumbent returned unchanged

    # Final exact nonsmooth steepest-ascent on the incumbent: escapes
    # simultaneous-tie stalls that coordinate ascent cannot, using only
    # bounded microsecond-scale HiGHS LPs, followed by a short greedy
    # settle. The incumbent is only ever replaced on strict improvement,
    # so a failure here costs nothing.
    if best is not None:
        # Strengthened end-stage LP steepest-ascent: a cyclic schedule of
        # exact active-set LP directions (each raising all tied triangles
        # simultaneously) alternated with coordinate-ascent re-settles on
        # the true normalized score. Runs only once on the single
        # incumbent, so it costs no basin-hopping throughput. Every step
        # of _lp_polish and _coord_polish is accepted only on strict
        # improvement of the true objective, and the incumbent is only
        # replaced on strict global improvement, so failure here is free.
        try:
            P_lp = _coord_polish(best, idx, rounds=15)
            v_lp = _score(P_lp, idx)
            if np.isfinite(v_lp) and v_lp > best_val:
                best_val, best = v_lp, P_lp.copy()
            for _cycle in range(3):
                if time.time() > t_end:
                    break
                P_lp = _lp_polish(best, idx, iters=200)
                P_lp = _coord_polish(P_lp, idx, rounds=20)
                v_lp = _score(P_lp, idx)
                if np.isfinite(v_lp) and v_lp > best_val + 1e-15:
                    best_val, best = v_lp, P_lp.copy()
                else:
                    break
        except Exception:
            pass  # incumbent returned unchanged

    if best is None:
        best = np.asarray(starts[0], dtype=float)
    best = np.asarray(best, dtype=float).reshape(n, 2)
    best -= best.mean(axis=0)
    scale = np.abs(best).max()
    if scale > 0:
        best /= scale
    return best


# EVOLVE-BLOCK-END
