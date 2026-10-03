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

    starts = []
    # Structured two-ring starts: (outer k, inner 13-k) with generous
    # inner radius. Previously the inner ring was too small (r=0.17),
    # which forces many near-degenerate triangles.
    for k, rin, phase in ((8, 0.26, 0.3), (9, 0.22, 0.0), (7, 0.30, 0.45)):
        a_out = 2 * np.pi * np.arange(k) / k
        outer = 0.5 + 0.45 * np.stack([np.cos(a_out), np.sin(a_out)], axis=1)
        m = n - k
        a_in = 2 * np.pi * np.arange(m) / m + phase
        inner = 0.5 + rin * np.stack([np.cos(a_in), np.sin(a_in)], axis=1)
        base = np.vstack([outer, inner])
        starts.append(base)
        # Symmetry-breaking perturbations of each structured start.
        for s in range(4):
            rng = np.random.default_rng(seed=500 + 17 * s + k)
            amp = 0.015 + 0.02 * s
            starts.append(base + amp * (rng.random((n, 2)) - 0.5))
    # Radial-jittered outer rings (breaks ring regularity, often key).
    for s in range(4):
        rng = np.random.default_rng(seed=2000 + s)
        rad = 0.42 + 0.07 * rng.random(9)
        a9 = 2 * np.pi * np.arange(9) / 9 + 0.05 * s
        outer = 0.5 + rad[:, None] * np.stack([np.cos(a9), np.sin(a9)], axis=1)
        a4 = 2 * np.pi * np.arange(4) / 4 + 0.3 + 0.1 * s
        inner = 0.5 + (0.18 + 0.06 * rng.random(4))[:, None] \
            * np.stack([np.cos(a4), np.sin(a4)], axis=1)
        starts.append(np.vstack([outer, inner]))
    # A few deterministic random starts.
    for s in range(10):
        rng = np.random.default_rng(seed=1000 + s)
        starts.append(rng.random((n, 2)))
    # Extra jittered variants of the best-performing structured family
    # (8 outer / 5 inner with large inner radius).
    base8 = starts[0]
    for s in range(6):
        rng = np.random.default_rng(seed=7000 + 31 * s)
        starts.append(base8 + (0.02 + 0.015 * s) * (rng.random((n, 2)) - 0.5))

    # Time guard: keep well inside the 360s budget.
    import time
    t_end = time.time() + 310.0

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
            P = _coord_polish(P, idx, rounds=60)
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
            t_bh = time.time() + 150.0
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

    if best is None:
        best = np.asarray(starts[0], dtype=float)
    best = np.asarray(best, dtype=float).reshape(n, 2)
    best -= best.mean(axis=0)
    scale = np.abs(best).max()
    if scale > 0:
        best /= scale
    return best


# EVOLVE-BLOCK-END
