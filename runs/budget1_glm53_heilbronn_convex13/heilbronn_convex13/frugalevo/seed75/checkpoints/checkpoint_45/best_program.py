# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull

_N = 13
_TRIPLES = np.array(
    [(i, j, k) for i in range(_N) for j in range(i + 1, _N) for k in range(j + 1, _N)],
    dtype=np.int64,
)


_N_HULL = 10
_N_IN = _N - _N_HULL


def _decode(x: np.ndarray) -> np.ndarray:
    """Decode 17 params to 13 points: topology-partitioned encoding.

    Params 0..9 : softmax logits of the 10 hull angular gaps (any positive
                  gaps summing to 2*pi -> arbitrary non-uniform hull spacing,
                  strictly convex unit-circle polygon guaranteed).
    Params 10..15: free (x,y) in [-0.9,0.9]^2 for 3 interior points
                  (scaled from [0,1]); ConvexHull re-detects the hull so
                  even a point straying outside stays valid.
    Param 16    : global rotation phase.

    Unlike the prior ring encoding (fixed cyclic rays, radius >= 0.5),
    this allows deep interior points and clustered hull vertices.
    """
    x = np.asarray(x, dtype=np.float64)
    logits = x[:_N_HULL] - np.max(x[:_N_HULL])
    e = np.exp(logits)
    gaps = 2.0 * np.pi * e / np.sum(e)
    ang = x[16] + np.concatenate([[0.0], np.cumsum(gaps)[:-1]])
    hull_pts = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    uv = np.clip(x[10:16].reshape(_N_IN, 2), 0.0, 1.0)
    inner = 1.8 * uv - 0.9
    # shrink any interior point that escaped the radius-0.9 open disk
    r = np.linalg.norm(inner, axis=1)
    scale = np.where(r > 0.9, 0.9 / np.maximum(r, 1e-12), 1.0)
    inner = inner * scale[:, None]
    return np.concatenate([hull_pts, inner], axis=0)


def _objective(x: np.ndarray) -> float:
    """Negative normalized minimum triangle area (to be minimized)."""
    pts = _decode(x)
    a = pts[_TRIPLES[:, 0]]
    b = pts[_TRIPLES[:, 1]]
    c = pts[_TRIPLES[:, 2]]
    areas2 = np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                    - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    hull_area = ConvexHull(pts).volume
    return -float(np.min(areas2) / (2.0 * hull_area))


def _score(pts: np.ndarray) -> float:
    """Exact evaluator objective: min triangle area / hull area; -inf if invalid."""
    if not np.all(np.isfinite(pts)):
        return -np.inf
    a = pts[_TRIPLES[:, 0]]
    b = pts[_TRIPLES[:, 1]]
    c = pts[_TRIPLES[:, 2]]
    areas2 = np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                    - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    try:
        hull_area = float(ConvexHull(pts).volume)
    except Exception:
        return -np.inf
    if not np.isfinite(hull_area) or hull_area < 1e-12:
        return -np.inf
    return float(np.min(areas2)) / (2.0 * hull_area)


def _rescale(x: np.ndarray) -> np.ndarray:
    """Scale configuration so hull area = 1 (removes flat scaling direction)."""
    try:
        ha = float(ConvexHull(x.reshape(_N, 2)).volume)
    except Exception:
        return x
    if np.isfinite(ha) and ha > 1e-12:
        return x / np.sqrt(ha)
    return x


def _score_smooth(pts: np.ndarray, kappa: float = 300.0,
                  band: float = 1.20, lift: float = 1e-5) -> float:
    """Two-tier lexicographic surrogate of the normalized min-area objective.

    Tier 1 is the exact minimum normalized area. Tier 2 is a log-sum-exp
    soft minimum (kappa) restricted to the near-minimal band of triangles
    with area <= band * min. The score is
        score = min + lift * softmin(band subset),
    with lift chosen so the tier-2 contribution (~lift * min ~ 3e-8) is
    strictly below typical min-area differences between nearby configs
    (~1e-6): the objective is genuinely lexicographic — the minimum always
    wins — yet along the active-set manifold, where the exact objective is
    exactly flat, the tier-2 term still has nonzero gradient, so strict-
    improvement pattern search accepts moves that thicken the near-minimal
    band. Those moves are precisely the ones from which a subsequent exact
    descent can lift the minimum itself.
    """
    if not np.all(np.isfinite(pts)):
        return -np.inf
    a = pts[_TRIPLES[:, 0]]
    b = pts[_TRIPLES[:, 1]]
    c = pts[_TRIPLES[:, 2]]
    areas2 = np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                    - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    try:
        hull_area = float(ConvexHull(pts).volume)
    except Exception:
        return -np.inf
    if not np.isfinite(hull_area) or hull_area < 1e-12:
        return -np.inf
    areas = areas2 / (2.0 * hull_area)
    m = float(np.min(areas))
    sub = areas[areas <= band * m]
    if sub.size <= 1:
        return m
    ms = float(np.min(sub))
    soft_sub = ms - np.log(float(np.sum(np.exp(-kappa * (sub - ms))))) / kappa
    return m + lift * soft_sub


def _hooke_jeeves(x: np.ndarray, sigma0: float, sigma_min: float,
                  budget: int, obj=None) -> tuple:
    """Hooke-Jeeves pattern search on 26 free coordinates.

    obj: callable pts(13,2) -> float to maximize; defaults to the exact
    evaluator objective _score. Coordinate sweeps accept strictly improving
    +-sigma moves (with hull-area renormalization); after each successful
    sweep an extrapolated pattern move with doubling line search accelerates
    progress. Sigma halves only when a full sweep + pattern phase accepts
    nothing. Returns (best_x, best_val).
    """
    if obj is None:
        obj = _score
    x = _rescale(np.asarray(x, dtype=np.float64).copy())
    best_val = obj(x.reshape(_N, 2))
    if not np.isfinite(best_val):
        return x, best_val
    sigma = sigma0
    while sigma > sigma_min and budget > 0:
        improved_any = False
        sweeping = True
        while sweeping and budget > 0:
            sweeping = False
            x_before = x.copy()
            for i in range(2 * _N):
                for s in (sigma, -sigma):
                    if budget <= 0:
                        break
                    budget -= 1
                    trial = x.copy()
                    trial[i] += s
                    trial = _rescale(trial)
                    v = obj(trial.reshape(_N, 2))
                    if v > best_val:
                        x, best_val = trial, v
                        improved_any = sweeping = True
                        break
            # Diagonal exploratory moves on cyclically adjacent coordinate
            # pairs (4 sign combos each). Pure axis-aligned sweeps cannot
            # represent improvements that require moving two coordinates
            # simultaneously; these diagonal probes remove that blindness
            # at a bounded cost (~4*2N evals per sweep).
            for i in range(2 * _N):
                j = (i + 1) % (2 * _N)
                for si in (sigma, -sigma):
                    for sj in (sigma, -sigma):
                        if budget <= 0:
                            break
                        budget -= 1
                        trial = x.copy()
                        trial[i] += si
                        trial[j] += sj
                        trial = _rescale(trial)
                        v = obj(trial.reshape(_N, 2))
                        if v > best_val:
                            x, best_val = trial, v
                            improved_any = sweeping = True
                            break
            direction = x - x_before
            if np.any(direction != 0.0):
                step = 2.0
                while budget > 0:
                    budget -= 1
                    trial = _rescale(x + step * direction)
                    v = obj(trial.reshape(_N, 2))
                    if v > best_val:
                        x, best_val = trial, v
                        step *= 2.0
                    else:
                        break
        if not improved_any:
            sigma *= 0.5
    return x, best_val


def _areas2_signed_jac(pts: np.ndarray) -> tuple:
    """Twice-areas of all 286 triangles plus their exact analytic Jacobian.

    The twice-area |(b-a) x (c-a)| is quadratic in the coordinates, but its
    gradient with respect to each coordinate is exact and cheap:
      dA/dx_i = s*(y_j - y_k), dA/dy_i = s*(x_k - x_j), ...
    where s = sign of the signed cross product at the current point (this
    resolves the |.| kink exactly for the active branch; a sign flip inside
    a solve is corrected by the next outer round's re-linearization).
    Returns (areas2, J) with shapes (286,) and (286, 2N).
    """
    ti = _TRIPLES[:, 0]
    tj = _TRIPLES[:, 1]
    tk = _TRIPLES[:, 2]
    a = pts[ti]
    b = pts[tj]
    c = pts[tk]
    sA = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) \
        - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    sg = np.where(sA >= 0.0, 1.0, -1.0)
    areas2 = sg * sA
    m = len(ti)
    J = np.zeros((m, 2 * _N), dtype=np.float64)
    np.add.at(J, (np.arange(m), 2 * ti), sg * (b[:, 1] - c[:, 1]))
    np.add.at(J, (np.arange(m), 2 * ti + 1), sg * (c[:, 0] - b[:, 0]))
    np.add.at(J, (np.arange(m), 2 * tj), sg * (c[:, 1] - a[:, 1]))
    np.add.at(J, (np.arange(m), 2 * tj + 1), sg * (a[:, 0] - c[:, 0]))
    np.add.at(J, (np.arange(m), 2 * tk), sg * (a[:, 1] - b[:, 1]))
    np.add.at(J, (np.arange(m), 2 * tk + 1), sg * (b[:, 0] - a[:, 0]))
    return areas2, J


def _hull_area_grad(pts: np.ndarray) -> tuple:
    """Convex-hull area H and its exact analytic gradient w.r.t. all 13 points.

    For a CCW polygon with vertices v_0..v_{m-1},
      H = 0.5 * sum_k (x_k*y_{k+1} - y_k*x_{k+1}),
      dH/dx_k = 0.5*(y_{k+1} - y_{k-1}),  dH/dy_k = 0.5*(x_{k-1} - x_{k+1}).
    Interior (non-hull) points receive zero gradient. Returns (H, grad[13,2]).
    """
    try:
        h = ConvexHull(pts)
    except Exception:
        return np.nan, None
    H = float(h.volume)
    if not np.isfinite(H) or H < 1e-12:
        return np.nan, None
    verts = np.asarray(h.vertices, dtype=np.int64)  # CCW in 2D
    m = len(verts)
    v = pts[verts]
    nxt = np.roll(np.arange(m), -1)
    prv = np.roll(np.arange(m), 1)
    gx = 0.5 * (v[nxt, 1] - v[prv, 1])
    gy = 0.5 * (v[prv, 0] - v[nxt, 0])
    grad = np.zeros((_N, 2), dtype=np.float64)
    grad[verts, 0] = gx
    grad[verts, 1] = gy
    return H, grad


def _epigraph_refine(pts: np.ndarray, rounds: int = 15) -> tuple:
    """Hull-area-coupled epigraph-minimax SLSQP with analytic Jacobians.

    Maximizes the scored ratio directly: solve for t subject to
      g_i(z) = areas2_i(pts) - 2*t*H(pts) >= 0  for all 286 triangles,
    where H(pts) is the hull area treated as a first-class differentiable
    quantity (analytic gradient via _hull_area_grad; hull combinatorics
    recomputed on every constraint call, cheap for 13 points). This adds
    the -2*t*dH/dz hull-shrink term that frozen-hull solves omit, so the
    stationary point of each SQP solve is a true local optimum of the
    normalized objective min-area/hull-area whenever hull vertices are on
    the active set. The constraint Jacobian rows are
      [J_i - 2*t*dH/dz , -2*H],
    both factors exact and re-linearized each outer round. Accept-only-if-
    better via exact _score; the _rescale guard keeps candidates valid.
    Returns (best_pts, best_score).
    """
    from scipy.optimize import minimize as _slsqp_min
    best_pts = pts.copy()
    best_sc = _score(best_pts)
    starts = [best_pts.copy()]
    for fx_, fy_, ang_ in ((1.03, 0.97, 0.0), (0.97, 1.03, 0.0),
                           (1.0, 1.0, np.pi / _N),
                           (1.01, 0.99, 0.0), (0.99, 1.01, 0.0)):
        c_, s_ = np.cos(ang_), np.sin(ang_)
        q = best_pts.copy()
        qx = q[:, 0] * fx_
        qy = q[:, 1] * fy_
        q[:, 0] = qx * c_ - qy * s_
        q[:, 1] = qx * s_ + qy * c_
        starts.append(q)

    ncol = 2 * _N + 1
    obj_grad = np.zeros(ncol, dtype=np.float64)
    obj_grad[2 * _N] = -1.0  # minimize -t

    for p0 in starts:
        cur = _rescale(p0.reshape(-1)).reshape(_N, 2).copy()
        if not np.isfinite(_score(cur)):
            continue
        stalls = 0  # consecutive non-improving SLSQP solves for this start
        for _ in range(rounds):
            H0, _g0 = _hull_area_grad(cur)
            if not np.isfinite(H0):
                break
            a2, _ = _areas2_signed_jac(cur)
            t0 = max(float(np.min(a2)) * 0.98 / (2.0 * H0), 1e-9)

            def _con(z: np.ndarray) -> np.ndarray:
                p = z[:2 * _N].reshape(_N, 2)
                H, _ = _hull_area_grad(p)
                if not np.isfinite(H):
                    return np.full(len(_TRIPLES), -1e12)
                return _areas2(p) - 2.0 * z[2 * _N] * H

            def _cjon(z: np.ndarray) -> np.ndarray:
                p = z[:2 * _N].reshape(_N, 2)
                _, J = _areas2_signed_jac(p)
                H, gH = _hull_area_grad(p)
                t = z[2 * _N]
                if not np.isfinite(H) or gH is None:
                    return np.zeros((J.shape[0], ncol), dtype=np.float64)
                rows = J - (2.0 * t) * gH.reshape(1, 2 * _N)
                return np.concatenate(
                    [rows, -2.0 * H * np.ones((J.shape[0], 1),
                                              dtype=np.float64)], axis=1)

            z0 = np.concatenate([cur.reshape(-1).astype(np.float64), [t0]])
            bounds = [(z0[i] - 0.5, z0[i] + 0.5) for i in range(2 * _N)] + \
                     [(0.0, max(t0 * 1.5, 1e-9))]
            try:
                res = _slsqp_min(
                    lambda z: -z[2 * _N], z0, method="SLSQP",
                    jac=lambda z: obj_grad,
                    bounds=bounds,
                    constraints=[{"type": "ineq", "fun": _con, "jac": _cjon}],
                    options={"maxiter": 300, "ftol": 1e-14},
                )
            except Exception:
                break
            if not np.all(np.isfinite(res.x)):
                break
            cand = _rescale(np.asarray(res.x[:2 * _N],
                                       dtype=np.float64)).reshape(_N, 2)
            cs = _score(cand)
            cur_sc = _score(cur)
            if np.isfinite(cs) and cs > cur_sc + 1e-12:
                cur = cand
                stalls = 0
                if cs > best_sc:
                    best_pts, best_sc = cand.copy(), cs
            else:
                # Do not abandon the whole active-set loop on the first
                # non-improving solve: a small deterministic kick lets the
                # next round re-detect the active set and re-solve from a
                # nearby point, which often yields further exact gains.
                stalls += 1
                if stalls > 3:
                    break
                ang_k = np.arange(_N, dtype=np.float64)
                q = cur.copy()
                q[:, 0] += 1e-4 * np.cos(1.7 * ang_k + 0.4)
                q[:, 1] += 1e-4 * np.sin(2.3 * ang_k + 1.1)
                kicked = _rescale(q.reshape(-1)).reshape(_N, 2)
                if np.isfinite(_score(kicked)):
                    cur = kicked
    return best_pts, best_sc


def _areas2(pts: np.ndarray) -> np.ndarray:
    """Vectorized twice-areas of all 286 triangles."""
    a = pts[_TRIPLES[:, 0]]
    b = pts[_TRIPLES[:, 1]]
    c = pts[_TRIPLES[:, 2]]
    return np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                  - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))


def heilbronn_convex13() -> np.ndarray:
    """
    Global stage + strengthened multi-start pattern refinement, n=13.

    Stage 1: DE on the 17-parameter topology-partitioned encoding
    (10 softmax-gap hull vertices on the unit circle + 3 free interior
    points in the open radius-0.9 disk + phase) with SLSQP polish. This
    removes the ring encoding's structural ceiling (fixed rays, r >= 0.5,
    equal angular base spacing) and permits deep interior points and
    non-uniform hull vertex clustering.
    Stage 2: smooth-surrogate escape phase — Hooke-Jeeves on the log-sum-exp
    softmin surrogate (kappa=60) from the DE incumbent and deterministic
    anisotropic perturbations; the smooth landscape lets moves slide past
    the nonsmooth min-max kinks that trap exact-objective search.
    Stage 3: exact multi-start Hooke-Jeeves (min triangle area / hull area)
    with hull-area renormalization and accept-only-if-better moves, from the
    surrogate-refined incumbents. Stage 4: four-stage exact polish ladder
    (sigma 0.05 -> 0.0005) from the overall best, so intermediate scales are
    re-explored between fine descents. Each start has a hard evaluation
    budget (~380k total cheap evals, well within timeout). The best valid
    incumbent is always returned; the regular 13-gon is the safe fallback.
    """
    # 17-parameter space: 10 gap logits + 6 interior coords + 1 phase
    n_par = _N_HULL + 2 * _N_IN + 1
    bounds = [(0.0, 1.0)] * n_par
    best_x, best_val = None, np.inf
    try:
        res = differential_evolution(
            _objective, bounds, seed=42, popsize=45, maxiter=500,
            tol=1e-10, polish=True, init="sobol",
        )
        if np.isfinite(res.fun):
            best_x, best_val = np.asarray(res.x, dtype=np.float64), float(res.fun)
    except Exception:
        pass

    if best_x is not None:
        try:
            pol = minimize(
                _objective, best_x, method="SLSQP",
                bounds=bounds, options={"maxiter": 200, "ftol": 1e-12},
            )
            if np.isfinite(pol.fun) and pol.fun < best_val:
                best_x, best_val = np.asarray(pol.x, dtype=np.float64), float(pol.fun)
        except Exception:
            pass

    theta0 = 2.0 * np.pi * np.arange(_N) / _N
    fallback = np.stack([np.cos(theta0), np.sin(theta0)], axis=1).astype(np.float64)

    pts = _decode(best_x) if best_x is not None else fallback

    # Multi-start Hooke-Jeeves: DE incumbent + deterministic perturbations
    # (anisotropic scalings plus a rotation), accept-only-if-better.
    starts = [pts.reshape(-1).copy()]
    for fx_, fy_, ang_ in ((1.06, 0.94, 0.0), (0.92, 1.08, 0.0),
                           (1.03, 1.0, 0.0), (1.0, 1.0, np.pi / _N)):
        c_, s_ = np.cos(ang_), np.sin(ang_)
        q = pts.copy()
        qx = q[:, 0] * fx_
        qy = q[:, 1] * fy_
        q[:, 0] = qx * c_ - qy * s_
        q[:, 1] = qx * s_ + qy * c_
        starts.append(q.reshape(-1).copy())

    # Stage 2: two-tier lexicographic escape from each start. The surrogate
    # score = min + 1e-5 * softmin(near-minimal band, band=1.2) is strictly
    # lexicographic (tier-2 contribution ~3e-8 << min differences ~1e-6) but
    # has nonzero gradient along the flat active-set manifold: the descent
    # first thickens the near-minimal band (coarse sigma), then sharpens
    # tier 2 (fine sigma, high kappa), handing off points from which the
    # exact objective can lift the minimum itself.
    refined = []
    for x0 in starts:
        try:
            sx, sv = _hooke_jeeves(x0, sigma0=0.10, sigma_min=1e-4,
                                   budget=10000,
                                   obj=lambda p: _score_smooth(p, 150.0))
            if np.isfinite(sv):
                sx2, _ = _hooke_jeeves(sx, sigma0=0.03, sigma_min=1e-5,
                                       budget=8000,
                                       obj=lambda p: _score_smooth(p, 500.0))
                refined.append(sx2)
            else:
                refined.append(x0)
        except Exception:
            refined.append(x0)

    # Stage 3: exact multi-start descent (min triangle area / hull area)
    # from the band-thickened incumbents, accept-only-if-better.
    best_pts, best_sc = pts.copy(), _score(pts)
    for x0 in refined:
        try:
            px, pv = _hooke_jeeves(x0, sigma0=0.08, sigma_min=1e-8,
                                   budget=120000)
            if np.isfinite(pv) and pv > best_sc:
                best_pts, best_sc = px.reshape(_N, 2), pv
        except Exception:
            pass

    # Deterministic perturbation-and-redescend cycle: structured fixed
    # kicks to the incumbent first enter a tier-aware surrogate descent
    # (thickens the near-minimal band from the kicked point), then an exact
    # descent lifts the minimum — a two-step path the flat single-tier
    # exact objective cannot express from the kicked point alone.
    for scale in (0.02, 0.01, 0.005):
        ang = np.arange(_N, dtype=np.float64)
        q = best_pts.copy()
        q[:, 0] += scale * np.cos(1.3 * ang + 0.7)
        q[:, 1] += scale * np.sin(2.1 * ang + 0.3)
        try:
            tx, tv = _hooke_jeeves(q.reshape(-1).copy(), sigma0=0.02,
                                   sigma_min=1e-5, budget=15000,
                                   obj=lambda p: _score_smooth(p, 300.0))
            rx, rv = _hooke_jeeves(tx, sigma0=0.02,
                                   sigma_min=1e-8, budget=30000)
            if np.isfinite(rv) and rv > best_sc:
                best_pts, best_sc = rx.reshape(_N, 2), rv
        except Exception:
            pass

    # Epigraph-minimax SLSQP refinement: maximize auxiliary t subject to
    # area(triple_i) - t >= 0 for all 286 triples (hull area fixed per
    # solve). SQP steps move all points along directions that raise every
    # active triangle simultaneously — a direction coordinate-wise pattern
    # search and simplex moves cannot express at the active-set kink.
    # Multi-start from the incumbent + 3 deterministic perturbations,
    # accept-only-if-better via exact _score, bounded rounds.
    try:
        eg_pts, eg_sc = _epigraph_refine(best_pts, rounds=15)
        if np.isfinite(eg_sc) and eg_sc > best_sc:
            best_pts, best_sc = eg_pts, eg_sc
            # Second escape cycle on the epigraph-refined incumbent: the
            # epigraph SLSQP solves hold hull combinatorics fixed, leaving
            # boundary kinks; smooth surrogate + exact redescend cross them.
            try:
                s2x, s2v = _hooke_jeeves(best_pts.reshape(-1).copy(),
                                         sigma0=0.03, sigma_min=1e-6,
                                         budget=30000, obj=_score_smooth)
                if np.isfinite(s2v):
                    e2x, e2v = _hooke_jeeves(s2x, sigma0=0.02,
                                             sigma_min=1e-9, budget=80000)
                    if np.isfinite(e2v) and e2v > best_sc:
                        best_pts, best_sc = e2x.reshape(_N, 2), e2v
            except Exception:
                pass
            for scale in (0.008, 0.003):
                ang = np.arange(_N, dtype=np.float64)
                q = best_pts.copy()
                q[:, 0] += scale * np.cos(2.6 * ang + 1.9)
                q[:, 1] += scale * np.sin(0.9 * ang + 2.4)
                try:
                    rx, rv = _hooke_jeeves(q.reshape(-1).copy(), sigma0=0.01,
                                           sigma_min=1e-9, budget=60000)
                    if np.isfinite(rv) and rv > best_sc:
                        best_pts, best_sc = rx.reshape(_N, 2), rv
                except Exception:
                    pass
    except Exception:
        pass

    # Final exact polish ladder after the epigraph stage: re-explore
    # intermediate sigma scales between broad and fine descents (a single
    # short cleanup leaves gains at unvisited scales on the table), each
    # stage accept-only-if-better on the exact objective, bounded budgets.
    for s0, smin, bud in ((0.05, 1e-6, 40000), (0.01, 1e-8, 40000),
                          (0.002, 1e-9, 30000), (0.0005, 1e-10, 25000)):
        try:
            fx2, fv2 = _hooke_jeeves(best_pts.reshape(-1).copy(),
                                     sigma0=s0, sigma_min=smin, budget=bud)
            if np.isfinite(fv2) and fv2 > best_sc:
                best_pts, best_sc = fx2.reshape(_N, 2), fv2
        except Exception:
            pass

    if np.all(np.isfinite(best_pts)) and _score(best_pts) > 0.0:
        return np.asarray(best_pts, dtype=np.float64)
    return fallback


# EVOLVE-BLOCK-END
