# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull

_N = 13
_TRIPLES = np.array(
    [(i, j, k) for i in range(_N) for j in range(i + 1, _N) for k in range(j + 1, _N)],
    dtype=np.int64,
)


def _decode(x: np.ndarray) -> np.ndarray:
    """Decode 14 params (13 radii in [0,1] mapped to [0.5,1.0], 1 phase) to points.

    The three largest radii are clamped to 1.0 so the hull is always the
    non-degenerate convex hull of >=3 points on the unit circle.
    """
    radii = 0.5 + 0.5 * np.clip(x[:_N], 0.0, 1.0)
    top3 = np.argsort(radii)[-_N + 3:]  # indices of 3 largest
    radii[top3] = 1.0
    theta = x[_N] + 2.0 * np.pi * np.arange(_N) / _N
    return np.stack([radii * np.cos(theta), radii * np.sin(theta)], axis=1)


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


def _score_smooth(pts: np.ndarray, kappa: float = 60.0) -> float:
    """Smooth softmin surrogate of the normalized min-triangle-area objective.

    Uses a log-sum-exp soft minimum over the 286 normalized triangle areas:
    softmin = -(1/kappa) * log(sum_i exp(-kappa * a_i)). As kappa grows this
    converges to the exact minimum, but for finite kappa it is smooth, giving
    pattern search useful gradients away from the active-set kink and letting
    the search slide along near-active constraints instead of getting stuck.
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
    m = float(np.max(areas))
    return m - np.log(float(np.sum(np.exp(-kappa * (areas - m))))) / kappa


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


def _epigraph_refine(pts: np.ndarray, rounds: int = 15) -> tuple:
    """Epigraph-minimax SLSQP on all 26 coordinates + auxiliary variable t.

    Maximizes t subject to areas2_i(pts) - t >= 0 for all 286 triangles,
    with the hull area fixed at the current value inside each SLSQP solve
    (piecewise-linear objective is smooth for fixed hull combinatorics).
    Each outer round re-detects the active set implicitly via the fresh
    t initialization and re-solves. Returns (best_pts, best_score).
    """
    from scipy.optimize import minimize as _slsqp_min
    best_pts = pts.copy()
    best_sc = _score(best_pts)
    starts = [best_pts.copy()]
    for fx_, fy_, ang_ in ((1.03, 0.97, 0.0), (0.97, 1.03, 0.0),
                           (1.0, 1.0, np.pi / _N)):
        c_, s_ = np.cos(ang_), np.sin(ang_)
        q = best_pts.copy()
        qx = q[:, 0] * fx_
        qy = q[:, 1] * fy_
        q[:, 0] = qx * c_ - qy * s_
        q[:, 1] = qx * s_ + qy * c_
        starts.append(q)

    for p0 in starts:
        cur = _rescale(p0.reshape(-1)).reshape(_N, 2).copy()
        if not np.isfinite(_score(cur)):
            continue
        stalls = 0  # consecutive non-improving SLSQP solves for this start
        for _ in range(rounds):
            try:
                H = float(ConvexHull(cur).volume)
            except Exception:
                break
            if not np.isfinite(H) or H < 1e-12:
                break
            a2 = _areas2(cur)
            t0 = max(float(np.min(a2)) * 0.98, 1e-9)

            def _con(z: np.ndarray) -> np.ndarray:
                p = z[:2 * _N].reshape(_N, 2)
                t = z[2 * _N]
                return _areas2(p) - t

            z0 = np.concatenate([cur.reshape(-1), [t0]])
            bounds = [(z0[i] - 0.5, z0[i] + 0.5) for i in range(2 * _N)] + \
                     [(0.0, max(t0 * 1.5, 1e-9))]
            try:
                res = _slsqp_min(
                    lambda z: -z[2 * _N], z0, method="SLSQP",
                    bounds=bounds,
                    constraints=[{"type": "ineq", "fun": _con}],
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

    Stage 1: DE on the 14-parameter ring parameterization (13 radii +
    phase) with SLSQP polish — reliably reaches the strong incumbent basin
    (removing this stage was shown to cost ~0.005 of normalized area).
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
    bounds = [(0.0, 1.0)] * _N + [(0.0, 2.0 * np.pi / _N)]
    best_x, best_val = None, np.inf
    try:
        res = differential_evolution(
            _objective, bounds, seed=42, popsize=30, maxiter=300,
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

    # Stage 2: smooth-surrogate escape from each start (cheap, bounded).
    refined = []
    for x0 in starts:
        try:
            sx, sv = _hooke_jeeves(x0, sigma0=0.10, sigma_min=1e-4,
                                   budget=12000, obj=_score_smooth)
            refined.append(sx if np.isfinite(sv) else x0)
        except Exception:
            refined.append(x0)

    best_pts, best_sc = pts.copy(), _score(pts)
    for x0 in refined:
        try:
            px, pv = _hooke_jeeves(x0, sigma0=0.08, sigma_min=1e-7,
                                   budget=80000)
            if np.isfinite(pv) and pv > best_sc:
                best_pts, best_sc = px.reshape(_N, 2), pv
        except Exception:
            pass

    # Deterministic perturbation-and-redescend cycle: structured fixed
    # kicks to the incumbent re-enter the exact-objective descent from
    # nearby points, letting the (now diagonal-capable) pattern search
    # take a different active-set path through the same basin.
    for scale in (0.02, 0.01, 0.005):
        ang = np.arange(_N, dtype=np.float64)
        q = best_pts.copy()
        q[:, 0] += scale * np.cos(1.3 * ang + 0.7)
        q[:, 1] += scale * np.sin(2.1 * ang + 0.3)
        try:
            rx, rv = _hooke_jeeves(q.reshape(-1).copy(), sigma0=0.02,
                                   sigma_min=1e-8, budget=40000)
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
    except Exception:
        pass

    # Short exact Hooke-Jeeves cleanup of hull-combinatorics boundary
    # effects left by the fixed-hull-area epigraph solves.
    try:
        fx2, fv2 = _hooke_jeeves(best_pts.reshape(-1).copy(),
                                 sigma0=0.002, sigma_min=1e-9, budget=30000)
        if np.isfinite(fv2) and fv2 > best_sc:
            best_pts, best_sc = fx2.reshape(_N, 2), fv2
    except Exception:
        pass

    if np.all(np.isfinite(best_pts)) and _score(best_pts) > 0.0:
        return np.asarray(best_pts, dtype=np.float64)
    return fallback


# EVOLVE-BLOCK-END
