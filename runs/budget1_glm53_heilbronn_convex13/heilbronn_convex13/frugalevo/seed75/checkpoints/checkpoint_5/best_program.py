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


def _hooke_jeeves(x: np.ndarray, sigma0: float, sigma_min: float,
                  budget: int) -> tuple:
    """Hooke-Jeeves pattern search on 26 free coordinates, exact objective.

    Coordinate sweeps accept strictly improving +-sigma moves (with hull-area
    renormalization); after each successful sweep an extrapolated pattern move
    with doubling line search accelerates progress. Sigma halves only when a
    full sweep + pattern phase accepts nothing. Returns (best_x, best_val).
    """
    x = _rescale(np.asarray(x, dtype=np.float64).copy())
    best_val = _score(x.reshape(_N, 2))
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
                    v = _score(trial.reshape(_N, 2))
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
                    v = _score(trial.reshape(_N, 2))
                    if v > best_val:
                        x, best_val = trial, v
                        step *= 2.0
                    else:
                        break
        if not improved_any:
            sigma *= 0.5
    return x, best_val


def heilbronn_convex13() -> np.ndarray:
    """
    Global stage + strengthened multi-start pattern refinement, n=13.

    Stage 1: DE on the 14-parameter ring parameterization (13 radii +
    phase) with SLSQP polish — reliably reaches the strong incumbent basin.
    Stage 2: deterministic multi-start Hooke-Jeeves on the full 26 free
    coordinates, exact evaluator objective (min triangle area / hull area),
    hull-area renormalization, accept-only-if-better moves. Starts: the
    lifted DE incumbent plus three deterministic perturbations (anisotropic
    scalings breaking radial symmetry). Each start has a hard evaluation
    budget; the best valid incumbent across starts is refined again at
    fine sigma. Total ~250k cheap objective evaluations (~30 s), well
    within the timeout. The best valid incumbent is always returned; the
    regular 13-gon is the safe fallback.
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

    # Multi-start Hooke-Jeeves: DE incumbent + 3 deterministic perturbations.
    starts = [pts.reshape(-1).copy()]
    for fx_, fy_ in ((1.06, 0.94), (0.92, 1.08), (1.03, 1.0)):
        q = pts.copy()
        q[:, 0] *= fx_
        q[:, 1] *= fy_
        starts.append(q.reshape(-1).copy())

    best_pts, best_sc = pts.copy(), _score(pts)
    for x0 in starts:
        try:
            px, pv = _hooke_jeeves(x0, sigma0=0.08, sigma_min=1e-6,
                                   budget=55000)
            if np.isfinite(pv) and pv > best_sc:
                best_pts, best_sc = px.reshape(_N, 2), pv
        except Exception:
            pass

    # Fine polish from the overall best incumbent.
    try:
        fx2, fv2 = _hooke_jeeves(best_pts.reshape(-1).copy(), sigma0=0.01,
                                 sigma_min=1e-7, budget=30000)
        if np.isfinite(fv2) and fv2 > best_sc:
            best_pts = fx2.reshape(_N, 2)
    except Exception:
        pass

    if np.all(np.isfinite(best_pts)) and _score(best_pts) > 0.0:
        return np.asarray(best_pts, dtype=np.float64)
    return fallback


# EVOLVE-BLOCK-END
