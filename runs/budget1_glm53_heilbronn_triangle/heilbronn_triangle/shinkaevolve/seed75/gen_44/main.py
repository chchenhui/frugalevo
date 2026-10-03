# EVOLVE-BLOCK-START
import numpy as np

try:
    from scipy.optimize import minimize as _scipy_minimize
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct an arrangement of n=11 points on or inside the equilateral triangle
    with vertices (0,0), (1,0), (0.5, sqrt(3)/2) maximizing the minimum triangle
    area over all point triplets.

    Novel approach ("active-set soft-min L-BFGS"):
      * Points are parametrized by barycentric weights (n x 3), projected onto
        the probability simplex (clip + renormalize) at every evaluation, so
        every candidate is feasible by construction without softmax distortion.
      * The objective is a temperature-annealed soft-min restricted to the
        bottom-k triangle areas ("active set"). This focuses the optimizer on
        the binding constraints only, which is crucial because with 165
        triples the naive soft-min drowns the signal of the few worst ones.
      * L-BFGS-B (numerical gradients, cheap in 33 dims) with progressively
        lower temperatures provides a smooth-to-sharp optimization schedule.
      * Deterministic multi-start: a hand-designed symmetric seed, reflected
        perturbations of it, and several seeded random starts.
      * Strictly-positive min-area guard with a rescue restart before any
        fallback, preventing silent degenerate outputs.

    Returns:
        np.ndarray of shape (11, 2) with the point coordinates.
    """
    n = 11
    sqrt3 = np.sqrt(3.0)
    V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, sqrt3 / 2.0]])
    area_T = sqrt3 / 4.0

    from itertools import combinations
    triples = np.array(list(combinations(range(n), 3)))   # (C,3)
    C = triples.shape[0]

    # ---------- geometry helpers ----------
    def all_areas(pts):
        p = pts[triples]                                   # (C,3,2)
        cross = ((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
                 - (p[:, 1, 1] - p[:, 0, 1]) * (p[:, 2, 0] - p[:, 0, 0]))
        return 0.5 * np.abs(cross)

    def min_area(pts):
        return float(np.min(all_areas(pts)))

    def unpack(W):
        """Barycentric weights -> points, with simplex projection."""
        W = np.asarray(W, dtype=float).reshape(n, 3)
        W = np.clip(W, 0.0, None)
        s = W.sum(axis=1, keepdims=True)
        s = np.where(s < 1e-12, 1.0, s)
        W = W / s
        return W @ V

    # ---------- active-set soft-min objective ----------
    K_ACTIVE = 40          # number of smallest triangles driving the objective

    def active_softmin(theta, temp):
        pts = unpack(theta)
        a = all_areas(pts)
        # bottom-k areas
        part = np.partition(a, K_ACTIVE - 1)[:K_ACTIVE]
        # soft-min of the active set: -temp*log(sum(exp(-a/temp)))
        m = part.min()
        e = np.exp(-(part - m) / temp)
        return temp * np.log(e.sum()) - m

    # ---------- seeds ----------
    rng = np.random.default_rng(20240517)

    base = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.5, sqrt3 / 2.0],
        [0.5, 0.0], [0.25, sqrt3 / 4.0], [0.75, sqrt3 / 4.0],
        [0.25, sqrt3 / 12.0], [0.75, sqrt3 / 12.0],
        [0.5, sqrt3 / 6.0], [0.125, sqrt3 / 8.0], [0.875, sqrt3 / 8.0],
    ])
    base_val = min_area(base)
    if base_val <= 0.0:
        # safety: nudge interior points so the fallback is non-degenerate
        for i in range(3, n):
            base[i] = base[i] + 0.02 * rng.normal(size=2)
        base[0], base[1], base[2] = V
        base_val = min_area(base)

    # barycentric coordinates of the seed
    x, y = base[:, 0], base[:, 1]
    wC = 2.0 * y / sqrt3
    wB = x - 0.5 * wC
    wA = 1.0 - wB - wC
    base_W = np.stack([wA, wB, wC], axis=1)

    starts = [base_W.ravel()]
    # deterministic perturbations + mirror-paired perturbations
    for _ in range(4):
        W = np.clip(base_W + 0.05 * rng.normal(size=(n, 3)), 1e-4, None)
        starts.append(W.ravel())
    for _ in range(3):
        W = np.clip(base_W + 0.05 * rng.normal(size=(n, 3)), 1e-4, None)
        # enforce reflection pairs across x = 0.5 (swap barycentric A,B)
        for i in range(0, n, 2):
            if i + 1 < n:
                W[i + 1] = W[i][[1, 0, 2]]
        starts.append(W.ravel())
    # uniform-random starts
    for _ in range(2):
        W = rng.random((n, 3)) + 0.05
        W /= W.sum(axis=1, keepdims=True)
        starts.append(W.ravel())
    # fully-symmetric start: exact mirror pairs + on-axis center point
    sym = base_W.copy()
    for i in range(0, n - 1, 2):
        sym[i + 1] = sym[i][[1, 0, 2]]
    starts.append(sym.ravel())
    # perturbed versions of the fully-symmetric start (mirror pairs preserved)
    for _ in range(4):
        W = np.clip(sym + 0.04 * rng.normal(size=(n, 3)), 1e-4, None)
        for i in range(0, n - 1, 2):
            W[i + 1] = W[i][[1, 0, 2]]
        starts.append(W.ravel())

    # ---------- optimization ----------
    temps = (2e-2 * area_T, 4e-3 * area_T, 8e-4 * area_T, 2e-4 * area_T)

    def run_start(theta0, iters=(150, 100, 80, 60)):
        """Coarse-to-fine Powell annealing: short budgets at coarse temperatures
        (precision irrelevant there), finer polish at low temperature."""
        theta = np.asarray(theta0, dtype=float).ravel()
        if not _HAVE_SCIPY:
            return unpack(theta)
        for temp, it in zip(temps, iters):
            try:
                res = _scipy_minimize(
                    active_softmin, theta, args=(temp,),
                    method="Powell",
                    options={"maxiter": it, "xtol": 1e-7, "ftol": 1e-10},
                )
                if np.all(np.isfinite(res.x)):
                    theta = res.x
            except Exception:
                pass
        return unpack(theta)

    def polish(theta0, it=250):
        """Low-temperature polish on the incumbent best solution."""
        theta = np.asarray(theta0, dtype=float).ravel()
        if not _HAVE_SCIPY:
            return unpack(theta)
        try:
            res = _scipy_minimize(
                active_softmin, theta, args=(1e-4 * area_T,),
                method="Powell",
                options={"maxiter": it, "xtol": 1e-9, "ftol": 1e-12},
            )
            if np.all(np.isfinite(res.x)):
                theta = res.x
        except Exception:
            pass
        return unpack(theta)

    best_pts = base
    best_val = base_val

    for s in starts:
        try:
            cand = run_start(s)
            if not np.all(np.isfinite(cand)):
                continue
            v = min_area(cand)
            if v > best_val:
                best_val = v
                best_pts = cand
        except Exception:
            continue

    # ---------- final polish of the incumbent ----------
    if _HAVE_SCIPY and best_val > 0.0:
        try:
            # back out barycentric weights of the incumbent for polishing
            x, y = best_pts[:, 0], best_pts[:, 1]
            wC = 2.0 * y / sqrt3
            wB = x - 0.5 * wC
            wA = 1.0 - wB - wC
            theta_best = np.stack([wA, wB, wC], axis=1).ravel()
            cand = polish(theta_best)
            if np.all(np.isfinite(cand)):
                v = min_area(cand)
                if v > best_val:
                    best_val = v
                    best_pts = cand
        except Exception:
            pass

    # ---------- strictly-positive guard with rescue restart ----------
    best_pts = np.asarray(best_pts, dtype=float)
    if (best_pts.shape != (n, 2) or not np.all(np.isfinite(best_pts))
            or min_area(best_pts) <= 0.0):
        ok = False
        for _ in range(3):
            try:
                W = rng.random((n, 3)) + 0.05
                W /= W.sum(axis=1, keepdims=True)
                cand = run_start(W.ravel())
                if (np.all(np.isfinite(cand)) and min_area(cand) > 0.0):
                    best_pts = cand
                    ok = True
                    break
            except Exception:
                continue
        if not ok:
            best_pts = base  # non-degenerate by construction

    return best_pts


# EVOLVE-BLOCK-END