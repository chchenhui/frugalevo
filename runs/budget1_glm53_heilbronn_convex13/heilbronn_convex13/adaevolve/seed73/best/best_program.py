# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations
from scipy.optimize import minimize, linprog

# Precomputed vertex-index triples for all C(13,3)=286 triangles.
_IDX = np.array(list(combinations(range(13), 3)))


def _softmin_val_grad(flat, tau):
    """Value and analytic gradient of smooth soft-min of triangle areas.

    V = m - tau*log(sum(exp(-(A_i - m)/tau))) with m = max area (max-shift
    trick for numerical stability). dV/dA_i = -w_i/Z, and each triangle
    area's derivative w.r.t. its vertices comes from the cross product.
    Returns (-V, -grad) so scipy minimize (which minimizes) maximizes V.
    """
    pts = flat.reshape(13, 2)
    i, j, k = _IDX[:, 0], _IDX[:, 1], _IDX[:, 2]
    a, b, c = pts[i], pts[j], pts[k]
    s = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    areas = 0.5 * np.abs(s)
    m = areas.max()
    w = np.exp(-(areas - m) / tau)
    Z = w.sum()
    val = m - tau * np.log(Z)
    # dV/ds_i = (dV/dA_i) * sign(s_i)/2
    g_s = -(w / Z) * (0.5 * np.sign(s))
    grad = np.zeros((13, 2))
    # ds/da = (c.y - b.y, b.x - c.x); ds/db = (a.y - c.y, c.x - a.x);
    # ds/dc = (b.y - a.y, a.x - b.x)
    np.add.at(grad, i, np.stack([c[:, 1] - b[:, 1], b[:, 0] - c[:, 0]], axis=1) * g_s[:, None])
    np.add.at(grad, j, np.stack([a[:, 1] - c[:, 1], c[:, 0] - a[:, 0]], axis=1) * g_s[:, None])
    np.add.at(grad, k, np.stack([b[:, 1] - a[:, 1], a[:, 0] - b[:, 0]], axis=1) * g_s[:, None])
    return -val, -grad.ravel()


def _areas_and_idx(points):
    """Areas of all C(13,3)=286 triangles plus their vertex-index triples."""
    idx = np.array(list(combinations(range(len(points)), 3)))
    a = points[idx[:, 0]]
    b = points[idx[:, 1]]
    c = points[idx[:, 2]]
    areas = 0.5 * np.abs(
        (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
        - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    )
    return areas, idx


def _soft_objective(points, m=10, alpha=1.0):
    """Weighted sum of the m smallest triangle areas (smooth surrogate for min)."""
    areas, idx = _areas_and_idx(points)
    order = np.argsort(areas)
    w = np.exp(-alpha * np.arange(m))
    w = w / w.sum()
    return float(np.sum(w * areas[order[:m]]))


def _min_area(points):
    return float(_areas_and_idx(points)[0].min())


def _slp(pts, iters=60, delta0=0.05):
    """Sequential Linear Programming (SLP) on the exact minimax epigraph.

    Formulation per iteration: variables z = [dx (26), t]; maximize t s.t.
    for every triangle i:  A_i(x0) + grad_i . dx >= t  (first-order
    linearization of the area, with orientation sign frozen at x0), plus
    trust-region bounds -delta <= dx <= delta. Solved exactly with
    scipy.optimize.linprog (HiGHS). Nonconvexity of the true problem is
    handled by the trust region: delta shrinks when the linearized
    prediction fails to improve the TRUE min area (acceptance criterion),
    and grows on success. This directly targets the bottleneck triangles,
    avoiding the surrogate gap of soft-min gradient methods.
    """
    pts = np.asarray(pts, dtype=float).copy()
    i, j, k = _IDX[:, 0], _IDX[:, 1], _IDX[:, 2]
    best = pts.copy()
    best_m = _min_area(pts)
    delta = delta0
    for _ in range(iters):
        a, b, c = pts[i], pts[j], pts[k]
        s = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        sg = np.where(s >= 0.0, 1.0, -1.0)
        areas = 0.5 * np.abs(s)
        # Constraint rows: grad(A_i) wrt 26 coords, then -1 for t.
        # dA/dx = 0.5*sg*ds/dx with ds derivatives as in the cross product.
        A_ub = np.zeros((len(_IDX), 27))
        A_ub[np.arange(len(_IDX)), 2 * i] = 0.5 * sg * (c[:, 1] - b[:, 1])
        A_ub[np.arange(len(_IDX)), 2 * i + 1] = 0.5 * sg * (b[:, 0] - c[:, 0])
        A_ub[np.arange(len(_IDX)), 2 * j] = 0.5 * sg * (a[:, 1] - c[:, 1])
        A_ub[np.arange(len(_IDX)), 2 * j + 1] = 0.5 * sg * (c[:, 0] - a[:, 0])
        A_ub[np.arange(len(_IDX)), 2 * k] = 0.5 * sg * (b[:, 1] - a[:, 1])
        A_ub[np.arange(len(_IDX)), 2 * k + 1] = 0.5 * sg * (a[:, 0] - b[:, 0])
        A_ub[:, 26] = -1.0
        b_ub = -areas  # grad.dx - t >= -A_i  <=>  A_i + grad.dx - t >= 0
        c_obj = np.zeros(27)
        c_obj[26] = -1.0  # maximize t
        # Trust region on dx; t free within [0, 1].
        bounds = [(-delta, delta)] * 26 + [(0.0, 1.0)]
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                          method="highs")
        except Exception:
            break
        if not res.success or not np.all(np.isfinite(res.x)):
            delta *= 0.5
            if delta < 1e-7:
                break
            continue
        dx = res.x[:26]
        cand = np.clip((pts.ravel() + dx).reshape(13, 2), 0.0, 1.0)
        m = _min_area(cand)
        if m > best_m + 1e-14:
            pts = cand
            best, best_m = cand.copy(), m
            delta = min(delta * 1.6, 0.08)
        else:
            delta *= 0.5
            if delta < 1e-7:
                break
    return best


def _polish(pts, steps=(0.002, 0.0008, 0.0003, 0.0001), max_it=40):
    """Greedy hill-climb on the TRUE minimum triangle area.

    Moves only vertices participating in the ~6 smallest triangles, trying
    16 directions per move at each step size of a coarse-to-fine ladder.
    Recovers the gap between the smooth soft-min surrogate optimum and the
    true non-smooth optimum at the end of each gradient continuation.
    """
    pts = pts.copy()
    for step in steps:
        improved = True
        it = 0
        while improved and it < max_it:
            improved = False
            it += 1
            areas, idx = _areas_and_idx(pts)
            order = np.argsort(areas)
            cur_min = areas[order[0]]
            cand = set()
            for t in idx[order[:6]]:
                cand.update(t.tolist())
            for p in cand:
                base = pts[p].copy()
                blv, blp = cur_min, base
                for d in range(16):
                    a2 = 2 * np.pi * d / 16
                    cand_pos = np.clip(base + step * np.array([np.cos(a2), np.sin(a2)]), 0, 1)
                    pts[p] = cand_pos
                    v = _min_area(pts)
                    if v > blv + 1e-15:
                        blv, blp = v, cand_pos
                pts[p] = blp
                if blv > cur_min + 1e-15:
                    cur_min = blv
                    improved = True
    return pts


def _hull_area(pts):
    """Area of the convex hull via the shoelace formula on hull vertices
    (deterministic, no qhull dependency)."""
    p = np.asarray(pts, dtype=float)
    if len(p) < 3:
        return 1e-12
    try:
        from scipy.spatial import ConvexHull
        return max(float(ConvexHull(p).volume), 1e-12)
    except Exception:
        # shoelace fallback
        best = 0.0
        for i in range(len(p)):
            for j in range(i + 1, len(p)):
                for k in range(j + 1, len(p)):
                    a = 0.5 * abs((p[j,0]-p[i,0])*(p[k,1]-p[i,1]) -
                                  (p[j,1]-p[i,1])*(p[k,0]-p[i,0]))
                    best = max(best, a)
        return max(best, 1e-12)


def _score(pts):
    """TRUE primary metric: smallest triangle area / convex hull area."""
    return _min_area(pts) / _hull_area(pts)


def _norm_polish(pts, steps=(0.02, 0.008, 0.003, 0.001, 0.0004, 0.0001),
                 max_it=30):
    """Greedy hill-climb directly on the NORMALIZED score.

    The pipeline above maximizes the raw min triangle area inside the unit
    square, but the metric is min_area / hull_area. Shrinking the hull
    (pulling extreme points inward) can raise the ratio even when the raw
    min area decreases slightly. Here every point (especially hull
    vertices) is tried in 16 directions at each step size, accepting a
    move only if the true normalized score strictly improves. This closes
    the gap between the raw-area optimum and the normalized-score optimum.
    """
    pts = np.asarray(pts, dtype=float).copy()
    best_s = _score(pts)
    for step in steps:
        improved = True
        it = 0
        while improved and it < max_it:
            improved = False
            it += 1
            for p in range(13):
                base = pts[p].copy()
                blv, blp = best_s, base
                for d in range(16):
                    a2 = 2 * np.pi * d / 16
                    cand = np.clip(base + step * np.array([np.cos(a2),
                                                           np.sin(a2)]), 0, 1)
                    pts[p] = cand
                    v = _score(pts)
                    if v > blv + 1e-15:
                        blv, blp = v, cand
                pts[p] = blp
                if blv > best_s + 1e-15:
                    best_s = blv
                    improved = True
    return pts


def heilbronn_convex13() -> np.ndarray:
    """Heilbronn triangle problem for n = 13 via gradient-based smooth
    optimization.

    Approach: maximize a smooth soft-min surrogate of the 286 triangle
    areas, V = m - tau*log(sum(exp(-(A_i - m)/tau)), using L-BFGS-B over
    the 26 point coordinates (bounds keep points in the unit square). The
    analytic gradient is vectorized over all triangles. A continuation
    homotopy shrinks tau from a broad average (0.05) down to ~1e-4 so the
    surrogate progressively concentrates on the bottleneck triangles,
    escaping the plateaus that stall single-vertex hill climbing, since
    gradients jointly move every point influencing the smallest triangles.
    The true min area is checked after every tau stage and the best
    configuration across all starts is kept. Deterministic via fixed seeds.

    Returns:
        np.ndarray of shape (13, 2) inside the unit square.
    """
    rng = np.random.default_rng(seed=12345)
    bounds = [(0.0, 1.0)] * 26

    def clip(pts):
        return np.clip(pts, 0.0, 1.0)

    starts = []
    # jittered 4x4 grid
    side = 4
    grid = [(x / (side - 1), y / (side - 1)) for x in range(side) for y in range(side)]
    starts.append(clip(np.array(grid[:13]) + rng.normal(0, 0.03, (13, 2))))
    # ring + center
    ang = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    ring = np.stack([0.5 + 0.45 * np.cos(ang), 0.5 + 0.45 * np.sin(ang)], axis=1)
    starts.append(clip(np.vstack([ring, [0.5, 0.5]]) + rng.normal(0, 0.02, (13, 2))))
    # hexagonal rings: 6 outer + 6 mid + center
    hexpts = []
    for r, m in ((0.47, 6), (0.24, 6)):
        a2 = np.linspace(0, 2 * np.pi, m, endpoint=False) + (np.pi / m if r < 0.4 else 0)
        for t in a2:
            hexpts.append((0.5 + r * np.cos(t), 0.5 + r * np.sin(t)))
    hexpts.append((0.5, 0.5))
    starts.append(clip(np.array(hexpts) + rng.normal(0, 0.02, (13, 2))))
    # boundary-heavy: 12 points on square perimeter + center
    per = []
    for i in range(4):
        for t in np.linspace(0.02, 0.98, 3):
            x, y = [(t, 0.0), (1.0, t), (t, 1.0), (0.0, t)][i]
            per.append((x, y))
    per.append((0.5, 0.5))
    starts.append(clip(np.array(per) + rng.normal(0, 0.015, (13, 2))))
    # random starts
    for _ in range(4):
        starts.append(rng.random((13, 2)))
    # Affine-invariance exploitation: the score (min triangle area / hull
    # area) is invariant under nonsingular affine maps, so anisotropic
    # stretched/sheared variants of the structured starts explore genuinely
    # different basins the isotropic starts cannot reach.
    base_starts = [s.copy() for s in starts[:5]]
    for s in base_starts:
        cent = s.mean(axis=0)
        for M in (np.array([[1.4, 0.3], [0.0, 0.7]]),
                  np.array([[0.7, -0.25], [0.35, 1.35]])):
            starts.append(np.clip((s - cent) @ M.T + cent, 0.0, 1.0))

    best_pts, best_min = None, -1.0
    # tau continuation: broad average -> concentrate on smallest triangles
    taus = (0.05, 0.02, 0.008, 0.003, 0.001, 3e-4, 1e-4)

    for pts in starts:
        flat = clip(pts).ravel()
        for tau in taus:
            try:
                res = minimize(
                    _softmin_val_grad, flat, args=(tau,), jac=True,
                    method="L-BFGS-B", bounds=bounds,
                    options={"maxiter": 300, "maxfun": 6000, "ftol": 1e-12, "gtol": 1e-10},
                )
                cand = np.asarray(res.x, dtype=float).reshape(13, 2)
            except Exception:
                cand = flat.reshape(13, 2)
            if np.all(np.isfinite(cand)):
                flat = clip(cand).ravel()
        # Polish THIS start's continuation result on the true min area
        # before comparing: the surrogate at the final tau leaves different
        # gaps for different basins, and the true optimum of a weaker basin
        # can surpass a stronger basin's unpolished surrogate optimum.
        cand = _polish(flat.reshape(13, 2))
        m = _min_area(cand)
        if m > best_min:
            best_min = m
            best_pts = cand.copy()

    # Final extra-fine polish on the overall best configuration.
    pts = _polish(best_pts, steps=(0.0008, 0.0003, 0.0001, 0.00005, 0.00002))
    if _min_area(pts) > best_min:
        best_pts = pts

    # SLP (linprog/HiGHS) on the exact minimax epigraph, warm-started from
    # the polished best; alternated with fine polish to re-linearize.
    for _ in range(3):
        cand = _slp(best_pts)
        cand = _polish(cand, steps=(0.0008, 0.0003, 0.0001, 0.00005), max_it=25)
        if _min_area(cand) > best_min + 1e-15:
            best_min = _min_area(cand)
            best_pts = cand.copy()
        else:
            break

    # Final refinement directly on the TRUE normalized metric
    # (min triangle area / hull area). The SLP stage above maximizes the
    # raw min area inside the unit square; pulling hull vertices inward
    # shrinks the denominator and can raise the normalized score beyond
    # what any raw-area optimization reaches.
    try:
        cand = _norm_polish(best_pts)
        if _score(cand) > _score(best_pts):
            best_pts = cand
        # a second pass after hull shrinkage re-balances interior points
        cand2 = _polish(best_pts, steps=(0.001, 0.0003, 0.0001), max_it=20)
        cand2 = _norm_polish(cand2, steps=(0.002, 0.0005, 0.0001), max_it=15)
        if _score(cand2) > _score(best_pts):
            best_pts = cand2
    except Exception:
        pass

    best_pts = np.asarray(best_pts, dtype=float)
    if not np.all(np.isfinite(best_pts)):
        best_pts = np.random.default_rng(0).random((13, 2))
    return best_pts


# EVOLVE-BLOCK-END
