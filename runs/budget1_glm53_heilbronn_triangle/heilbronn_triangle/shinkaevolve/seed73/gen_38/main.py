# EVOLVE-BLOCK-START
import numpy as np
import itertools

# Precompute all 165 triplet index combinations once at module level.
_TRIPLETS = np.array(list(itertools.combinations(range(11), 3)), dtype=int)
_TRI_AREA = np.sqrt(3.0) / 4.0


def _clip_to_triangle(pts: np.ndarray) -> np.ndarray:
    """Project points into the equilateral triangle with vertices
    (0,0), (1,0), (0.5, sqrt(3)/2) using barycentric coordinates."""
    h = np.sqrt(3.0) / 2.0
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, h])
    # barycentric coordinates
    v0 = B - A
    v1 = C - A
    v2 = pts - A
    d00 = v0 @ v0
    d01 = v0 @ v1
    d11 = v1 @ v1
    d20 = v2 @ v0
    d21 = v2 @ v1
    denom = d00 * d11 - d01 * d01
    b = (d11 * d20 - d01 * d21) / denom  # weight of B
    c = (d00 * d21 - d01 * d20) / denom  # weight of C
    a = 1.0 - b - c
    w = np.stack([a, b, c], axis=-1)
    w = np.clip(w, 0.0, 1.0)
    w /= w.sum(axis=-1, keepdims=True)
    out = w[:, 0:1] * A + w[:, 1:2] * B + w[:, 2:3] * C
    return out


def _all_areas(pts: np.ndarray) -> np.ndarray:
    """Vectorized (doubled) areas of all 165 triplets, precomputed indices."""
    p = pts[_TRIPLETS]  # (165, 3, 2)
    cross = (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - \
            (p[:, 1, 1] - p[:, 0, 1]) * (p[:, 2, 0] - p[:, 0, 0])
    return np.abs(cross)


def _min_area(pts: np.ndarray) -> float:
    """Doubled area of the smallest triangle among all triplets."""
    return _all_areas(pts).min()


def _softmin(pts: np.ndarray, k: float = 60.0) -> float:
    """Smoothed (soft) minimum over the smallest ~15% of doubled areas.
    log( sum_{sel} exp(-k*a) ) / (-k): a smooth lower-bound surrogate
    that gradients the whole near-minimal set of triangles upward."""
    a = _all_areas(pts)
    m = a.min()
    sel = a <= m + 0.05 * _TRI_AREA * 2.0
    return m - np.log(np.exp(-k * (a[sel] - m)).sum()) / k


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct an arrangement of 11 points on or inside an equilateral triangle
    maximizing the smallest triangle area (Heilbronn problem, n=11).

    Returns:
        points: np.ndarray of shape (11,2) with the x,y coordinates of the points.
    """
    n = 11
    h = np.sqrt(3.0) / 2.0

    # Heuristic starting layout: vertices + edge points + interior spread
    pts = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, h],
        [1.0 / 3.0, 0.0],
        [2.0 / 3.0, 0.0],
        [1.0 / 6.0, h / 3.0],
        [5.0 / 6.0, h / 3.0],
        [1.0 / 3.0, 2.0 * h / 3.0],
        [2.0 / 3.0, 2.0 * h / 3.0],
        [0.5, h / 3.0],
        [0.5, 0.0],
    ])

    rng = np.random.default_rng(12345)
    best = _clip_to_triangle(pts)
    best_val = _min_area(best)

    cur = best.copy()
    cur_val = best_val

    # ---- Phase 1: soft-min coarse exploration (smoothed landscape) ----
    soft_cur = _softmin(cur)
    for it in range(15000):
        step = 0.05 * (1.0 - it / 15000) + 1e-3
        trial = cur.copy()
        if rng.random() < 0.20:
            # Coordinated group move: perturb the vertices of the current
            # smallest-area triangle simultaneously. The soft-min objective
            # tolerates temporary hard-min worsenings, so these correlated
            # moves can escape pinning configurations single-point moves
            # cannot.
            a_all = _all_areas(cur)
            t = int(np.argmin(a_all))
            i0, i1, i2 = _TRIPLETS[t]
            verts = cur[[i0, i1, i2]]
            cen = verts.mean(axis=0)
            # shared translation (keeps most other triangle areas ~invariant)
            shared = rng.normal(0.0, 0.6 * step, size=2)
            for vi, v in zip((i0, i1, i2), verts):
                d = v - cen
                nd = np.hypot(d[0], d[1])
                if nd > 1e-12:
                    d = d / nd
                else:
                    d = np.array([0.0, 1.0])
                # outward expansion along centroid direction + noise
                delta = shared + d * step * (0.5 + rng.random()) \
                        + rng.normal(0.0, 0.35 * step, size=2)
                trial[vi] = trial[vi] + delta
        else:
            i = rng.integers(n)
            trial[i] += rng.normal(0.0, step, size=2)
        trial = _clip_to_triangle(trial)
        s_new = _softmin(trial)
        if s_new >= soft_cur or rng.random() < np.exp((s_new - soft_cur) * 40.0):
            cur, soft_cur = trial, s_new
            v = _min_area(cur)
            if v > best_val:
                best, best_val = cur.copy(), v

    # Restart hard phase from the soft-phase result
    cur, cur_val = best.copy(), best_val

    # Deterministic local search with annealing-style shrinking steps.
    # Bias point selection toward vertices of the currently-smallest
    # triangles: these are the binding constraints of the objective.
    step0 = 0.05
    iters = 30000
    for it in range(iters):
        step = step0 * (1.0 - it / iters) + 1e-4
        if rng.random() < 0.5:
            a_all = _all_areas(cur)
            order = np.argsort(a_all)[:8]
            t = order[rng.integers(len(order))]
            i = _TRIPLETS[t][rng.integers(3)]
        else:
            i = rng.integers(n)
        delta = rng.normal(0.0, step, size=2)
        trial = cur.copy()
        trial[i] += delta
        trial = _clip_to_triangle(trial)
        val = _min_area(trial)
        if val >= cur_val - 1e-12:
            cur, cur_val = trial, val
            if val > best_val + 1e-15:
                best, best_val = trial.copy(), val

    # Final fine polishing around the best configuration
    cur, cur_val = best.copy(), best_val
    for it in range(10000):
        step = 0.002 * (1.0 - it / 10000) + 1e-5
        i = rng.integers(n)
        delta = rng.normal(0.0, step, size=2)
        trial = cur.copy()
        trial[i] += delta
        trial = _clip_to_triangle(trial)
        val = _min_area(trial)
        if val >= cur_val - 1e-12:
            cur, cur_val = trial, val
            if val > best_val + 1e-15:
                best, best_val = trial.copy(), val

    # ---- Phase 4: pair-exchange escape from plateaued basins ----
    # Swap a binding vertex (in the smallest triangle) with a "free" point
    # whose incident triangles are all comfortably large, then re-polish.
    def _vertex_binding_scores(pts):
        """Cheap per-vertex score: the smallest doubled area among the
        triangles incident to each point (small = tightly binding)."""
        a = _all_areas(pts)                       # (165,)
        scores = np.full(n, np.inf)
        flat = _TRIPLETS.reshape(-1)              # (495,) vertex indices
        rep = np.repeat(a, 3)                     # (495,) area per slot
        order = np.argsort(flat, kind="stable")
        fs, rs = flat[order], rep[order]
        # min-reduce per vertex
        best_per_v = np.full(n, np.inf)
        np.minimum.at(best_per_v, fs, rs)
        return best_per_v

    def _short_polish(pts, iters=2500, s0=0.01, s1=1e-4, seed=999):
        rng2 = np.random.default_rng(seed)
        cur = pts.copy()
        cur_val = _min_area(cur)
        out, out_val = cur.copy(), cur_val
        for it in range(iters):
            step = s0 * (s1 / s0) ** (it / iters)
            a_all = _all_areas(cur)
            order = np.argsort(a_all)[:6]
            t = order[rng2.integers(len(order))]
            i = _TRIPLETS[t][rng2.integers(3)]
            trial = cur.copy()
            trial[i] += rng2.normal(0.0, step, size=2)
            trial = _clip_to_triangle(trial)
            val = _min_area(trial)
            if val >= cur_val - 1e-12:
                cur, cur_val = trial, val
                if val > out_val + 1e-15:
                    out, out_val = trial.copy(), val
        return out, out_val

    try:
        rng_swap = np.random.default_rng(777)
        for trial_no in range(12):
            if best_val >= 0.040:
                break
            vs = _vertex_binding_scores(best)
            # binding point: vertex of a smallest triangle
            a_all = _all_areas(best)
            t = int(np.argmin(a_all))
            binding_pts = _TRIPLETS[t]
            bp = binding_pts[rng_swap.integers(3)]
            # free point: whose binding score is largest (least constrained)
            free_order = np.argsort(vs)[::-1]
            fp = int(free_order[rng_swap.integers(min(4, n))])
            if fp == bp:
                continue
            swapped = best.copy()
            swapped[bp], swapped[fp] = swapped[fp].copy(), swapped[bp].copy()
            cand, cand_val = _short_polish(swapped, seed=1000 + trial_no)
            if cand_val > best_val + 1e-15:
                best, best_val = cand, cand_val
    except Exception:
        pass

    # Guarantee validity of output even if optimization failed somehow
    if best is None or not np.all(np.isfinite(best)):
        best = _clip_to_triangle(np.zeros((n, 2)) + np.array([0.4, 0.3]))

    return best


# EVOLVE-BLOCK-END