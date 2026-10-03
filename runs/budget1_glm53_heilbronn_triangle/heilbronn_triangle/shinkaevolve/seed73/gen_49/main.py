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

    # --- Multi-start: several structurally distinct seeds (deterministic) ---
    def _lattice_seed():
        q = []
        rows = [4, 3, 2, 1, 1]
        for r, cnt in enumerate(rows):
            y = h * (r + 0.5) / len(rows)
            xs = np.linspace(0.0, 1.0, cnt + 2)[1:-1]
            halfw = 0.5 * (1.0 - y / h)
            for x in xs:
                q.append((0.5 + (x - 0.5) * (2 * halfw) * 0.98, y))
        return np.array(q)

    def _rings_seed():
        out = [(0.0, 0.0), (1.0, 0.0), (0.5, h),
               (0.5, 0.0), (0.25, h / 2.0), (0.75, h / 2.0)]
        cx, cy = 0.5, h / 3.0
        for s in (0.5, 0.25):
            for (vx, vy) in [(0.0, 0.0), (1.0, 0.0), (0.5, h)]:
                out.append((cx + s * (vx - cx), cy + s * (vy - cy)))
        out.append((cx, cy))
        return np.array(out[:11])

    def _rand_start(seed):
        r = np.random.default_rng(seed)
        u, v = r.random(11), r.random(11)
        su = np.sqrt(u)
        w1, w2 = su * (1 - v), su * v
        return np.stack([w1 + 0.5 * w2, h * w2], axis=1)

    starts = [_clip_to_triangle(pts), _clip_to_triangle(_lattice_seed()),
              _clip_to_triangle(_rings_seed()), _rand_start(777),
              _rand_start(4242), _rand_start(2024)]

    best = None
    best_val = -1.0

    for si, start in enumerate(starts):
        rng = np.random.default_rng(12345 + si)  # per-start deterministic stream
        cur = _clip_to_triangle(start.copy())
        cur_val = _min_area(cur)
        if cur_val > best_val:
            best, best_val = cur.copy(), cur_val

        # ---- Phase 1: soft-min coarse exploration (smoothed landscape) ----
        soft_cur = _softmin(cur)
        for it in range(4000):
            step = 0.05 * (1.0 - it / 4000) + 1e-3
            trial = cur.copy()
            if rng.random() < 0.20:
                # Coordinated group move: perturb the vertices of the current
                # smallest-area triangle simultaneously.
                a_all = _all_areas(cur)
                t = int(np.argmin(a_all))
                i0, i1, i2 = _TRIPLETS[t]
                verts = cur[[i0, i1, i2]]
                cen = verts.mean(axis=0)
                shared = rng.normal(0.0, 0.6 * step, size=2)
                for vi, v in zip((i0, i1, i2), verts):
                    d = v - cen
                    nd = np.hypot(d[0], d[1])
                    if nd > 1e-12:
                        d = d / nd
                    else:
                        d = np.array([0.0, 1.0])
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

        # ---- Phase 2: hard-min annealed local search ----
        cur, cur_val = best.copy(), best_val
        step0 = 0.05
        iters = 8000
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

        # ---- Phase 3: fine polish ----
        cur, cur_val = best.copy(), best_val
        for it in range(3000):
            step = 0.002 * (1.0 - it / 3000) + 1e-5
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

    # ---- Global final polish on the winner across all starts ----
    rng = np.random.default_rng(999)
    cur, cur_val = best.copy(), best_val
    for it in range(6000):
        step = 0.001 * (1.0 - it / 6000) + 1e-5
        if rng.random() < 0.6:
            a_all = _all_areas(cur)
            order = np.argsort(a_all)[:5]
            t = order[rng.integers(len(order))]
            i = _TRIPLETS[t][rng.integers(3)]
        else:
            i = rng.integers(n)
        trial = cur.copy()
        trial[i] += rng.normal(0.0, step, size=2)
        trial = _clip_to_triangle(trial)
        val = _min_area(trial)
        if val >= cur_val - 1e-12:
            cur, cur_val = trial, val
            if val > best_val + 1e-15:
                best, best_val = trial.copy(), val

    # Guarantee validity of output even if optimization failed somehow
    if best is None or not np.all(np.isfinite(best)):
        best = _clip_to_triangle(np.zeros((n, 2)) + np.array([0.4, 0.3]))

    return best


# EVOLVE-BLOCK-END