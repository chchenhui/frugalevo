# EVOLVE-BLOCK-START
import numpy as np
import itertools


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
    """Vectorized doubled areas of all triplets."""
    idx = np.array(list(itertools.combinations(range(len(pts)), 3)))
    p = pts[idx]  # (C(n,3), 3, 2)
    cross = (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - \
            (p[:, 1, 1] - p[:, 0, 1]) * (p[:, 2, 0] - p[:, 0, 0])
    return np.abs(cross)


def _min_area(pts: np.ndarray) -> float:
    """Doubled area of the smallest triangle among all triplets."""
    return _all_areas(pts).min()


def _soft_obj(pts: np.ndarray, k: float = 60.0, frac: float = 0.05) -> float:
    """Smooth lower-bound surrogate: soft-min over the smallest fraction of areas.
    Lifting many near-minimal triangles at once escapes the hard-min plateaus."""
    areas = _all_areas(pts)
    m = max(3, int(len(areas) * frac))
    smallest = np.partition(areas, m - 1)[:m]
    return -np.log(np.mean(np.exp(-smallest * k))) / k


def _seeds(n: int, h: float) -> list:
    """A few structured, deterministic starting layouts."""
    seeds = []
    # Triangular lattice rows (4+3+2+1 = 10) plus one extra interior point
    pts = []
    rows = 4
    for r in range(rows):
        y = h * r / (rows - 1)
        cnt = rows - r
        for c in range(cnt):
            x = (c + 0.5) / cnt if r > 0 else c / (cnt - 1)
            pts.append([x * (1.0 - 0.5 * r / (rows - 1)) + 0.25 * r / (rows - 1), y])
    pts = np.array(pts)
    pts = np.vstack([pts, [0.5, h / 3.0]])[:n]
    seeds.append(_clip_to_triangle(pts))
    # Edge-heavy layout: many points on the boundary
    t = np.linspace(0.0, 1.0, 7)
    edge = []
    for u in t:
        edge.append([u, 0.0])
    for u in t[1:-1]:
        edge.append([0.5 * (1 + u), h * u])
        edge.append([0.5 * (1 - u), h * u])
    edge = np.array(edge)
    # take 11 spread along the boundary
    sel = np.linspace(0, len(edge) - 1, n).round().astype(int)
    seeds.append(_clip_to_triangle(edge[sel]))
    # Interior-spread layout
    rng = np.random.default_rng(777)
    u, v = rng.random(n), rng.random(n)
    su = np.sqrt(u)
    w0, w1, w2 = 1 - su, su * (1 - v), su * v
    seeds.append(_clip_to_triangle(np.stack([w1 + 0.5 * w2, h * w2], axis=1)))
    return seeds


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct an arrangement of 11 points on or inside an equilateral triangle
    maximizing the smallest triangle area (Heilbronn problem, n=11).

    Returns:
        points: np.ndarray of shape (11,2) with the x,y coordinates of the points.
    """
    n = 11
    h = np.sqrt(3.0) / 2.0

    rng = np.random.default_rng(12345)
    best, best_val = None, -1.0

    for seed_id, start in enumerate(_seeds(n, h)):
        s_rng = np.random.default_rng(12345 + 1000 * seed_id)
        cur = _clip_to_triangle(start.copy())

        # Phase 1: soft-min coarse search (smooth landscape, larger steps)
        cur_val = _soft_obj(cur, k=30.0)
        iters = 12000
        step0 = 0.05
        k_lo, k_hi = 30.0, 120.0
        for it in range(iters):
            frac_it = it / iters
            step = step0 * (1.0 - frac_it) + 1e-3
            k_it = k_lo + (k_hi - k_lo) * frac_it
            i = s_rng.integers(n)
            trial = cur.copy()
            trial[i] += s_rng.normal(0.0, step, size=2)
            trial = _clip_to_triangle(trial)
            val = _soft_obj(trial, k=k_it)
            if val >= cur_val - 1e-12:
                cur, cur_val = trial, val

        # Phase 2: hard-min fine search (shrink steps)
        cur_val = _min_area(cur)
        iters = 10000
        step0 = 0.01
        for it in range(iters):
            step = step0 * (1.0 - it / iters) + 1e-5
            i = s_rng.integers(n)
            trial = cur.copy()
            trial[i] += s_rng.normal(0.0, step, size=2)
            trial = _clip_to_triangle(trial)
            val = _min_area(trial)
            if val >= cur_val - 1e-13:
                cur, cur_val = trial, val

        if cur_val > best_val:
            best, best_val = cur.copy(), cur_val

    # Phase 3: deterministic coordinate-wise greedy polish on the best layout
    step = 0.005
    cur, cur_val = best.copy(), best_val
    for _ in range(60):
        improved = False
        for i in range(n):
            for d in range(2):
                for s in (+1, -1):
                    trial = cur.copy()
                    trial[i, d] += s * step
                    trial = _clip_to_triangle(trial)
                    val = _min_area(trial)
                    if val > cur_val + 1e-15:
                        cur, cur_val = trial, val
                        improved = True
        if not improved:
            step *= 0.5
            if step < 1e-7:
                break
    if cur_val > best_val:
        best, best_val = cur.copy(), cur_val

    # Guarantee validity of output even if optimization failed somehow
    if best is None or not np.all(np.isfinite(best)):
        best = _clip_to_triangle(np.zeros((n, 2)) + np.array([0.4, 0.3]))

    return best


# EVOLVE-BLOCK-END