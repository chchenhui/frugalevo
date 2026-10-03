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


def _min_area(pts: np.ndarray) -> float:
    """Vectorized area of the smallest triangle among all triplets (doubled area)."""
    idx = np.array(list(itertools.combinations(range(len(pts)), 3)))
    p = pts[idx]  # (C(n,3), 3, 2)
    cross = (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - \
            (p[:, 1, 1] - p[:, 0, 1]) * (p[:, 2, 0] - p[:, 0, 0])
    return np.abs(cross).min()


def _triangular_seed(n=11):
    """Triangular-lattice style seed: 4 rows (1+2+3+... pattern trimmed to n)."""
    h = np.sqrt(3.0) / 2.0
    pts = []
    rows = 4
    for r in range(rows):
        y = h * r / (rows - 1)
        cnt = rows - r
        for c in range(cnt):
            x = (c + 0.5 * r / (rows - 1)) / (rows - 1)
            pts.append([x, y])
    pts = np.array(pts)
    if len(pts) > n:
        cen = pts.mean(axis=0)
        d = np.linalg.norm(pts - cen, axis=1)
        keep = np.sort(np.argsort(d)[:n])
        pts = pts[keep]
    return _clip_to_triangle(pts)


def _random_seed(n, rng):
    """Uniform sample of n points in the triangle via barycentric coords."""
    h = np.sqrt(3.0) / 2.0
    u, v = rng.random(n), rng.random(n)
    su = np.sqrt(u)
    w0, w1, w2 = 1.0 - su, su * (1.0 - v), su * v
    return _clip_to_triangle(np.stack([w1 + 0.5 * w2, h * w2], axis=1))


def _anneal(pts, best, best_val, seed, iters, step0, step_min):
    """Deterministic annealed hill-climb; returns improved best config."""
    n = len(pts)
    rng = np.random.default_rng(seed)
    cur = pts.copy()
    cur_val = _min_area(cur)
    if cur_val > best_val:
        best, best_val = cur.copy(), cur_val
    for it in range(iters):
        step = step0 * (1.0 - it / iters) + step_min
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
    return best, best_val


def _optimize():
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
    starts = [_clip_to_triangle(pts), _triangular_seed(n)]
    for s in range(4):
        starts.append(_random_seed(n, np.random.default_rng(1000 + s)))

    best, best_val = None, -1.0
    per_start = 5000
    for s_i, start in enumerate(starts):
        best, best_val = _anneal(start, best, best_val,
                                 seed=20240000 + s_i * 7919,
                                 iters=per_start, step0=0.05, step_min=1e-4)
    # Fine polish on the best configuration found
    best, best_val = _anneal(best, best, best_val,
                             seed=99, iters=10000, step0=0.002, step_min=1e-5)

    # Guarantee validity of output even if optimization failed somehow
    if best is None or not np.all(np.isfinite(best)):
        best = _clip_to_triangle(np.zeros((n, 2)) + np.array([0.4, 0.3]))
    return best


try:
    _POINTS = _optimize()
except Exception:
    _POINTS = _clip_to_triangle(np.zeros((11, 2)) + np.array([0.4, 0.3]))


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct an arrangement of 11 points on or inside an equilateral triangle
    maximizing the smallest triangle area (Heilbronn problem, n=11).

    Returns:
        points: np.ndarray of shape (11,2) with the x,y coordinates of the points.
    """
    return _POINTS.copy()

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

    # Deterministic local search with annealing-style shrinking steps
    step0 = 0.05
    iters = 30000
    for it in range(iters):
        step = step0 * (1.0 - it / iters) + 1e-4
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

    # Guarantee validity of output even if optimization failed somehow
    if best is None or not np.all(np.isfinite(best)):
        best = _clip_to_triangle(np.zeros((n, 2)) + np.array([0.4, 0.3]))

    return best


# EVOLVE-BLOCK-END