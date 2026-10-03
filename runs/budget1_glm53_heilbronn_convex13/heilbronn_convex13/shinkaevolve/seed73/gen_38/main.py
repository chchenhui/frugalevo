# EVOLVE-BLOCK-START
import numpy as np


_TRI_IDX = np.array(
    [t for i in range(11) for j in range(i + 1, 12) for t in
     ((i, j, k) for k in range(j + 1, 13))], dtype=np.int32)


def _min_triple_area(points: np.ndarray) -> tuple:
    """Return (min_area, argmin_index) over all 286 triples, fully vectorized."""
    p = points[_TRI_IDX]                      # (286, 3, 2)
    a = p[:, 1] - p[:, 0]
    b = p[:, 2] - p[:, 0]
    areas = 0.5 * np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
    k = int(np.argmin(areas))
    return float(areas[k]), k


def _hull_area(points: np.ndarray) -> float:
    """Area of convex hull via the shoelace formula on hull vertices."""
    pts = points[np.lexsort((points[:, 1], points[:, 0]))]
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = np.array(lower[:-1] + upper[:-1])
    x, y = hull[:, 0], hull[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _score(points: np.ndarray) -> float:
    ha = _hull_area(points)
    if ha <= 1e-12:
        return 0.0
    return _min_triple_area(points) / ha


_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1),
         (1, 1), (1, -1), (-1, 1), (-1, -1))


def _polish(points: np.ndarray, steps: np.ndarray,
            iters_per_step: int = 60) -> tuple:
    """Greedy hill-climb on min-triangle/hull-area score: perturb the
    vertices of the current worst triangle until no improvement at this
    step size, then shrink the step. Returns (points, score)."""
    best = points.copy()
    cur = _score(best)
    for step in steps:
        for _ in range(iters_per_step):
            _, k = _min_triple_area(best)
            improved = False
            for pi in _TRI_IDX[k]:
                for dx, dy in _DIRS:
                    trial = best.copy()
                    trial[pi, 0] += dx * step
                    trial[pi, 1] += dy * step
                    sc = _score(trial)
                    if sc > cur + 1e-14:
                        best = trial
                        cur = sc
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
    return best, cur


def _gen_seeds(rng) -> list:
    """72 deterministic seeds: random interior, jittered circles, and
    3-fold-symmetric concentric rings (center + 4 + 4 + 4)."""
    seeds = []
    for _ in range(30):
        seeds.append(rng.uniform(0.05, 0.95, size=(13, 2)))
    for _ in range(15):
        ang = np.sort(rng.uniform(0.0, 2.0 * np.pi, 13))
        r = rng.uniform(0.25, 0.47)
        pts = np.column_stack([0.5 + r * np.cos(ang), 0.5 + r * np.sin(ang)])
        pts = pts + rng.normal(0.0, 0.02, pts.shape)
        seeds.append(np.clip(pts, 0.01, 0.99))
    for _ in range(27):
        pts = np.empty((13, 2))
        radii = [rng.uniform(0.32, 0.47), rng.uniform(0.16, 0.30),
                 rng.uniform(0.03, 0.12)]
        counts = [5, 4, 4]
        a0 = rng.uniform(0.0, 2.0 * np.pi)
        idx = 0
        for ring, (r, c) in enumerate(zip(radii, counts)):
            for i in range(c):
                a = a0 + 0.7 * ring + 2.0 * np.pi * i / c
                pts[idx] = [0.5 + r * np.cos(a), 0.5 + r * np.sin(a)]
                idx += 1
        pts[idx] = [0.5, 0.5]
        pts = pts + rng.normal(0.0, 0.015, pts.shape)
        seeds.append(np.clip(pts, 0.01, 0.99))
    return seeds


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points maximizing the smallest triangle
    area normalized by convex hull area, via a deterministic multi-start
    funnel: 72 seeds -> coarse polish of top 24 -> exact polish of top 10.
    """
    rng = np.random.default_rng(seed=20250113)
    seeds = _gen_seeds(rng)

    # Stage 1: score all seeds, keep top 24.
    scored = sorted(seeds, key=_score, reverse=True)[:24]

    # Stage 2: coarse polish on survivors, keep top 10.
    coarse_steps = np.geomspace(0.10, 0.005, 12)
    polished = []
    for pts in scored:
        polished.append(_polish(pts, coarse_steps, iters_per_step=40))
    polished.sort(key=lambda t: t[1], reverse=True)
    polished = polished[:10]

    # Stage 3: exact fine polish on the top candidates.
    fine_steps = np.geomspace(0.05, 0.0002, 40)
    best_pts, best_val = None, -1.0
    for pts, _ in polished:
        p, v = _polish(pts, fine_steps, iters_per_step=60)
        if v > best_val:
            best_val, best_pts = v, p

    # Safety fallback if the funnel failed for any reason.
    if best_pts is None or not np.all(np.isfinite(best_pts)):
        ang = 2.0 * np.pi * np.arange(12) / 12.0
        best_pts = np.zeros((13, 2))
        best_pts[:12, 0] = 0.5 + 0.45 * np.cos(ang)
        best_pts[:12, 1] = 0.5 + 0.45 * np.sin(ang)
        best_pts[12] = 0.5
    return best_pts


# EVOLVE-BLOCK-END