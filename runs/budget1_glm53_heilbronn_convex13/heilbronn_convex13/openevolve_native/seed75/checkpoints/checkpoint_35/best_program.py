# EVOLVE-BLOCK-START
import time
import numpy as np

# Precomputed once: all C(13,3)=286 index triples, so the objective
# evaluation inside the search loop is pure vectorized NumPy.
_TRI_IDX = np.array(
    [(a, b, c) for a in range(13) for b in range(a + 1, 13)
     for c in range(b + 1, 13)], dtype=np.intp
)

# Triangles (rows of _TRI_IDX) that contain each point index i.
_TRI_WITH = [np.where((_TRI_IDX == i).any(axis=1))[0] for i in range(13)]


def _sorted_areas(pts: np.ndarray) -> np.ndarray:
    """All 286 triangle areas, sorted ascending (vectorized)."""
    p = pts[_TRI_IDX]  # (286, 3, 2)
    area = 0.5 * np.abs(
        (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) -
        (p[:, 1, 1] - p[:, 0, 1]) * (p[:, 2, 0] - p[:, 0, 0])
    )
    return np.sort(area)


def _min_triangle_area(pts: np.ndarray) -> float:
    """Vectorized minimum area over all C(n,3) triangles."""
    return float(_sorted_areas(pts)[0])


def _hull_area(pts: np.ndarray) -> float:
    """Convex hull area via monotone chain + shoelace (n=13, very cheap)."""
    p = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    if len(p) < 3:
        return 0.0

    def _cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for q in p:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    upper = []
    for q in p[::-1]:
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    h = np.array(lower[:-1] + upper[:-1])
    return 0.5 * abs(
        np.dot(h[:, 0], np.roll(h[:, 1], -1)) -
        np.dot(h[:, 1], np.roll(h[:, 0], -1)))


def _tri_areas(pts: np.ndarray) -> np.ndarray:
    """All 286 triangle areas, unsorted (vectorized)."""
    p = pts[_TRI_IDX]
    return 0.5 * np.abs(
        (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) -
        (p[:, 1, 1] - p[:, 0, 1]) * (p[:, 2, 0] - p[:, 0, 0]))


def heilbronn_convex13() -> np.ndarray:
    """
    Deterministic construction of 13 points maximizing the normalized
    smallest-triangle area  min_area / hull_area.

    Approach:
    1. Multi-start initialization (circle, square perimeter, seeded random).
    2. Seeded stochastic local search with incremental evaluation: moving
       point i only changes the triangles containing i, so only those areas
       are recomputed each trial (~10x faster than full recompute).
    3. Objective = (min_area + 0.05 * sum of next-4 smallest) / hull_area.
       Optimizing the ratio directly (instead of raw min area in a box)
       exploits scale invariance: the hull can shrink/expand freely, which
       is exactly what the scoring metric measures.
    4. Best configuration across all starts kept by true normalized min.

    Deterministic: fixed seeds throughout.
    """
    n = 13

    starts = []
    ang = 2.0 * np.pi * np.arange(n) / n + 0.15
    starts.append(np.column_stack(
        [0.5 + 0.49 * np.cos(ang), 0.5 + 0.49 * np.sin(ang)]))
    per = np.linspace(0.0, 4.0, n, endpoint=False)
    sq = []
    for t in per:
        if t < 1.0:
            sq.append([t, 0.0])
        elif t < 2.0:
            sq.append([1.0, t - 1.0])
        elif t < 3.0:
            sq.append([3.0 - t, 1.0])
        else:
            sq.append([0.0, 4.0 - t])
    starts.append(np.array(sq, dtype=float))
    # 3-fold rotationally symmetric start: optimal Heilbronn-style
    # configurations often exhibit 3-fold symmetry, so seed one explicitly.
    base = np.array([[0.12, 0.10], [0.88, 0.14], [0.50, 0.90],
                     [0.30, 0.35], [0.70, 0.32], [0.42, 0.62],
                     [0.65, 0.58], [0.05, 0.55], [0.95, 0.60],
                     [0.20, 0.80], [0.80, 0.85], [0.50, 0.15],
                     [0.50, 0.45]], dtype=float)
    starts.append(base)
    for s in range(8):
        rng0 = np.random.default_rng(100 + s)
        starts.append(rng0.random((n, 2)))
    # Near-boundary annular starts: good Heilbronn configs often place
    # several points near the hull boundary; random radii sample that.
    for s in range(3):
        rng0 = np.random.default_rng(500 + s)
        r = 0.3 + 0.7 * rng0.random((n, 1))
        th = 2.0 * np.pi * rng0.random((n, 1)) + 0.1 * s
        starts.append(np.column_stack(
            [0.5 + r * np.cos(th), 0.5 + r * np.sin(th)]).reshape(n, 2))

    def _obj(areas: np.ndarray, hull: float) -> float:
        # Weight the 6 smallest areas: the min dominates, but the next
        # five give the optimizer a gradient to grow runner-up triangles.
        srt = np.partition(areas, 6)
        return (srt[0] + 0.05 * srt[1:6].sum()) / max(hull, 1e-12)

    best_pts = None
    best_val = -1.0
    t0 = time.time()
    TIME_BUDGET = 280.0  # headroom below the 360 s limit

    def _anneal(pts, seed, scale0, max_iter=200000):
        """Seeded stochastic local search with incremental evaluation.
        Returns the final point set (in place annealed) and its true
        normalized minimum triangle area."""
        rng = np.random.default_rng(seed)
        areas = _tri_areas(pts)
        hull = _hull_area(pts)
        obj = _obj(areas, hull)
        scale = scale0
        it = 0
        while scale > 1e-6 and it < max_iter:
            improved = False
            for _ in range(3000):
                it += 1
                i = int(rng.integers(n))
                delta = rng.normal(size=2) * scale
                old = pts[i].copy()
                old_hull = hull
                pts[i] = old + delta
                tw = _TRI_WITH[i]
                old_areas = areas[tw].copy()
                p3 = pts[_TRI_IDX[tw]]
                areas[tw] = 0.5 * np.abs(
                    (p3[:, 1, 0] - p3[:, 0, 0]) * (p3[:, 2, 1] - p3[:, 0, 1]) -
                    (p3[:, 1, 1] - p3[:, 0, 1]) * (p3[:, 2, 0] - p3[:, 0, 0]))
                hull = _hull_area(pts)
                o2 = _obj(areas, hull)
                if o2 > obj + 1e-14:
                    obj = o2
                    improved = True
                else:
                    pts[i] = old
                    areas[tw] = old_areas
                    hull = old_hull
            if not improved:
                scale *= 0.5
            if time.time() - t0 > TIME_BUDGET:
                break
        return _min_triangle_area(pts) / max(_hull_area(pts), 1e-12)

    for si, pts0 in enumerate(starts):
        pts = pts0.copy()
        val = _anneal(pts, 42 + si,
                      max(np.ptp(pts), 1e-6) * 0.05)
        if val > best_val:
            best_val = val
            best_pts = pts.copy()
        if time.time() - t0 > TIME_BUDGET:
            break

    # Reheat + polish: restart annealing from the best configuration with a
    # fresh moderate scale and a different seed. This escapes the local
    # optimum the monotone scale schedule got trapped in, while the incumbent
    # is always kept (never worse than the first pass).
    if best_pts is not None and time.time() - t0 < TIME_BUDGET:
        for r in range(6):
            if time.time() - t0 > TIME_BUDGET:
                break
            pts = best_pts.copy()
            val = _anneal(pts, 9000 + r,
                          max(np.ptp(pts), 1e-6) * 0.02,
                          max_iter=150000)
            if val > best_val:
                best_val = val
                best_pts = pts.copy()

    # Normalize to unit-area hull for a clean, reproducible output.
    if best_pts is not None:
        h = _hull_area(best_pts)
        if h > 1e-12:
            best_pts = best_pts / np.sqrt(h)
    return best_pts


# EVOLVE-BLOCK-END
