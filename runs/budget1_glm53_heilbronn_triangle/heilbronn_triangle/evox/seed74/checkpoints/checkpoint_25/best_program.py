# EVOLVE-BLOCK-START
import numpy as np
import time


def _min_area(points: np.ndarray) -> float:
    """Area of smallest triangle among all triplets (normalized: triangle area = 1)."""
    n = points.shape[0]
    best = np.inf
    for i in range(n - 2):
        p = points[i]
        for j in range(i + 1, n - 1):
            q = points[j] - p
            r = points[j + 1:] - p
            areas = 0.5 * np.abs(q[0] * r[:, 1] - q[1] * r[:, 0])
            m = areas.min()
            if m < best:
                best = m
    return best


def _min_area_k(points: np.ndarray, kk: int = 4):
    """kk-th smallest triangle area and sorted set of vertex indices involved."""
    n = points.shape[0]
    vals, idxs = [], []
    for i in range(n - 2):
        p = points[i]
        for j in range(i + 1, n - 1):
            q = points[j] - p
            r = points[j + 1:] - p
            areas = 0.5 * np.abs(q[0] * r[:, 1] - q[1] * r[:, 0])
            for t in range(areas.shape[0]):
                vals.append(float(areas[t]))
                idxs.append((i, j, j + 1 + t))
    order = np.argsort(vals)[:kk]
    crit = set()
    for o in order:
        crit.update(idxs[o])
    return vals[order[-1]], sorted(crit)


def _in_triangle(pts: np.ndarray) -> np.ndarray:
    """Project points back inside the unit equilateral triangle."""
    x, y = pts[:, 0].copy(), pts[:, 1].copy()
    s = np.sqrt(3.0)
    x = np.clip(x, 0.0, 1.0)
    y = np.clip(y, 0.0, None)
    ymax = np.minimum(s * x, s * (1.0 - x))
    y = np.minimum(y, np.maximum(ymax, 0.0))
    return np.stack([x, y], axis=1)


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct 11 points in the unit equilateral triangle maximizing the minimum
    triangle area.

    Approach: deterministic multi-start annealing local search with targeted
    moves. At each step the few smallest triangles are located and their
    vertices are perturbed first (only they can raise the minimum); other
    points follow in random order. Moves improving or nearly equal to the
    current minimum are accepted (annealing tolerance) to escape local optima;
    the best configuration ever seen is tracked and returned. Each start gets
    a capped share of the time budget, plus a final polish on the overall
    best. Fixed RNG seed for reproducibility.
    """
    n = 11
    rng = np.random.default_rng(12345)
    deadline = time.time() + 200.0  # seconds budget

    s = np.sqrt(3.0) / 2.0
    tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, s]])

    starts = []

    # Start 1: edge points plus interior
    pts = []
    m = 4
    for k in range(1, m):
        t = k / m
        pts.append([t, 0.0])
        pts.append([0.5 + (t - 0.5) * 0.5, s * 2.0 * t])
        pts.append([0.5 - (t - 0.5) * 0.5, s * 2.0 * t])
    pts = np.array(pts)
    interior = np.array([[0.25, 0.25], [0.75, 0.25]])
    starts.append(np.vstack([pts, interior]))

    # Start 2: random barycentric
    for _ in range(3):
        b = rng.dirichlet(np.ones(3), size=n)
        starts.append(b @ tri)

    # Start 3: structured rows
    pts = []
    rows = [(0.0, [0.5]), (0.2, [0.2, 0.5, 0.8]), (0.45, [0.1, 0.35, 0.6, 0.85]),
            (0.65, [0.25, 0.5, 0.75])]
    for y, xs in rows:
        for x in xs:
            pts.append([x, y])
    starts.append(np.array(pts[:n]))

    best_pts, best_val = None, -1.0

    per_start = max((deadline - time.time()) / len(starts), 1.0)
    for pts0 in starts:
        if time.time() >= deadline:
            break
        stop = min(deadline, time.time() + per_start)
        pts = _in_triangle(np.array(pts0[:n], dtype=float))
        cur = _min_area(pts)
        run_best_pts, run_best_val = pts.copy(), cur
        scale = 0.05
        temp = 1e-4  # annealing tolerance for near-equal/slightly-worse moves
        while scale > 1e-6 and time.time() < stop:
            improved = False
            # perturb vertices of the few smallest triangles first
            _, crit = _min_area_k(pts, 4)
            order = list(rng.permutation(n))
            order = crit + [i for i in order if i not in crit]
            for i in order:
                base = pts[i].copy()
                for trial in range(10):
                    if trial % 2 == 0:
                        delta = rng.normal(0.0, scale, size=2)
                    else:
                        # axis-aligned move (helps slide along edges)
                        delta = np.zeros(2)
                        delta[rng.integers(0, 2)] = rng.normal(0.0, scale * 1.5)
                    cand = pts.copy()
                    cand[i] = _in_triangle((base + delta)[None, :])[0]
                    v = _min_area(cand)
                    if v > cur - temp:
                        pts = cand
                        cur = v
                        if v > run_best_val:
                            run_best_val = v
                            run_best_pts = pts.copy()
                        improved = True
                        break
            if not improved:
                scale *= 0.6
                temp *= 0.5
        if run_best_val > best_val:
            best_val = run_best_val
            best_pts = run_best_pts.copy()

    # Final polish: fine greedy refinement of the overall best configuration
    if best_pts is not None and time.time() < deadline:
        pts = best_pts.copy()
        cur = best_val
        scale = 0.01
        while scale > 1e-7 and time.time() < deadline:
            improved = False
            _, crit = _min_area_k(pts, 4)
            order = list(rng.permutation(n))
            order = crit + [i for i in order if i not in crit]
            for i in order:
                base = pts[i].copy()
                for _ in range(12):
                    delta = rng.normal(0.0, scale, size=2)
                    cand = pts.copy()
                    cand[i] = _in_triangle((base + delta)[None, :])[0]
                    v = _min_area(cand)
                    if v > cur + 1e-15:
                        pts, cur = cand, v
                        improved = True
                        break
            if not improved:
                scale *= 0.5
        if cur > best_val:
            best_val = cur
            best_pts = pts.copy()

    if best_pts is None:
        best_pts = starts[0]
    return np.asarray(best_pts, dtype=float)


# EVOLVE-BLOCK-END
