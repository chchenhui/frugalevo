# EVOLVE-BLOCK-START
import numpy as np
import time


from itertools import combinations

_TRIP = np.array(list(combinations(range(11), 3)), dtype=int)  # (165,3) triplet indices
# For each point index, the triplet row indices that involve it (45 per point).
_TRIP_WITH = [np.flatnonzero((i == _TRIP).any(axis=1)) for i in range(11)]


def _all_areas(points: np.ndarray) -> np.ndarray:
    """Vectorized areas of all 165 point triplets (triangle area normalized to 1)."""
    i, j, k = _TRIP[:, 0], _TRIP[:, 1], _TRIP[:, 2]
    dx1 = points[j, 0] - points[i, 0]
    dy1 = points[j, 1] - points[i, 1]
    dx2 = points[k, 0] - points[i, 0]
    dy2 = points[k, 1] - points[i, 1]
    return 0.5 * np.abs(dx1 * dy2 - dy1 * dx2)


def _min_area(points: np.ndarray):
    """Smallest triangle area and the indices (i,j,k) achieving it (triangle area normalized to 1)."""
    a = _all_areas(points)
    m = int(np.argmin(a))
    return float(a[m]), tuple(_TRIP[m])


def _min_area_k(points: np.ndarray, kk: int = 4):
    """Value of the kk-th smallest triangle area and sorted set of all vertex
    indices involved in those kk smallest triangles."""
    a = _all_areas(points)
    order = np.argsort(a)[:kk]
    crit = sorted(set(_TRIP[order].ravel().tolist()))
    return float(a[order[-1]]), crit


def _min_area_after_move(points: np.ndarray, moved: int, full_areas: np.ndarray):
    """Incrementally update all-triplet areas after moving a single point and
    return (new minimum area, updated area vector). Only the 45 triplets
    containing `moved` are recomputed instead of all 165."""
    a = full_areas.copy()
    idx = _TRIP_WITH[moved]
    i, j, k = _TRIP[idx, 0], _TRIP[idx, 1], _TRIP[idx, 2]
    dx1 = points[j, 0] - points[i, 0]
    dy1 = points[j, 1] - points[i, 1]
    dx2 = points[k, 0] - points[i, 0]
    dy2 = points[k, 1] - points[i, 1]
    a[idx] = 0.5 * np.abs(dx1 * dy2 - dy1 * dx2)
    return float(a.min()), a


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

    Approach: deterministic multi-start local search with targeted moves.
    At each step the few smallest triangles are located and their vertices
    are perturbed first (only they can raise the minimum); other points
    follow in random order. Both Gaussian and axis-aligned moves are
    tried. A mild annealing tolerance (accepting slightly worse
    configurations) helps escape local optima; the best configuration
    ever seen is tracked and returned. Each start gets a capped share of
    the time budget so all starts are refined. Fixed RNG seed.
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
        pts.append([0.5 + (t - 0.5) * 0.5, s * t])
        pts.append([0.5 - (t - 0.5) * 0.5, s * t])
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

    per_start = (deadline - time.time()) / max(len(starts), 1)

    for pts0 in starts:
        stop = min(deadline, time.time() + per_start)
        pts = _in_triangle(np.array(pts0[:n], dtype=float))
        cur, _ = _min_area(pts)
        cur_areas = _all_areas(pts)
        run_best_pts, run_best_val = pts.copy(), cur
        scale = 0.05
        temp = 1e-4  # annealing tolerance for near-equal/slightly-worse moves
        while scale > 1e-6 and time.time() < stop:
            improved = False
            # prioritize vertices of the few smallest triangles, then all points
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
                    v, new_a = _min_area_after_move(cand, i, cur_areas)
                    # accept improving or near-equal moves; track best ever
                    if v > cur - temp:
                        pts = cand
                        cur = v
                        cur_areas = new_a
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

    # Basin-hopping polish: fine greedy refinement (deterministic axis
    # line-moves plus random trials), then kick the best and re-polish
    # until the budget is exhausted. Best configuration ever seen wins.
    if best_pts is not None and time.time() < deadline:
        pts = best_pts.copy()
        cur = best_val
        cur_areas = _all_areas(pts)
        while time.time() < deadline:
            scale = 0.008
            while scale > 1e-7 and time.time() < deadline:
                improved = False
                _, crit = _min_area_k(pts, 6)
                order = crit + [i for i in range(n) if i not in crit]
                for i in order:
                    base = pts[i].copy()
                    # deterministic coordinate line moves (slide along edges)
                    for d in (scale, -scale, 2.0 * scale, -2.0 * scale,
                              0.5 * scale, -0.5 * scale):
                        done = False
                        for axis in (0, 1):
                            cand = pts.copy()
                            mv = base.copy()
                            mv[axis] += d
                            cand[i] = _in_triangle(mv[None, :])[0]
                            v, new_a = _min_area_after_move(cand, i, cur_areas)
                            if v > cur + 1e-15:
                                pts, cur, cur_areas = cand, v, new_a
                                improved = True
                                done = True
                                break
                        if done:
                            break
                    if not improved:
                        for trial in range(10):
                            delta = rng.normal(0.0, scale, size=2)
                            cand = pts.copy()
                            cand[i] = _in_triangle((base + delta)[None, :])[0]
                            v, new_a = _min_area_after_move(cand, i, cur_areas)
                            if v > cur + 1e-15:
                                pts, cur, cur_areas = cand, v, new_a
                                improved = True
                                break
                if not improved:
                    scale *= 0.5
            if cur > best_val:
                best_val = cur
                best_pts = pts.copy()
            # kick the best configuration and re-polish (basin hopping)
            pts = best_pts.copy()
            pts = _in_triangle(pts + rng.normal(0.0, 0.003, size=(n, 2)))
            cur, _ = _min_area(pts)
            cur_areas = _all_areas(pts)

    if best_pts is None:
        best_pts = starts[0]
    return np.asarray(best_pts, dtype=float)


# EVOLVE-BLOCK-END
