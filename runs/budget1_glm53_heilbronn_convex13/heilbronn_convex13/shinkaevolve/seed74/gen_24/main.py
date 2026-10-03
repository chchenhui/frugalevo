# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations

_N = 13
_IDX = np.array(list(combinations(range(_N), 3)), dtype=int)  # (286, 3)
K_SMALL = 12


def _all_tri_areas(pts):
    a = pts[_IDX[:, 0]]
    b = pts[_IDX[:, 1]]
    c = pts[_IDX[:, 2]]
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    return 0.5 * np.abs(cross)


def _hull_area(points):
    pts = points[np.lexsort((points[:, 1], points[:, 0]))]
    if len(pts) <= 2:
        return 0.0

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = np.array(lower[:-1] + upper[:-1])
    x, y = hull[:, 0], hull[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _init_ring(alt):
    ang = np.arange(12) * (2.0 * np.pi / 12.0)
    rad = np.where(np.arange(12) % 2 == 0, 1.0, 0.72) if alt else np.ones(12)
    pts = np.zeros((_N, 2))
    pts[:12, 0] = rad * np.cos(ang)
    pts[:12, 1] = rad * np.sin(ang)
    pts[12] = (0.0, 0.0)
    return pts


def _init_triangle():
    pts = np.zeros((_N, 2))
    a0 = np.pi / 2.0
    for i in range(3):
        a = a0 + i * 2.0 * np.pi / 3.0
        pts[i] = (1.05 * np.cos(a), 1.05 * np.sin(a))
    for i in range(6):
        a = a0 + np.pi / 6.0 + i * np.pi / 3.0
        pts[3 + i] = (0.62 * np.cos(a), 0.62 * np.sin(a))
    for i in range(3):
        a = a0 + i * 2.0 * np.pi / 3.0
        pts[9 + i] = (0.30 * np.cos(a), 0.30 * np.sin(a))
    pts[12] = (0.0, 0.0)
    return pts


def _init_grid(rng):
    g = []
    for i in range(4):
        for j in range(4):
            g.append((i / 3.0, j / 3.0))
    g = np.array(g, dtype=float)
    keep = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14]
    pts = g[keep] * 2.0 - 1.0
    pts += rng.normal(0, 0.06, pts.shape)
    return pts


def _init_pentagon(rng):
    pts = np.zeros((_N, 2))
    for i in range(5):
        a = -np.pi / 2.0 + i * 2 * np.pi / 5.0
        pts[i] = (np.cos(a), np.sin(a))
    for i in range(5):
        a = -np.pi / 2.0 + 0.3 + i * 2 * np.pi / 5.0
        r = 0.55
        pts[5 + i] = (r * np.cos(a), r * np.sin(a))
    pts[10] = (0.25, 0.15)
    pts[11] = (-0.22, -0.18)
    pts[12] = (0.0, 0.0)
    return pts + rng.normal(0, 0.02, pts.shape)


def _targeted_new(pts, t, i, step, rng):
    old = pts[i]
    others = [int(x) for x in t if x != i]
    j, k2 = others[0], others[1]
    d = pts[k2] - pts[j]
    nrm = np.hypot(d[0], d[1]) + 1e-12
    nd = np.array([-d[1], d[0]]) / nrm
    side = d[0] * (old[1] - pts[j, 1]) - d[1] * (old[0] - pts[j, 0])
    if side < 0:
        nd = -nd
    return old + step * nd + rng.normal(0, step * 0.35, 2)


def _anneal(init, rng, iters, step0, temp0):
    """Targeted annealing on the smooth score (mean of K smallest areas)."""
    pts = init.copy()
    a = _all_tri_areas(pts)
    smooth = float(np.sort(a)[:K_SMALL].mean())
    best = pts.copy()
    best_min = float(a.min())
    step = step0
    T = temp0
    decay = temp0 ** (1.0 / iters)

    for it in range(iters):
        a = _all_tri_areas(pts)
        order = np.argsort(a)
        t = _IDX[order[rng.integers(K_SMALL)]]
        i = int(t[rng.integers(3)])
        old = pts[i].copy()

        if rng.random() < 0.55:
            new = _targeted_new(pts, t, i, step, rng)
        else:
            new = old + rng.normal(0, step, 2)

        dists = np.hypot(*(pts - new).T)
        dists[i] = 10.0
        if dists.min() < 1e-6:
            T *= decay
            continue

        pts[i] = new
        a2 = _all_tri_areas(pts)
        s2 = float(np.sort(a2)[:K_SMALL].mean())
        if s2 >= smooth or rng.random() < np.exp((s2 - smooth) / max(T, 1e-12)):
            smooth = s2
            m2 = float(a2.min())
            if m2 > best_min:
                best_min = m2
                best = pts.copy()
        else:
            pts[i] = old
        T *= decay
        if it % 500 == 499:
            step = max(step * 0.88, 0.0015)

    return best, best_min


def _polish(pts, rng, iters, K, step0=0.01, step_min=0.0004):
    """Greedy polish on the mean of the K smallest triangle areas, tracking
    the best true minimum encountered. Staged K lets pressure tighten."""
    pts = pts.copy()
    a = _all_tri_areas(pts)
    cur_soft = float(np.sort(a)[:K].mean())
    best = pts.copy()
    best_min = float(a.min())
    step = step0
    for it in range(iters):
        order = np.argsort(a)
        t = _IDX[order[rng.integers(min(K, 6))]]
        i = int(t[rng.integers(3)])
        old = pts[i].copy()
        new = _targeted_new(pts, t, i, step * 0.7, rng)
        dists = np.hypot(*(pts - new).T)
        dists[i] = 10.0
        if dists.min() < 1e-6:
            continue
        pts[i] = new
        a2 = _all_tri_areas(pts)
        s2 = float(np.sort(a2)[:K].mean())
        m2 = float(a2.min())
        if s2 > cur_soft or (s2 >= cur_soft and m2 > best_min):
            cur_soft = s2
            a = a2
            if m2 > best_min:
                best_min = m2
                best = pts.copy()
        else:
            pts[i] = old
        if it % 300 == 299:
            step = max(step * 0.9, step_min)
    return best, best_min


def _full_cycle(init, rng, iters_a, step0, temp0):
    pts, m = _anneal(init, rng, iters=iters_a, step0=step0, temp0=temp0)
    pts, m = _polish(pts, rng, iters=5000, K=8)
    pts, m = _polish(pts, rng, iters=3000, K=4)
    return pts, m


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points maximizing the smallest triangle
    area normalized by convex hull area. Multi-start targeted anneal with
    two-tier polish, plus crossover reheat passes for basin mixing.
    """
    rng = np.random.default_rng(seed=42)
    inits = [_init_ring(False), _init_ring(True), _init_triangle(),
             _init_grid(rng), _init_pentagon(rng)]

    # --- Phase 1: multi-start full cycles (anneal + K=8 polish + K=4 polish) ---
    pool = []
    global_min = -1.0
    global_best = None
    for init in inits:
        pts, m = _full_cycle(init, rng, iters_a=9000, step0=0.05, temp0=0.004)
        pool.append((m, pts))
        if m > global_min:
            global_min = m
            global_best = pts

    pool.sort(key=lambda t: t[0], reverse=True)
    # keep the top half for crossover mixing (cheap exploration)
    npool = max(3, len(pool) // 2)
    pool = pool[:npool]

    # --- Phase 2: 2 round-robin crossover reheat passes over the pool ---
    for pass_num in range(2):
        new_entries = []
        for ii in range(npool):
            jj = (ii + 1 + pass_num) % npool
            if jj == ii:
                jj = (ii + 2) % npool
            child = 0.7 * pool[ii][1] + 0.3 * pool[jj][1]
            child += rng.normal(0, 0.01, child.shape)
            pts, m = _anneal(child, rng, iters=4000,
                             step0=0.02, temp0=0.002 * (0.6 ** pass_num))
            pts, m = _polish(pts, rng, iters=3000, K=8)
            pts, m = _polish(pts, rng, iters=2000, K=4)
            new_entries.append((m, pts))
            if m > global_min:
                global_min = m
                global_best = pts
        combined = pool + new_entries
        combined.sort(key=lambda t: t[0], reverse=True)
        pool = combined[:npool]
        if all(not np.array_equal(p[1], global_best) for p in pool):
            pool[-1] = (global_min, global_best)

    # --- Phase 3: reheat cycles from the global best ---
    for cyc in range(2):
        pts, m = _anneal(global_best.copy(), rng, iters=4000,
                         step0=0.02, temp0=0.002 * (0.5 ** cyc))
        pts, m = _polish(pts, rng, iters=4000, K=8)
        pts, m = _polish(pts, rng, iters=2500, K=4)
        if m > global_min:
            global_min = m
            global_best = pts

    best_pts = global_best.copy()

    # rescale so the convex hull has unit area (scale-invariant problem)
    hull = _hull_area(best_pts)
    if hull > 1e-12:
        s = 1.0 / np.sqrt(hull)
        best_pts = (best_pts - best_pts.mean(axis=0)) * s + best_pts.mean(axis=0)

    return best_pts


# EVOLVE-BLOCK-END