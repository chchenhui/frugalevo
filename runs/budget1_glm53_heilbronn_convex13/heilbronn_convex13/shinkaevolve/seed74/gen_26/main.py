# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations

_N = 13
_IDX = np.array(list(combinations(range(_N), 3)))  # (286, 3)


def _all_tri_areas(pts):
    """Vectorized areas of all C(13,3) triangles."""
    a = pts[_IDX[:, 0]]
    b = pts[_IDX[:, 1]]
    c = pts[_IDX[:, 2]]
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    return 0.5 * np.abs(cross)


def _hull_area(pts):
    p = sorted(map(tuple, pts), key=lambda t: (t[0], t[1]))
    p = list(dict.fromkeys(p))
    if len(p) < 3:
        return 0.0

    def cross(o, x, y):
        return (x[0]-o[0])*(y[1]-o[1]) - (x[1]-o[1])*(y[0]-o[0])

    lower = []
    for pt in p:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], pt) <= 0:
            lower.pop()
        lower.append(pt)
    upper = []
    for pt in reversed(p):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], pt) <= 0:
            upper.pop()
        upper.append(pt)
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return 0.0
    h = np.array(hull)
    s = 0.0
    for i in range(len(h)):
        x1, y1 = h[i]
        x2, y2 = h[(i + 1) % len(h)]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def _score(pts):
    ha = _hull_area(pts)
    if ha <= 1e-12:
        return 0.0
    return _all_tri_areas(pts).min() / ha


def _seeds():
    cands = []
    # 1) hexagonal rings: 6 outer + 6 inner + center (3-fold symmetric)
    t = np.linspace(0, 2 * np.pi, 7)[:-1]
    outer = np.stack([0.5 + 0.47 * np.cos(t), 0.5 + 0.47 * np.sin(t)], axis=1)
    inner = np.stack([0.5 + 0.26 * np.cos(t + np.pi / 6),
                      0.5 + 0.26 * np.sin(t + np.pi / 6)], axis=1)
    cands.append(np.vstack([outer, inner, [[0.5, 0.5]]]))

    # 2) two staggered hexagons, different radii
    inner2 = np.stack([0.5 + 0.33 * np.cos(t), 0.5 + 0.33 * np.sin(t)], axis=1)
    outer2 = np.stack([0.5 + 0.48 * np.cos(t + np.pi / 6),
                       0.5 + 0.48 * np.sin(t + np.pi / 6)], axis=1)
    cands.append(np.vstack([outer2, inner2, [[0.5, 0.5]]]))

    # 3) regular 13-gon on circle (boundary-only baseline)
    ang = np.linspace(0, 2 * np.pi, 14)[:-1]
    poly = np.stack([0.5 + 0.48 * np.cos(ang), 0.5 + 0.48 * np.sin(ang)], axis=1)
    cands.append(poly)

    # 4) jittered grid, fixed seed
    rng = np.random.default_rng(123)
    g = []
    for i in range(4):
        for j in range(4):
            g.append((0.125 + i * 0.25 + rng.random() * 0.05,
                      0.125 + j * 0.25 + rng.random() * 0.05))
    cands.append(np.array(g[:13]))

    # 5) triangular lattice, 13 points
    pts = []
    r = 0
    y = 0.08
    while len(pts) < 13:
        cnt = min(5 - abs(r - 2), 13 - len(pts))
        xs = np.linspace(0.1, 0.9, max(cnt, 1)) + 0.03 * (r % 2)
        for x in xs[:cnt]:
            pts.append((min(max(x, 0.02), 0.98), y))
        r += 1
        y += 0.19
    cands.append(np.array(pts))
    return cands


def _push_move(pts, tri, vi, step, rng):
    """Targeted geometric move: push vertex vi away from the line through the
    other two vertices of the small triangle `tri` (increases its area)."""
    others = [int(x) for x in tri if x != vi]
    j, k2 = others[0], others[1]
    d = pts[k2] - pts[j]
    nrm = np.hypot(d[0], d[1]) + 1e-12
    nd = np.array([-d[1], d[0]]) / nrm
    old = pts[vi]
    side = d[0] * (old[1] - pts[j, 1]) - d[1] * (old[0] - pts[j, 0])
    if side < 0:
        nd = -nd
    new = old + step * nd + rng.normal(0, step * 0.35, 2)
    return np.clip(new, 0.0, 1.0)


def _anneal(pts, iters=3000, t0=0.004, t1=1e-5, rng=None, step0=0.05):
    """Worst-triangle targeted simulated annealing on a SMOOTH objective
    (mean of the K smallest triangle areas) with adaptive step control.

    Adaptive step: acceptance rate over the last 200 proposals is monitored;
    <15% -> halve the step, >50% -> scale by 1.5 (capped at 0.05). This keeps
    the anneal in the productive acceptance band instead of a fixed ladder.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    pts = pts.copy()
    K = 12
    areas = _all_tri_areas(pts)
    smooth = float(np.sort(areas)[:K].mean())
    # best tracked on the TRUE minimum area (hull is ~constant during moves)
    best_min = float(areas.min())
    best_pts = pts.copy()

    step = step0
    temp = t0
    decay = (t1 / t0) ** (1.0 / iters)
    acc_hist = []  # acceptance booleans for adaptive step control

    for it in range(iters):
        areas = _all_tri_areas(pts)
        order = np.argsort(areas)
        tri = _IDX[order[rng.integers(K)]]
        vi = int(tri[rng.integers(3)])
        old = pts[vi].copy()

        if rng.random() < 0.55:
            pts[vi] = _push_move(pts, tri, vi, step, rng)
        else:
            pts[vi] = np.clip(old + rng.normal(0, step, 2), 0.0, 1.0)

        # reject degenerate (duplicate) configurations
        dists = np.linalg.norm(pts - pts[vi], axis=1)
        dists[vi] = 10.0
        if dists.min() < 1e-6:
            pts[vi] = old
            acc_hist.append(0.0)
        else:
            a2 = _all_tri_areas(pts)
            s2 = float(np.sort(a2)[:K].mean())
            if s2 >= smooth or rng.random() < np.exp((s2 - smooth) / max(temp, 1e-12)):
                smooth = s2
                m2 = float(a2.min())
                if m2 > best_min:
                    best_min = m2
                    best_pts = pts.copy()
                acc_hist.append(1.0)
            else:
                pts[vi] = old
                acc_hist.append(0.0)

        temp *= decay

        # adaptive step control from the last 200 proposals
        if len(acc_hist) >= 200 and len(acc_hist) % 200 == 0:
            rate = float(np.mean(acc_hist[-200:]))
            if rate < 0.15:
                step = max(step * 0.5, 0.0015)
            elif rate > 0.50:
                step = min(step * 1.5, 0.05)
            # decay the history so the next window is evaluated fresh
            acc_hist = acc_hist[-200:]

    return best_pts, best_min


def _soft_score(pts, k=10):
    """Soft objective: mean of the k smallest triangle areas (pressure on bottom tail)."""
    ha = _hull_area(pts)
    if ha <= 1e-12:
        return 0.0
    areas = np.partition(_all_tri_areas(pts), k - 1)[:k]
    return areas.mean() / ha


def _polish(pts, rng, iters=1200):
    """Strict local search: accept only targeted moves that increase the true
    minimum triangle area. Uses the same geometric push move as the annealer."""
    pts = pts.copy()
    areas = _all_tri_areas(pts)
    cur_min = float(areas.min())
    step = 0.01
    K = 6
    for it in range(iters):
        order = np.argsort(areas)
        tri = _IDX[order[rng.integers(K)]]
        vi = int(tri[rng.integers(3)])
        old = pts[vi].copy()
        pts[vi] = _push_move(pts, tri, vi, step * 0.7, rng)
        dists = np.linalg.norm(pts - pts[vi], axis=1)
        dists[vi] = 10.0
        if dists.min() < 1e-6:
            pts[vi] = old
            continue
        a2 = _all_tri_areas(pts)
        m2 = float(a2.min())
        if m2 > cur_min:
            cur_min = m2
            areas = a2
        else:
            pts[vi] = old
        if it % 300 == 299:
            step = max(step * 0.9, 0.0005)
    return pts, cur_min


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points maximizing the smallest triangle
    area normalized by convex hull area, via worst-triangle targeted
    simulated annealing with structured multi-starts.
    """
    best_pts, best_val = None, -1.0
    seeds = _seeds()
    for si, cand in enumerate(seeds):
        rng = np.random.default_rng(100 + si)
        p, v = _anneal(cand, iters=4000, rng=rng)
        # short reheat then decay, with adaptive steps restarting small
        p2, v2 = _anneal(p, iters=1500, t0=0.002, t1=1e-6,
                         rng=np.random.default_rng(200 + si), step0=0.02)
        if v2 > v:
            p, v = p2, v2
        p, v = _polish(p, rng)
        if v > best_val:
            best_val, best_pts = v, p
    if best_pts is None:
        best_pts = np.random.default_rng(seed=42).random((13, 2))
        best_pts = np.clip(best_pts, 0.02, 0.98)

    # rescale so the convex hull has unit area (problem is scale-invariant)
    ha = _hull_area(best_pts)
    if ha > 1e-12:
        s = 1.0 / np.sqrt(ha)
        best_pts = (best_pts - best_pts.mean(axis=0)) * s + best_pts.mean(axis=0)
    return best_pts


# EVOLVE-BLOCK-END