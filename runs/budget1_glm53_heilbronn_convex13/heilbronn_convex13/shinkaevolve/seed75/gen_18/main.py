# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations

TRI_IDX = np.array(list(combinations(range(13), 3)))


def _hull_area(pts):
    """Shoelace area of convex hull via monotone chain."""
    P = sorted(map(tuple, pts))
    P = list(dict.fromkeys(P))
    if len(P) < 3:
        return 0.0
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lo, up = [], []
    for p in P:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(P):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    hull = lo[:-1] + up[:-1]
    if len(hull) < 3:
        return 0.0
    H = np.array(hull)
    x, y = H[:, 0], H[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _min_tri(pts):
    p = pts[TRI_IDX]
    a = p[:, 0]
    v1 = p[:, 1] - a
    v2 = p[:, 2] - a
    return 0.5 * np.abs(v1[:, 0]*v2[:, 1] - v1[:, 1]*v2[:, 0]).min()


def _score(pts):
    ha = _hull_area(pts)
    if ha <= 1e-12:
        return 0.0
    return _min_tri(pts) / ha


def _bottleneck_vertex(pts):
    """Index of a vertex participating in the smallest triangle."""
    p = pts[TRI_IDX]
    a = p[:, 0]
    v1 = p[:, 1] - a
    v2 = p[:, 2] - a
    cr = np.abs(v1[:, 0]*v2[:, 1] - v1[:, 1]*v2[:, 0])
    k = int(np.argmin(cr))
    return int(TRI_IDX[k][0])


def _seeds(rng):
    n = 13
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    seeds = []
    # regular 13-gon
    seeds.append(np.column_stack([np.cos(t), np.sin(t)]))
    # ellipse
    seeds.append(np.column_stack([1.3*np.cos(t), 0.75*np.sin(t)]))
    # 12-gon ring + center
    t12 = np.linspace(0, 2*np.pi, 12, endpoint=False)
    ring = np.column_stack([np.cos(t12), np.sin(t12)])
    seeds.append(np.vstack([ring, [[0.0, 0.0]]]))
    # outer ring 7 + inner ring 6 (3-fold style)
    t7 = np.linspace(0, 2*np.pi, 7, endpoint=False)
    t6 = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6
    outer = np.column_stack([np.cos(t7), np.sin(t7)])
    inner = 0.45*np.column_stack([np.cos(t6), np.sin(t6)])
    seeds.append(np.vstack([outer, inner]))
    # random
    seeds.append(rng.random((n, 2)))
    return seeds


def _bottleneck_tri(pts):
    """Indices (i,j,k) of the smallest-area triangle."""
    p = pts[TRI_IDX]
    a = p[:, 0]
    v1 = p[:, 1] - a
    v2 = p[:, 2] - a
    cr = np.abs(v1[:, 0]*v2[:, 1] - v1[:, 1]*v2[:, 0])
    return TRI_IDX[int(np.argmin(cr))]


def _directed_candidate(pts, rng, sigma):
    """Directed move: push a vertex of the bottleneck triangle away from
    the opposite edge (increases that triangle's altitude/area)."""
    cand = pts.copy()
    if rng.random() < 0.25:
        # occasional random move for exploration
        i = int(rng.integers(13))
        cand[i] += rng.normal(0, sigma, 2)
        return cand
    tri = _bottleneck_tri(pts)
    i = int(rng.choice(tri))
    other = [j for j in tri if j != i]
    d = pts[other[1]] - pts[other[0]]
    nd = np.linalg.norm(d)
    if nd < 1e-9:
        cand[i] += rng.normal(0, sigma, 2)
        return cand
    perp = np.array([-d[1], d[0]]) / nd
    # sign the perpendicular so the move increases |cross|
    v = pts[i] - pts[other[0]]
    if np.dot(perp, v) < 0:
        perp = -perp
    cand[i] = pts[i] + perp * abs(rng.normal(0, sigma)) + rng.normal(0, 0.3*sigma, 2)
    return cand


def _anneal(pts, rng, deadline, T0=0.01, T_min=1e-6):
    """Simulated annealing with bottleneck-directed proposals."""
    pts = pts.copy()
    sc = _score(pts)
    best, best_sc = pts.copy(), sc
    T = T0
    sigma = 0.1
    since_best = 0
    while time.time() < deadline:
        for _ in range(150):
            cand = _directed_candidate(pts, rng, sigma)
            csc = _score(cand)
            d = csc - sc
            if d > 0 or (T > 0 and rng.random() < np.exp(d / max(T, 1e-12))):
                pts, sc = cand, csc
                if sc > best_sc + 1e-14:
                    best, best_sc = cand.copy(), sc
                    since_best = 0
        since_best += 1
        T = max(T * 0.92, T_min)
        if since_best > 4:
            sigma = max(sigma * 0.7, 1e-4)
            since_best = 0
        elif since_best == 0:
            sigma = min(sigma * 1.25, 0.3)
    return best, best_sc


def _shrink_polish(pts):
    """Deterministic isotropic shrink line search: since the score is
    min_area/hull_area, shrinking the configuration about the hull
    centroid can raise the ratio; keep only improvements."""
    best, best_sc = pts.copy(), _score(pts)
    c = pts.mean(axis=0)
    for f in (0.99, 0.97, 0.94, 0.90, 0.85, 0.80):
        cand = c + f * (pts - c)
        sc = _score(cand)
        if sc > best_sc + 1e-15:
            best, best_sc = cand, sc
    return best, best_sc


def heilbronn_convex13() -> np.ndarray:
    rng = np.random.default_rng(seed=42)
    t_start = time.time()
    time_limit = 3.0

    best_pts, best_sc = None, -1.0
    seeds = _seeds(rng)
    per_seed = time_limit / len(seeds)
    for k, s in enumerate(seeds):
        deadline = min(t_start + (k + 1) * per_seed, t_start + time_limit)
        pts, sc = _anneal(s, rng, deadline)
        if sc > best_sc:
            best_sc, best_pts = sc, pts.copy()

    # Basin-hopping restarts from the best configuration until time runs out
    while time.time() - t_start < time_limit - 0.05:
        kick = best_pts + rng.normal(0, 0.05, best_pts.shape)
        deadline = min(time.time() + 0.25, t_start + time_limit)
        pts, sc = _anneal(kick, rng, deadline)
        if sc > best_sc:
            best_sc, best_pts = sc, pts.copy()

    # Deterministic shrink polish on the final configuration
    best_pts, best_sc = _shrink_polish(best_pts)

    best_pts = np.asarray(best_pts, dtype=float)
    if not np.all(np.isfinite(best_pts)) or best_pts.shape != (13, 2):
        t = np.linspace(0, 2*np.pi, 13, endpoint=False)
        best_pts = np.column_stack([np.cos(t), np.sin(t)])
    return best_pts


# EVOLVE-BLOCK-END