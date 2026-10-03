# EVOLVE-BLOCK-START
import numpy as np
import time


def _convex_hull_area(pts):
    """Monotone chain convex hull; returns hull area (0 if degenerate)."""
    P = sorted(map(tuple, np.round(pts, 12)))
    P = list(dict.fromkeys(P))
    if len(P) < 3:
        return 0.0
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower = []
    for p in P:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(P):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return 0.0
    H = np.array(hull)
    x, y = H[:, 0], H[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _min_triangle_area(pts, tri_idx):
    p = pts[tri_idx]                      # (286, 3, 2)
    a = p[:, 0]
    v1 = p[:, 1] - a
    v2 = p[:, 2] - a
    cross = np.abs(v1[:, 0]*v2[:, 1] - v1[:, 1]*v2[:, 0])
    return cross.min() * 0.5


def _score(pts, tri_idx):
    ha = _convex_hull_area(pts)
    if ha <= 1e-12:
        return 0.0
    return _min_triangle_area(pts, tri_idx) / ha


def heilbronn_convex13() -> np.ndarray:
    n = 13
    rng = np.random.default_rng(seed=42)
    from itertools import combinations
    tri_idx = np.array(list(combinations(range(n), 3)))

    # --- Boundary-biased initializations ---
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    seeds = []
    # 1: regular 13-gon on circle
    seeds.append(np.column_stack([np.cos(t), np.sin(t)]))
    # 2: ellipse (stretched), boundary emphasis
    seeds.append(np.column_stack([1.3*np.cos(t), 0.75*np.sin(t)]))
    # 3: polygon + slight interior perturbations
    s = np.column_stack([np.cos(t), np.sin(t)])
    s += rng.normal(0, 0.08, s.shape)
    seeds.append(s)
    # 4: fully random
    seeds.append(rng.random((n, 2)))

    # extra structural seed: 12-gon ring + center (strong prior for n=13)
    t12 = np.linspace(0, 2*np.pi, n - 1, endpoint=False)
    ring = np.column_stack([np.cos(t12), np.sin(t12)])
    seeds.append(np.vstack([ring, [[0.0, 0.0]]]))

    best_pts, best_score = seeds[0], _score(seeds[0], tri_idx)
    time_limit = 4.5
    t_start = time.time()

    def _smallest_tri_idx(pts):
        p = pts[tri_idx]
        a = p[:, 0]
        v1 = p[:, 1] - a
        v2 = p[:, 2] - a
        cr = np.abs(v1[:, 0]*v2[:, 1] - v1[:, 1]*v2[:, 0])
        return tri_idx[int(np.argmin(cr))]

    def _climb(pts, sc, sigma0, deadline):
        """Adaptive-sigma strict-improvement hill climb with
        bottleneck-triangle-targeted moves (~50% of proposals)."""
        sigma = sigma0
        while time.time() < deadline:
            improved = False
            for _ in range(200):
                cand = pts.copy()
                if rng.random() < 0.5:
                    # target a vertex of the current bottleneck triangle
                    tri = _smallest_tri_idx(pts)
                    i = int(rng.choice(tri))
                else:
                    i = int(rng.integers(n))
                cand[i] += rng.normal(0, sigma, 2)
                csc = _score(cand, tri_idx)
                if csc > sc + 1e-12:
                    pts, sc = cand, csc
                    improved = True
            if improved:
                sigma = min(sigma * 1.3, 0.3)
            else:
                sigma *= 0.6
            if sigma < 1e-5:
                break
        return pts, sc

    per_seed = time_limit / (2 * len(seeds))
    for k, seed_pts in enumerate(seeds):
        deadline = min(t_start + (k + 1) * per_seed, t_start + time_limit)
        if time.time() - t_start > time_limit:
            break
        pts, sc = _climb(seed_pts.copy(), _score(seed_pts, tri_idx),
                        0.15, deadline)
        if sc > best_score:
            best_score, best_pts = sc, pts.copy()

    # Second phase: basin-hopping with targeted kicks.
    # Kicking ALL coordinates at small sigma preserves the incumbent basin
    # (the perturbation averages out). Instead kick only 3-4 vertices at
    # larger sigma to force real basin transitions, and occasionally
    # rigidly translate the bottleneck triangle's 3 vertices (a correlated
    # move single-vertex noise cannot explore).
    while time.time() - t_start < time_limit - 0.05:
        kick = best_pts.copy()
        u = rng.random()
        if u < 0.20:
            # coordinated rigid translation of the bottleneck triangle
            tri = _smallest_tri_idx(best_pts)
            shift = rng.normal(0, 0.13, 2)
            for vi in tri:
                kick[vi] = kick[vi] + shift
        else:
            # kick a small subset (3-4 vertices) at larger sigma
            k_vert = int(rng.integers(3, 5))
            idxs = rng.choice(n, size=k_vert, replace=False)
            kick[idxs] += rng.normal(0, 0.13, (k_vert, 2))
        deadline = min(time.time() + 0.4, t_start + time_limit - 0.05)
        pts, sc = _climb(kick, _score(kick, tri_idx), 0.10, deadline)
        if sc > best_score:
            best_score, best_pts = sc, pts.copy()

    # Deterministic reprojection/clamp: nothing needed (points already valid),
    # but ensure finite values.
    best_pts = np.asarray(best_pts, dtype=float)
    if not np.all(np.isfinite(best_pts)):
        t = np.linspace(0, 2*np.pi, n, endpoint=False)
        best_pts = np.column_stack([np.cos(t), np.sin(t)])
    return best_pts


# EVOLVE-BLOCK-END