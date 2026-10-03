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


def _bottleneck_vertices(pts, tri_idx, k=3):
    """Indices of points participating in the k smallest triangles."""
    p = pts[tri_idx]
    a = p[:, 0]
    v1 = p[:, 1] - a
    v2 = p[:, 2] - a
    cr = np.abs(v1[:, 0]*v2[:, 1] - v1[:, 1]*v2[:, 0])
    order = np.argsort(cr)[:k]
    return np.unique(tri_idx[order])


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
    # 2: 12-gon ring + center (classic Heilbronn-style boundary-heavy layout)
    t12 = np.linspace(0, 2*np.pi, 12, endpoint=False)
    ring = np.column_stack([np.cos(t12), np.sin(t12)])
    seeds.append(np.vstack([ring, [[0.0, 0.0]]]))
    # 3: two rings 7 outer + 6 inner (near 3-fold symmetric)
    t7 = np.linspace(0, 2*np.pi, 7, endpoint=False)
    t6 = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6
    outer = np.column_stack([np.cos(t7), np.sin(t7)])
    inner = 0.45*np.column_stack([np.cos(t6), np.sin(t6)])
    seeds.append(np.vstack([outer, inner]))
    # 4: ellipse (stretched), boundary emphasis
    seeds.append(np.column_stack([1.3*np.cos(t), 0.75*np.sin(t)]))
    # 5: polygon + slight interior perturbations
    s = np.column_stack([np.cos(t), np.sin(t)])
    s += rng.normal(0, 0.08, s.shape)
    seeds.append(s)
    # 6: fully random
    seeds.append(rng.random((n, 2)))

    best_pts, best_score = seeds[0], _score(seeds[0], tri_idx)
    time_limit = 3.0
    t_start = time.time()

    per_seed = time_limit / len(seeds)
    for si, seed_pts in enumerate(seeds):
        deadline = min(t_start + (si + 1) * per_seed, t_start + time_limit)
        pts = seed_pts.copy()
        sc = _score(pts, tri_idx)
        # scale-invariant initial step size
        diam = float(np.ptp(pts, axis=0).max()) + 1e-9
        sigma = 0.08 * diam
        # simulated annealing with bottleneck-focused moves
        T = 0.02 * max(sc, 1e-3)
        since_best = 0
        while time.time() < deadline:
            improved = False
            for _ in range(120):
                cand = pts.copy()
                # target a vertex of the smallest triangles half the time
                if rng.random() < 0.5:
                    bn = _bottleneck_vertices(pts, tri_idx, k=3)
                    i = int(rng.choice(bn))
                else:
                    i = int(rng.integers(n))
                cand[i] += rng.normal(0, sigma, 2)
                csc = _score(cand, tri_idx)
                d = csc - sc
                if d > 0 or rng.random() < np.exp(d / max(T, 1e-12)):
                    pts, sc = cand, csc
                    if sc > best_score + 1e-14:
                        best_score, best_pts = sc, cand.copy()
                        since_best = 0
                        improved = True
            if not improved:
                since_best += 1
                sigma *= 0.75
                if since_best > 3:
                    # reheat to escape local optimum
                    T = max(T, 0.01 * max(sc, 1e-3))
                    sigma = 0.08 * diam
                    since_best = 0
            else:
                sigma = min(sigma * 1.2, 0.2 * diam)
            T = max(T * 0.97, 1e-7)
            if sigma < 1e-6:
                sigma = 1e-6
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