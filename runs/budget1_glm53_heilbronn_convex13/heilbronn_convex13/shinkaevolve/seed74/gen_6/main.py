# EVOLVE-BLOCK-START
import numpy as np


def _min_triangle_area(points):
    """Area of smallest triangle among all triples, via vectorized cross products."""
    n = len(points)
    best = np.inf
    for i in range(n - 2):
        # vectors from point i to all later points
        v = points[i + 1:] - points[i]
        # cross products between all pairs of later vectors
        # area = 0.5 * |cross|
        x = v[:, 0]
        y = v[:, 1]
        # cross[i+1+a, i+1+b] = x[a]*y[b] - y[a]*x[b]
        cross = x[:, None] * y[None, :] - y[:, None] * x[None, :]
        # take upper triangle (a < b)
        m = cross.shape[0]
        if m < 2:
            continue
        iu = np.triu_indices(m, k=1)
        vals = np.abs(cross[iu])
        if vals.size:
            best = min(best, vals.min())
    return 0.5 * best


def heilbronn_convex13() -> np.ndarray:
    n = 13
    rng = np.random.default_rng(seed=42)

    # Seed: regular 13-gon on unit circle, slightly perturbed deterministically
    angles = np.arange(n) * (2 * np.pi / n)
    points = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    points += rng.normal(0, 0.02, points.shape)

    cur_area = _min_triangle_area(points)
    best_points = points.copy()
    best_area = cur_area

    # Simulated annealing: perturb points, keep inside large convex region (disk),
    # accept per Metropolis rule. Scale-invariance: normalize at the end.
    T0, T1 = 0.05, 1e-5
    iters = 60000
    for it in range(iters):
        T = T0 * (T1 / T0) ** (it / iters)
        idx = rng.integers(n)
        step = rng.normal(0, T, 2)
        cand = points.copy()
        cand[idx] += step
        # keep point near unit disk
        r = np.hypot(*cand[idx])
        if r > 1.0:
            cand[idx] /= r
        a = _min_triangle_area(cand)
        if a >= cur_area or rng.random() < np.exp((a - cur_area) / max(T, 1e-12)):
            points = cand
            cur_area = a
            if a > best_area:
                best_area = a
                best_points = cand.copy()

    # Normalize: translate centroid to origin, scale so convex hull has unit area.
    pts = best_points - best_points.mean(axis=0)
    # hull area via gift wrapping (small n, simple)
    P = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    hull = []
    for p in P:
        while len(hull) >= 2 and cross(hull[-2], hull[-1], p) <= 0:
            hull.pop()
        hull.append(p)
    lower = []
    for p in P[::-1]:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    hull = np.array(hull[:-1] + lower[:-1])
    area = 0.0
    for i in range(len(hull)):
        o, a, b = hull[i], hull[(i+1) % len(hull)], hull[(i+2) % len(hull)]
    h = hull
    area = 0.5 * abs(np.sum(h[:, 0] * np.roll(h[:, 1], -1) - np.roll(h[:, 0], -1) * h[:, 1]))
    pts = pts / np.sqrt(area)
    return pts


# EVOLVE-BLOCK-END
