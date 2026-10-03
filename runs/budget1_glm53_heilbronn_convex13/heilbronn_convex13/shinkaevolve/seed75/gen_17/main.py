# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations

TRI = np.array(list(combinations(range(13), 3)))
N = 13


def hull_area(pts):
    """Shoelace area of convex hull via monotone chain (vectorized-ish)."""
    P = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    if len(P) < 3:
        return 0.0
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lo, up = [], []
    for p in P:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in P[::-1]:
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    H = np.array(lo[:-1] + up[:-1])
    if len(H) < 3:
        return 0.0
    x, y = H[:, 0], H[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def tri_areas(pts):
    p = pts[TRI]
    a = p[:, 0]
    v1 = p[:, 1] - a
    v2 = p[:, 2] - a
    return 0.5 * np.abs(v1[:, 0]*v2[:, 1] - v1[:, 1]*v2[:, 0])


def exact_score(pts):
    ha = hull_area(pts)
    if ha <= 1e-12:
        return 0.0
    return tri_areas(pts).min() / ha


def points_from_params(theta, ab):
    """Angles -> points on ellipse with semi-axes ab=(a,b)."""
    return np.column_stack([ab[0]*np.cos(theta), ab[1]*np.sin(theta)])


def smooth_objective(theta, ab, p):
    """Soft-min of triangle areas minus log hull area (log domain)."""
    pts = points_from_params(theta, ab)
    # hull area of points on ellipse = polygon area (exact, sorted by angle)
    order = np.argsort(np.mod(theta, 2*np.pi))
    P = pts[order]
    x, y = P[:, 0], P[:, 1]
    ha = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    if ha <= 1e-12:
        return -1e9, pts, ha
    ar = tri_areas(pts)
    # soft min in log domain for numerical stability
    la = np.log(ar + 1e-300)
    m = la.min()
    softmin = m - np.log(np.sum(np.exp(p*(la - m)))) / p
    return softmin - np.log(ha), pts, ha


def numeric_grad(theta, ab, p, h=1e-5):
    base, _, _ = smooth_objective(theta, ab, p)
    g = np.zeros(15)
    for i in range(13):
        t2 = theta.copy(); t2[i] += h
        g[i] = (smooth_objective(t2, ab, p)[0] - base) / h
    for i in range(2):
        a2 = ab.copy(); a2[i] += h
        g[13+i] = (smooth_objective(theta, a2, p)[0] - base) / h
    return g


def adam_optimize(theta0, ab0, deadline, rng):
    theta = theta0.copy()
    ab = ab0.copy()
    m = np.zeros(15); v = np.zeros(15)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lr = 0.02
    p = 4.0
    step = 0
    best_pts, best_sc = None, -1e18
    while time.time() < deadline:
        step += 1
        g = numeric_grad(theta, ab, p)
        m = b1*m + (1-b1)*g
        v = b2*v + (1-b2)*g*g
        mh = m / (1 - b1**step)
        vh = v / (1 - b2**step)
        upd = lr * mh / (np.sqrt(vh) + eps)
        theta += upd[:13]
        ab = np.maximum(ab + upd[13:], 0.05)
        # exact score tracking
        pts = points_from_params(theta, ab)
        sc = exact_score(pts)
        if sc > best_sc:
            best_sc = sc
            best_pts = pts.copy()
        # anneal: sharpen softmin, decay lr
        if step % 60 == 0:
            p = min(p * 1.5, 400.0)
            lr *= 0.6
        if lr < 1e-5:
            break
    return best_pts, best_sc


def heilbronn_convex13() -> np.ndarray:
    rng = np.random.default_rng(seed=42)
    t_start = time.time()
    time_limit = 2.0

    # deterministic angle seeds
    base = np.linspace(0, 2*np.pi, N, endpoint=False)
    seeds = [
        (base.copy(), np.array([1.0, 1.0])),
        (base + rng.normal(0, 0.15, N), np.array([1.0, 1.0])),
        (base + rng.normal(0, 0.15, N), np.array([1.4, 0.7])),
        (base + rng.normal(0, 0.3, N), np.array([1.2, 0.85])),
        (base + rng.normal(0, 0.5, N), np.array([1.0, 1.0])),
        (np.sort(rng.uniform(0, 2*np.pi, N)), np.array([1.0, 1.0])),
        (base + np.pi/N, np.array([0.8, 1.25])),
    ]

    best_pts, best_sc = None, -1.0
    per = time_limit / len(seeds)
    for k, (th, ab) in enumerate(seeds):
        deadline = min(t_start + (k+1)*per, t_start + time_limit)
        if time.time() - t_start > time_limit - 0.02:
            break
        pts, sc = adam_optimize(th, ab, deadline, rng)
        if sc > best_sc:
            best_sc, best_pts = sc, pts.copy()

    # fine polish: coordinate descent on exact score from best config
    if best_pts is not None:
        theta = np.arctan2(best_pts[:, 1], best_pts[:, 0]) + \
                (best_pts[:, 0] < 0) * np.pi
        # recover ellipse axes via PCA
        C = best_pts.T @ best_pts
        w, V = np.linalg.eigh(C)
        R = V @ np.diag(np.sqrt(np.maximum(w, 1e-12)))
        pts = best_pts @ np.linalg.inv(R)
        theta = np.mod(np.arctan2(pts[:, 1], pts[:, 0]), 2*np.pi)
        a = np.max(np.hypot(pts[:, 0], pts[:, 1]) * np.abs(np.cos(theta)))
        b = np.max(np.hypot(pts[:, 0], pts[:, 1]) * np.abs(np.sin(theta)))
        deadline = t_start + time_limit
        pts2, sc2 = adam_optimize(theta, np.array([a, b]), deadline, rng)
        if sc2 > best_sc:
            best_sc, best_pts = sc2, pts2.copy()

    if best_pts is None or not np.all(np.isfinite(best_pts)):
        t = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(t), np.sin(t)])
    return np.asarray(best_pts, dtype=float)


# EVOLVE-BLOCK-END
