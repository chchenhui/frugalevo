# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations

_N = 13
_TRI = np.array(list(combinations(range(_N), 3)))


def _hull_area(pts):
    idx = np.argsort(pts[:, 0], kind="stable")
    P = pts[idx]
    # remove duplicates
    keep = np.ones(len(P), dtype=bool)
    keep[1:] = np.any(P[1:] != P[:-1], axis=1)
    P = P[keep]
    if len(P) < 3:
        return 0.0

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    def half(seq):
        h = []
        for p in seq:
            while len(h) >= 2 and cross(h[-2], h[-1], p) <= 0:
                h.pop()
            h.append(p)
        return h

    lower = half(P)
    upper = half(P[::-1])
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return 0.0
    H = np.array(hull)
    x, y = H[:, 0], H[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _areas(pts):
    p = pts[_TRI]
    a = p[:, 0]
    v1 = p[:, 1] - a
    v2 = p[:, 2] - a
    return np.abs(v1[:, 0]*v2[:, 1] - v1[:, 1]*v2[:, 0]) * 0.5


def _score(pts):
    ha = _hull_area(pts)
    if ha <= 1e-12:
        return 0.0
    return _areas(pts).min() / ha


def _rescale(pts):
    ha = _hull_area(pts)
    if ha > 1e-12:
        pts = pts / np.sqrt(ha)
    return pts


def _make_inits(rng):
    inits = []
    cx, cy = 0.0, 0.0
    t = np.linspace(0, 2*np.pi, _N, endpoint=False)
    # regular 13-gon
    inits.append(np.column_stack([np.cos(t), np.sin(t)]))
    # stretched rings
    inits.append(np.column_stack([1.2*np.cos(t), 0.8*np.sin(t)]))
    # center + 3-ring + 9-ring (3-fold symmetry)
    pts = [(0.0, 0.0)]
    for i in range(3):
        a = 2*np.pi*i/3
        pts.append((0.30*np.cos(a), 0.30*np.sin(a)))
    for i in range(9):
        a = 2*np.pi*i/9
        pts.append((np.cos(a), np.sin(a)))
    inits.append(np.array(pts, dtype=float))
    # square-hugging ring
    pts = []
    for i in range(12):
        a = 2*np.pi*i/12
        r = 1.0 / max(abs(np.cos(a)), abs(np.sin(a)))
        pts.append((r*np.cos(a), r*np.sin(a)))
    pts.append((0.0, 0.0))
    inits.append(np.array(pts, dtype=float))
    # random + small jittered ring
    inits.append(rng.random((_N, 2)) - 0.5)
    s = np.column_stack([np.cos(t), np.sin(t)]) + rng.normal(0, 0.05, (_N, 2))
    inits.append(s)
    return inits


def heilbronn_convex13() -> np.ndarray:
    rng = np.random.default_rng(seed=42)
    t_start = time.time()
    time_limit = 2.4

    inits = _make_inits(rng)
    best_pts, best_sc = None, -1.0

    for init in inits:
        if time.time() - t_start > time_limit:
            break
        pts = _rescale(np.asarray(init, dtype=float).copy())
        sc = _score(pts)
        sigma = 0.05
        t_seed = time.time()
        seed_budget = (time_limit - (t_seed - t_start)) / max(1, len(inits))

        while time.time() - t_start < time_limit and time.time() - t_seed < seed_budget and sigma > 1e-6:
            improved = False
            for _ in range(40):
                cand = pts.copy()
                if rng.random() < 0.5:
                    # bottleneck-targeted move: perturb a vertex of min triangle
                    k = int(np.argmin(_areas(pts)))
                    i, j, l = _TRI[k]
                    v = int(rng.choice([i, j, l]))
                    cand[v] += rng.normal(0, sigma, 2)
                else:
                    k = int(rng.integers(1, 4))
                    idxs = rng.choice(_N, size=k, replace=False)
                    cand[idxs] += rng.normal(0, sigma, (k, 2))
                cand = _rescale(cand)
                csc = _score(cand)
                if csc > sc + 1e-12:
                    pts, sc = cand, csc
                    improved = True
            if not improved:
                sigma *= 0.5
            else:
                sigma *= 1.15
                sigma = min(sigma, 0.05)

        if sc > best_sc:
            best_sc = sc
            best_pts = pts.copy()

    best_pts = np.asarray(best_pts, dtype=float)
    if not np.all(np.isfinite(best_pts)):
        t = np.linspace(0, 2*np.pi, _N, endpoint=False)
        best_pts = np.column_stack([np.cos(t), np.sin(t)])
    return best_pts


# EVOLVE-BLOCK-END
