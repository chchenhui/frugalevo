# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations

_N = 13
_idx = np.array(list(combinations(range(_N), 3)), dtype=np.intp)
_I, _J, _K = _idx[:, 0].copy(), _idx[:, 1].copy(), _idx[:, 2].copy()


def _hull_area(P: np.ndarray) -> float:
    """Convex hull area of the point set (Andrew's monotone chain)."""
    pts = sorted(set(map(tuple, np.asarray(P, dtype=float).tolist())))
    if len(pts) < 3:
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
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return 0.0
    s = 0.0
    m = len(hull)
    for i in range(m):
        x1, y1 = hull[i]
        x2, y2 = hull[(i + 1) % m]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def _tri_areas(P: np.ndarray) -> np.ndarray:
    """Vectorized areas of all C(13,3) triangles."""
    x, y = P[:, 0], P[:, 1]
    return 0.5 * np.abs((x[_I] - x[_K]) * (y[_J] - y[_K]) -
                        (y[_I] - y[_K]) * (x[_J] - x[_K]))


def _obj(P: np.ndarray) -> float:
    """min triangle area / hull area (affine-invariant score)."""
    a = _tri_areas(P)
    h = _hull_area(P)
    return float(a.min() / h) if h > 1e-12 else 0.0


def _soft_obj(P: np.ndarray, beta: float = 200.0) -> float:
    """Smooth soft-min surrogate: -log(sum(exp(-beta*a)))/beta approximates min.

    Gives nonzero gradient information on the ~285 non-critical triangles,
    helping annealing escape flat basins of the hard min objective.
    """
    a = _tri_areas(P)
    h = _hull_area(P)
    if h <= 1e-12:
        return 0.0
    m = a.min()
    soft = m - np.log(np.sum(np.exp(-beta * (a - m)))) / beta
    return float(soft / h)


def _sa(P: np.ndarray, iters: int, t0: float, t1: float, rng,
        soft_frac: float = 0.5) -> tuple:
    """Simulated annealing on the 26 coordinates; returns (score, points).

    Continuation scheme: the first `soft_frac` of iterations use the smooth
    soft-min surrogate (beta annealed from 50 to 400) to explore, then switch
    to the hard min objective for final refinement. Best-so-far is always
    tracked with the hard objective.
    """
    P = P.copy()
    f = _obj(P)
    best, bestP = f, P.copy()
    n_soft = int(iters * soft_frac)
    for it in range(iters):
        T = t0 * (t1 / t0) ** (it / max(iters - 1, 1))
        sig = 0.6 * np.sqrt(T) + 1e-5
        j = int(rng.integers(_N))
        old = P[j].copy()
        P[j] = old + sig * rng.standard_normal(2)
        if it < n_soft:
            beta = 50.0 * (400.0 / 50.0) ** (it / max(n_soft - 1, 1))
            f2 = _soft_obj(P, beta)
            # accept on soft score, but track hard score for the record
            hard = _obj(P)
            if f2 >= f or rng.random() < np.exp(min((f2 - f) / T, 0.0)):
                f = f2
                if hard > best:
                    best, bestP = hard, P.copy()
            else:
                P[j] = old
        else:
            f2 = _obj(P)
            if f2 >= f or rng.random() < np.exp(min((f2 - f) / T, 0.0)):
                f = f2
                if f2 > best:
                    best, bestP = f2, P.copy()
            else:
                P[j] = old
    return best, bestP


def _polish(P: np.ndarray, steps) -> tuple:
    """Greedy pattern search: try 8 directions per point with shrinking steps."""
    P = P.copy()
    best = _obj(P)
    dirs = np.array([[1, 0], [-1, 0], [0, 1], [0, -1],
                     [1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=float) / np.sqrt(2)
    for st in steps:
        improved = True
        while improved:
            improved = False
            for j in range(_N):
                for d in dirs:
                    cand = P.copy()
                    cand[j] = P[j] + st * d
                    f2 = _obj(cand)
                    if f2 > best + 1e-12:
                        P, best = cand, f2
                        improved = True
    return best, P


def _seed(kind: str, rng) -> np.ndarray:
    """Structured / random initial configurations."""
    if kind == 'gon13':
        ang = np.linspace(0, 2 * np.pi, 13, endpoint=False)
        return np.stack([np.cos(ang), np.sin(ang)], axis=1)
    if kind == 'gon12c':
        ang = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        ring = np.stack([np.cos(ang), np.sin(ang)], axis=1)
        return np.vstack([ring, [[0.0, 0.0]]])
    if kind == 'ring931':
        a1 = np.linspace(0, 2 * np.pi, 9, endpoint=False)
        outer = np.stack([np.cos(a1), np.sin(a1)], axis=1)
        a2 = np.linspace(0, 2 * np.pi, 3, endpoint=False) + np.pi / 9
        inner = 0.5 * np.stack([np.cos(a2), np.sin(a2)], axis=1)
        return np.vstack([outer, inner, [[0.0, 0.0]]])
    # random points in the unit disk
    r = np.sqrt(rng.random(_N))
    t = rng.random(_N) * 2 * np.pi
    return np.stack([r * np.cos(t), r * np.sin(t)], axis=1)


def heilbronn_convex13() -> np.ndarray:
    """
    Construct 13 points maximizing the minimum triangle area normalized by
    the convex hull area.

    Approach: the score is affine-invariant, so we optimize the ratio directly.
    Multi-start simulated annealing (vectorized evaluation of all 286 triangle
    areas) from structured seeds (regular 13-gon, 12-gon+center, 9+3+1 double
    ring, random disks), followed by greedy pattern-search polishing and a
    low-temperature re-annealing refinement of the global best. Fully
    deterministic (fixed seeds, fixed iteration counts).
    """
    best_f, best_P = -1.0, None
    seeds = ['gon12c', 'gon13', 'ring931'] + ['rand'] * 9
    for s_i, kind in enumerate(seeds):
        rng = np.random.default_rng(1000 + s_i)
        P0 = _seed(kind, rng)
        if kind == 'rand':
            P0 += 0.03 * rng.standard_normal((13, 2))
        f, P = _sa(P0, 50000, 4e-3, 2e-5, rng, soft_frac=0.5)
        f, P = _polish(P, [2e-2, 5e-3, 1e-3, 2e-4, 5e-5])
        if f > best_f:
            best_f, best_P = f, P
    # low-temperature refinement around the global best (hard objective only)
    for r_i in range(5):
        rng = np.random.default_rng(777 + r_i)
        f, P = _sa(best_P, 60000, 6e-4, 1e-5, rng, soft_frac=0.0)
        f, P = _polish(P, [1e-2, 3e-3, 7e-4, 1e-4, 2e-5, 5e-6, 1e-6])
        if f > best_f:
            best_f, best_P = f, P
    # guard against pathological results
    if best_P is None or not np.all(np.isfinite(best_P)) or best_f <= 0:
        ang = np.linspace(0, 2 * np.pi, 13, endpoint=False)
        best_P = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    # rescale so the convex hull has exactly unit area
    h = _hull_area(best_P)
    if h > 1e-12:
        best_P = best_P / np.sqrt(h)
    return np.ascontiguousarray(best_P, dtype=float)


# EVOLVE-BLOCK-END
