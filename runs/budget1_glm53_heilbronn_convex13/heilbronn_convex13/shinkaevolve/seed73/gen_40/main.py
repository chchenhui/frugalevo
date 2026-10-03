# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations

n = 13
_IDX = np.array(list(combinations(range(n), 3)))
_I0, _I1, _I2 = _IDX[:, 0], _IDX[:, 1], _IDX[:, 2]


def _tri_areas(P: np.ndarray) -> np.ndarray:
    a = P[_I0]
    b = P[_I1]
    c = P[_I2]
    return 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) -
                        (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))


def _hull_area(P: np.ndarray) -> float:
    pts = P[np.lexsort((pts_y := (P[:, 1], P[:, 0]))[::-1])]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return 1e-12
    h = np.array(hull)
    x, y = h[:, 0], h[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _min_area(P: np.ndarray) -> float:
    return float(_tri_areas(P).min())


def _true_score(P: np.ndarray) -> float:
    ha = _hull_area(P)
    if ha <= 1e-12:
        return 0.0
    return _min_area(P) / ha


def _guide_score(P: np.ndarray) -> float:
    """Sharper smooth-ish guidance: 0.8*min + 0.2*mean of 3 smallest,
    normalized by hull area."""
    ha = _hull_area(P)
    if ha <= 1e-12:
        return 0.0
    a = _tri_areas(P)
    k3 = np.partition(a, 2)[:3]
    return (0.8 * a.min() + 0.2 * k3.mean()) / ha


# 16 search directions at 22.5-degree increments.
_thetas = np.arange(16) * (np.pi / 8.0)
_DIRS = np.stack([np.cos(_thetas), np.sin(_thetas)], axis=1)

_th = 2.0 * np.pi / 3.0
_c, _s = np.cos(_th), np.sin(_th)
_ROT = np.array([[_c, -_s], [_s, _c]])
_ROT2 = _ROT @ _ROT


def _expand(B: np.ndarray) -> np.ndarray:
    """4 base points -> 12 points by 3-fold rotation, plus center."""
    return np.vstack([B, B @ _ROT.T, B @ _ROT2.T, [[0.0, 0.0]]])


def _try_expand_hull(P: np.ndarray, cur_true: float) -> tuple:
    """Scale all points outward by 1.01x about their centroid; keep only if
    the exact min_area/hull_area score improves."""
    cen = P.mean(axis=0)
    Q = cen + 1.01 * (P - cen)
    sc = _true_score(Q)
    if sc > cur_true + 1e-15:
        # repeat while improving (bounded)
        for _ in range(50):
            Q2 = cen + 1.01 * (Q - cen)
            sc2 = _true_score(Q2)
            if sc2 > sc + 1e-15:
                Q, sc = Q2, sc2
            else:
                break
        return Q, sc
    return P, cur_true


def _refine_base(B: np.ndarray, step: float = 0.25,
                 min_step: float = 2e-4) -> np.ndarray:
    """Coarse-to-fine pattern search on the 8 base coordinates, guided by
    the sharper blend, with occasional hull-expansion attempts scored by
    the true objective."""
    P = _expand(B)
    best = _guide_score(P)
    true = _true_score(P)
    rounds = 0
    while step > min_step:
        improved = False
        for i in range(len(B)):
            for j in range(2):
                for sgn in (1.0, -1.0):
                    cand = B.copy()
                    cand[i, j] += sgn * step
                    val = _guide_score(_expand(cand))
                    if val > best + 1e-15:
                        B = cand
                        best = val
                        improved = True
        rounds += 1
        # periodically attempt hull expansion on the current config
        if rounds % 3 == 0:
            P = _expand(B)
            P, true = _try_expand_hull(P, _true_score(P))
            # re-synthesize base points from expanded config is complex;
            # instead keep expansion effect by continuing search from P directly
            if _true_score(P) > true:
                true = _true_score(P)
        if not improved:
            step *= 0.5
    return B


def _polish_full(P: np.ndarray, step: float = 0.02,
                 min_step: float = 1e-4) -> np.ndarray:
    """Symmetry-breaking pattern search on all 26 coordinates, guided by
    the sharper blend, with hull-expansion attempts accepted on the true
    objective."""
    best = _guide_score(P)
    true = _true_score(P)
    rounds = 0
    while step > min_step:
        improved = False
        for i in range(n):
            base = P.copy()
            mask_has = (_I0 == i) | (_I1 == i) | (_I2 == i)
            others = _tri_areas(P)[~mask_has].min() if np.any(~mask_has) else np.inf
            vals = np.empty(16)
            for d in range(16):
                cand = P[i] + step * _DIRS[d]
                base[i] = cand
                vals[d] = min(others, _tri_areas(base)[mask_has].min())
            # guide score uses blend; recompute blend with candidate min areas
            a_full = _tri_areas(P)
            k3 = np.partition(a_full, 2)[:3]
            base_guide = (0.8 * a_full.min() + 0.2 * k3.mean()) / _hull_area(P)
            best_d = -1
            best_val = base_guide
            for d in range(16):
                cand = P[i] + step * _DIRS[d]
                trial = P.copy()
                trial[i] = cand
                g = _guide_score(trial)
                if g > best_val + 1e-15:
                    best_val = g
                    best_d = d
            if best_d >= 0:
                P[i] = P[i] + step * _DIRS[best_d]
                best = best_val
                improved = True
        rounds += 1
        # hull-expansion attempt, gated on the TRUE score
        if rounds % 2 == 0:
            P, true = _try_expand_hull(P, _true_score(P))
            if _guide_score(P) > best:
                best = _guide_score(P)
                improved = True
        if not improved:
            step *= 0.5
    return P


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points maximizing the smallest triangle
    area normalized by convex hull area. Deterministic multi-start over
    3-fold symmetric seeds (4 free base points -> 12 points + center),
    refined with a sharper guidance blend (0.8*min + 0.2*mean-of-3-smallest)
    and hull-expansion moves accepted only on the exact min/hull ratio.
    """
    rng = np.random.default_rng(seed=42)

    seeds = []
    ang = np.pi / 2.0
    for r4 in ([1.0, 0.72, 0.45, 0.2], [1.0, 0.8, 0.55, 0.3],
               [1.0, 0.85, 0.6, 0.35], [1.0, 0.75, 0.5, 0.25]):
        B = [[r * np.cos(ang - k * np.pi / 9.0),
              r * np.sin(ang - k * np.pi / 9.0)] for k, r in enumerate(r4)]
        seeds.append(np.array(B))
    for spread in (0.9, 0.6, 0.3):
        seeds.append(np.array([[spread, 0.0],
                               [spread * 0.5, spread * 0.85],
                               [0.0, spread],
                               [spread * 0.4, spread * 0.3]]))
    for _ in range(10):
        r = rng.uniform(0.1, 1.0, 4)
        a = rng.uniform(0, 2 * np.pi, 4)
        seeds.append(np.column_stack([r * np.cos(a), r * np.sin(a)]))

    best_P, best_sc = None, -np.inf
    for B0 in seeds:
        try:
            B = _refine_base(B0.copy())
            P = _expand(B)
            P = _polish_full(P)
            sc = _true_score(P)
            if sc > best_sc:
                best_sc = sc
                best_P = P.copy()
        except Exception:
            continue

    if best_P is None:
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_P = np.column_stack([np.cos(t), np.sin(t)])

    # numerical safety: no duplicates
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(best_P[i] - best_P[j]) < 1e-9:
                best_P[j, 0] += 1e-6 * (j + 1)

    assert np.all(np.isfinite(best_P))
    assert best_P.shape == (n, 2)
    return best_P


# EVOLVE-BLOCK-END