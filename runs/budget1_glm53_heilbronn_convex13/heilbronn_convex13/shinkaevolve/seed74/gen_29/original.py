# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations

_N = 13
_IDX = np.array(list(combinations(range(_N), 3)))  # (286, 3)
_I0, _I1, _I2 = _IDX[:, 0], _IDX[:, 1], _IDX[:, 2]


def _all_tri_areas(pts):
    a = pts[_I0]
    b = pts[_I1]
    c = pts[_I2]
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    return 0.5 * np.abs(cross)


def _hull_area(pts):
    # Convex hull area via monotone chain on numpy input.
    P = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    # dedupe
    keep = np.ones(len(P), dtype=bool)
    keep[1:] = np.any(P[1:] != P[:-1], axis=1)
    P = P[keep]
    if len(P) < 3:
        return 0.0

    def build(seq):
        h = []
        for pt in seq:
            while len(h) >= 2:
                o, x = h[-2], h[-1]
                if (x[0]-o[0])*(pt[1]-o[1]) - (x[1]-o[1])*(pt[0]-o[0]) <= 0:
                    h.pop()
                else:
                    break
            h.append(pt)
        return h

    lower = build(P)
    upper = build(P[::-1])
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return 0.0
    h = np.asarray(hull)
    x, y = h[:, 0], h[:, 1]
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    return abs(np.sum(x * y2 - x2 * y)) / 2.0


def _score(pts, ha=None):
    if ha is None:
        ha = _hull_area(pts)
    if ha <= 1e-12:
        return 0.0, 0.0
    areas = _all_tri_areas(pts)
    return areas.min() / ha, ha


def _seeds():
    cands = []
    t = np.linspace(0, 2 * np.pi, 7)[:-1]
    # 1) hexagonal rings: 6 outer + 6 inner + center
    outer = np.stack([0.5 + 0.47 * np.cos(t), 0.5 + 0.47 * np.sin(t)], axis=1)
    inner = np.stack([0.5 + 0.26 * np.cos(t + np.pi / 6),
                      0.5 + 0.26 * np.sin(t + np.pi / 6)], axis=1)
    cands.append(np.vstack([outer, inner, [[0.5, 0.5]]]))
    # 2) staggered hexagons
    inner2 = np.stack([0.5 + 0.33 * np.cos(t), 0.5 + 0.33 * np.sin(t)], axis=1)
    outer2 = np.stack([0.5 + 0.48 * np.cos(t + np.pi / 6),
                       0.5 + 0.48 * np.sin(t + np.pi / 6)], axis=1)
    cands.append(np.vstack([outer2, inner2, [[0.5, 0.5]]]))
    # 3) jittered grid, fixed seed
    rng = np.random.default_rng(123)
    g = []
    for i in range(4):
        for j in range(4):
            g.append((0.125 + i * 0.25 + rng.random() * 0.05,
                      0.125 + j * 0.25 + rng.random() * 0.05))
    cands.append(np.array(g[:13]))
    return cands


def _anneal(pts, iters, t0, t1, rng, step0=0.05, step1=0.002, k_bot=8):
    """Worst-triangle targeted simulated annealing with best-ever fallback."""
    pts = pts.copy()
    ha = _hull_area(pts)
    best, ha = _score(pts, ha)
    cur = best
    best_pts = pts.copy()
    for it in range(iters):
        frac = it / iters
        temp = t0 * (t1 / t0) ** frac
        step = step0 * (step1 / step0) ** frac + 0.0005
        areas = _all_tri_areas(pts)
        order = np.argsort(areas)[:k_bot]
        verts = np.unique(_IDX[order].ravel())
        vi = verts[rng.integers(len(verts))]
        cand = pts.copy()
        cand[vi] += rng.normal(0, step, 2)
        cand[vi] = np.clip(cand[vi], 0.0, 1.0)
        if np.min(np.linalg.norm(np.delete(cand, vi, axis=0) - cand[vi], axis=1)) < 1e-6:
            continue
        ha_new = _hull_area(cand)
        s_new, ha_new = _score(cand, ha_new)
        if s_new >= cur or rng.random() < np.exp((s_new - cur) / max(temp, 1e-12)):
            pts, cur, ha = cand, s_new, ha_new
            if s_new > best:
                best = s_new
                best_pts = cand.copy()
    return best_pts, best


def _polish(pts, iters=1200):
    """Greedy targeted refinement at small steps."""
    pts = pts.copy()
    ha = _hull_area(pts)
    best, ha = _score(pts, ha)
    for _ in range(iters):
        areas = _all_tri_areas(pts)
        order = np.argsort(areas)[:6]
        verts = np.unique(_IDX[order].ravel())
        improved = False
        for vi in verts:
            for d in range(2):
                for sgn in (1, -1):
                    for step in (0.003, 0.0008):
                        cand = pts.copy()
                        cand[vi, d] = np.clip(cand[vi, d] + sgn * step, 0.0, 1.0)
                        ha_new = _hull_area(cand)
                        s, ha_new = _score(cand, ha_new)
                        if s > best + 1e-12:
                            pts, best, ha, improved = cand, s, ha_new, True
        if not improved:
            break
    return pts, best


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points maximizing the smallest triangle
    area normalized by convex hull area, via worst-triangle targeted
    simulated annealing with reheat restarts and a final greedy polish.
    """
    best_pts, best_val = None, -1.0
    seeds = _seeds()
    for si, cand in enumerate(seeds):
        rng = np.random.default_rng(100 + si)
        p, v = _anneal(cand, iters=2200, t0=0.004, t1=1e-5, rng=rng)
        # adaptive reheat restarts: short reheats, keep best-ever
        for r in range(3):
            rng_r = np.random.default_rng(300 + 10 * si + r)
            p2, v2 = _anneal(p, iters=500, t0=0.002, t1=1e-6,
                             rng=rng_r, step0=0.05, step1=0.002)
            if v2 > v:
                p, v = p2, v2
        p, v = _polish(p)
        if v > best_val:
            best_val, best_pts = v, p
    if best_pts is None:
        best_pts = np.random.default_rng(seed=42).random((13, 2))
    return best_pts


# EVOLVE-BLOCK-END