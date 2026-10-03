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
    P = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
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
    outer = np.stack([0.5 + 0.47 * np.cos(t), 0.5 + 0.47 * np.sin(t)], axis=1)
    inner = np.stack([0.5 + 0.26 * np.cos(t + np.pi / 6),
                      0.5 + 0.26 * np.sin(t + np.pi / 6)], axis=1)
    cands.append(np.vstack([outer, inner, [[0.5, 0.5]]]))
    inner2 = np.stack([0.5 + 0.33 * np.cos(t), 0.5 + 0.33 * np.sin(t)], axis=1)
    outer2 = np.stack([0.5 + 0.48 * np.cos(t + np.pi / 6),
                      0.5 + 0.48 * np.sin(t + np.pi / 6)], axis=1)
    cands.append(np.vstack([outer2, inner2, [[0.5, 0.5]]]))
    rng = np.random.default_rng(123)
    g = []
    for i in range(4):
        for j in range(4):
            g.append((0.125 + i * 0.25 + rng.random() * 0.05,
                      0.125 + j * 0.25 + rng.random() * 0.05))
    cands.append(np.array(g[:13]))
    t13 = 2 * np.pi * np.arange(13) / 13
    cands.append(np.stack([0.5 + 0.48 * np.cos(t13),
                           0.5 + 0.48 * np.sin(t13)], axis=1))
    t10 = 2 * np.pi * np.arange(10) / 10
    outer10 = np.stack([0.5 + 0.48 * np.cos(t10),
                        0.5 + 0.48 * np.sin(t10)], axis=1)
    inner3 = np.array([[0.5, 0.28], [0.32, 0.62], [0.68, 0.62]])
    cands.append(np.vstack([outer10, inner3]))
    return cands


def _anneal(pts, iters, t0, t1, rng, step0=0.05, step1=0.002, k_bot=8,
            win=200, lo_acc=0.15, hi_acc=0.50, smax=0.05, smin=0.002):
    """Worst-triangle targeted SA with adaptive acceptance-rate step control
    plus coordinate-axis fallback proposals.

    The step size is NOT a fixed geometric decay: it is steered by the
    acceptance rate over the last `win` proposals (random-direction and
    axis-fallback acceptances both count). Low acceptance halves the step,
    high acceptance grows it by 1.5x, keeping the search inside a productive
    move band so the cheap axis directions are tried usefully often.
    """
    pts = pts.copy()
    ha = _hull_area(pts)
    best, ha = _score(pts, ha)
    cur = best
    best_pts = pts.copy()
    step = step0
    acc_hist = []  # 1.0 accepted, 0.0 rejected

    def _try(cand, cur, temp):
        ha_new = _hull_area(cand)
        s_new, ha_new = _score(cand, ha_new)
        if s_new >= cur or rng.random() < np.exp((s_new - cur) / max(temp, 1e-12)):
            return True, s_new, ha_new
        return False, s_new, ha_new

    for it in range(iters):
        frac = it / iters
        temp = t0 * (t1 / t0) ** frac
        # floor the step so late-stage refinement still moves
        eff_step = max(step, step1 * 0.5)

        areas = _all_tri_areas(pts)
        order = np.argsort(areas)[:k_bot]
        verts = np.unique(_IDX[order].ravel())
        vi = verts[rng.integers(len(verts))]
        cand = pts.copy()
        cand[vi] += rng.normal(0, eff_step, 2)
        cand[vi] = np.clip(cand[vi], 0.0, 1.0)
        if np.min(np.linalg.norm(np.delete(cand, vi, axis=0) - cand[vi], axis=1)) < 1e-6:
            continue

        accepted, s_new, ha_new = _try(cand, cur, temp)
        if accepted:
            pts, cur, ha = cand, s_new, ha_new
            if s_new > best:
                best = s_new
                best_pts = cand.copy()
            acc_hist.append(1.0)
        else:
            # coordinate-axis fallback: +/-x, +/-y at the same magnitude
            done = False
            for d in range(2):
                for sgn in (1, -1):
                    if done:
                        break
                    cand2 = pts.copy()
                    cand2[vi, d] = np.clip(cand2[vi, d] + sgn * eff_step, 0.0, 1.0)
                    if np.min(np.linalg.norm(np.delete(cand2, vi, axis=0) - cand2[vi], axis=1)) < 1e-6:
                        continue
                    ok, s2, ha2 = _try(cand2, cur, temp)
                    if ok:
                        pts, cur, ha = cand2, s2, ha2
                        if s2 > best:
                            best = s2
                            best_pts = cand2.copy()
                        done = True
            acc_hist.append(1.0 if done else 0.0)

        # --- adaptive acceptance-rate step controller ---
        if len(acc_hist) > win:
            acc_hist.pop(0)
        if (it + 1) % 25 == 0 and len(acc_hist) >= 50:
            rate = float(np.mean(acc_hist))
            if rate < lo_acc:
                step = max(step * 0.5, smin)
            elif rate > hi_acc:
                step = min(step * 1.5, smax)
            # mid band: leave step alone

    return best_pts, best


def _polish(pts, iters=1200):
    """Greedy targeted refinement with adaptive step ladder."""
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
                    for step in (0.006, 0.003, 0.001, 0.0004):
                        cand = pts.copy()
                        cand[vi, d] = np.clip(cand[vi, d] + sgn * step, 0.0, 1.0)
                        ha_new = _hull_area(cand)
                        s, ha_new = _score(cand, ha_new)
                        if s > best + 1e-12:
                            pts, best, ha, improved = cand, s, ha_new, True
        if not improved:
            break
    return pts, best


def _pool_add(pool, pts, val, max_pool=4, min_dist=0.08):
    """Keep top distinct configurations in the pool."""
    for _, (qp, qv) in enumerate(pool):
        if qv == val:
            return
        d = np.mean(np.abs(np.sort(pts, axis=0) - np.sort(qp, axis=0)))
        if d < min_dist:
            if val > qv:
                pool.remove((qp, qv))
            else:
                return
    pool.append((pts, val))
    pool.sort(key=lambda t: -t[1])
    while len(pool) > max_pool:
        pool.pop()


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points maximizing the smallest triangle
    area normalized by convex hull area.

    Pipeline: multi-seed targeted simulated annealing with an adaptive
    acceptance-rate step controller and coordinate-axis fallback proposals,
    a distinct-basin pool, reheats from every pool basin, and a final
    greedy multi-scale polish.
    """
    pool = []
    seeds = _seeds()
    for si, cand in enumerate(seeds):
        rng = np.random.default_rng(100 + si)
        p, v = _anneal(cand, iters=2200, t0=0.004, t1=1e-5, rng=rng)
        for r in range(2):
            rng_r = np.random.default_rng(300 + 10 * si + r)
            p2, v2 = _anneal(p, iters=500, t0=0.002, t1=1e-6,
                             rng=rng_r, step0=0.04, step1=0.002)
            if v2 > v:
                p, v = p2, v2
        _pool_add(pool, p, v)

    restarts = [p for p, v in pool]
    if pool:
        restarts.append(pool[0][0])
    best_pts, best_val = None, -1.0
    for ri, p in enumerate(restarts):
        rng_r = np.random.default_rng(700 + ri)
        p2, v2 = _anneal(p, iters=900, t0=0.003, t1=1e-6,
                         rng=rng_r, step0=0.04, step1=0.001)
        p2, v2 = _polish(p2)
        if v2 > best_val:
            best_val, best_pts = v2, p2
    if best_pts is None:
        best_pts = np.random.default_rng(seed=42).random((13, 2))
    return best_pts


# EVOLVE-BLOCK-END