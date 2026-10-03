# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations


def _min_triangle_area(points, tri_idx):
    """Exact minimum (unsigned) area over all C(13,3) triangles, vectorized."""
    a = points[tri_idx[:, 0]]
    b = points[tri_idx[:, 1]]
    c = points[tri_idx[:, 2]]
    cross = ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
             - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    return 0.5 * np.min(np.abs(cross))


def _hull_area(points):
    """Area of the convex hull via the shoelace formula on hull vertices."""
    p = points[np.argsort(points[:, 0] + 1e-9 * points[:, 1])]
    # Andrew's monotone chain
    def half(pts):
        h = []
        for q in pts:
            while len(h) >= 2 and ((h[-1][0] - h[-2][0]) * (q[1] - h[-2][1])
                                   - (h[-1][1] - h[-2][1]) * (q[0] - h[-2][0])) <= 0:
                h.pop()
            h.append(q)
        return h
    lower = half(p)
    upper = half(p[::-1])
    hull = np.array(lower[:-1] + upper[:-1])
    x, y = hull[:, 0], hull[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _score(pts, tri_idx):
    """Actual metric: min triangle area / convex hull area."""
    ha = _hull_area(pts)
    if ha <= 0 or not np.isfinite(ha):
        return 0.0
    return _min_triangle_area(pts, tri_idx) / ha


def _climb(pts, tri_idx, rng, iters=4000, step0=0.05):
    """Hill-climb directly on the normalized objective min_area/hull_area.

    Moves vertices of the current worst triangle (mostly), accepting only
    strict improvements; step size anneals on repeated failures.
    The initial step size `step0` allows fine refinement passes.
    """
    pts = pts.copy()
    best = _score(pts, tri_idx)
    step = step0
    fails = 0
    for it in range(iters):
        a = pts[tri_idx[:, 0]]
        b = pts[tri_idx[:, 1]]
        c = pts[tri_idx[:, 2]]
        cross = np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                       - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
        worst = tri_idx[int(np.argmin(cross))]
        if rng.random() < 0.7:
            cand = worst
        else:
            cand = tri_idx[int(rng.integers(len(tri_idx)))]
        i = cand[int(rng.integers(3))]
        old = pts[i].copy()
        r = rng.random()
        if r < 0.5:
            # isotropic Gaussian move
            pts[i] = old + step * rng.standard_normal(2)
        elif r < 0.8:
            # axis-aligned probe (good for sliding along hull edges)
            d = int(rng.integers(4))
            delta = step * np.array([1.0, 0.0, -1.0, 0.0][d] if d % 2 == 0
                                     else [0.0, 1.0, 0.0, -1.0][d])
            pts[i] = old + delta
        else:
            # occasionally move two vertices of the candidate triangle jointly
            j = cand[int(rng.integers(3))]
            while j == i:
                j = cand[int(rng.integers(3))]
            oldj = pts[j].copy()
            pts[i] = old + step * rng.standard_normal(2)
            pts[j] = oldj + step * rng.standard_normal(2)
            np.clip(pts[i], -1.6, 1.6, out=pts[i])
            np.clip(pts[j], -1.6, 1.6, out=pts[j])
            val = _score(pts, tri_idx)
            if val > best + 1e-14:
                best = val
                fails = 0
            else:
                pts[i] = old
                pts[j] = oldj
                fails += 1
                if fails % 300 == 0:
                    step *= 0.75
            if step < 1e-4:
                break
            continue
        np.clip(pts[i], -1.6, 1.6, out=pts[i])
        val = _score(pts, tri_idx)
        if val > best + 1e-14:
            best = val
            fails = 0
        else:
            pts[i] = old
            fails += 1
            if fails % 300 == 0:
                step *= 0.75
        if step < 1e-4:
            break
    return pts, best


def heilbronn_convex13() -> np.ndarray:
    """
    Deterministic multi-start hill-climbing for 13 points maximizing the
    smallest triangle area normalized by the convex hull area.

    Approach:
    1. Several structured initializations: rings of 11/12/13 points
       (optionally elliptical) with small seeded perturbations, plus
       ring+interior variants.
    2. For each start, hill-climb directly on the true metric
       (min triangle area / hull area), prioritizing vertices of the
       current worst triangle.
    3. Keep the best configuration and rescale to unit hull area.
    """
    n = 13
    tri_idx = np.array(list(combinations(range(n), 3)))
    ts = np.linspace(0, 2 * np.pi, 13, endpoint=False)
    unit_ring = np.stack([np.cos(ts), np.sin(ts)], axis=1)

    def tri_fold_start(rot_seed):
        """3-fold symmetric start: 3 petals of 4 points + center.

        Each petal is a small diamond centered on a 3-fold axis; this
        symmetry is frequently seen in near-optimal Heilbronn configs.
        """
        rng = np.random.default_rng(seed=rot_seed)
        pts = [[0.0, 0.0]]
        for k in range(3):
            ang = 2 * np.pi * k / 3
            ca, sa = np.cos(ang), np.sin(ang)
            # local petal: outer, mid, inner, side point
            local = np.array([[1.0, 0.0],
                              [0.62, 0.28],
                              [0.30, 0.0],
                              [0.62, -0.28]])
            for lx, ly in local:
                pts.append([lx * ca - ly * sa, lx * sa + ly * ca])
        return np.array(pts) + 0.02 * rng.standard_normal((n, 2)), rng

    starts = []
    for seed in range(15):
        rng = np.random.default_rng(seed=100 + seed)
        if seed == 13:
            # hexagonal-ish triangular lattice patch (new diverse basin)
            ax_ = np.array([1.0, 0.0])
            ay_ = np.array([0.5, np.sqrt(3) / 2])
            pts = []
            for i in range(-2, 3):
                for j in range(-2, 3):
                    if abs(i + j) <= 2:
                        pts.append(i * ax_ + j * ay_)
            pts = np.array(pts[:n])
        elif seed == 14:
            # 4-fold symmetric: square ring of 8 + inner square of 4 + center
            sq = np.array([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]])
            t8 = np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.pi / 8
            outer = np.stack([np.cos(t8), np.sin(t8)], axis=1)
            pts = np.vstack([outer, 0.45 * sq, [[0.0, 0.0]]])
        elif seed % 5 == 4:
            # two concentric rings: 7 outer + 6 inner
            t7 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
            t6 = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi / 6
            pts = np.vstack([np.stack([np.cos(t7), np.sin(t7)], axis=1),
                            0.55 * np.stack([np.cos(t6), np.sin(t6)], axis=1)])
        elif seed % 4 == 0:
            # full ring of 13
            pts = unit_ring.copy()
        elif seed % 4 == 1:
            # ring of 11 + 2 interior points
            t11 = np.linspace(0, 2 * np.pi, 11, endpoint=False)
            ring = np.stack([np.cos(t11), 1.15 * np.sin(t11)], axis=1)
            pts = np.vstack([ring, [[0.0, 0.35], [0.0, -0.35]]])
        elif seed % 4 == 2:
            # elliptical ring of 12 + center
            t12 = np.linspace(0, 2 * np.pi, 12, endpoint=False)
            ring = np.stack([np.cos(t12), 1.1 * np.sin(t12)], axis=1)
            pts = np.vstack([ring, [[0.0, 0.0]]])
        else:
            # 3-fold symmetric petal configuration
            pts, rng = tri_fold_start(200 + seed)
            starts.append((pts, rng))
            continue
        pts = pts + 0.02 * rng.standard_normal((n, 2))
        starts.append((pts, rng))

    # Keep the top-3 distinct starts so refinement isn't locked into one basin
    results = []
    for pts0, rng in starts:
        pts, val = _climb(pts0, tri_idx, rng, iters=5000)
        results.append((val, pts))
    results.sort(key=lambda t: -t[0])
    best_val, best_pts = results[0]
    # Refine each of the top-3 candidates with fine passes; keep the global best
    for cand_val, cand_pts in results[:3]:
        rng2 = np.random.default_rng(seed=999)
        p2, v2 = _climb(cand_pts, tri_idx, rng2, iters=2500, step0=0.01)
        rng3 = np.random.default_rng(seed=1234)
        p2, v2 = _climb(p2, tri_idx, rng3, iters=2000, step0=0.003)
        rng4 = np.random.default_rng(seed=4321)
        p2, v2 = _climb(p2, tri_idx, rng4, iters=2000, step0=0.001)
        if v2 > best_val:
            best_val, best_pts = v2, p2

    # Fine refinement passes on the winner with smaller initial steps
    if best_pts is not None:
        rng2 = np.random.default_rng(seed=999)
        pts2, val2 = _climb(best_pts, tri_idx, rng2, iters=4000, step0=0.01)
        if val2 > best_val:
            best_val, best_pts = val2, pts2
        rng3 = np.random.default_rng(seed=1234)
        pts3, val3 = _climb(best_pts, tri_idx, rng3, iters=3000, step0=0.003)
        if val3 > best_val:
            best_val, best_pts = val3, pts3
        rng4 = np.random.default_rng(seed=4321)
        pts4, val4 = _climb(best_pts, tri_idx, rng4, iters=3000, step0=0.001)
        if val4 > best_val:
            best_val, best_pts = val4, pts4

    # Deterministic basin hopping: perturb the winner, re-climb, keep best.
    # Kicks of decreasing magnitude escape shallow local optima while
    # preserving the best-so-far configuration (monotone improvement).
    if best_pts is not None:
        kick_schedule = [(0.05, 3500, 0.02), (0.03, 3000, 0.01),
                         (0.02, 2500, 0.005), (0.012, 2500, 0.003),
                         (0.008, 2000, 0.001), (0.005, 2000, 0.0008)]
        for kick_round, (kick, k_iters, k_step) in enumerate(kick_schedule):
            rngk = np.random.default_rng(seed=7000 + kick_round)
            trial = best_pts + kick * rngk.standard_normal((n, 2))
            trial, tval = _climb(trial, tri_idx, rngk, iters=k_iters, step0=k_step)
            if tval > best_val:
                best_val, best_pts = tval, trial
        # One final ultra-fine polish to squeeze out remaining gains.
        rngf = np.random.default_rng(seed=424242)
        pf, vf = _climb(best_pts, tri_idx, rngf, iters=2500, step0=0.0003)
        if vf > best_val:
            best_val, best_pts = vf, pf

    if best_pts is None or not np.all(np.isfinite(best_pts)):
        # fallback: deterministic regular 13-gon (unit hull area)
        return unit_ring / np.sqrt(np.pi)

    ha = _hull_area(best_pts)
    if ha > 0 and np.isfinite(ha):
        best_pts = best_pts / np.sqrt(ha)
    else:
        best_pts = unit_ring / np.sqrt(np.pi)
    return best_pts


# EVOLVE-BLOCK-END
