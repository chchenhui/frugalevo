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
    """Hill-climb on min_area/hull_area with incremental cross-product caching.

    All 286 triangle cross-products are cached; when a point moves, only
    the ~66 triangles containing it are recomputed (~4x speedup), letting
    us afford many more iterations. Moves target the current worst
    triangle's vertices (mostly); only strict improvements are accepted
    and the step anneals on repeated failures.
    """
    pts = pts.copy()
    n = len(pts)
    # cache of |cross| for all triangles
    a = pts[tri_idx[:, 0]]
    b = pts[tri_idx[:, 1]]
    c = pts[tri_idx[:, 2]]
    cross = np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                   - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    # triangles containing each point (for incremental updates)
    tri_of = [np.flatnonzero((tri_idx == i).any(axis=1)) for i in range(n)]

    def refresh(i):
        t = tri_of[i]
        ai = pts[tri_idx[t, 0]]
        bi = pts[tri_idx[t, 1]]
        ci = pts[tri_idx[t, 2]]
        cross[t] = np.abs((bi[:, 0] - ai[:, 0]) * (ci[:, 1] - ai[:, 1])
                          - (bi[:, 1] - ai[:, 1]) * (ci[:, 0] - ai[:, 0]))

    ha = _hull_area(pts)
    best = 0.5 * cross.min() / ha if ha > 0 else 0.0
    step = step0
    fails = 0
    for it in range(iters):
        worst = tri_idx[int(np.argmin(cross))]
        if rng.random() < 0.7:
            cand = worst
        else:
            cand = tri_idx[int(rng.integers(len(tri_idx)))]
        i = cand[int(rng.integers(3))]
        old = pts[i].copy()
        r = rng.random()
        if r < 0.5:
            pts[i] = old + step * rng.standard_normal(2)
        elif r < 0.8:
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
            refresh(i)
            refresh(j)
            ha = _hull_area(pts)
            val = 0.5 * cross.min() / ha if ha > 0 else 0.0
            if val > best + 1e-14:
                best = val
                fails = 0
            else:
                pts[i] = old
                pts[j] = oldj
                refresh(i)
                refresh(j)
                fails += 1
                if fails % 300 == 0:
                    step *= 0.75
            if step < 1e-4:
                break
            continue
        np.clip(pts[i], -1.6, 1.6, out=pts[i])
        refresh(i)
        ha = _hull_area(pts)
        val = 0.5 * cross.min() / ha if ha > 0 else 0.0
        if val > best + 1e-14:
            best = val
            fails = 0
        else:
            pts[i] = old
            refresh(i)
            fails += 1
            if fails % 300 == 0:
                step *= 0.75
        if step < 1e-4:
            break
    return pts, best


def _polish(pts, tri_idx, deltas=(0.004, 0.002, 0.001, 0.0005, 0.0002, 0.0001)):
    """Deterministic pattern-search polish: for each delta, try +/-delta moves
    on each coordinate of each point, accepting any strict improvement.
    Systematic (non-random) local refinement that complements hill-climbing;
    monotone by construction."""
    pts = pts.copy()
    best = _score(pts, tri_idx)
    for delta in deltas:
        improved = True
        while improved:
            improved = False
            for i in range(len(pts)):
                for ax in range(2):
                    for sgn in (1.0, -1.0):
                        old = pts[i, ax]
                        pts[i, ax] = old + sgn * delta
                        val = _score(pts, tri_idx)
                        if val > best + 1e-15:
                            best = val
                            improved = True
                        else:
                            pts[i, ax] = old
    return pts, best


def _soft_climb(pts, tri_idx, rng, iters=3000, step0=0.02, k=4):
    """Climb on a smoothed objective: sum of the k smallest triangle areas
    (normalized by hull area). When many triangles tie at the hard minimum,
    the min-only landscape is flat; the soft objective gives a gradient that
    lifts the whole bottom tier, after which hard-min polishing can push
    further. Deterministic given the seeded rng."""
    pts = pts.copy()
    def soft_score(p):
        ha = _hull_area(p)
        if ha <= 0 or not np.isfinite(ha):
            return 0.0
        a = p[tri_idx[:, 0]]
        b = p[tri_idx[:, 1]]
        c = p[tri_idx[:, 2]]
        cross = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                             - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
        return np.sum(np.sort(cross)[:k]) / ha
    best = soft_score(pts)
    step = step0
    fails = 0
    for it in range(iters):
        a = pts[tri_idx[:, 0]]
        b = pts[tri_idx[:, 1]]
        c = pts[tri_idx[:, 2]]
        cross = np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                       - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
        order = np.argsort(cross)[:k]
        cand = tri_idx[int(order[int(rng.integers(len(order)))])]
        i = cand[int(rng.integers(3))]
        old = pts[i].copy()
        pts[i] = old + step * rng.standard_normal(2)
        np.clip(pts[i], -1.6, 1.6, out=pts[i])
        val = soft_score(pts)
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
    return pts, _score(pts, tri_idx)


def heilbronn_convex13() -> np.ndarray:
    """
    Deterministic multi-start hill-climbing for 13 points maximizing the
    smallest triangle area normalized by the convex hull area.

    Approach:
    1. Structured starts: rings of 11/12/13 points (some elliptical, some
       with interior points) and 3-fold symmetric petal configurations.
    2. Hill-climb directly on the true metric (min triangle area / hull
       area), prioritizing vertices of the current worst triangle.
    3. Fine refinement, soft-objective basin exploration, and pattern-search
       polish on the winner; rescale to unit hull area at the end.
    """
    n = 13
    tri_idx = np.array(list(combinations(range(n), 3)))
    ts = np.linspace(0, 2 * np.pi, 13, endpoint=False)
    unit_ring = np.stack([np.cos(ts), np.sin(ts)], axis=1)

    def tri_fold_start(rot_seed):
        """3-fold symmetric start: 3 petals of 4 points + center."""
        rng = np.random.default_rng(seed=rot_seed)
        pts = [[0.0, 0.0]]
        for k in range(3):
            ang = 2 * np.pi * k / 3
            ca, sa = np.cos(ang), np.sin(ang)
            local = np.array([[1.0, 0.0],
                              [0.62, 0.28],
                              [0.30, 0.0],
                              [0.62, -0.28]])
            for lx, ly in local:
                pts.append([lx * ca - ly * sa, lx * sa + ly * ca])
        return np.array(pts) + 0.02 * rng.standard_normal((n, 2)), rng

    starts = []
    for seed in range(13):
        rng = np.random.default_rng(seed=100 + seed)
        if seed % 5 == 4:
            # two concentric rings: 7 outer + 6 inner
            t7 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
            t6 = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi / 6
            pts = np.vstack([np.stack([np.cos(t7), np.sin(t7)], axis=1),
                            0.55 * np.stack([np.cos(t6), np.sin(t6)], axis=1)])
        elif seed % 4 == 0:
            pts = unit_ring.copy()
        elif seed % 4 == 1:
            t11 = np.linspace(0, 2 * np.pi, 11, endpoint=False)
            ring = np.stack([np.cos(t11), 1.15 * np.sin(t11)], axis=1)
            pts = np.vstack([ring, [[0.0, 0.35], [0.0, -0.35]]])
        elif seed % 4 == 2:
            t12 = np.linspace(0, 2 * np.pi, 12, endpoint=False)
            ring = np.stack([np.cos(t12), 1.1 * np.sin(t12)], axis=1)
            pts = np.vstack([ring, [[0.0, 0.0]]])
        else:
            pts, rng = tri_fold_start(200 + seed)
            starts.append((pts, rng))
            continue
        pts = pts + 0.02 * rng.standard_normal((n, 2))
        starts.append((pts, rng))

    best_pts, best_val = None, -1.0
    for pts0, rng in starts:
        pts, val = _climb(pts0, tri_idx, rng, iters=8000)
        if val > best_val:
            best_val, best_pts = val, pts

    # Fine refinement ladder on the winner: alternating coarse/fine steps
    # let the climb escape shallow ridges while polishing; each pass only
    # replaces the incumbent on strict improvement of the true metric.
    # The incremental-cache climb is ~4x faster, so we can afford larger
    # iteration budgets in the same wall time.
    if best_pts is not None:
        for seed, iters, step0 in [(999, 6000, 0.01), (1234, 4500, 0.003),
                                    (4321, 4500, 0.001), (777, 4500, 0.02),
                                    (2024, 4500, 0.005), (555, 4000, 0.0005),
                                    (31337, 4000, 0.015), (8888, 4000, 0.002)]:
            rp = np.random.default_rng(seed=seed)
            cand_pts, cand_val = _climb(best_pts, tri_idx, rp,
                                        iters=iters, step0=step0)
            if cand_val > best_val:
                best_val, best_pts = cand_val, cand_pts

    # Deterministic basin hopping: perturb the winner, re-climb, keep best.
    # Kicks of decreasing magnitude escape shallow local optima while
    # preserving the best-so-far configuration (monotone improvement).
    if best_pts is not None:
        kick_schedule = [(0.05, 5000, 0.02), (0.03, 4500, 0.01),
                         (0.02, 4000, 0.005), (0.012, 4000, 0.003),
                         (0.008, 3500, 0.001), (0.005, 3500, 0.0008)]
        for kick_round, (kick, k_iters, k_step) in enumerate(kick_schedule):
            rngk = np.random.default_rng(seed=7000 + kick_round)
            trial = best_pts + kick * rngk.standard_normal((n, 2))
            trial, tval = _climb(trial, tri_idx, rngk, iters=k_iters, step0=k_step)
            if tval > best_val:
                best_val, best_pts = tval, trial
        # One final ultra-fine polish to squeeze out remaining gains.
        rngf = np.random.default_rng(seed=424242)
        pf, vf = _climb(best_pts, tri_idx, rngf, iters=4000, step0=0.0003)
        if vf > best_val:
            best_val, best_pts = vf, pf

    # Soft-objective basin exploration: climb on the sum of the k smallest
    # triangle areas to escape the flat plateau where many triangles tie at
    # the hard minimum, then hard-polish the result. Monotone on true metric.
    if best_pts is not None:
        for soft_round, (kick, k) in enumerate([(0.01, 3), (0.02, 4),
                                                (0.006, 5), (0.015, 6)]):
            rngs = np.random.default_rng(seed=9000 + soft_round)
            trial = best_pts + kick * rngs.standard_normal((n, 2))
            trial, tval = _soft_climb(trial, tri_idx, rngs,
                                      iters=2500, step0=0.02, k=k)
            # hard polish the soft result with the true metric
            rngh = np.random.default_rng(seed=9500 + soft_round)
            trial, tval = _climb(trial, tri_idx, rngh,
                                 iters=2000, step0=0.005)
            pp, pv = _polish(trial, tri_idx,
                             deltas=(0.002, 0.001, 0.0005, 0.0002))
            if pv > tval:
                tval = pv
                trial = pp
            if tval > best_val:
                best_val, best_pts = tval, trial

    # Deterministic pattern-search polish: systematic fine-grained refinement
    # that random hill-climbing often misses near a local optimum.
    if best_pts is not None:
        pp, pv = _polish(best_pts, tri_idx)
        if pv > best_val:
            best_val, best_pts = pv, pp
        # One more random pass after polish (polish can unlock new directions)
        rngz = np.random.default_rng(seed=31415)
        pz, vz = _climb(best_pts, tri_idx, rngz, iters=1500, step0=0.002)
        if vz > best_val:
            best_val, best_pts = vz, pz
        pp, pv = _polish(best_pts, tri_idx)
        if pv > best_val:
            best_val, best_pts = pv, pp
        # Final ultra-fine pattern polish down to machine-precision steps.
        pp, pv = _polish(best_pts, tri_idx,
                         deltas=(0.0002, 0.0001, 0.00005, 0.00002))
        if pv > best_val:
            best_val, best_pts = pv, pp

    if best_pts is None or not np.all(np.isfinite(best_pts)):
        return unit_ring / np.sqrt(np.pi)

    ha = _hull_area(best_pts)
    if ha > 0 and np.isfinite(ha):
        best_pts = best_pts / np.sqrt(ha)
    else:
        best_pts = unit_ring / np.sqrt(np.pi)
    return best_pts


# EVOLVE-BLOCK-END
