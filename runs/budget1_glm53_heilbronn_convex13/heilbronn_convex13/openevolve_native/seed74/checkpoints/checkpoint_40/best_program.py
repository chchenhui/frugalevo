# EVOLVE-BLOCK-START
import numpy as np


def _hull_area(pts: np.ndarray) -> float:
    """Area of the convex hull via the shoelace formula on hull vertices."""
    try:
        from scipy.spatial import ConvexHull
        ch = ConvexHull(pts)
        v = pts[ch.vertices]
    except Exception:
        c = pts.mean(axis=0)
        d = pts - c
        order = np.argsort(np.arctan2(d[:, 1], d[:, 0]))
        v = pts[order]
    x, y = v[:, 0], v[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _tri_areas(pts: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Absolute areas of all C(13,3) triangles (vectorized cross products)."""
    a = pts[tri[:, 0]]
    b = pts[tri[:, 1]]
    c = pts[tri[:, 2]]
    return 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                        - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))


def heilbronn_convex13() -> np.ndarray:
    """
    Construct 13 points maximizing the smallest normalized triangle area.

    Approach: deterministic multi-start simulated annealing with a
    geometry-aware "critical-triangle" move.
    - Objective: min triangle area / convex hull area (scale-invariant).
    - Critical move: locate the currently smallest triangle and push each
      of its vertices perpendicular away from its opposite edge -- the
      steepest direction for raising that triangle's area.
    - Starts: regular 13-gon, two-ring (concentric circle) configurations
      (these break the single-ring symmetry that traps pure hill-climbing
      at ~0.0176), 3-fold symmetric layouts, and a few seeded random sets.
    - Each start gets annealing + greedy polish; the global best then gets
      a long final polish plus perturbation restarts.
    - Fixed RNG seed for full reproducibility; ~280s time budget.
    """
    import time
    n = 13
    rng = np.random.default_rng(seed=42)
    t_start = time.time()
    budget = 280.0

    tri = np.array([(i, j, k)
                    for i in range(n) for j in range(i + 1, n)
                    for k in range(j + 1, n)], dtype=int)

    def score(p):
        return _tri_areas(p, tri).min() / _hull_area(p)

    def critical_move(cur, step):
        """Push vertices of one of the few smallest triangles away from
        its opposite edge; among a handful of candidates, return the one
        with the best score (mini line search over critical triangles)."""
        areas = _tri_areas(cur, tri)
        cand_idx = np.argsort(areas)[:4]
        best_c, best_s = None, -np.inf
        for ci in cand_idx:
            t = tri[ci]
            cand = cur.copy()
            for vi, a, b in ((t[0], t[1], t[2]),
                             (t[1], t[0], t[2]),
                             (t[2], t[0], t[1])):
                e = cur[b] - cur[a]
                d = np.array([-e[1], e[0]])
                L = np.hypot(d[0], d[1])
                if L < 1e-12:
                    continue
                d = d / L
                if np.dot(cur[vi] - cur[a], d) < 0:
                    d = -d
                cand[vi] = cand[vi] + step * d * rng.uniform(0.5, 1.5)
            s = score(cand)
            if s > best_s:
                best_c, best_s = cand, s
        return best_c if best_c is not None else cur.copy()

    def anneal(pts, s0, iters=8000, t0=0.02):
        """Annealing mixing random single-point and critical-triangle moves."""
        cur, cur_s = pts.copy(), s0
        bst, bst_s = cur.copy(), cur_s
        step = 0.05
        for it in range(iters):
            if (it & 127) == 0 and time.time() - t_start > budget:
                break
            temp = t0 * (1 - it / iters) + 1e-4
            if rng.random() < 0.5:
                cand = cur.copy()
                cand[rng.integers(n)] += rng.normal(0, step, size=2)
            else:
                cand = critical_move(cur, step)
            s = score(cand)
            if s > cur_s or rng.random() < np.exp((s - cur_s) / temp):
                cur, cur_s = cand, s
                if s > bst_s:
                    bst, bst_s = cand.copy(), s
            if it % 500 == 499:
                step *= 0.85
        return bst, bst_s

    def polish(pts, s0, iters=15000):
        """Greedy hill-climb: per-point Gaussian + critical-triangle moves."""
        cur, cur_s = pts.copy(), s0
        step = 0.02
        since = 0
        for it in range(iters):
            if (it & 127) == 0 and time.time() - t_start > budget:
                break
            if it % 3 == 2:
                cand = critical_move(cur, step)
            else:
                cand = cur.copy()
                cand[it % n] += rng.normal(0, step, size=2)
            s = score(cand)
            if s > cur_s:
                cur, cur_s = cand, s
                since = 0
            else:
                since += 1
                if since > 3 * n:
                    step *= 0.8
                    since = 0
                    if step < 1e-5:
                        break
        return cur, cur_s

    # --- Diverse initializations ---
    starts = []
    ang = 2 * np.pi * np.arange(n) / n
    starts.append(np.column_stack([np.cos(ang), np.sin(ang)]))
    # Two-ring configs: m outer + (13-m) inner, various radii/rotations
    for m in (7, 8, 9, 10):
        for r in (0.25, 0.4, 0.55, 0.7):
            for rot in (0.0, 0.3, 0.7):
                ao = rot + 2 * np.pi * np.arange(m) / m
                ai = rot + np.pi / (n - m) + 2 * np.pi * np.arange(n - m) / (n - m)
                p = np.vstack([np.column_stack([np.cos(ao), np.sin(ao)]),
                               r * np.column_stack([np.cos(ai), np.sin(ai)])])
                starts.append(p)
    # 3-fold symmetric layouts (often near-optimal for triangle problems)
    for r1 in (0.45, 0.6, 0.75):
        for r2 in (0.15, 0.3):
            anchors = np.column_stack([np.cos(2*np.pi*np.arange(3)/3),
                                       np.sin(2*np.pi*np.arange(3)/3)])
            a1 = 0.5 + 2*np.pi*np.arange(6)/6
            a2 = 2*np.pi*np.arange(4)/4
            starts.append(np.vstack([
                anchors,
                r1*np.column_stack([np.cos(a1), np.sin(a1)]),
                r2*np.column_stack([np.cos(a2), np.sin(a2)])]))
    # Perturbed regular 13-gons (break symmetry gently)
    for _ in range(4):
        starts.append(np.column_stack([np.cos(ang), np.sin(ang)])
                      + rng.normal(0, 0.12, size=(n, 2)))
    # A few seeded random starts
    for _ in range(5):
        starts.append(rng.uniform(-1, 1, size=(n, 2)))

    best, best_score = None, -1.0
    for st in starts:
        if time.time() - t_start > budget:
            break
        s = score(st)
        p, sp = anneal(st, s)
        p, sp = polish(p, sp)
        if sp > best_score:
            best, best_score = p, sp

    # Long final polish of the global best (fresh step schedule).
    best, best_score = polish(best, best_score, iters=40000)

    # Perturbation restarts around the global best: kick it with small
    # noise and re-polish; keep any improvement. Escapes shallow optima.
    for _ in range(6):
        if time.time() - t_start > budget:
            break
        kick = best + rng.normal(0, 0.03, size=(n, 2))
        p, sp = polish(kick, score(kick), iters=12000)
        if sp > best_score:
            best, best_score = p, sp
    best, best_score = polish(best, best_score, iters=20000)

    # Normalize so the convex hull has unit area (affine scaling preserves
    # the normalized score).
    a = _hull_area(best)
    best = best / np.sqrt(a)

    return best


# EVOLVE-BLOCK-END
