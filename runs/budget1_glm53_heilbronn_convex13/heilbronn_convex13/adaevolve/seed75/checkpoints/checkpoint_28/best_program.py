# EVOLVE-BLOCK-START
import time

import numpy as np

try:
    from scipy.optimize import linprog
except Exception:  # scipy expected, but degrade gracefully if absent
    linprog = None


def _tri_areas(pts, tri_idx):
    """Vectorized areas over given triples."""
    a = pts[tri_idx[:, 0]]
    b = pts[tri_idx[:, 1]]
    c = pts[tri_idx[:, 2]]
    return 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                        - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))


def _hull_area(pts):
    """Area of the convex hull of a small point set (monotone chain).

    Exact shoelace area over the hull vertices; collinear boundary points
    are dropped, which does not change the area. Fast enough (13 points)
    to call once per candidate move inside the local search.
    """
    P = [tuple(p) for p in pts[np.lexsort((pts[:, 1], pts[:, 0]))]]
    if len(P) < 3:
        return 0.0

    def build(seq):
        h = []
        for x, y in seq:
            while len(h) >= 2:
                ox, oy = h[-2]
                ax, ay = h[-1]
                if (ax - ox) * (y - oy) - (ay - oy) * (x - ox) <= 0:
                    h.pop()
                else:
                    break
            h.append((x, y))
        return h

    lower = build(P)
    upper = build(P[::-1])
    hull = lower[:-1] + upper[:-1]
    m = len(hull)
    if m < 3:
        return 0.0
    s = 0.0
    for i in range(m):
        j = (i + 1) % m
        s += hull[i][0] * hull[j][1] - hull[j][0] * hull[i][1]
    return 0.5 * abs(s)


def _local_search(pts, tri_idx, pt_tris, pt_rest, rng, deadline):
    """Greedy coordinate descent on the affine-invariant ratio objective
    r = (smallest triangle area) / (convex hull area).

    Incremental evaluation: moving point i only changes the 66 triangles
    containing i; the two smallest areas among the other 220 triangles are
    cached per point sweep. The hull area (13 points) is recomputed per
    candidate via a monotone chain. Accepts strict ratio improvements, or
    near-tie moves that raise the second-smallest triangle area (plateau
    escape). No container clipping is used: the ratio is scale-invariant,
    so points are optimized in a free frame. Deterministic given pts, rng.
    """
    pts = pts.copy()
    h0 = _hull_area(pts)
    if h0 < 1e-9:
        return pts, 0.0
    s0 = np.sort(_tri_areas(pts, tri_idx))
    best, best2 = s0[0] / h0, s0[1] / h0
    steps = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001,
             0.0005, 0.0002]
    dirs = np.array([[1, 0], [-1, 0], [0, 1], [0, -1],
                     [1, 1], [1, -1], [-1, 1], [-1, -1]]) / np.sqrt(2)
    for step in steps:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in rng.permutation(13):
                rest = pt_rest[i]
                if len(rest):
                    ra = np.sort(_tri_areas(pts, rest))[:2]
                else:
                    ra = np.array([np.inf, np.inf])
                tri_i = pt_tris[i]
                for d in dirs:
                    cand = pts.copy()
                    cand[i] = cand[i] + step * d
                    h = _hull_area(cand)
                    if h < 1e-9:
                        continue
                    ai = _tri_areas(cand, tri_i)
                    v1 = min(ra[0], ai.min())
                    r = v1 / h
                    if r > best + 1e-9:
                        pts = cand
                        best = r
                        s = np.sort(np.concatenate([ai, ra]))
                        best2 = s[1] / h
                        improved = True
                    elif r > best - 1e-9:
                        # plateau move: accept if second-smallest improves
                        s = np.sort(np.concatenate([ai, ra]))
                        v2 = s[1] / h
                        if v2 > best2 + 1e-9:
                            pts = cand
                            best, best2 = r, v2
                            improved = True
                if time.time() >= deadline:
                    return pts, best
    return pts, best


def _hull_idx(pts):
    """Indices (into pts) of the convex hull vertices (monotone chain)."""
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    P = pts[order]
    if len(P) < 3:
        return np.array([], dtype=int)

    def build(seq):
        h = []
        for k in seq:
            while len(h) >= 2:
                o, a = h[-2], h[-1]
                if ((P[a, 0] - P[o, 0]) * (P[k, 1] - P[o, 1])
                        - (P[a, 1] - P[o, 1]) * (P[k, 0] - P[o, 0]) <= 0):
                    h.pop()
                else:
                    break
            h.append(k)
        return h

    lower = build(list(range(len(P))))
    upper = build(list(range(len(P)))[::-1])
    hull = lower[:-1] + upper[:-1]
    if not hull:
        return np.array([], dtype=int)
    return order[np.array(hull, dtype=int)]


def _lp_refine(pts, tri_idx, pt_jk, pt_rest, deadline):
    """LP bottleneck refinement (breakthrough method).

    For each point i, the signed area of every triangle containing i is an
    affine function of (x_i, y_i) while orientation signs stay fixed:
        D = x*(y_j-y_k) - y*(x_k-x_j) + (x_j*y_k - y_j*x_k).
    A linear program (HiGHS) maximizes t subject to
        s*D >= 2t  for all triangles containing i  (s = current sign),
        t <= min area of triangles not containing i (unchanged by move),
        |p_i - p_i^cur| <= delta                     (box trust region),
    which is the EXACT best single-point move for the minimum-area
    bottleneck within the trust region — no direction/step discretization
    error. Degenerate (near-zero) constraints are dropped; the true ratio
    (min area / hull area) is recomputed and moves accepted only on strict
    improvement, guarding against orientation flips and hull changes.
    Deterministic (HiGHS is deterministic).
    """
    if linprog is None:
        return pts, float(_tri_areas(pts, tri_idx).min()) / _hull_area(pts)
    pts = pts.copy()
    h = _hull_area(pts)
    if h < 1e-9:
        return pts, 0.0
    best = float(_tri_areas(pts, tri_idx).min()) / h

    def _solve_point(pts, i, delta):
        """Solve the exact best single-point LP move for point i within
        trust region delta. Returns (new_pts, ratio) or (None, best) if
        no strict improvement. Pure function of (pts, i, delta): the LP
        maximizes t s.t. signed-area constraints (fixed orientation) and
        t <= min area of triangles not containing i."""
        rest = pt_rest[i]
        mrest = float(_tri_areas(pts, rest).min()) if len(rest) else np.inf
        jk = pt_jk[i]
        J = pts[jk[:, 0]]
        K = pts[jk[:, 1]]
        xi, yi = pts[i]
        D0 = (xi * (J[:, 1] - K[:, 1]) - yi * (J[:, 0] - K[:, 0])
              + J[:, 0] * K[:, 1] - J[:, 1] * K[:, 0])
        s = np.sign(D0)
        keep = np.abs(D0) > 1e-12
        if not keep.any():
            return None, -1.0
        Jk, Kk, sk = J[keep], K[keep], s[keep]
        c0 = Jk[:, 0] * Kk[:, 1] - Jk[:, 1] * Kk[:, 0]
        A_ub = np.column_stack([
            -sk * (Jk[:, 1] - Kk[:, 1]),
            -sk * (Kk[:, 0] - Jk[:, 0]),
            np.full(len(sk), 2.0),
        ])
        b_ub = sk * c0
        bounds = [(xi - delta, xi + delta), (yi - delta, yi + delta),
                  (None, mrest)]
        try:
            res = linprog([0.0, 0.0, -1.0], A_ub=A_ub, b_ub=b_ub,
                          bounds=bounds, method="highs")
        except Exception:
            return None, -1.0
        if not res.success or res.x is None:
            return None, -1.0
        xn, yn = res.x[0], res.x[1]
        if not (np.isfinite(xn) and np.isfinite(yn)):
            return None, -1.0
        cand = pts.copy()
        cand[i] = [xn, yn]
        hc = _hull_area(cand)
        if hc < 1e-9:
            return None, -1.0
        rc = float(_tri_areas(cand, tri_idx).min()) / hc
        return cand, rc

    for delta in (0.15, 0.08, 0.04, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(13):
                if time.time() >= deadline:
                    return pts, best
                cand, rc = _solve_point(pts, i, delta)
                if cand is not None and rc > best + 1e-9:
                    pts, best, improved = cand, rc, True
                    # Line search: the LP move was productive, so try the
                    # same point again with a doubled trust region (capped).
                    # This captures large beneficial displacements that the
                    # fixed delta schedule would never reach, while the
                    # strict true-ratio acceptance guards every step.
                    d2 = min(delta * 2.0, 0.4)
                    while d2 > delta and time.time() < deadline:
                        cand2, rc2 = _solve_point(pts, i, d2)
                        if cand2 is not None and rc2 > best + 1e-9:
                            pts, best = cand2, rc2
                            d2 = min(d2 * 2.0, 0.4)
                        else:
                            break
    return pts, best


def _hull_pull(pts, tri_idx, deadline):
    """Hull-shape reduction moves: pull hull vertices toward the centroid.

    The ratio (min triangle area)/(hull area) can improve by shrinking or
    rounding the hull even when the smallest triangle shrinks slightly.
    Each candidate pull is accepted only on strict true-ratio improvement.
    """
    pts = pts.copy()
    h = _hull_area(pts)
    if h < 1e-9:
        return pts, 0.0
    best = float(_tri_areas(pts, tri_idx).min()) / h
    cen = pts.mean(axis=0)
    for _ in range(3):
        if time.time() >= deadline:
            break
        moved = False
        for i in _hull_idx(pts):
            for f in (0.02, 0.05, 0.10, 0.20):
                cand = pts.copy()
                cand[i] = pts[i] + f * (cen - pts[i])
                hc = _hull_area(cand)
                if hc < 1e-9:
                    continue
                rc = float(_tri_areas(cand, tri_idx).min()) / hc
                if rc > best + 1e-9:
                    pts, best, moved = cand, rc, True
        if not moved:
            break
    return pts, best


def heilbronn_convex13() -> np.ndarray:
    """
    Construct 13 points maximizing (smallest triangle area) / (convex hull area).

    The objective is affine-invariant, so the search runs in a free frame
    (no container clipping) and the winning configuration is finally affinely
    normalized into the unit square, which preserves the ratio exactly.

    Approach: deterministic multi-start greedy coordinate descent on the
    ratio objective. Starts include a regular 13-gon, a 3-fold symmetric
    nest (center + 4 concentric equilateral triangles = 13 points), a
    double-hexagon configuration (center + 6 + 6), their perturbations, and
    seeded random clouds. Each start is refined by coordinate descent with
    shrinking step sizes; an intensification phase then perturbs and
    re-optimizes around the incumbent for a fixed number of rounds (also
    bounded by a time budget well under the execution limit). Fixed seeds
    and fixed iteration orders make results reproducible. Falls back to
    the regular 13-gon on any failure.
    """
    n = 13
    # Precompute all triangle index triples
    tri_idx = np.array([(i, j, k) for i in range(n) for j in range(i + 1, n)
                       for k in range(j + 1, n)], dtype=int)
    # Per-point triangle index arrays: triangles containing i, and the rest
    contains = (tri_idx == np.arange(n)[:, None, None]).any(axis=2)  # (n, 286)
    pt_tris = [tri_idx[contains[i]] for i in range(n)]
    pt_rest = [tri_idx[~contains[i]] for i in range(n)]
    # For each point i: (j, k) index pairs of the other two vertices of
    # every triangle containing i (used for the LP linearization).
    pt_jk = [np.array([[x for x in row if x != i] for row in tri_idx
                      if i in row], dtype=int) for i in range(n)]

    t0 = time.time()
    deadline = t0 + 280.0   # overall budget, safely under the 360 s limit
    phase_cd = t0 + 80.0    # multi-start coordinate descent
    phase_int = t0 + 150.0  # perturb-and-reoptimize intensification
    rng = np.random.default_rng(seed=42)

    # Regular 13-gon on the unit circle
    ang = 2 * np.pi * np.arange(n) / n
    gon = np.column_stack([np.cos(ang), np.sin(ang)])
    # 3-fold symmetric nest: center + 4 concentric equilateral triangles
    nest = [[0.0, 0.0]]
    for t in range(4):
        rr = 0.22 + 0.26 * t
        for k in range(3):
            a = t * np.pi / 9.0 + 2 * np.pi * k / 3.0
            nest.append([rr * np.cos(a), rr * np.sin(a)])
    nest = np.asarray(nest)
    # Center + two hexagons (outer one rotated 30 degrees)
    hexes = [[0.0, 0.0]]
    for k in range(6):
        a = 2 * np.pi * k / 6.0
        hexes.append([0.45 * np.cos(a), 0.45 * np.sin(a)])
    for k in range(6):
        a = np.pi / 6.0 + 2 * np.pi * k / 6.0
        hexes.append([np.cos(a), np.sin(a)])
    hexes = np.asarray(hexes)

    starts = [gon, nest, hexes]
    for s in range(45):
        if s < 12:
            p = gon + 0.08 * rng.standard_normal((n, 2))
        elif s < 24:
            p = nest + 0.08 * rng.standard_normal((n, 2))
        elif s < 33:
            p = hexes + 0.08 * rng.standard_normal((n, 2))
        else:
            p = rng.standard_normal((n, 2))
        starts.append(p)

    best_pts, best_val = gon, -1.0
    for start in starts:
        if time.time() >= phase_cd:
            break
        p, v = _local_search(start, tri_idx, pt_tris, pt_rest, rng, phase_cd)
        if v > best_val:
            best_val, best_pts = v, p

    # Intensification: perturb-and-reoptimize around the incumbent.
    for r in range(220):
        if time.time() >= phase_int:
            break
        scale = max(0.05 * (0.985 ** r), 0.0015)
        p0 = best_pts + scale * rng.standard_normal((n, 2))
        p, v = _local_search(p0, tri_idx, pt_tris, pt_rest, rng, phase_int)
        if v > best_val:
            best_val, best_pts = v, p

    # ---- LP bottleneck refinement (breakthrough method) ----
    # Exact best single-point moves via linear programming on the active
    # area constraints, alternating with hull-shape pulls and perturbation
    # restarts until the time budget is exhausted.
    if linprog is not None:
        try:
            p, v = _lp_refine(best_pts, tri_idx, pt_jk, pt_rest, deadline)
            if v > best_val:
                best_val, best_pts = v, p
            p, v = _hull_pull(best_pts, tri_idx, deadline)
            if v > best_val:
                best_val, best_pts = v, p
        except Exception:
            pass
        r = 0
        while time.time() < deadline - 2.0:
            r += 1
            scale = max(0.03 * (0.99 ** r), 0.001)
            p0 = best_pts + scale * rng.standard_normal((n, 2))
            p, v = _local_search(p0, tri_idx, pt_tris, pt_rest, rng, deadline)
            if v > best_val:
                best_val, best_pts = v, p
            try:
                p, v = _lp_refine(best_pts, tri_idx, pt_jk, pt_rest, deadline)
                if v > best_val:
                    best_val, best_pts = v, p
                p, v = _hull_pull(best_pts, tri_idx, deadline)
                if v > best_val:
                    best_val, best_pts = v, p
            except Exception:
                pass

    # Affine normalization into the unit square (preserves the ratio).
    pts = np.asarray(best_pts, dtype=float)
    if pts.shape != (n, 2) or not np.all(np.isfinite(pts)):
        pts = gon  # safe fallback
    lo = pts.min(axis=0)
    span = float((pts - lo).max())
    if span > 1e-12:
        pts = (pts - lo) / span
    return pts


# EVOLVE-BLOCK-END
