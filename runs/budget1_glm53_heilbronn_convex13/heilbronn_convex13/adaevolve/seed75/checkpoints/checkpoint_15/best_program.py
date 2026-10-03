# EVOLVE-BLOCK-START
import numpy as np
import time


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

    deadline = time.time() + 240.0  # well under the 360 s limit
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
        if time.time() >= deadline:
            break
        p, v = _local_search(start, tri_idx, pt_tris, pt_rest, rng, deadline)
        if v > best_val:
            best_val, best_pts = v, p

    # Intensification: perturb-and-reoptimize around the incumbent.
    for r in range(350):
        if time.time() >= deadline:
            break
        scale = max(0.05 * (0.985 ** r), 0.0015)
        p0 = best_pts + scale * rng.standard_normal((n, 2))
        p, v = _local_search(p0, tri_idx, pt_tris, pt_rest, rng, deadline)
        if v > best_val:
            best_val, best_pts = v, p

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
