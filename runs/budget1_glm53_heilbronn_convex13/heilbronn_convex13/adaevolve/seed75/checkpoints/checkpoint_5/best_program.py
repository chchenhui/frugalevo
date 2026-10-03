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


def _min_triangle_area(pts, tri_idx):
    """Vectorized smallest triangle area over all C(13,3)=286 triples."""
    return _tri_areas(pts, tri_idx).min()


def _min2_triangle_area(pts, tri_idx):
    """Smallest and second-smallest triangle areas (soft tiebreaker)."""
    s = np.sort(_tri_areas(pts, tri_idx))
    return s[0], s[1]


def _local_search(pts, tri_idx, pt_tris, pt_rest, rng, deadline):
    """Greedy coordinate-descent: move one point at a time to raise the min area.

    Incremental evaluation: when moving point i, only the 66 triangles that
    contain i change; the minimum over the other triangles is cached and
    recomputed only when a move is accepted. Accepts a move if it strictly
    increases the smallest triangle area, or increases the second-smallest
    while keeping the smallest unchanged (plateau-escaping tiebreaker).
    Uses shrinking step sizes and clips points to the unit square.
    Deterministic given pts and rng.
    """
    pts = pts.copy()
    best, best2 = _min2_triangle_area(pts, tri_idx)
    steps = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
    dirs = np.array([[1, 0], [-1, 0], [0, 1], [0, -1],
                     [1, 1], [1, -1], [-1, 1], [-1, -1]]) / np.sqrt(2)
    for step in steps:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            order = rng.permutation(13)
            for i in order:
                # minimum over triangles not involving point i (cached)
                if len(pt_rest[i]):
                    rest_min = _tri_areas(pts, pt_rest[i]).min()
                else:
                    rest_min = np.inf
                tri_i = pt_tris[i]
                for d in dirs:
                    cand = pts.copy()
                    cand[i] = np.clip(cand[i] + step * d, 0.0, 1.0)
                    ai = _tri_areas(cand, tri_i)
                    v1 = min(rest_min, ai.min())
                    if v1 > best + 1e-12:
                        pts = cand
                        best = v1
                        best2 = _min2_triangle_area(pts, tri_idx)[1]
                        improved = True
                    elif v1 > best - 1e-12:
                        # plateau move: accept if second-smallest improves
                        s = np.sort(np.concatenate([ai, [rest_min]]))
                        v2 = s[1] if len(s) > 1 else s[0]
                        if v2 > best2 + 1e-12:
                            pts = cand
                            best, best2 = v1, v2
                            improved = True
                if time.time() >= deadline:
                    return pts, best
    return pts, best


def heilbronn_convex13() -> np.ndarray:
    """
    Construct 13 points in the unit square maximizing the smallest triangle area.

    Approach: multi-start greedy local search. Starting configurations include
    a circle, perturbed circles, and seeded random points. Each start is refined
    by coordinate descent (move one point at a time in 8 directions with
    shrinking step sizes), keeping moves that increase the minimum triangle
    area over all 286 triples, with a second-smallest-area tiebreaker to
    escape plateaus. An intensification phase perturbs and re-optimizes around
    the incumbent. Fully deterministic (fixed seeds, fixed iteration orders).
    Falls back to the circle configuration on any failure.
    """
    n = 13
    # Precompute all triangle index triples
    tri_idx = np.array([(i, j, k) for i in range(n) for j in range(i + 1, n)
                        for k in range(j + 1, n)], dtype=int)
    # Per-point triangle index arrays: triangles containing i, and the rest
    contains = (tri_idx == np.arange(n)[:, None, None]).any(axis=2)  # (n, 286)
    pt_tris = [tri_idx[contains[i]] for i in range(n)]
    pt_rest = [tri_idx[~contains[i]] for i in range(n)]

    deadline = time.time() + 100.0  # generous but well under the limit
    rng = np.random.default_rng(seed=42)

    starts = []
    # Circle start (rescaled into unit square)
    ang = 2 * np.pi * np.arange(n) / n + 0.12
    circ = np.column_stack([0.5 + 0.49 * np.cos(ang), 0.5 + 0.49 * np.sin(ang)])
    starts.append(circ)
    # Perturbed circle starts, grid starts, and random starts
    for s in range(48):
        if s < 8:
            p = circ + 0.03 * rng.standard_normal((n, 2))
        elif s < 16:
            # perturbed hexagonal/triangular lattice
            gx, gy = np.meshgrid(np.linspace(0.05, 0.95, 4),
                                 np.linspace(0.05, 0.95, 4))
            p = np.column_stack([gx.ravel(), gy.ravel()])[:n]
            p = p + 0.05 * rng.standard_normal((n, 2))
        else:
            p = rng.random((n, 2))
        starts.append(np.clip(p, 0.0, 1.0))

    best_pts, best_val = circ, _min_triangle_area(circ, tri_idx)
    for start in starts:
        if time.time() >= deadline:
            break
        p, v = _local_search(start, tri_idx, pt_tris, pt_rest, rng, deadline)
        if v > best_val:
            best_val, best_pts = v, p

    # Intensification: perturb-and-reoptimize around the incumbent.
    for r in range(200):
        if time.time() >= deadline:
            break
        scale = max(0.02 * (1.0 - r / 200.0), 0.001)
        p0 = np.clip(best_pts + scale * rng.standard_normal((n, 2)), 0.0, 1.0)
        p, v = _local_search(p0, tri_idx, pt_tris, pt_rest, rng, deadline)
        if v > best_val:
            best_val, best_pts = v, p

    pts = np.asarray(best_pts, dtype=float)
    if pts.shape != (n, 2) or not np.all(np.isfinite(pts)):
        pts = circ  # safe fallback
    return pts


# EVOLVE-BLOCK-END
