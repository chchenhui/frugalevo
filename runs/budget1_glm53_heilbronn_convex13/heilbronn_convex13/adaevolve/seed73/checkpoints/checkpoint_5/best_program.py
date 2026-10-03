# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations


def _areas_and_idx(points):
    """Areas of all C(13,3)=286 triangles plus their vertex-index triples."""
    idx = np.array(list(combinations(range(len(points)), 3)))
    a = points[idx[:, 0]]
    b = points[idx[:, 1]]
    c = points[idx[:, 2]]
    areas = 0.5 * np.abs(
        (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
        - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    )
    return areas, idx


def _soft_objective(points, m=10, alpha=1.0):
    """Weighted sum of the m smallest triangle areas (smooth surrogate for min)."""
    areas, idx = _areas_and_idx(points)
    order = np.argsort(areas)
    w = np.exp(-alpha * np.arange(m))
    w = w / w.sum()
    return float(np.sum(w * areas[order[:m]]))


def _min_area(points):
    return float(_areas_and_idx(points)[0].min())


def heilbronn_convex13() -> np.ndarray:
    """
    Heilbronn triangle problem for n = 13.

    Approach: deterministic multi-start local search. Start from structured
    configurations (jittered grids, ring+center, random seeds), then hill-climb
    on a soft-min objective (weighted sum of smallest triangle areas) with a
    coarse-to-fine step schedule, moving points that participate in the
    smallest triangles. Finally, a greedy pass directly maximizes the true
    minimum triangle area. Points are kept in the unit square (convex region
    of unit area). Fully deterministic via fixed seeds.

    Returns:
        points: np.ndarray of shape (13,2) inside the unit square.
    """
    n = 13
    rng = np.random.default_rng(seed=12345)

    def clip(pts):
        return np.clip(pts, 0.0, 1.0)

    starts = []
    # jittered 4x4 grid
    side = 4
    grid = [(x / (side - 1), y / (side - 1)) for x in range(side) for y in range(side)]
    starts.append(clip(np.array(grid[:13]) + rng.normal(0, 0.03, (13, 2))))
    # ring + center
    ang = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    ring = np.stack([0.5 + 0.45 * np.cos(ang), 0.5 + 0.45 * np.sin(ang)], axis=1)
    starts.append(clip(np.vstack([ring, [0.5, 0.5]]) + rng.normal(0, 0.02, (13, 2))))
    # jittered triangular lattice
    tri = []
    for r in range(5):
        for s in range(5 - r):
            tri.append((0.1 + 0.2 * s + 0.1 * r, 0.1 + 0.2 * r * np.sqrt(3) / 2))
    tri = tri[:13]
    if len(tri) < 13:
        tri = tri + [(0.9, 0.9)] * (13 - len(tri))
    starts.append(clip(np.array(tri) + rng.normal(0, 0.02, (13, 2))))
    # random starts
    for _ in range(5):
        starts.append(rng.random((13, 2)))

    best_pts, best_min = None, -1.0

    for pts in starts:
        pts = pts.copy()
        cur = _soft_objective(pts)
        for step in (0.05, 0.02, 0.008, 0.003, 0.0012, 0.0005, 0.0002, 0.0001):
            improved = True
            it = 0
            while improved and it < 150:
                improved = False
                it += 1
                areas, idx = _areas_and_idx(pts)
                order = np.argsort(areas)
                cand = set()
                for t in idx[order[:10]]:
                    cand.update(t.tolist())
                for p in cand:
                    base = pts[p].copy()
                    blv, blp = cur, base
                    for d in range(16):
                        a2 = 2 * np.pi * d / 16 + rng.uniform(-0.2, 0.2)
                        cand_pos = clip(base + step * np.array([np.cos(a2), np.sin(a2)]))
                        pts[p] = cand_pos
                        v = _soft_objective(pts)
                        if v > blv:
                            blv, blp = v, cand_pos
                    pts[p] = blp
                    if blv > cur + 1e-12:
                        cur = blv
                        improved = True

        # Greedy fine-tune directly on the true minimum area
        for step in (0.002, 0.0008, 0.0003, 0.0001):
            improved = True
            it = 0
            while improved and it < 60:
                improved = False
                it += 1
                areas, idx = _areas_and_idx(pts)
                order = np.argsort(areas)
                cur_min = areas[order[0]]
                cand = set()
                for t in idx[order[:6]]:
                    cand.update(t.tolist())
                for p in cand:
                    base = pts[p].copy()
                    blv, blp = cur_min, base
                    for d in range(12):
                        a2 = 2 * np.pi * d / 12
                        cand_pos = clip(base + step * np.array([np.cos(a2), np.sin(a2)]))
                        pts[p] = cand_pos
                        v = _min_area(pts)
                        if v > blv + 1e-15:
                            blv, blp = v, cand_pos
                    pts[p] = blp
                    if blv > cur_min + 1e-15:
                        cur_min = blv
                        improved = True

        m = _min_area(pts)
        if m > best_min:
            best_min = m
            best_pts = pts.copy()

    best_pts = np.asarray(best_pts, dtype=float)
    if not np.all(np.isfinite(best_pts)):
        best_pts = np.random.default_rng(0).random((13, 2))
    return best_pts


# EVOLVE-BLOCK-END
