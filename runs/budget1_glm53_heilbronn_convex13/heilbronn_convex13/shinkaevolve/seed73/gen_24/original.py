# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    rng = np.random.default_rng(seed=42)

    # Unit-area regular hexagon centered at origin as the convex region.
    # Regular hexagon with circumradius R has area (3*sqrt(3)/2) R^2 => R for area 1:
    R = np.sqrt(2.0 / (3.0 * np.sqrt(3.0)))
    hull_area = 1.0
    hex_dir = np.array([np.cos(np.pi / 3 * k + np.pi / 6) for k in range(6)])
    hex_y = np.array([np.sin(np.pi / 3 * k + np.pi / 6) for k in range(6)])

    def clip(pts):
        # Project points back inside hexagon: scale toward origin until inside.
        out = pts.copy()
        for k in range(6):
            nx, ny = hex_dir[k], hex_y[k]
            d = out[:, 0] * nx + out[:, 1] * ny - R
            viol = d > 1e-12
            if np.any(viol):
                out[viol, 0] -= d[viol] * nx
                out[viol, 1] -= d[viol] * ny
        return out

    # Initialization: center point + 12 points on two concentric rings with 3-fold symmetric jitter.
    pts = np.zeros((n, 2))
    pts[0] = 0.0
    ring1_r, ring2_r = 0.55 * R, 0.95 * R
    for i in range(6):
        a1 = np.pi / 3 * i
        a2 = np.pi / 3 * i + np.pi / 6
        pts[1 + i] = ring1_r * np.array([np.cos(a1), np.sin(a1)])
        pts[7 + i] = ring2_r * np.array([np.cos(a2), np.sin(a2)])

    # Small deterministic jitter to break degeneracies (collinear triples with center).
    pts += 0.02 * R * rng.standard_normal((n, 2))
    pts = clip(pts)

    idx = np.array([(i, j, k) for i in range(n) for j in range(i + 1, n) for k in range(j + 1, n)])

    def min_tri(p):
        a = p[idx[:, 0]]
        b = p[idx[:, 1]]
        c = p[idx[:, 2]]
        areas = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) -
                             (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
        return areas

    # Subgradient ascent on the minimum triangle area.
    step = 0.02 * R
    best_pts = pts.copy()
    best_val = min_tri(pts).min()
    for it in range(600):
        areas = min_tri(pts)
        m = areas.min()
        if m > best_val:
            best_val = m
            best_pts = pts.copy()
        # soft-min gradient: weight triangles near the minimum
        w = np.exp(-(areas - m) / max(1e-6, 0.02 * m + 1e-9))
        w /= w.sum()
        grads = np.zeros_like(pts)
        a = pts[idx[:, 0]]
        b = pts[idx[:, 1]]
        c = pts[idx[:, 2]]
        s = np.sign((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) -
                    (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
        ga = 0.5 * s[:, None] * np.stack([c[:, 1] - b[:, 1], b[:, 0] - c[:, 0]], axis=1)
        gb = 0.5 * s[:, None] * np.stack([a[:, 1] - c[:, 1], c[:, 0] - a[:, 0]], axis=1)
        gc = 0.5 * s[:, None] * np.stack([b[:, 1] - a[:, 1], a[:, 0] - b[:, 0]], axis=1)
        np.add.at(grads, idx[:, 0], w[:, None] * ga)
        np.add.at(grads, idx[:, 1], w[:, None] * gb)
        np.add.at(grads, idx[:, 2], w[:, None] * gc)
        # normalize gradient scale
        gn = np.linalg.norm(grads, axis=1, keepdims=True)
        gn[gn < 1e-12] = 1.0
        grads /= gn
        pts = clip(pts + step * grads)
        step *= 0.995

    pts = clip(best_pts)

    # ---- Direct pattern-search refinement on the true min-triangle objective ----
    # 16 directions at 22.5-degree increments break the ring symmetry and let
    # points settle into irregular optimal placements.
    ang = np.arange(16) * np.pi / 16.0
    dirs = np.stack([np.cos(ang), np.sin(ang)], axis=1)  # (16,2)

    def min_area_val(p):
        return min_tri(p).min()

    cur_val = min_area_val(pts)
    step2 = 0.01 * R
    while step2 > 1e-5 * R:
        improved = False
        for i in range(n):
            # candidate moves for point i: (16,2)
            cand = np.repeat(pts[i][None, :], 16, axis=0) + step2 * dirs
            cand = clip(cand)
            # evaluate all candidates at once: 16 x 286 areas
            vals = np.empty(16)
            base = pts.copy()
            for d in range(16):
                base[i] = cand[d]
                vals[d] = min_tri(base).min()
            d = int(np.argmax(vals))
            if vals[d] > cur_val + 1e-15:
                pts[i] = cand[d]
                cur_val = vals[d]
                improved = True
        if not improved:
            step2 *= 0.5

    pts = clip(pts)

    # Validate: finite, distinct
    assert np.all(np.isfinite(pts))
    assert pts.shape == (13, 2)
    return pts


# EVOLVE-BLOCK-END