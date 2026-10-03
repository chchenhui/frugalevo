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

    idx = np.array([(i, j, k) for i in range(n) for j in range(i + 1, n) for k in range(j + 1, n)])

    def min_tri(p):
        a = p[idx[:, 0]]
        b = p[idx[:, 1]]
        c = p[idx[:, 2]]
        areas = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) -
                             (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
        return areas

    def optimize(p0):
        # Hard-min vertex repulsion: enlarge only the few smallest triangles,
        # with backtracking so the true minimum never decreases.
        p = clip(p0.copy())
        bv = min_tri(p).min()
        bp = p.copy()
        step = 0.05 * R
        for it in range(2500):
            areas = min_tri(p)
            order = np.argsort(areas)
            m = areas[order[0]]
            if m > bv:
                bv = m
                bp = p.copy()
            grads = np.zeros_like(p)
            for t in order[:8]:
                i, j, l = idx[t]
                a, b, c = p[i], p[j], p[l]
                s = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
                s = 1.0 if s >= 0 else -1.0
                grads[i] += 0.5 * s * np.array([c[1] - b[1], b[0] - c[0]])
                grads[j] += 0.5 * s * np.array([a[1] - c[1], c[0] - a[0]])
                grads[l] += 0.5 * s * np.array([b[1] - a[1], a[0] - b[0]])
            gn = np.linalg.norm(grads, axis=1, keepdims=True)
            gn[gn < 1e-12] = 1.0
            grads /= gn
            cand = clip(p + step * grads)
            if min_tri(cand).min() >= m - 1e-12:
                p = cand
                step *= 0.999
            else:
                step *= 0.6
                if step < 1e-5 * R:
                    break
        return clip(bp), bv

    # Multi-start over ring geometries and deterministic seeds; keep global best.
    best_pts = None
    best_val = -1.0
    for r1f, r2f, jig in [(0.55, 0.95, 0.02), (0.40, 0.90, 0.05),
                          (0.60, 1.00, 0.03), (0.30, 0.85, 0.06),
                          (0.50, 1.00, 0.00)]:
        for seed in (42, 123, 7):
            rr = np.random.default_rng(seed=seed)
            p0 = np.zeros((n, 2))
            for i in range(6):
                a1 = np.pi / 3 * i
                a2 = np.pi / 3 * i + np.pi / 6
                p0[1 + i] = r1f * R * np.array([np.cos(a1), np.sin(a1)])
                p0[7 + i] = r2f * R * np.array([np.cos(a2), np.sin(a2)])
            p0 += jig * R * rr.standard_normal((n, 2))
            bp, bv = optimize(p0)
            if bv > best_val:
                best_val = bv
                best_pts = bp
    pts = clip(best_pts)

    # Validate: finite, distinct
    assert np.all(np.isfinite(pts))
    assert pts.shape == (13, 2)
    return pts


# EVOLVE-BLOCK-END