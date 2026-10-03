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

    def hull_area(p):
        order = np.lexsort((p[:, 1], p[:, 0]))
        hidx = []
        for chain_in in (order, order[::-1]):
            chain = []
            for i in chain_in:
                while len(chain) >= 2:
                    o, a = p[chain[-2]], p[chain[-1]]
                    if (a[0]-o[0])*(p[i,1]-o[1]) - (a[1]-o[1])*(p[i,0]-o[0]) <= 0:
                        chain.pop()
                    else:
                        break
                chain.append(i)
            hidx.extend(chain[:-1])
        hidx = sorted(set(hidx))
        V = p[hidx]
        m = len(hidx)
        s = 0.0
        for i in range(m):
            x1, y1 = V[i]
            x2, y2 = V[(i + 1) % m]
            s += x1 * y2 - x2 * y1
        return abs(s) * 0.5

    def score(p):
        ha = hull_area(p)
        if ha < 1e-12:
            return 0.0
        return min_tri(p).min() / ha

    def min_grad(p):
        # Ascent gradient of the smallest-triangle area wrt each point.
        areas = min_tri(p)
        order = np.argsort(areas)
        grads = np.zeros_like(p)
        for t in order[:12]:
            i, j, l = idx[t]
            a, b, c = p[i], p[j], p[l]
            s = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            s = 1.0 if s >= 0 else -1.0
            grads[i] += 0.5 * s * np.array([c[1] - b[1], b[0] - c[0]])
            grads[j] += 0.5 * s * np.array([a[1] - c[1], c[0] - a[0]])
            grads[l] += 0.5 * s * np.array([b[1] - a[1], a[0] - b[0]])
        gn = np.linalg.norm(grads, axis=1, keepdims=True)
        gn[gn < 1e-12] = 1.0
        return grads / gn

    def optimize(p0, seed, sweeps=300):
        # Maximize score = min_tri_area / hull_area with adaptive per-point steps.
        rng = np.random.default_rng(seed)
        p = p0.copy()
        cur = score(p)
        bp, bv = p.copy(), cur
        steps = np.full((n, 1), 0.05 * R)
        for sw in range(sweeps):
            improved_any = False
            skipped = np.zeros(n, dtype=bool)
            for i in range(n):
                improved = False
                # Gradient-guided trial
                g = min_grad(p)
                cand = p.copy()
                cand[i] += steps[i, 0] * g[i]
                v = score(cand)
                if v > cur + 1e-14:
                    p, cur, improved_any, improved = cand, v, True, True
                else:
                    # Random perturbation trial
                    for _ in range(3):
                        cand = p.copy()
                        cand[i] += steps[i, 0] * rng.standard_normal(2)
                        v = score(cand)
                        if v > cur + 1e-14:
                            p, cur, improved_any, improved = cand, v, True, True
                            break
                if improved:
                    steps[i, 0] = min(steps[i, 0] * 1.3, 0.25 * R)
                    if cur > bv:
                        bv, bp = cur, p.copy()
                else:
                    steps[i, 0] *= 0.5
                    skipped[i] = True
            # Re-grow steps for points stuck for many sweeps
            if np.all(steps < 1e-6 * R):
                break
            if sw % 40 == 39:
                steps = np.maximum(steps, 0.01 * R)
        return bp, bv

    # Multi-start over ring geometries and deterministic seeds; keep global best.
    best_pts = None
    best_val = -1.0
    for r1f, r2f, jig in [(0.55, 0.95, 0.02), (0.40, 0.90, 0.05),
                          (0.60, 1.00, 0.03), (0.30, 0.85, 0.06),
                          (0.50, 1.00, 0.00)]:
        for seed in (42, 123, 7):
            rr = np.random.default_rng(seed=seed)
            p0 = np.zeros((n, 2))
            p0[0] = 0.0
            for i in range(6):
                a1 = np.pi / 3 * i
                a2 = np.pi / 3 * i + np.pi / 6
                p0[1 + i] = r1f * R * np.array([np.cos(a1), np.sin(a1)])
                p0[7 + i] = r2f * R * np.array([np.cos(a2), np.sin(a2)])
            p0 += jig * R * rr.standard_normal((n, 2))
            p0 = clip(p0)
            bp, bv = optimize(p0, seed=seed, sweeps=250)
            # Also try a purely random interior start
            p0b = clip(0.8 * R * (2 * rr.random((n, 2)) - 1.0))
            bp2, bv2 = optimize(p0b, seed=seed + 1, sweeps=250)
            if bv2 > bv:
                bp, bv = bp2, bv2
            if bv > best_val:
                best_val = bv
                best_pts = bp
    pts = clip(best_pts)
    # Normalize so the convex hull of the points has unit area (score invariant).
    ha = hull_area(pts)
    if ha > 1e-12:
        c = pts.mean(axis=0)
        pts = (pts - c) / np.sqrt(ha) + c
        pts = clip(pts)

    # Validate: finite, distinct
    assert np.all(np.isfinite(pts))
    assert pts.shape == (13, 2)
    return pts


# EVOLVE-BLOCK-END