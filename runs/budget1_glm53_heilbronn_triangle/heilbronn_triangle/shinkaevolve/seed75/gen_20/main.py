# EVOLVE-BLOCK-START
import numpy as np

VERTS = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2.0]])


def _in_triangle(pts, tol=1e-12):
    (x1, y1), (x2, y2), (x3, y3) = VERTS
    s = (x2 - x1) * (pts[:, 1] - y1) - (y2 - y1) * (pts[:, 0] - x1)
    t = (x3 - x2) * (pts[:, 1] - y2) - (y3 - y2) * (pts[:, 0] - x2)
    u = (x1 - x3) * (pts[:, 1] - y3) - (y1 - y3) * (pts[:, 0] - x3)
    return ((s >= -tol) & (t >= -tol) & (u >= -tol))


def _bary_repair(pts):
    A = np.array([[1.0, VERTS[0, 0], VERTS[0, 1]],
                  [1.0, VERTS[1, 0], VERTS[1, 1]],
                  [1.0, VERTS[2, 0], VERTS[2, 1]]])
    b = np.column_stack([np.ones(len(pts)), pts])
    lam = np.linalg.solve(A.T, b.T).T  # barycentric coords
    lam = np.clip(lam, 0.0, None)
    lam /= lam.sum(axis=1, keepdims=True)
    return lam @ VERTS


def _min_area(pts):
    n = len(pts)
    best = np.inf
    worst = None
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                a = 0.5 * abs((pts[j, 0] - pts[i, 0]) * (pts[k, 1] - pts[i, 1]) -
                              (pts[k, 0] - pts[i, 0]) * (pts[j, 1] - pts[i, 1]))
                if a < best:
                    best = a
                    worst = (i, j, k)
    return best, worst


def _initial_layouts():
    """Several deterministic, well-spread seed layouts (no near-degenerate triplets)."""
    s3 = np.sqrt(3.0)
    layouts = []
    # Layout A: hand-built known-good configuration (vertices + edge + interior)
    layouts.append(np.array([
        [0.0, 0.0], [1.0, 0.0], [0.5, s3 / 2.0],
        [0.5, 0.0], [0.25, s3 / 4.0], [0.75, s3 / 4.0],
        [0.25, s3 / 12.0], [0.75, s3 / 12.0],
        [0.5, s3 / 6.0], [0.125, s3 / 8.0], [0.875, s3 / 8.0],
    ]))
    # Layout B: uniform barycentric lattice-ish interior points
    rng = np.random.default_rng(777)
    layouts.append(np.array([
        [0.0, 0.0], [1.0, 0.0], [0.5, s3 / 2.0],
        [0.5, 0.0], [0.25, s3 / 4.0], [0.75, s3 / 4.0],
        [1.0 / 3.0, s3 / 9.0], [2.0 / 3.0, s3 / 9.0],
        [0.5, s3 / 3.0], [0.5, s3 / 18.0], [0.25, s3 / 36.0],
    ]))
    # Layout C: jittered version of A for extra diversity
    jitter = layouts[0] + rng.normal(scale=0.02, size=layouts[0].shape)
    layouts.append(jitter)
    return layouts


def _polish(pts, iters=1500, step=0.015):
    """Gradient-informed polish: move worst-triplet points perpendicular
    away from the line through the other two points (the exact direction
    that increases that triplet's signed area)."""
    for _ in range(iters):
        cur, worst = _min_area(pts)
        if worst is None:
            break
        i, j, k = worst
        p = pts
        # perpendicular "escape" directions for each of the 3 points
        dirs = {}
        dirs[i] = _perp(p[j], p[k], p[i])
        dirs[j] = _perp(p[i], p[k], p[j])
        dirs[k] = _perp(p[i], p[j], p[k])
        improved = False
        # try moving all three, then pairs, then single points
        trials = [(i, j, k), (i, j), (j, k), (i, k), (i,), (j,), (k,)]
        for trial in trials:
            cand = pts.copy()
            for m_ in trial:
                cand[m_] = pts[m_] + step * dirs[m_]
            cand = _bary_repair(cand)
            new, _ = _min_area(cand)
            if new > cur + 1e-15:
                pts = cand
                improved = True
                break
        if not improved:
            step *= 0.85
            if step < 1e-7:
                break
    return pts


def _perp(a, b, c):
    """Unit direction from point c away from line through a,b (increases |area|)."""
    ex, ey = b[0] - a[0], b[1] - a[1]
    L = np.hypot(ex, ey) + 1e-12
    ux, uy = ex / L, ey / L          # unit vector along edge
    dx, dy = c[0] - a[0], c[1] - a[1]
    # signed perpendicular distance component
    px, py = -uy * dx + ux * dy, ux * dx + uy * dy
    # perpendicular pointing away from edge, on the side where c lies
    if px >= 0:
        return np.array([-uy, ux])
    return np.array([uy, -ux])


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct 11 points inside the unit equilateral triangle maximizing the
    minimum triangle area (Heilbronn problem, n=11).

    Returns:
        points: np.ndarray of shape (11,2) of x,y coordinates.
    """
    s3 = np.sqrt(3.0)
    fallback = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.5, s3 / 2.0],
        [0.5, 0.0], [0.25, s3 / 4.0], [0.75, s3 / 4.0],
        [0.25, s3 / 12.0], [0.75, s3 / 12.0],
        [0.5, s3 / 6.0], [0.125, s3 / 8.0], [0.875, s3 / 8.0],
    ])
    try:
        best_pts, best_val = None, -np.inf
        for layout in _initial_layouts():
            cand = _bary_repair(np.asarray(layout, dtype=float))
            cand = _polish(cand)
            val, _ = _min_area(cand)
            if val > best_val:
                best_val, best_pts = val, cand
        if best_pts is None or best_pts.shape != (11, 2) or not np.all(np.isfinite(best_pts)):
            raise RuntimeError("bad result")
        pts = best_pts
    except Exception:
        pts = fallback
    return np.asarray(pts, dtype=float)


# EVOLVE-BLOCK-END