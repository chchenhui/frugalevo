# EVOLVE-BLOCK-START
import numpy as np

S3 = np.sqrt(3.0)
AREA = S3 / 4.0  # container area


def _triplet_indices(n=11):
    ii, jj, kk = np.array(np.meshgrid(np.arange(n), np.arange(n), np.arange(n))).reshape(3, -1)
    mask = (ii < jj) & (jj < kk)
    return ii[mask], jj[mask], kk[mask]


def _min_area(pts, idx):
    i, j, k = idx
    pi, pj, pk = pts[i], pts[j], pts[k]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    a = np.abs(cross)
    m = int(np.argmin(a))
    return 0.5 * a[m] / AREA, (int(i[m]), int(j[m]), int(k[m]))


def _bary_clip(pts):
    """Project points into the triangle via clipped barycentric coords."""
    h = S3 / 2.0
    x, y = pts[:, 0], pts[:, 1]
    a = y / h
    c = x - 0.5 * a
    b = 1.0 - a - c
    lam = np.stack([a, b, c], axis=1)
    lam = np.clip(lam, 0.0, None)
    lam /= lam.sum(axis=1, keepdims=True)
    a, b, c = lam[:, 0], lam[:, 1], lam[:, 2]
    return np.column_stack([c + 0.5 * a, a * h])


def _perp_dir(pa, pb, pc):
    """Unit vector from c perpendicular to line ab, on c's side (increases |area|)."""
    ex, ey = pb[0] - pa[0], pb[1] - pa[1]
    L = np.hypot(ex, ey) + 1e-12
    ux, uy = ex / L, ey / L
    dx, dy = pc[0] - pa[0], pc[1] - pa[1]
    side = -uy * dx + ux * dy
    if side >= 0:
        return np.array([-uy, ux])
    return np.array([uy, -ux])


def _polish(pts, idx, rng, iters=900, step=0.02):
    best_val, _ = _min_area(pts, idx)
    for it in range(iters):
        cur, worst = _min_area(pts, idx)
        i, j, k = worst
        d = {
            i: _perp_dir(pts[j], pts[k], pts[i]),
            j: _perp_dir(pts[i], pts[k], pts[j]),
            k: _perp_dir(pts[i], pts[j], pts[k]),
        }
        improved = False
        for trial in [(i, j, k), (i, j), (i, k), (j, k), (i,), (j,), (k,)]:
            cand = pts.copy()
            for m in trial:
                cand[m] = pts[m] + step * d[m]
            cand = _bary_clip(cand)
            new, _ = _min_area(cand, idx)
            if new > cur + 1e-14:
                pts = cand
                improved = True
                break
        if not improved:
            step *= 0.85
            if step < 1e-7:
                break
        # occasional random kick to escape local optima
        if it % 250 == 249 and it + 1 < iters:
            v, _ = _min_area(pts, idx)
            if v >= best_val - 1e-15:
                best_val = v
                best_pts = pts.copy()
            cand = pts.copy()
            m = rng.integers(0, 11)
            cand[m] = cand[m] + rng.normal(0.0, 0.02, size=2)
            cand = _bary_clip(cand)
            nv, _ = _min_area(cand, idx)
            if nv > 0.5 * v:
                pts = cand
                step = max(step, 0.01)
    v, _ = _min_area(pts, idx)
    return pts, v


def _seeds():
    s = S3
    layouts = []
    # A: vertices + edge midpoints + symmetric interior cluster
    layouts.append(np.array([
        [0.0, 0.0], [1.0, 0.0], [0.5, s / 2.0],
        [0.5, 0.0], [0.25, s / 4.0], [0.75, s / 4.0],
        [0.5, s / 18.0], [0.5, s / 4.0],
        [0.25, s / 12.0], [0.75, s / 12.0], [0.5, s / 3.0],
    ]))
    # B: centroid + two pentagonal rings (symmetric-ish)
    cx, cy = 0.5, s / 6.0
    B = [[cx, cy]]
    for r, rot in ((0.30, 0.0), (0.18, np.pi / 5.0)):
        for t in np.linspace(0, 2 * np.pi, 5, endpoint=False) + rot:
            B.append([cx + r * np.cos(t), cy + r * np.sin(t)])
    layouts.append(np.array(B))
    # C: jitter of A
    rng = np.random.default_rng(777)
    layouts.append(layouts[0] + rng.normal(scale=0.02, size=layouts[0].shape))
    # D: coarse triangular lattice, 11 of 15 nodes
    D = []
    m = 4
    for i in range(m + 1):
        for j in range(m + 1 - i):
            kk = m - i - j
            D.append([(j + 0.5 * kk) / m, (kk * s / 2.0) / m])
    layouts.append(np.array(D[:11]))
    return layouts


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct an arrangement of 11 points on or inside an equilateral triangle
    maximizing the smallest triangle area (Heilbronn problem, n = 11).

    Returns:
        points: np.ndarray of shape (11,2).
    """
    n = 11
    s = S3
    fallback = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.5, s / 2.0],
        [0.5, 0.0], [0.25, s / 4.0], [0.75, s / 4.0],
        [0.25, s / 12.0], [0.75, s / 12.0],
        [0.5, s / 6.0], [0.125, s / 8.0], [0.875, s / 8.0],
    ])
    try:
        idx = _triplet_indices(n)
        rng = np.random.default_rng(20240517)
        best_pts, best_val = None, -np.inf
        for layout in _seeds():
            pts = _bary_clip(np.asarray(layout, dtype=float))
            # ensure 11 distinct points (nudge duplicates)
            for a in range(n):
                for b in range(a + 1, n):
                    if np.linalg.norm(pts[a] - pts[b]) < 1e-9:
                        pts[b] = pts[b] + np.array([1e-4, 1e-4])
            pts = _bary_clip(pts)
            pts, val = _polish(pts, idx, rng)
            if val > best_val:
                best_val, best_pts = val, pts
        # final validity checks
        if best_pts is None or best_pts.shape != (11, 2) or not np.all(np.isfinite(best_pts)):
            raise RuntimeError("bad result")
        # confirm points inside triangle (barycentric nonneg)
        chk = _bary_clip(best_pts)
        if not np.allclose(chk, best_pts, atol=1e-7):
            best_pts = chk
        v, _ = _min_area(best_pts, idx)
        if v <= 0:
            raise RuntimeError("degenerate")
        return best_pts
    except Exception:
        return fallback


# EVOLVE-BLOCK-END