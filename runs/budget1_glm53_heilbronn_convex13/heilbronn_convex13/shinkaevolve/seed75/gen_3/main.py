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
    from itertools import combinations

    n = int(n)

    tri_idx = np.array(list(combinations(range(n), 3)))

    def hull_area(pts):
        P = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
        def cross(o, a, b):
            return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
        lower = []
        for p in P:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        upper = []
        for p in P[::-1]:
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
        hull = np.array(lower[:-1] + upper[:-1])
        s = 0.0
        m = len(hull)
        for i in range(m):
            x1, y1 = hull[i]
            x2, y2 = hull[(i+1) % m]
            s += x1*y2 - x2*y1
        return abs(s) / 2.0

    def min_tri_area(pts):
        a = pts[tri_idx[:, 0]]
        b = pts[tri_idx[:, 1]]
        c = pts[tri_idx[:, 2]]
        areas = 0.5 * np.abs(
            (b[:, 0]-a[:, 0])*(c[:, 1]-a[:, 1]) -
            (b[:, 1]-a[:, 1])*(c[:, 0]-a[:, 0])
        )
        return float(np.min(areas))

    def score(pts):
        ha = hull_area(pts)
        if ha <= 1e-12:
            return -1.0
        return min_tri_area(pts) / ha

    def clip(pts):
        return np.clip(pts, 0.0, 1.0)

    # ---- candidate generation ----
    candidates = []

    # 1. 3-fold symmetric: triangle corners + points along edges + interior ring
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3.0)/2.0]])
    ctr = corners.mean(axis=0)
    for ts in ([1/3, 2/3], [0.25, 0.75]):
        for radii in ([0.30, 0.30, 0.30, 0.30], [0.25, 0.35, 0.25, 0.35]):
            pts = list(corners)
            for e in range(3):
                p = corners[e]
                q = corners[(e+1) % 3]
                for t in ts:
                    pts.append(p + t*(q-p))
            for k, r in enumerate(radii):
                a = np.pi/2 + 2*np.pi*k/len(radii)
                pts.append(ctr + r*np.array([np.cos(a), np.sin(a)]) / np.sqrt(3.0))
            cand = np.array(pts[:n])
            if cand.shape == (n, 2):
                candidates.append(cand)

    # 2. ring configurations with jitter
    for m_ring in (12, 13):
        for jitter in (0.0, 0.03):
            angles = np.linspace(0, 2*np.pi, m_ring, endpoint=False) + 0.3
            pts = 0.5 + 0.48*np.stack([np.cos(angles), np.sin(angles)], axis=1)
            if m_ring < n:
                pts = np.vstack([pts, [[0.5, 0.5]]])
            if jitter > 0:
                pts = pts + rng.normal(0, jitter, pts.shape)
            candidates.append(np.clip(pts[:n], 0, 1))

    # 3. random uniform seeds
    for _ in range(5):
        candidates.append(rng.uniform(0.02, 0.98, size=(n, 2)))

    # 4. perturbed grid
    gx, gy = np.meshgrid(np.linspace(0.1, 0.9, 4), np.linspace(0.1, 0.9, 4))
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1)[:n]
    candidates.append(grid + rng.normal(0, 0.03, grid.shape))

    # ---- deterministic hill-climbing refinement ----
    def refine(pts, init_step=0.02, iters=300):
        pts = clip(np.asarray(pts, dtype=float).copy())
        best = score(pts)
        step = init_step
        for _ in range(iters):
            improved = False
            for i in rng.permutation(n):
                for _ in range(8):
                    d = rng.normal(size=2)
                    d /= (np.linalg.norm(d) + 1e-12)
                    trial = pts.copy()
                    trial[i] = pts[i] + step*d
                    trial = clip(trial)
                    s = score(trial)
                    if s > best + 1e-12:
                        pts = trial
                        best = s
                        improved = True
            if not improved:
                step *= 0.5
                if step < 1e-6:
                    break
        return pts, best

    best_pts = None
    best_score = -np.inf
    for cand in candidates:
        try:
            p, s = refine(cand)
        except Exception:
            continue
        if s > best_score:
            best_score = s
            best_pts = p

    if best_pts is None or best_pts.shape != (n, 2) or not np.all(np.isfinite(best_pts)):
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        best_pts = 0.5 + 0.48*np.stack([np.cos(angles), np.sin(angles)], axis=1)

    points = np.asarray(best_pts, dtype=float)
    return points


# EVOLVE-BLOCK-END