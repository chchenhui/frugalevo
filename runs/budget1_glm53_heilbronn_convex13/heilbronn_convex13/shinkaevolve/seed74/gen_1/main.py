# EVOLVE-BLOCK-START
import numpy as np


def _triangle_areas(pts):
    """Vectorized areas of all C(n,3) triangles."""
    n = len(pts)
    i, j, k = np.triu_indices(n, 3)[:3] if False else np.array(
        [(a, b, c) for a in range(n) for b in range(a + 1, n)
         for c in range(b + 1, n)]).T
    ax, ay = pts[i, 0], pts[i, 1]
    bx, by = pts[j, 0], pts[j, 1]
    cx, cy = pts[k, 0], pts[k, 1]
    area = 0.5 * np.abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
    return area, (i, j, k, ax, ay, bx, by, cx, cy)


def _softmin_grad(pts, i, j, k, beta=200.0):
    """Value & gradient of -1/beta * logsumexp(-beta*area_signed_asc)."""
    n = len(pts)
    ax, ay = pts[i, 0], pts[i, 1]
    bx, by = pts[j, 0], pts[j, 1]
    cx, cy = pts[k, 0], pts[k, 1]
    signed = 0.5 * ((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
    # weight triangles near the minimum (softmin weights)
    w = np.exp(-beta * np.abs(signed))
    w /= w.sum()
    # gradient of signed area wrt points
    grad = np.zeros_like(pts)
    gx = 0.5 * (by - cy)
    gy = 0.5 * (cx - bx)
    hx = 0.5 * (cy - ay)
    hy = 0.5 * (ax - cx)
    ix = 0.5 * (ay - by)
    iy = 0.5 * (bx - ax)
    s = np.sign(signed)
    grad[i, 0] += np.sum(w * gx * s)
    grad[i, 1] += np.sum(w * gy * s)
    grad[j, 0] += np.sum(w * hx * s)
    grad[j, 1] += np.sum(w * hy * s)
    grad[k, 0] += np.sum(w * ix * s)
    grad[k, 1] += np.sum(w * iy * s)
    val = np.min(np.abs(signed))
    return val, grad


def _barycentric_clip(pts, A, B, C):
    """Project points back inside triangle ABC (soft: shrink toward centroid)."""
    centroid = (A + B + C) / 3.0
    # iterative: if outside, move toward centroid until inside
    for _ in range(30):
        v0 = C - A
        v1 = B - A
        v2 = pts - A
        d00 = v0 @ v0
        d01 = v0 @ v1
        d11 = v1 @ v1
        d20 = v2 @ v0
        d21 = v2 @ v1
        den = d00 * d11 - d01 * d01
        u = (d11 * d20 - d01 * d21) / den  # coef on C
        v = (d00 * d21 - d01 * d20) / den  # coef on B
        inside = (u >= 0) & (v >= 0) & (u + v <= 1)
        if inside.all():
            break
        bad = ~inside
        pts[bad] = 0.5 * (pts[bad] + centroid)
    return pts


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on/inside a unit-area convex region
    (equilateral triangle) maximizing the smallest triangle area via
    deterministic softmin gradient ascent.

    Returns:
        points: np.ndarray of shape (13,2)
    """
    n = 13
    rng = np.random.default_rng(seed=42)

    # Unit-area equilateral triangle: side s, area = sqrt(3)/4 s^2 = 1
    s = np.sqrt(4.0 / np.sqrt(3.0))
    h = np.sqrt(3.0) / 2.0 * s
    A = np.array([0.0, 0.0])
    B = np.array([s, 0.0])
    C = np.array([s / 2.0, h])

    # --- Initialization: 3 corners + points along edges + interior lattice ---
    pts = [A, B, C]
    # 2 points on each edge (9 boundary total)
    for (P, Q) in [(A, B), (B, C), (C, A)]:
        for t in (1.0 / 3.0, 2.0 / 3.0):
            pts.append(P + t * (Q - P))
    # 4 interior points: centroid + 3 symmetric around centroid
    G = (A + B + C) / 3.0
    pts.append(G.copy())
    for (P, Q) in [(A, B), (B, C), (C, A)]:
        M = 0.5 * (P + Q)
        pts.append(G + 0.55 * (M - G))
    pts = np.array(pts, dtype=float)

    idx = np.array([(a, b, c) for a in range(n) for b in range(a + 1, n)
                    for c in range(b + 1, n)]).T
    i, j, k = idx

    best_pts = pts.copy()
    _, areas = _triangle_areas(pts)
    best_val = areas.min()

    # --- Multi-scale softmin gradient ascent ---
    lr = 0.004
    for it in range(400):
        beta = 150.0 + 50.0 * it / 400.0
        val, grad = _softmin_grad(pts, i, j, k, beta=beta)
        # normalize gradient scale
        gn = np.linalg.norm(grad)
        if gn > 0:
            grad /= gn
        new_pts = pts + lr * grad * s
        # random tiny perturbation for exploration (deterministic seed)
        new_pts += rng.normal(0, 1e-4 * s, new_pts.shape)
        new_pts = _barycentric_clip(new_pts, A, B, C)
        # keep corners pinned on vertices to preserve hull
        new_pts[0] = A
        new_pts[1] = B
        new_pts[2] = C
        _, areas = _triangle_areas(new_pts)
        nv = areas.min()
        if nv > best_val:
            best_val = nv
            best_pts = new_pts.copy()
            pts = new_pts
            lr *= 1.05
        else:
            lr *= 0.93
            if lr < 1e-6:
                break
            pts = new_pts  # allow exploration

    # ensure corners fixed (hull = triangle of area 1)
    best_pts[0], best_pts[1], best_pts[2] = A, B, C
    return best_pts


# EVOLVE-BLOCK-END
