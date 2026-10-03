# EVOLVE-BLOCK-START
import numpy as np

_N = 13
_BENCH = 0.030936889034895654
_EPS = 1e-9


def _hull_area(pts):
    p = np.vstack([pts, pts[0]])
    return 0.5 * abs(np.sum(p[:-1, 0] * p[1:, 1] - p[:-1, 1] * p[1:, 0]))


def _inside_hull(pts, verts, iters=8):
    """Project points back inside convex hull (triangle verts) via clamping + clipping iterations."""
    verts = np.asarray(verts, float)
    centroid = verts.mean(axis=0)
    out = pts.copy()
    # scale hull so area = 1? we keep unit triangle; points scaled inside later
    for _ in range(iters):
        moved = False
        for i in range(len(verts)):
            a, b = verts[i], verts[(i + 1) % len(verts)]
            e = b - a
            nrm = np.array([e[1], -e[0]])
            nrm = nrm / (np.linalg.norm(nrm) + _EPS)
            # inside direction: toward centroid
            s = np.dot(centroid - a, nrm)
            if s < 0:
                nrm = -nrm
            d = (out - a) @ nrm
            bad = d > -1e-9
            if np.any(bad):
                proj = out[bad] - np.outer(d[bad] + 1e-9, nrm)
                out[bad] = proj
                moved = True
        if not moved:
            break
    return out


def _min_tri_area(pts, k=5):
    n = len(pts)
    A = []
    idx = []
    for i in range(n):
        for j in range(i + 1, n):
            for k2 in range(j + 1, n):
                idx.append((i, j, k2))
                A.append(abs(np.cross(pts[j] - pts[i], pts[k2] - pts[i])) * 0.5)
    return np.array(A), idx


def _score(pts, beta=800.0):
    areas, _ = _min_tri_area(pts)
    ha = _hull_area(pts)
    if ha < _EPS:
        return -1e9, 0.0
    soft = -np.log(np.sum(np.exp(-beta * areas / ha))) / beta
    return soft, areas.min() / ha


def _grad(pts, beta=800.0):
    n = len(pts)
    areas, idx = _min_tri_area(pts)
    ha = _hull_area(pts)
    na = areas / ha
    w = np.exp(-beta * (na - na.min()))
    w = w / (np.sum(w) + _EPS)
    G = np.zeros_like(pts)
    for t, (i, j, k) in enumerate(idx):
        a = areas[t]
        if a < _EPS:
            continue
        p_i, p_j, p_k = pts[i], pts[j], pts[k]
        # d(area)/d(points) for |cross|*0.5
        cr = np.cross(p_j - p_i, p_k - p_i)
        s = np.sign(cr) if abs(cr) > 1e-15 else 0.0
        if s == 0.0:
            continue
        gi = s * 0.5 * np.array([-(p_k[1] - p_j[1]), (p_k[0] - p_j[0])]) / ha
        gj = s * 0.5 * np.array([(p_k[1] - p_i[1]), -(p_k[0] - p_i[0])]) / ha
        gk = s * 0.5 * np.array([(p_j[1] - p_i[1]), -(p_j[0] - p_i[0])]) / ha
        c = w[t] * a / (ha * ha)  # chain via na = a/ha treated ha const
        G[i] += w[t] * gi - c * 0  # simplified: main term
        G[j] += w[t] * gj
        G[k] += w[t] * gk
    return G


def _optimize(pts0, verts, iters=300, lr=0.02):
    pts = pts0.copy()
    best = pts.copy()
    best_m = 0.0
    for it in range(iters):
        G = _grad(pts)
        # ascent on soft min-area (gradient signs: we maximize)
        pts = pts + lr * G * _N
        pts = _inside_hull(pts, verts)
        _, m = _score(pts)
        if m > best_m:
            best_m = m
            best = pts.copy()
        lr *= 0.995
    return best, best_m


def heilbronn_convex13() -> np.ndarray:
    # Unit-area equilateral-ish triangle vertices (regular triangle, area 1)
    s = np.sqrt(4.0 / np.sqrt(3.0))  # side for area 1
    h = np.sqrt(3.0) / 2.0 * s
    verts = np.array([[0.0, 0.0], [s, 0.0], [s / 2.0, h]])
    centroid = verts.mean(axis=0)

    def to_bary_dir(t):
        # place point by barycentric-like coords then clamp
        return t

    rng = np.random.default_rng(seed=42)
    starts = []
    # 3-fold symmetric start: rings of points in triangle
    for r, cnt in [(0.18, 3), (0.45, 3), (0.75, 3), (1.0, 3)]:
        ring = []
        for m2 in range(3):
            ang = 2 * np.pi * m2 / 3
            v = np.array([np.cos(ang), np.sin(ang)]) * r
            ring.append(centroid + v * s * 0.5)
        ring = _inside_hull(np.array(ring), verts)
        starts.extend(list(ring))
    base = np.array(starts[:12])
    center = centroid.reshape(1, 2)
    s0 = np.vstack([base, center])

    best_pts, best_m = _optimize(s0, verts, iters=350, lr=0.02)

    # a few perturbed restarts (deterministic seed)
    for t in range(4):
        pert = s0 + rng.normal(0, 0.03, s0.shape)
        pert = _inside_hull(pert, verts)
        p, m = _optimize(pert, verts, iters=250, lr=0.02)
        if m > best_m + 1e-6:
            best_pts, best_m = p, m

    # ensure final points inside hull & valid
    best_pts = _inside_hull(best_pts, verts)
    # tiny anti-degeneracy jitter avoided; keep deterministic
    if not np.all(np.isfinite(best_pts)):
        best_pts = s0
    return best_pts.astype(float)


# EVOLVE-BLOCK-END
