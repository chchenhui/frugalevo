# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations

N = 13
IDX = np.array(list(combinations(range(N), 3)), dtype=int)  # (286,3)
I0, I1, I2 = IDX[:, 0], IDX[:, 1], IDX[:, 2]


def _areas(pts):
    a = pts[I0]; b = pts[I1]; c = pts[I2]
    return 0.5 * np.abs((b[:,0]-a[:,0])*(c[:,1]-a[:,1]) - (b[:,1]-a[:,1])*(c[:,0]-a[:,0]))


def _hull_area(pts):
    P = pts[np.lexsort((pts[:,1], pts[:,0]))]
    keep = np.ones(len(P), bool)
    keep[1:] = np.any(P[1:] != P[:-1], axis=1)
    P = P[keep]
    if len(P) < 3: return 1e-12
    def build(seq):
        h = []
        for pt in seq:
            while len(h) >= 2 and (
                (h[-1][0]-h[-2][0])*(pt[1]-h[-2][1]) -
                (h[-1][1]-h[-2][1])*(pt[0]-h[-2][0])) <= 0:
                h.pop()
            h.append(pt)
        return h
    hull = build(P)[:-1] + build(P[::-1])[:-1]
    if len(hull) < 3: return 1e-12
    h = np.asarray(hull)
    x, y = h[:,0], h[:,1]
    return 0.5*abs(np.dot(x, np.roll(y,-1)) - np.dot(y, np.roll(x,-1)))


def _score(pts, ha=None):
    if ha is None: ha = _hull_area(pts)
    if ha < 1e-12: return 0.0
    return _areas(pts).min() / ha


def _worst_verts(pts, k_tri=10):
    a = _areas(pts)
    order = np.argsort(a)[:k_tri]
    return np.unique(IDX[order].ravel()), a


def _relax(pts, rng, iters, step0, step1, t0=0.0, t1=0.0, k_tri=10):
    """Targeted moves: perturb only vertices of the worst triangles."""
    pts = pts.copy()
    ha = _hull_area(pts)
    best = _areas(pts).min() / ha
    best_pts = pts.copy()
    cur = best
    for it in range(iters):
        f = it / iters
        step = step0 * (step1/step0)**f
        temp = t0 * (t1/t0)**f if t0 > 0 else 0.0
        verts, _ = _worst_verts(pts, k_tri)
        vi = verts[rng.integers(len(verts))]
        old = pts[vi].copy()
        # bias move away from centroid of other worst-triangle vertices
        _, a = _worst_verts(pts, k_tri)
        worst_set = IDX[np.argsort(a)[:k_tri]]
        mask = np.any(worst_set == vi, axis=1)
        others = pts[np.unique(worst_set[mask].ravel())]
        direction = pts[vi] - others.mean(axis=0)
        nrm = np.linalg.norm(direction)
        if nrm > 1e-9:
            direction = direction / nrm
            cand = old + step * (0.6*direction + 0.4*rng.normal(0, 1, 2))
        else:
            cand = old + rng.normal(0, step, 2)
        pts[vi] = cand
        ha2 = _hull_area(pts)
        s = _areas(pts).min() / ha2 if ha2 > 1e-12 else 0.0
        acc = s >= cur or (temp > 0 and rng.random() < np.exp((s-cur)/max(temp,1e-12)))
        if acc:
            cur = s; ha = ha2
            if s > best:
                best = s; best_pts = pts.copy()
        else:
            pts[vi] = old
    return best_pts, best


def _polish(pts, max_rounds=400):
    """Fine greedy: exact coordinate + normal-direction moves on worst vertices."""
    pts = pts.copy()
    ha = _hull_area(pts)
    best = _areas(pts).min() / ha
    step = 0.004
    for _ in range(max_rounds):
        verts, a = _worst_verts(pts, 12)
        improved = False
        for vi in verts:
            for mode in ("x", "y", "n"):
                if mode in ("x", "y"):
                    dirs = [np.eye(2)[0 if mode == "x" else 1]]
                else:
                    # push away from the line through the two other worst-tri verts
                    worst_set = IDX[np.argsort(a)[:12]]
                    m = np.any(worst_set == vi, axis=1)
                    tri = worst_set[m][0]
                    j, k2 = [int(x) for x in tri if x != vi]
                    d = pts[k2] - pts[j]
                    nn = np.hypot(*d) + 1e-12
                    nd = np.array([-d[1], d[0]]) / nn
                    side = d[0]*(pts[vi,1]-pts[j,1]) - d[1]*(pts[vi,0]-pts[j,0])
                    if side < 0: nd = -nd
                    dirs = [nd, -nd]
                for d in dirs:
                    for sgn in (1, -1):
                        cand = pts.copy()
                        cand[vi] = pts[vi] + sgn * step * d
                        ha2 = _hull_area(cand)
                        if ha2 < 1e-12: continue
                        s = _areas(cand).min() / ha2
                        if s > best + 1e-15:
                            pts, best, ha = cand, s, ha2
                            improved = True
        if not improved:
            step *= 0.5
            if step < 1e-5:
                break
    return pts, best


def _seeds():
    seeds = []
    # double rings (hexagon + hexagon + center)
    for r in (0.5, 0.58, 0.66, 0.42):
        t = np.linspace(0, 2*np.pi, 7)[:-1]
        outer = np.c_[np.cos(t), np.sin(t)]
        inner = np.c_[r*np.cos(t+np.pi/6), r*np.sin(t+np.pi/6)]
        seeds.append(np.vstack([outer, inner, [[0, 0]]]))
    # triangle-based: 3 corners + 6 mid-edges + 3 inner + center
    for rm, ri in ((0.55, 0.30), (0.62, 0.25), (0.48, 0.34)):
        pts = np.zeros((N, 2))
        for i in range(3):
            a = np.pi/2 + i*2*np.pi/3
            pts[i] = [np.cos(a), np.sin(a)]
        for i in range(6):
            a = np.pi/2 + np.pi/6 + i*np.pi/3
            pts[3+i] = [rm*np.cos(a), rm*np.sin(a)]
        for i in range(3):
            a = np.pi/2 + np.pi/3 + i*2*np.pi/3
            pts[9+i] = [ri*np.cos(a), ri*np.sin(a)]
        seeds.append(pts)
    # 12-ring + center
    t = np.linspace(0, 2*np.pi, 13)[:-1]
    seeds.append(np.vstack([np.c_[np.cos(t), np.sin(t)], [[0, 0]]]))
    return seeds


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points maximizing the smallest triangle
    area normalized by convex hull area. Deterministic targeted relaxation
    focused on the worst (smallest-area) triangles.
    """
    best_pts, best_val = None, -1.0
    for si, seed in enumerate(_seeds()):
        rng = np.random.default_rng(1000 + si)
        pts, v = _relax(seed, rng, iters=1500, step0=0.06, step1=0.003,
                        t0=0.003, t1=1e-6, k_tri=10)
        # reheat pass
        rng2 = np.random.default_rng(2000 + si)
        pts2, v2 = _relax(pts, rng2, iters=800, step0=0.02, step1=0.001,
                          t0=0.001, t1=1e-7, k_tri=8)
        if v2 > v: pts, v = pts2, v2
        pts, v = _polish(pts)
        if v > best_val:
            best_val, best_pts = v, pts

    # joint refinement pass from global best
    pts = best_pts.copy()
    for T, it in ((0.0008, 1200), (0.00005, 800)):
        rng3 = np.random.default_rng(9999)
        pts, v = _relax(pts, rng3, iters=it, step0=0.004, step1=0.0005,
                        t0=T, t1=1e-8, k_tri=8)
    pts, v = _polish(pts)
    if v > best_val:
        best_val, best_pts = v, pts

    # normalize hull to unit area (scale-invariance)
    ha = _hull_area(best_pts)
    if ha > 1e-12:
        c = best_pts.mean(axis=0)
        best_pts = (best_pts - c) / np.sqrt(ha) + c
    return best_pts


# EVOLVE-BLOCK-END
