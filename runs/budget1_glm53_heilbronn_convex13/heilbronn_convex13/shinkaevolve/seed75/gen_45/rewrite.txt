# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations

_N = 13
_TRI = np.array(list(combinations(range(_N), 3))))

# Precompute row indices of triangles containing each point
_WITH = [np.where((_TRI == i).any(axis=1))[0] for i in range(_N)]


def _hull_area(pts):
    P = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    keep = np.ones(len(P), dtype=bool)
    if len(P) > 1:
        keep[1:] = np.any(P[1:] != P[:-1], axis=1)
    P = P[keep]
    if len(P) < 3:
        return 0.0
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lo = []
    for p in P:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    up = []
    for p in reversed(P):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    hull = lo[:-1] + up[:-1]
    if len(hull) < 3:
        return 0.0
    H = np.asarray(hull)
    x, y = H[:, 0], H[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _all_areas(pts):
    p = pts[_TRI]
    a = p[:, 0]
    v1 = p[:, 1] - a
    v2 = p[:, 2] - a
    return 0.5 * np.abs(v1[:, 0]*v2[:, 1] - v1[:, 1]*v2[:, 0])


def _point_areas(pts, i, rows):
    """Areas of triangles containing point i (66 of them)."""
    others = [j for j in range(_N) if j != i]
    o = np.asarray(others)
    jj, kk = np.triu_indices(len(o), 1)
    pj = pts[o[jj]]
    pk = pts[o[kk]]
    pi = pts[i]
    return 0.5 * np.abs((pj[:, 0]-pi[0])*(pk[:, 1]-pi[1]) -
                        (pj[:, 1]-pi[1])*(pk[:, 0]-pi[0]))


def _rescale(pts):
    ha = _hull_area(pts)
    if ha > 1e-12:
        pts = pts / np.sqrt(ha)
    return pts


def _tri_grad(pts, tri):
    """Exact gradient of (unsigned) triangle area wrt each of the 3 vertices.
    Returns (3,2) array: each vertex pushed away from its opposite edge."""
    a, b, c = pts[tri[0]], pts[tri[1]], pts[tri[2]]
    v1, v2 = b - a, c - a
    s = v1[0]*v2[1] - v1[1]*v2[0]
    sgn = np.sign(s) if abs(s) > 1e-15 else 0.0
    # dA/d(vertex) for A = |cross|/2
    g = np.zeros((3, 2))
    if sgn == 0.0:
        return g
    # cross = (b-a)x(c-a)
    # d/da = (v2 - v1) rotated: (v2y - v1y, v1x - v2x)
    g[0] = 0.5*sgn*np.array([v2[1]-v1[1], v1[0]-v2[0]])
    g[1] = 0.5*sgn*np.array([-v2[1], v2[0]])
    g[2] = 0.5*sgn*np.array([v1[1], -v1[0]])
    return g


def heilbronn_convex13() -> np.ndarray:
    rng = np.random.default_rng(seed=42)
    t_start = time.time()
    budget = 6.0
    n = _N

    # ---- Seeds (structural priors for n=13) ----
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    seeds = [np.column_stack([np.cos(t), np.sin(t)])]
    seeds.append(np.column_stack([1.3*np.cos(t), 0.75*np.sin(t)]))
    t12 = np.linspace(0, 2*np.pi, 12, endpoint=False)
    seeds.append(np.vstack([np.column_stack([np.cos(t12), np.sin(t12)]),
                            [[0.0, 0.0]]]))
    t7 = np.linspace(0, 2*np.pi, 7, endpoint=False)
    t6 = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6
    seeds.append(np.vstack([np.column_stack([np.cos(t7), np.sin(t7)]),
                           0.5*np.column_stack([np.cos(t6), np.sin(t6)])]))
    s = np.column_stack([np.cos(t), np.sin(t)]) + rng.normal(0, 0.08, (n, 2))
    seeds.append(s)

    def fast_score(pts, areas=None, ha=None):
        if areas is None:
            areas = _all_areas(pts)
        if ha is None:
            ha = _hull_area(pts)
        if ha <= 1e-12:
            return 0.0, areas, ha
        return areas.min()/ha, areas, ha

    def grad_polish(pts, deadline, gstep0=0.02):
        """Exact gradient ascent on the bottleneck triangle + light noise,
        using incremental single-vertex updates for speed."""
        pts = _rescale(pts.copy())
        areas = _all_areas(pts)
        ha = _hull_area(pts)
        sc = areas.min()/ha
        gstep = gstep0
        nstep = gstep0
        while time.time() < deadline:
            improved = False
            # pick among the 4 smallest triangles for diversification
            order = np.argsort(areas)[:4]
            for tri_i in order[:2]:
                tri = _TRI[tri_i]
                for attempt in range(3):
                    cand = pts.copy()
                    mode = rng.random()
                    if mode < 0.6:
                        # coordinated gradient move on all 3 vertices
                        g = _tri_grad(pts, tri)
                        for c in range(3):
                            cand[tri[c]] += gstep * g[c] / (np.abs(g[c]).max() + 1e-12) if np.abs(g[c]).max() > 0 else 0
                        # add tiny correlated noise to escape flat directions
                        cand[tri] += rng.normal(0, 0.2*gstep, (3, 2))
                    else:
                        # single-vertex gradient push
                        g = _tri_grad(pts, tri)
                        c = int(rng.integers(3))
                        cand[tri[c]] += gstep * 2.0 * g[c]
                    cand = _rescale(cand)
                    ca = _all_areas(cand)
                    cha = _hull_area(cand)
                    csc = ca.min()/cha if cha > 1e-12 else 0.0
                    if csc > sc + 1e-13:
                        pts, areas, ha, sc = cand, ca, cha, csc
                        improved = True
                        break
            # occasional random single-point move (incremental)
            if not improved:
                for _ in range(30):
                    cand = pts.copy()
                    i = int(rng.integers(n))
                    cand[i] += rng.normal(0, nstep, 2)
                    cand = _rescale(cand)
                    rows = _WITH[i]
                    # incremental area update
                    new_a = areas.copy()
                    new_a[rows] = _point_areas(cand, i, rows)
                    cha = _hull_area(cand)
                    csc = new_a.min()/cha if cha > 1e-12 else 0.0
                    if csc > sc + 1e-13:
                        pts, areas, ha, sc = cand, new_a, cha, csc
                        improved = True
                        break
            if improved:
                gstep = min(gstep*1.25, 0.08)
                nstep = min(nstep*1.15, 0.06)
            else:
                gstep *= 0.55
                nstep *= 0.6
            if gstep < 1e-5:
                break
        return pts, sc

    best_pts, best_sc = None, -1.0

    # ---- Phase 1: seed sweep with gradient polish ----
    per = budget*0.5/len(seeds)
    for k, seed in enumerate(seeds):
        dl = min(t_start + (k+1)*per, t_start + budget)
        if time.time() > dl:
            continue
        pts, sc = grad_polish(np.asarray(seed, float), dl,
                              gstep0=0.03 if k < 2 else 0.05)
        if sc > best_sc:
            best_sc, best_pts = sc, pts.copy()

    if best_pts is None:
        best_pts = seeds[0]

    # ---- Phase 2: basin-hopping with targeted kicks ----
    while time.time() - t_start < budget - 0.05:
        kick = best_pts.copy()
        u = rng.random()
        if u < 0.25:
            # rigidly translate the bottleneck triangle (correlated move)
            kmin = int(np.argmin(_all_areas(best_pts)))
            tri = _TRI[kmin]
            kick[tri] += rng.normal(0, 0.14, (3, 2))
        else:
            # kick 3-4 random vertices at large sigma
            kv = int(rng.integers(3, 5))
            idxs = rng.choice(n, size=kv, replace=False)
            kick[idxs] += rng.normal(0, 0.13, (kv, 2))
        rem = budget - (time.time() - t_start)
        dl = min(time.time() + 0.35, t_start + budget - 0.05)
        pts, sc = grad_polish(kick, dl, gstep0=0.012)
        if sc > best_sc + 1e-13:
            best_sc, best_pts = sc, pts.copy()

    best_pts = np.asarray(best_pts, dtype=float)
    if not np.all(np.isfinite(best_pts)) or best_pts.shape != (n, 2):
        t = np.linspace(0, 2*np.pi, n, endpoint=False)
        best_pts = np.column_stack([np.cos(t), np.sin(t)])
    return best_pts

# EVOLVE-BLOCK-END