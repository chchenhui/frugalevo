# EVOLVE-BLOCK-START
import numpy as np
import time


def _convex_hull_area(pts):
    """Monotone chain convex hull; returns hull area (0 if degenerate)."""
    P = sorted(map(tuple, np.round(pts, 12)))
    P = list(dict.fromkeys(P))
    if len(P) < 3:
        return 0.0
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower = []
    for p in P:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(P):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return 0.0
    H = np.array(hull)
    x, y = H[:, 0], H[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _min_triangle_area(pts, tri_idx):
    p = pts[tri_idx]                      # (286, 3, 2)
    a = p[:, 0]
    v1 = p[:, 1] - a
    v2 = p[:, 2] - a
    cross = np.abs(v1[:, 0]*v2[:, 1] - v1[:, 1]*v2[:, 0])
    return cross.min() * 0.5


def _score(pts, tri_idx):
    ha = _convex_hull_area(pts)
    if ha <= 1e-12:
        return 0.0
    return _min_triangle_area(pts, tri_idx) / ha


# --- Analytic-gradient core (soft-min of all 286 triangle areas) ---
from itertools import combinations

N = 13
TRI_IDX = np.array(list(combinations(range(N), 3)))
I0 = TRI_IDX[:, 0]
I1 = TRI_IDX[:, 1]
I2 = TRI_IDX[:, 2]


def _rescale_unit_hull(pts):
    ha = _convex_hull_area(pts)
    if ha > 1e-12:
        pts = pts / np.sqrt(ha)
    return pts


def _tri_signed2(pts):
    A = pts[I0]; B = pts[I1]; C = pts[I2]
    u = B - A
    v = C - A
    return u[:, 0]*v[:, 1] - u[:, 1]*v[:, 0], u, v


def _exact_score(pts):
    ha = _convex_hull_area(pts)
    if ha <= 1e-12:
        return 0.0
    c, _, _ = _tri_signed2(pts)
    return 0.5*np.abs(c).min() / ha


def _smooth_grad(pts, p):
    """Soft-min S = -(1/p) log sum exp(-p a) and exact gradient dS/dpts."""
    c, u, v = _tri_signed2(pts)
    s = np.sign(c); s[s == 0] = 1.0
    a = 0.5 * np.abs(c)
    z = -p * a
    zmax = z.max()
    w = np.exp(z - zmax)
    w /= w.sum()
    S = -(zmax + np.log(np.exp(z - zmax).sum())) / p
    q = 0.5 * s * w          # dS/da * da/dc
    gA_t = np.stack([q*(u[:, 1] - v[:, 1]), q*(v[:, 0] - u[:, 0])], axis=1)
    gB_t = np.stack([q*v[:, 1], -q*v[:, 0]], axis=1)
    gC_t = np.stack([-q*u[:, 1], q*u[:, 0]], axis=1)
    g = np.zeros((N, 2))
    np.add.at(g, I0, gA_t)
    np.add.at(g, I1, gB_t)
    np.add.at(g, I2, gC_t)
    return S, g


def _adam_run(pts, deadline, lr=0.02, p0=30.0):
    pts = _rescale_unit_hull(pts.copy())
    m = np.zeros((N, 2)); vv = np.zeros((N, 2))
    b1, b2, eps = 0.9, 0.999, 1e-8
    p = p0
    step = 0
    best_pts, best_sc = pts.copy(), _exact_score(pts)
    while time.time() < deadline:
        for _ in range(20):
            step += 1
            S, g = _smooth_grad(pts, p)
            m = b1*m + (1-b1)*g
            vv = b2*vv + (1-b2)*g*g
            mh = m / (1 - b1**step)
            vh = vv / (1 - b2**step)
            pts = pts + lr * mh / (np.sqrt(vh) + eps)
            pts = _rescale_unit_hull(pts)
            sc = _exact_score(pts)
            if sc > best_sc + 1e-14:
                best_sc = sc
                best_pts = pts.copy()
        p = min(p * 1.15, 600.0)
        lr *= 0.93
        if lr < 1e-5:
            break
    return best_pts, best_sc


def _polish(pts, rng, deadline):
    """Exact-score hill climb: bottleneck-vertex-targeted + random moves."""
    pts = _rescale_unit_hull(pts.copy())
    sc = _exact_score(pts)
    sigma = 0.02
    while time.time() < deadline:
        improved = False
        for _ in range(100):
            cand = pts.copy()
            if rng.random() < 0.5:
                c, _, _ = _tri_signed2(pts)
                tri = TRI_IDX[int(np.argmin(np.abs(c)))]
                i = int(rng.choice(tri))
            else:
                i = int(rng.integers(N))
            cand[i] += rng.normal(0, sigma, 2)
            cand = _rescale_unit_hull(cand)
            csc = _exact_score(cand)
            if csc > sc + 1e-14:
                pts, sc = cand, csc
                improved = True
        if improved:
            sigma = min(sigma*1.3, 0.05)
        else:
            sigma *= 0.6
        if sigma < 1e-6:
            break
    return pts, sc


def _seeds(rng):
    t = np.linspace(0, 2*np.pi, N, endpoint=False)
    seeds = []
    seeds.append(np.column_stack([np.cos(t), np.sin(t)]))              # 13-gon
    seeds.append(np.column_stack([1.3*np.cos(t), 0.75*np.sin(t)]))     # ellipse
    s = np.column_stack([np.cos(t), np.sin(t)]) + rng.normal(0, 0.08, (N, 2))
    seeds.append(s)                                                    # perturbed 13-gon
    # 7+6 two-arc cluster
    tA = np.linspace(-0.6*np.pi, 0.4*np.pi, 7)
    tB = np.linspace(0.4*np.pi, 1.4*np.pi, 6)
    seeds.append(np.vstack([np.column_stack([np.cos(tA), np.sin(tA)]),
                            0.9*np.column_stack([np.cos(tB), np.sin(tB)])]))
    # 9+4 two-arc cluster (different ratio)
    tA9 = np.linspace(-0.5*np.pi, 0.5*np.pi, 9)
    tB4 = np.linspace(0.5*np.pi, 1.5*np.pi, 4)
    seeds.append(np.vstack([np.column_stack([np.cos(tA9), np.sin(tA9)]),
                            0.85*np.column_stack([np.cos(tB4), np.sin(tB4)])]))
    # rounded-square superellipse
    u = t
    seeds.append(np.column_stack([np.sign(np.cos(u))*np.abs(np.cos(u))**0.5,
                                  np.sign(np.sin(u))*np.abs(np.sin(u))**0.5]))
    # spiral (original chirality)
    k = np.arange(N)
    rad = 0.15 + 0.85*k/(N-1)
    ang = 2.4*k
    seeds.append(np.column_stack([rad*np.cos(ang), rad*np.sin(ang)]))
    # reversed-chirality spiral
    ang2 = -2.4*k + 0.7
    seeds.append(np.column_stack([rad*np.cos(ang2), rad*np.sin(ang2)]))
    # 7 outer + 6 inner
    t7 = np.linspace(0, 2*np.pi, 7, endpoint=False)
    t6 = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6
    seeds.append(np.vstack([np.column_stack([np.cos(t7), np.sin(t7)]),
                            0.45*np.column_stack([np.cos(t6), np.sin(t6)])]))
    # 12-ring + center
    t12 = np.linspace(0, 2*np.pi, 12, endpoint=False)
    seeds.append(np.vstack([np.column_stack([np.cos(t12), np.sin(t12)]),
                            [[0.0, 0.0]]]))
    return seeds


def heilbronn_convex13() -> np.ndarray:
    n = N
    rng = np.random.default_rng(seed=42)
    t_start = time.time()
    time_limit = 2.8

    seeds = _seeds(rng)
    per_seed = 0.8 * time_limit / len(seeds)
    # phase 1: Adam descent from each seed; keep top-2 basins
    top = []  # list of (score, pts) sorted desc, at most 2
    for k, s in enumerate(seeds):
        deadline = min(t_start + (k+1)*per_seed, t_start + time_limit)
        if time.time() - t_start > time_limit - 0.1:
            break
        pts, sc = _adam_run(np.asarray(s, dtype=float).copy(), deadline)
        top.append((sc, pts))
        top.sort(key=lambda z: -z[0])
        top = top[:2]

    if not top:
        t = np.linspace(0, 2*np.pi, N, endpoint=False)
        return np.column_stack([np.cos(t), np.sin(t)])

    # phase 2: alternate short re-anneals on the top-2 basins
    which = 0
    while time.time() - t_start < time_limit - 0.25:
        base_sc, base_pts = top[which]
        kick = base_pts + rng.normal(0, 0.05, base_pts.shape)
        deadline = min(time.time() + 0.25, t_start + time_limit)
        pts, sc = _adam_run(kick, deadline, lr=0.005, p0=200.0)
        if sc > base_sc:
            top[which] = (sc, pts)
        top.sort(key=lambda z: -z[0])
        top = top[:2]
        which = (which + 1) % len(top)

    # phase 3: final bottleneck polish on the incumbent
    best_sc, best_pts = top[0]
    deadline = t_start + time_limit
    pts, sc = _polish(best_pts.copy(), rng, deadline)
    if sc > best_sc:
        best_sc, best_pts = sc, pts.copy()

    best_pts = np.asarray(best_pts, dtype=float)
    if best_pts is None or not np.all(np.isfinite(best_pts)) \
            or best_pts.shape != (13, 2):
        t = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(t), np.sin(t)])
    return best_pts


# EVOLVE-BLOCK-END