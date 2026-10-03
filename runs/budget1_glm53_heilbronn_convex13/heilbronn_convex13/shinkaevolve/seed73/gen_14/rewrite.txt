# EVOLVE-BLOCK-START
import numpy as np

_N = 13
# Precomputed triangle index arrays (i<j<k), shape (286,)
_idx = np.array([(i, j, k) for i in range(_N - 2)
                 for j in range(i + 1, _N - 1)
                 for k in range(j + 1, _N)], dtype=np.int64)
_I = _idx[:, 0]
_J = _idx[:, 1]
_K = _idx[:, 2]


def _cross_all(P):
    """Signed area of all 286 triangles (vectorized)."""
    ax, ay = P[_I, 0], P[_I, 1]
    bx, by = P[_J, 0], P[_J, 1]
    cx, cy = P[_K, 0], P[_K, 1]
    return 0.5 * ((bx - ax) * (cy - ay) - (cx - ax) * (by - ay))


def _min_tri_area(P):
    return np.abs(_cross_all(P)).min()


def _hull(P):
    """Monotone chain hull; returns (hull_area, hull_vertex_indices)."""
    order = np.lexsort((P[:, 1], P[:, 0]))
    idx = []
    for h in (order, order[::-1]):
        chain = []
        for i in h:
            while len(chain) >= 2:
                o, a = P[chain[-2]], P[chain[-1]]
                if (a[0]-o[0])*(P[i,1]-o[1]) - (a[1]-o[1])*(P[i,0]-o[0]) <= 0:
                    chain.pop()
                else:
                    break
            chain.append(i)
        idx.extend(chain[:-1])
    idx = np.array(sorted(set(idx)), dtype=np.int64)
    V = P[idx]
    m = len(idx)
    area = 0.0
    for i in range(m):
        x1, y1 = V[i]
        x2, y2 = V[(i + 1) % m]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5, idx


def _hull_area(P):
    a, _ = _hull(P)
    return a


def _ratio(P):
    ha = _hull_area(P)
    if ha < 1e-12:
        return 0.0
    return _min_tri_area(P) / ha


def _softmin_grad(P):
    """Gradient (ascent) of a smooth approximation of min |area|."""
    A = _cross_all(P)
    absA = np.abs(A)
    amin = absA.min()
    scale = absA.mean() + 1e-12
    alpha = 12.0 / scale
    w = np.exp(-alpha * (absA - amin))
    w /= w.sum()
    s = w * np.sign(A)
    gI = 0.5 * np.stack([P[_J, 1] - P[_K, 1], P[_K, 0] - P[_J, 0]], axis=1)
    gJ = 0.5 * np.stack([P[_K, 1] - P[_I, 1], P[_I, 0] - P[_K, 0]], axis=1)
    gK = 0.5 * np.stack([P[_I, 1] - P[_J, 1], P[_J, 0] - P[_I, 0]], axis=1)
    G = np.zeros_like(P)
    np.add.at(G, _I, s[:, None] * gI)
    np.add.at(G, _J, s[:, None] * gJ)
    np.add.at(G, _K, s[:, None] * gK)
    return G, amin


def _log_hull_grad(P):
    """Gradient of log(hull area) wrt all points (zero for interior points)."""
    ha, hidx = _hull(P)
    if ha < 1e-12:
        return np.zeros_like(P), ha
    V = P[hidx]
    m = len(hidx)
    G = np.zeros_like(P)
    hg = np.zeros((m, 2))
    for i in range(m):
        prev = V[(i - 1) % m]
        nxt = V[(i + 1) % m]
        hg[i, 0] = 0.5 * (nxt[1] - prev[1])
        hg[i, 1] = 0.5 * (prev[0] - nxt[0])
    np.add.at(G, hidx, hg)
    return G / ha, ha


def _optimize(init, iters=1500, step0=0.05):
    """Gradient ascent on log(min area) - log(hull area)."""
    P = init.copy()
    best_P = P.copy()
    best_val = _ratio(P)
    step = step0
    for it in range(iters):
        Gmin, amin = _softmin_grad(P)
        Ghull, _ = _log_hull_grad(P)
        G = Gmin / (amin + 1e-9) - Ghull
        nrm = np.linalg.norm(G)
        if nrm < 1e-15:
            break
        P = P + step * G / nrm
        val = _ratio(P)
        if val > best_val:
            best_val = val
            best_P = P.copy()
        step *= 0.999
    return best_P, best_val


def _polish(P, rounds=400, seed=12345, amp0=0.02):
    """Coordinate-level stochastic hill climbing on the ratio."""
    rng = np.random.default_rng(seed)
    P = P.copy()
    best = _ratio(P)
    amp = amp0
    for r in range(rounds):
        improved = False
        for i in range(_N):
            for _ in range(6):
                d = rng.normal(size=2) * amp
                Q = P.copy()
                Q[i] += d
                v = _ratio(Q)
                if v > best + 1e-12:
                    best = v
                    P = Q
                    improved = True
        amp *= 0.995
        if not improved and amp < 1e-4:
            break
    return P, best


def _sym_config(params, center=(0.0, 0.0)):
    """3-fold symmetric configuration: center + 4 orbits of 3 points.
    params: [r0,r1,r2,r3, t0,t1,t2,t3]"""
    r = params[:4]
    th = params[4:8]
    pts = [[center[0], center[1]]]
    for i in range(4):
        for k in range(3):
            a = th[i] + 2.0 * np.pi * k / 3.0
            pts.append([r[i] * np.cos(a), r[i] * np.sin(a)])
    return np.array(pts)


def _sym_search(rng, samples=1500):
    """Random search over the 8-D symmetric parametrization; return top candidates."""
    cands = []
    for _ in range(samples):
        r = np.sort(rng.random(4)) * 1.4 + 0.05   # radii in (0.05, 1.45], sorted
        th = rng.random(4) * 2 * np.pi / 3.0      # phase within one sector
        P = _sym_config(np.concatenate([r, th]))
        cands.append((_ratio(P), np.concatenate([r, th])))
    cands.sort(key=lambda c: -c[0])
    return cands[:6]


def _inits(seed):
    """General (non-symmetric) initializations for diversity."""
    rng = np.random.default_rng(seed)
    inits = []
    th = 2 * np.pi * np.arange(_N) / _N + 0.1
    inits.append(np.stack([np.cos(th), np.sin(th)], axis=1))
    th2 = th + rng.normal(scale=0.08, size=_N)
    inits.append(np.stack([np.cos(th2), np.sin(th2)], axis=1))
    u = rng.random(_N) * 2 * np.pi
    rr = np.sqrt(rng.random(_N))
    inits.append(np.stack([rr * np.cos(u), rr * np.sin(u)], axis=1))
    return inits


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points in a convex region maximizing
    the smallest triangle area (Heilbronn problem, n=13).

    Returns:
        points: np.ndarray of shape (13,2) with x,y coordinates.
    """
    best_P = None
    best_val = -1.0

    # --- Stage 1: symmetric-parametrization global search + gradient refinement
    rng = np.random.default_rng(42)
    for val0, params in _sym_search(rng, samples=1200):
        P0 = _sym_config(params)
        P, val = _optimize(P0, iters=800)
        if val > best_val:
            best_val = val
            best_P = P.copy()

    # --- Stage 2: general initializations (polygons, disks), gradient refinement
    for seed in (42, 7):
        for init in _inits(seed):
            P, val = _optimize(init, iters=800)
            if val > best_val:
                best_val = val
                best_P = P.copy()

    # --- Fallback safety
    if best_P is None or best_val <= 0:
        th = 2 * np.pi * np.arange(_N) / _N
        best_P = np.stack([np.cos(th), np.sin(th)], axis=1)
        best_val = _ratio(best_P)

    # --- Stage 3: coordinate polish on the overall best
    best_P, best_val = _polish(best_P, rounds=300, seed=12345)

    # Rescale so convex hull has unit area
    ha = _hull_area(best_P)
    if ha > 1e-12:
        c = best_P.mean(axis=0)
        best_P = (best_P - c) / np.sqrt(ha) + c
    else:
        th = 2 * np.pi * np.arange(_N) / _N
        best_P = np.stack([np.cos(th), np.sin(th)], axis=1) / np.sqrt(
            _N / 2 * np.sin(2 * np.pi / _N))
    return best_P


# EVOLVE-BLOCK-END