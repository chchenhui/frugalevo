# EVOLVE-BLOCK-START
import numpy as np

SQRT3 = np.sqrt(3.0)
TRI_AREA = SQRT3 / 4.0
VERTS = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, SQRT3 / 2.0]])
N = 11

I, J, K = np.array([(i, j, k) for i in range(N) for j in range(i + 1, N)
                    for k in range(j + 1, N)]).T
T_ = np.array([(i, j, k) for i in range(N) for j in range(i + 1, N)
               for k in range(j + 1, N)])


def from_bary(u):
    """(N,3) barycentric weights (each row sums to 1) -> (N,2) xy."""
    u = np.abs(u)
    u = u / u.sum(axis=1, keepdims=True)
    return u @ VERTS


def to_bary(pts):
    A, B, C = VERTS
    v0, v1 = B - A, C - A
    d = v0[0] * v1[1] - v0[1] * v1[0]
    ap = pts - A
    b = (ap[:, 0] * v1[1] - ap[:, 1] * v1[0]) / d
    c = (v0[0] * ap[:, 1] - v0[1] * ap[:, 0]) / d
    return np.stack([1 - b - c, b, c], axis=1)


def areas(pts):
    a, b, c = pts[I], pts[J], pts[K]
    cr = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - \
         (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    return 0.5 * np.abs(cr)


def min_area(pts):
    return areas(pts).min() / TRI_AREA


def grad_min_area(pts):
    """Gradient of mean-of-bottom-k softmin (ascent dir) wrt pts."""
    ar = areas(pts) / TRI_AREA
    order = np.argsort(ar)
    k = max(4, len(ar) // 50)
    sel = order[:k]
    w = np.zeros(len(ar))
    w[sel] = np.exp(-(ar[sel] - ar[sel].min()) * 50.0)
    w /= w.sum()
    a, b, c = pts[I], pts[J], pts[K]
    cr = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - \
         (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    s = np.sign(cr)
    co = (0.5 * s * w / TRI_AREA)
    g = np.zeros_like(pts)
    for col, (P, Q, R) in ((I, (b, c)), (J, (c, a)), (K, (a, b))):
        pass  # handled below explicitly
    # vertex gradients of area
    def sc(idx, vec):
        np.add.at(g, idx, (co[:, None] * vec))
    sc(I, np.stack([b[:, 1] - c[:, 1], c[:, 0] - b[:, 0]], axis=1))
    sc(J, np.stack([c[:, 1] - a[:, 1], a[:, 0] - c[:, 0]], axis=1))
    sc(K, np.stack([a[:, 1] - b[:, 1], b[:, 0] - a[:, 0]], axis=1))
    return g


def _starts():
    rng = np.random.default_rng(2024)
    st = []
    # triangular lattice
    lat = []
    for r in range(4):
        y = r * SQRT3 / 6.0
        for c in range(r + 1):
            x = 0.5 * (1 - r / 3.0) + c / 3.0
            lat.append((x, y))
    lat = np.array(lat)
    lat = lat[[0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 6]][:11]
    st.append(lat)
    # boundary ring
    t = np.linspace(0, 1, 12)
    ring = [(f, 0.0) for f in t] + \
           [(0.5 + 0.5 * f, SQRT3 / 2 * f) for f in t[1:-1]] + \
           [(0.5 * (1 - f), SQRT3 / 2 * f) for f in t[1:-1]]
    ring = np.array(ring)
    st.append(ring[[0, 2, 4, 6, 8, 10, 12, 17, 21, 26, 30]][:11])
    # random
    for _ in range(4):
        u = rng.random((N, 3))
        u = np.sort(u, axis=1)
        st.append(u @ VERTS)
    return st


def _slsqp_stage(pts):
    from scipy.optimize import minimize
    u = to_bary(pts).flatten()
    best = (pts.copy(), min_area(pts))
    try:
        for p in (8.0, 25.0, 80.0):
            def obj(x):
                P = from_bary(x.reshape(N, 3))
                ar = areas(P) / TRI_AREA
                m = ar.min()
                return -(m - np.log(np.sum(np.exp(-p * (ar - m)))) / p)

            def jac(x):
                P = from_bary(x.reshape(N, 3))
                ar = areas(P) / TRI_AREA
                m = ar.min()
                w = np.exp(-p * (ar - m))
                w /= w.sum()
                a, b, c = P[I], P[J], P[K]
                cr = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - \
                     (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
                s = np.sign(cr)
                co = (0.5 * s * w / TRI_AREA)[:, None]
                g = np.zeros_like(P)

                def sc(idx, vec):
                    np.add.at(g, idx, co * vec)
                sc(I, np.stack([b[:, 1] - c[:, 1], c[:, 0] - b[:, 0]], axis=1))
                sc(J, np.stack([c[:, 1] - a[:, 1], a[:, 0] - c[:, 0]], axis=1))
                sc(K, np.stack([a[:, 1] - b[:, 1], b[:, 0] - a[:, 0]], axis=1))
                # chain rule to barycentric (u -> xy linear): dxy = (V2-V1, V3-V1) per bary pair
                g3 = np.empty((N, 3))
                g3[:, 1] = g[:, 0] * (VERTS[1, 0] - VERTS[0, 0]) + \
                    g[:, 1] * (VERTS[1, 1] - VERTS[0, 1])
                g3[:, 2] = g[:, 0] * (VERTS[2, 0] - VERTS[0, 0]) + \
                    g[:, 1] * (VERTS[2, 1] - VERTS[0, 1])
                g3[:, 0] = -(g3[:, 1] + g3[:, 2])
                return -g3.flatten()

            res = minimize(obj, u, jac=jac, method='L-BFGS-B',
                           bounds=[(0.0, 1.0)] * (3 * N),
                           options={'maxiter': 150, 'ftol': 1e-14})
            if np.isfinite(res.x).all():
                u = res.x
            P = from_bary(u.reshape(N, 3))
            v = min_area(P)
            if v > best[1]:
                best = (P.copy(), v)
    except Exception:
        pass
    return best


def _polish(pts, rng, iters=900):
    n = N
    best = pts.copy()
    bv = min_area(pts)
    cur, step, stall = bv, 0.02, 0

    def clip(p):
        x = np.clip(p[0], 0.0, 1.0)
        yhi = SQRT3 * x if x <= 0.5 else SQRT3 * (1.0 - x)
        return np.array([x, np.clip(p[1], 0.0, yhi)])

    def batch_min(base, idx, cand):
        B = len(idx)
        P = np.broadcast_to(base, (B, n, 2)).copy()
        P[np.arange(B), idx] = cand
        A, Bv, Cv = P[:, I], P[:, J], P[:, K]
        cr = (Bv[:, :, 0] - A[:, :, 0]) * (Cv[:, :, 1] - A[:, :, 1]) - \
             (Bv[:, :, 1] - A[:, :, 1]) * (Cv[:, :, 0] - A[:, :, 0])
        return 0.5 * np.abs(cr).min(axis=1) / TRI_AREA

    for it in range(iters):
        ar = areas(pts) / TRI_AREA
        order = np.argsort(ar)
        wid = int(order[0])
        hot = np.unique(T_[order[:8]].ravel())
        # directed: perpendicular escapes from worst line
        i, j, k = T_[wid]
        mi = []
        mv = []
        for pi in hot:
            t = np.argmin([ar[m] if pi in T_[m] else np.inf for m in order[:30]] or [0])
        # simpler: perpendicular moves for the three worst-triplet vertices
        tri = T_[order[:6]]
        for t in tri:
            for pi in T_[t]:
                o = [q for q in T_[t] if q != pi]
                d = pts[o[1]] - pts[o[0]]
                L = np.hypot(*d)
                if L < 1e-12:
                    continue
                perp = np.array([-d[1], d[0]]) / L
                sgn = np.sign(perp @ (pts[pi] - pts[o[0]])) or 1.0
                for sc in (step, 0.5 * step, 2 * step):
                    mi.append(pi)
                    mv.append(clip(pts[pi] + sc * sgn * perp))
        # random moves
        for _ in range(20):
            pi = int(rng.integers(n))
            mi.append(pi)
            mv.append(clip(pts[pi] + rng.normal(0, step, 2)))
        mi = np.array(mi)
        mv = np.array(mv)
        vals = batch_min(pts, mi, mv)
        b = int(np.argmax(vals))
        if vals[b] > cur + 1e-14:
            pts = pts.copy()
            pts[mi[b]] = mv[b]
            cur = float(vals[b])
            stall = 0
            if cur > bv:
                bv, best = cur, pts.copy()
            continue
        stall += 1
        # teleport a hot point
        if stall % 12 == 0:
            wi = int(T_[order[0]][rng.integers(3)])
            u = rng.random(3)
            u /= u.sum()
            cand = pts.copy()
            cand[wi] = u @ VERTS
            v = min_area(cand)
            if v > cur * 0.98:
                pts, cur = cand, max(v, cur)
                if v > bv:
                    bv, best = v, pts.copy()
                step = max(step, 0.01)
                stall = 0
        if stall % 8 == 0:
            step *= 0.75
        if step < 3e-6:
            break
    return best, bv


def heilbronn_triangle11() -> np.ndarray:
    rng = np.random.default_rng(7)
    best_pts, best_val = None, -1.0
    for s in _starts():
        try:
            P, v = _slsqp_stage(np.asarray(s, float))
        except Exception:
            continue
        try:
            P, v = _polish(P, rng)
        except Exception:
            pass
        if v > best_val:
            best_val, best_val_ = v, v
            best_pts = P
    if best_pts is None:
        best_pts = _starts()[0]
    # exact feasibility: snap to triangle
    out = np.array([p for p in best_pts])
    out[:, 1] = np.clip(out[:, 1], 0.0, None)
    out[:, 1] = np.minimum(out[:, 1], np.minimum(SQRT3 * out[:, 0],
                                                  SQRT3 * (1.0 - out[:, 0])))
    out[:, 0] = np.clip(out[:, 0], 0.0, 1.0)
    return out
# EVOLVE-BLOCK-END