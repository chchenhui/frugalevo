# EVOLVE-BLOCK-START
import numpy as np

SQRT3 = np.sqrt(3.0)
TRI_AREA = SQRT3 / 4.0  # area of the unit equilateral triangle


def _all_triples(n):
    idxs = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                idxs.append((i, j, k))
    return np.array(idxs)


TRIPLES = _all_triples(11)


def _areas(pts):
    a = pts[TRIPLES[:, 0]]
    b = pts[TRIPLES[:, 1]]
    c = pts[TRIPLES[:, 2]]
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    return 0.5 * np.abs(cross)


def _constraints_flat(x):
    """Linear inequality constraints keeping all points inside the triangle."""
    p = x.reshape(-1, 2)
    xs, ys = p[:, 0], p[:, 1]
    return np.concatenate([
        ys,                              # y >= 0
        SQRT3 * xs - ys,                 # below left edge (y <= sqrt3 x)
        SQRT3 * (1.0 - xs) - ys,         # below right edge
    ])


def _min_area(x):
    return _areas(x.reshape(-1, 2)).min() / TRI_AREA


def _optimize(start, iters):
    from scipy.optimize import minimize
    x = start.flatten().copy()
    best_x, best_val = x.copy(), _min_area(x)
    try:
        for p in (10.0, 30.0, 100.0):
            def obj(xx):
                ar = _areas(xx.reshape(-1, 2)) / TRI_AREA
                # smooth min: (1/p) * log sum exp(-p * a)
                m = ar.min()
                return (np.log(np.sum(np.exp(-p * (ar - m)))) - p * m) / (-p)

            def jac(xx):
                ar = _areas(xx.reshape(-1, 2)) / TRI_AREA
                m = ar.min()
                w = np.exp(-p * (ar - m))
                w = w / w.sum()
                pts = xx.reshape(-1, 2)
                a = pts[TRIPLES[:, 0]]
                b = pts[TRIPLES[:, 1]]
                c = pts[TRIPLES[:, 2]]
                cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
                s = np.sign(cross)
                # vectorized scatter-add gradient of |cross|/2 / TRI_AREA
                co = (0.5 * s * w / TRI_AREA)[:, None]
                g = np.zeros_like(pts)
                I = TRIPLES[:, 0]
                J = TRIPLES[:, 1]
                K = TRIPLES[:, 2]
                np.add.at(g, (I, np.zeros_like(I)),
                          co * np.stack([a[:, 1] - c[:, 1], c[:, 0] - a[:, 0]], axis=1))
                np.add.at(g, (J, np.zeros_like(J)),
                          co * np.stack([c[:, 1] - b[:, 1], b[:, 0] - c[:, 0]], axis=1))
                np.add.at(g, (K, np.zeros_like(K)),
                          co * np.stack([b[:, 1] - a[:, 1], a[:, 0] - b[:, 0]], axis=1))
                return -g.flatten()  # obj returns -softmin (to be minimized)

            cons = {'type': 'ineq', 'fun': lambda xx: _constraints_flat(xx),
                    'jac': None}
            res = minimize(obj, x, jac=jac, method='SLSQP',
                           constraints=[cons],
                           bounds=[(0.0, 1.0)] * 22,
                           options={'maxiter': iters, 'ftol': 1e-12})
            if np.isfinite(res.x).all():
                x = res.x
            v = _min_area(x)
            if v > best_val:
                best_val, best_x = v, x.copy()
    except Exception:
        pass
    return best_x, best_val


def _lattice_starts():
    """Deterministic starting layouts inside the triangle."""
    starts = []
    # Triangular lattice rows 1..5 (15 pts) -> pick 11 spread out
    lat = []
    for r in range(5):
        y = r * (SQRT3 / 2) / 4 * (SQRT3 / SQRT3)  # rows at dy = sqrt3/8
        y = r * (SQRT3 / 2) / 4
        for c in range(r + 1):
            x = 0.5 * (1 - r / 4) + c * (1.0 / 4)
            lat.append((x, y))
    lat = np.array(lat)
    # keep 11 well-spread points (drop every other interior one)
    keep = [0, 1, 2, 3, 4, 5, 6, 8, 9, 12, 14]
    starts.append(lat[keep])
    # boundary ring: 3 corners + points along edges
    t = np.linspace(0, 1, 13)
    ring = []
    for f in t:
        ring.append((f, 0.0))                                   # bottom edge
    for f in t[1:-1]:
        ring.append((0.5 * (1 - f) + 1.0 * f, SQRT3 / 2 * f))  # right edge
    for f in t[1:-1]:
        ring.append((0.5 * (1 - f), SQRT3 / 2 * f))            # left edge
    ring = np.array(ring)
    starts.append(ring[[0, 2, 4, 6, 8, 10, 12, 16, 20, 23, 26]][:11] if len(ring) > 26 else ring[:11])
    # small interior grid
    g = []
    for yy in [0.1, 0.3, 0.5, 0.7]:
        for xx in [0.15, 0.5, 0.85]:
            if yy <= SQRT3 * xx and yy <= SQRT3 * (1 - xx):
                g.append((xx, yy))
    g = np.array(g)
    starts.append(g[:11] if len(g) >= 11 else np.vstack([g, lat[:11 - len(g)]]))
    return starts


def heilbronn_triangle11() -> np.ndarray:
    n = 11
    rng = np.random.default_rng(12345)

    starts = _lattice_starts()
    # add seeded random feasible starts
    for _ in range(3):
        pts = rng.random((n, 2))
        pts[:, 1] *= np.minimum(SQRT3 * pts[:, 0], SQRT3 * (1 - pts[:, 0])) * 0.99 + 1e-4
        starts.append(pts)

    best_x, best_val = None, -1.0
    for s in starts:
        try:
            x, v = _optimize(np.asarray(s, dtype=float), iters=120)
        except Exception:
            continue
        if v > best_val:
            best_val, best_x = v, x

    if best_x is None:
        # fallback: lattice points (always valid)
        best_x = _lattice_starts()[0].flatten()
        best_x = np.asarray(best_x, dtype=float).reshape(n, 2).flatten()

    pts = best_x.reshape(n, 2)
    # project into triangle to guarantee feasibility
    pts[:, 1] = np.clip(pts[:, 1], 0.0, None)
    pts[:, 1] = np.minimum(pts[:, 1], np.minimum(SQRT3 * pts[:, 0], SQRT3 * (1 - pts[:, 0])))

    # ---- k-worst-triplet guided batched polish ----
    def _clip_pt(p):
        x = np.clip(p[0], 0.0, 1.0)
        yhi = SQRT3 * x if x <= 0.5 else SQRT3 * (1.0 - x)
        return np.array([x, np.clip(p[1], 0.0, yhi)])

    def _min_area_batch(base_pts, move_idx, move_pts):
        """Min area over a batch of candidates, each moving one point."""
        B = move_idx.shape[0]
        P = np.broadcast_to(base_pts, (B, n, 2)).copy()
        P[np.arange(B), move_idx] = move_pts
        A = P[:, TRIPLES[:, 0]]
        Bv = P[:, TRIPLES[:, 1]]
        Cv = P[:, TRIPLES[:, 2]]
        cross = (Bv[:, :, 0] - A[:, :, 0]) * (Cv[:, :, 1] - A[:, :, 1]) - \
                (Bv[:, :, 1] - A[:, :, 1]) * (Cv[:, :, 0] - A[:, :, 0])
        return 0.5 * np.abs(cross).min(axis=1) / TRI_AREA

    # precompute per-point worst-triplet lookup helpers
    def _worst_triplet_for_point(areas, pi):
        """Index of the worst triplet containing point pi."""
        mask = (TRIPLES[:, 0] == pi) | (TRIPLES[:, 1] == pi) | (TRIPLES[:, 2] == pi)
        tmask = np.where(mask)[0]
        return tmask[np.argmin(areas[tmask])] if len(tmask) else -1

    def _escape_dirs(pts, areas, hot):
        """Analytic escape directions for hot points: move perpendicular to the
        line through the other two vertices of that point's worst triplet,
        away from the line (increases that triplet's area)."""
        dirs = []
        idxs = []
        for pi in hot:
            t = _worst_triplet_for_point(areas, int(pi))
            if t < 0:
                continue
            i, j, k = TRIPLES[t]
            others = [q for q in (i, j, k) if q != pi]
            o1, o2 = pts[others[0]], pts[others[1]]
            d = o2 - o1
            L = np.hypot(*d)
            if L < 1e-12:
                continue
            perp = np.array([-d[1], d[0]]) / L
            rel = pts[pi] - o1
            sgn = np.sign(perp @ rel)
            if sgn == 0:
                sgn = 1.0
            dirs.append(sgn * perp)
            idxs.append(int(pi))
        if not dirs:
            return np.empty(0, dtype=int), np.empty((0, 2))
        return np.array(idxs), np.array(dirs)

    cur = _min_area(pts)
    best_pts = pts.copy()
    best_val = cur
    step = 0.02
    stall = 0
    total = 0
    max_polish = 600
    batch = 24
    while total < max_polish and step > 1e-5:
        total += 1
        areas = _areas(pts) / TRI_AREA
        order = np.argsort(areas)
        # hot points: vertices of worst triplet, sometimes bottom-10 union
        if rng.random() < 0.35:
            hot = np.unique(TRIPLES[order[:10]].ravel())
        else:
            hot = np.unique(TRIPLES[order[:1]].ravel())
        # split budget: half random, half directed perpendicular escapes
        n_rand = max(1, batch // 2)
        n_dir = batch - n_rand
        # random candidates
        reps = max(1, n_rand // len(hot))
        rand_idx = np.tile(hot, reps)[:n_rand]
        noise = rng.normal(0.0, step, size=(rand_idx.shape[0], 2))
        rand_pts = np.array([_clip_pt(pts[i] + d) for i, d in zip(rand_idx, noise)])
        # directed candidates at several scales
        esc_idx, esc_dir = _escape_dirs(pts, areas, hot)
        if len(esc_idx) > 0 and n_dir > 0:
            scales = np.array([step, 0.5 * step, 2.0 * step, 0.25 * step])
            rep = int(np.ceil(n_dir / (len(esc_idx) * len(scales))))
            dir_idx_list = []
            dir_off_list = []
            for _ in range(rep):
                perm = rng.permutation(len(esc_idx))
                for pi in perm:
                    for sc in scales:
                        if len(dir_idx_list) >= n_dir:
                            break
                        dir_idx_list.append(esc_idx[pi])
                        dir_off_list.append(sc * esc_dir[pi])
            dir_idx = np.array(dir_idx_list[:n_dir])
            dir_pts = np.array([_clip_pt(pts[i] + o) for i, o in zip(dir_idx, dir_off_list)])
        else:
            dir_idx = np.empty(0, dtype=int)
            dir_pts = np.empty((0, 2))
        move_idx = np.concatenate([rand_idx, dir_idx]).astype(int)
        move_pts = np.vstack([rand_pts, dir_pts])
        dmin = np.array([
            np.min(np.linalg.norm(np.delete(pts, i, axis=0) - mp, axis=1))
            for i, mp in zip(move_idx, move_pts)
        ])
        ok = dmin > 1e-7
        if np.any(ok):
            mi, mp = move_idx[ok], move_pts[ok]
            vals = _min_area_batch(pts, mi, mp)
            b = int(np.argmax(vals))
            if vals[b] > cur + 1e-14:
                pts = pts.copy()
                pts[mi[b]] = mp[b]
                cur = float(vals[b])
                stall = 0
                if cur > best_val:
                    best_val = cur
                    best_pts = pts.copy()
                continue
        stall += 1
        # occasional global relocation: teleport a worst-triplet point
        if stall % 15 == 0 and step < 5e-2:
            wi = int(TRIPLES[order[0]][rng.integers(3)])
            u, v = rng.random(), rng.random()
            if u + v > 1.0:
                u, v = 1.0 - u, 1.0 - v
            cand = pts.copy()
            cand[wi] = [v + 0.5 * (1 - u - v), (1 - u - v) * SQRT3 / 2.0]
            a = _min_area(cand)
            if a > cur:
                pts, cur = cand, a
                step = max(step, 0.02)
                stall = 0
                if cur > best_val:
                    best_val = cur
                    best_pts = pts.copy()
        if stall % 10 == 0 and stall > 0:
            step *= 0.7
    pts = best_pts
    # final feasibility projection
    pts = np.array([_clip_pt(p) for p in pts])
    return pts


# EVOLVE-BLOCK-END