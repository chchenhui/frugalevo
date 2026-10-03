# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations

S3 = np.sqrt(3.0)
AREA = S3 / 4.0


def _all_triples(n):
    return np.array(list(combinations(range(n), 3)), dtype=int)


def _areas(points, triples):
    p = points[triples]
    a = p[:, 1] - p[:, 0]
    b = p[:, 2] - p[:, 0]
    return 0.5 * np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])


def _project(pts):
    orig_shape = pts.shape
    flat = pts.reshape(-1, 2)
    h = S3 / 2.0
    x, y = flat[:, 0], flat[:, 1]
    a = y / h
    c = x - 0.5 * a
    b = 1.0 - a - c
    lam = np.clip(np.stack([a, b, c], axis=1), 0.0, None)
    lam /= lam.sum(axis=1, keepdims=True)
    a, b, c = lam[:, 0], lam[:, 1], lam[:, 2]
    out = np.column_stack([c + 0.5 * a, a * h])
    return out.reshape(orig_shape)


def _init_points(rng, n):
    u = rng.random((n, 2))
    su = np.sqrt(u[:, 0])
    b = np.stack([1 - su, su * (1 - u[:, 1]), su * u[:, 1]], axis=1)
    verts = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, S3 / 2.0]])
    return b @ verts


def _perp_dir(pa, pb, pc):
    ex, ey = pb[0] - pa[0], pb[1] - pa[1]
    L = np.hypot(ex, ey) + 1e-12
    ux, uy = ex / L, ey / L
    dx, dy = pc[0] - pa[0], pc[1] - pa[1]
    if -uy * dx + ux * dy >= 0:
        return np.array([-uy, ux])
    return np.array([uy, -ux])


def heilbronn_triangle11() -> np.ndarray:
    n = 11
    rng = np.random.default_rng(987654321)
    verts = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, S3 / 2.0]])
    triples = _all_triples(n)
    T = len(triples)
    norm = 1.0 / AREA

    contains = []
    inc_pairs = []
    for i in range(n):
        rows = np.where((triples == i).any(axis=1))[0]
        others = np.array([[p for p in triples[t] if p != i] for t in rows], dtype=int)
        contains.append(rows)
        inc_pairs.append((rows, others))

    t0 = time.time()
    time_budget = 9.0

    def eval_full(pts):
        return _areas(pts, triples) * norm

    def try_candidates(pts, area_vec, pi, base):
        """Evaluate K candidate placements for point pi; return best improving one."""
        rows, pairs = inc_pairs[pi]
        pj = base[:, pairs[:, 0], :]
        pk = base[:, pairs[:, 1], :]
        pvi = base[:, pi, :][:, None, :]
        cross = ((pj - pvi)[:, :, 0] * (pk - pvi)[:, :, 1]
                 - (pj - pvi)[:, :, 1] * (pk - pvi)[:, :, 0])
        new_areas = 0.5 * np.abs(cross) * norm
        unaffected_min = np.min(np.delete(area_vec, rows)) if T - len(rows) > 0 else np.inf
        vals = np.minimum(unaffected_min, new_areas.min(axis=1))
        others = np.delete(base, pi, axis=1)
        d = np.linalg.norm(base[:, pi, :][:, None, :] - others, axis=2)
        bad = d.min(axis=1) < 1e-7
        vals = np.where(bad, -1.0, vals)
        j = int(np.argmax(vals))
        if vals[j] > area_vec.min() + 1e-13:
            cand_vec = area_vec.copy()
            cand_vec[rows] = new_areas[j]
            return base[j], cand_vec, True
        return pts, area_vec, False

    def best_of_k_move(pts, area_vec, pi, step, K=24):
        base = np.repeat(pts[None], K, axis=0)
        base[:, pi, :] += rng.normal(0.0, step, size=(K, 2))
        base = _project(base)
        return try_candidates(pts, area_vec, pi, base)

    def guided_move(pts, area_vec, pi, step, dirs):
        """Perpendicular-direction guided moves for point pi (multiple step sizes)."""
        base = np.repeat(pts[None], len(dirs) * 2, axis=0)
        for d_, d in enumerate(dirs):
            base[2 * d_, pi, :] += step * d
            base[2 * d_ + 1, pi, :] += 0.4 * step * d
        base = _project(base)
        return try_candidates(pts, area_vec, pi, base)

    def local_search(pts, budget_deadline):
        area_vec = eval_full(pts)
        step = 0.05
        while step > 1e-6 and time.time() - t0 < budget_deadline:
            improved = False
            w = int(np.argmin(area_vec))
            i, j, k = triples[w]
            dirs = {
                i: _perp_dir(pts[j], pts[k], pts[i]),
                j: _perp_dir(pts[i], pts[k], pts[j]),
                k: _perp_dir(pts[i], pts[j], pts[k]),
            }
            for pi in (i, j, k):
                pts, area_vec, imp = guided_move(pts, area_vec, pi, step, [dirs[pi]])
                if imp:
                    improved = True
            if not improved:
                for oi in rng.permutation(3):
                    pi = triples[w][oi]
                    pts, area_vec, imp = best_of_k_move(pts, area_vec, pi, step)
                    if imp:
                        improved = True
                        break
            if not improved:
                pi = int(rng.integers(n))
                pts, area_vec, imp = best_of_k_move(pts, area_vec, pi, step)
                if not imp:
                    step *= 0.65
        return pts, area_vec.min()

    # ---------- warm starts ----------
    starts = []
    starts.append(_project(np.vstack([verts, _init_points(rng, n - 3)])))
    for m in (3, 4, 5):
        lat = []
        for i in range(m + 1):
            for j in range(m + 1 - i):
                kk = m - i - j
                lat.append([(j + 0.5 * kk) / m, (kk * S3 / 2.0) / m])
        lat = np.array(lat)
        if len(lat) >= n:
            p = lat[:n]
        else:
            p = np.vstack([lat, _init_points(rng, n - len(lat))])
        starts.append(_project(p + rng.normal(0.0, 0.01, size=p.shape)))
    # symmetric structured seeds from inspiration script
    sym = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.5, S3 / 2.0],
        [0.5, 0.0], [0.25, S3 / 4.0], [0.75, S3 / 4.0],
        [0.5, S3 / 18.0], [0.5, S3 / 4.0],
        [0.25, S3 / 12.0], [0.75, S3 / 12.0], [0.5, S3 / 3.0],
    ])
    starts.append(_project(sym))
    cx, cy = 0.5, S3 / 6.0
    ring = [[cx, cy]]
    for r, rot in ((0.30, 0.0), (0.18, np.pi / 5.0)):
        for th in np.linspace(0, 2 * np.pi, 5, endpoint=False) + rot:
            ring.append([cx + r * np.cos(th), cy + r * np.sin(th)])
    starts.append(_project(np.array(ring)))
    # edge-heavy start
    edge_pts = []
    for (P, Q, cnt) in [(verts[0], verts[1], 3), (verts[1], verts[2], 2), (verts[2], verts[0], 2)]:
        for tt in range(1, cnt + 1):
            edge_pts.append(P + (Q - P) * (tt / (cnt + 1)))
    edge_pts = np.array(edge_pts)
    starts.append(_project(np.vstack([verts, edge_pts, _init_points(rng, n - 3 - len(edge_pts))])))

    results = []
    si = 0
    while time.time() - t0 < time_budget * 0.55:
        if si < len(starts):
            p = starts[si].copy()
        else:
            p = _init_points(rng, n)
        si += 1
        p, val = local_search(p, time_budget * 0.55)
        results.append((val, p.copy()))

    if not results:
        return _project(_init_points(np.random.default_rng(0), n))

    results.sort(key=lambda r: -r[0])
    best_val, best_pts = results[0]

    # Phase 2: merge polish on top-3 configurations
    top = results[:3]
    ci = 0
    while time.time() - t0 < time_budget and ci < 6:
        ci += 1
        (v1, p1), (v2, p2) = top[ci % len(top)], top[(ci + 1) % len(top)]
        perm = rng.permutation(n)
        merged = p1.copy()
        merged[perm[5:]] = p2[perm[5:]]
        merged = _project(merged + rng.normal(0.0, 0.005, size=merged.shape))
        p, val = local_search(merged, time_budget)
        if val > best_val:
            best_val, best_pts = val, p.copy()

    best_pts = _project(best_pts)
    if best_pts.shape != (11, 2) or not np.all(np.isfinite(best_pts)):
        return _project(_init_points(np.random.default_rng(0), n))
    return np.asarray(best_pts, dtype=float)


# EVOLVE-BLOCK-END