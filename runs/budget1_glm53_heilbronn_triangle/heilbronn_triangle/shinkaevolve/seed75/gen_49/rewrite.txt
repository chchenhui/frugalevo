# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations

S3 = np.sqrt(3.0)
AREA = S3 / 4.0


def _all_triples(n):
    return np.array(list(combinations(range(n), 3)), dtype=int)


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


def heilbronn_triangle11() -> np.ndarray:
    n = 11
    rng = np.random.default_rng(987654321)
    verts = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, S3 / 2.0]])
    triples = _all_triples(n)
    T = len(triples)
    norm = 1.0 / AREA
    ti, tj, tk = triples[:, 0], triples[:, 1], triples[:, 2]

    # incremental pair lists for single-point updates
    inc_pairs = []
    for i in range(n):
        rows = np.where((triples == i).any(axis=1))[0]
        others = np.array([[p for p in triples[t] if p != i] for t in rows], dtype=int)
        inc_pairs.append((rows, others))

    t0 = time.time()
    time_budget = 9.0

    def signed_areas(pts):
        p = pts[triples]
        a = p[:, 1] - p[:, 0]
        b = p[:, 2] - p[:, 0]
        return 0.5 * (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])

    def softmin_grad(pts, tau):
        """Value and gradient of -tau*log(sum exp(-A/tau)) w.r.t. points."""
        sa = signed_areas(pts)
        A = np.abs(sa) * norm
        sgn = np.where(sa >= 0, 1.0, -1.0)
        w = np.exp(-(A - A.min()) / tau)
        w /= w.sum()
        val = A.min() - tau * np.log(np.sum(np.exp(-(A - A.min()) / tau)))
        grad = np.zeros_like(pts)
        # dA_t/dp: for cross product c = 0.5*|cross|*norm
        p = pts[triples]
        for r in range(3):
            # d cross / d p_r  (signed, times sgn*0.5*norm)
            i0, i1, i2 = r, (r + 1) % 3, (r + 2) % 3
            p0, p1, p2 = p[:, i0], p[:, i1], p[:, i2]
            # cross = (p1-p0)x(p2-p0)
            dc = np.zeros((T, 2))
            dc[:, 0] = -(p1[:, 1] - p2[:, 1])
            dc[:, 1] = (p1[:, 0] - p2[:, 0])
            coef = w * sgn * 0.5 * norm
            contrib = coef[:, None] * dc
            np.add.at(grad, ti if r == 0 else (tj if r == 1 else tk), contrib)
        return val, grad

    def try_candidates(pts, area_vec, pi, base):
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

    def best_of_k_move(pts, area_vec, pi, step, K=32):
        base = np.repeat(pts[None], K, axis=0)
        base[:, pi, :] += rng.normal(0.0, step, size=(K, 2))
        base = _project(base)
        return try_candidates(pts, area_vec, pi, base)

    def triplet_refine(pts, iters=20):
        """Discrete refinement of the current worst triplet only."""
        for _ in range(iters):
            area_vec = np.abs(signed_areas(pts)) * norm
            w = int(np.argmin(area_vec))
            if area_vec[w] <= 0:
                # degenerate: perturb the triplet strongly
                for pi in triples[w]:
                    pts, _av, imp = best_of_k_move(pts, area_vec, pi, 0.02, 32)
                continue
            improved = False
            step = 0.01
            for pi in rng.permutation(triples[w]):
                pts, area_vec, imp = best_of_k_move(pts, area_vec, int(pi), step, 32)
                if imp:
                    improved = True
            if not improved:
                return pts
        return pts

    def optimize(pts, deadline):
        """Alternating softmin gradient ascent and discrete triplet refinement."""
        lr = 0.002
        tau = 0.02
        while time.time() - t0 < deadline:
            # --- softmin gradient stage (with backtracking line search) ---
            val, g = softmin_grad(pts, tau)
            step_pts = pts + lr * g
            step_pts = _project(step_pts)
            nv, _ = softmin_grad(step_pts, tau)
            if nv < val:
                lr *= 0.5
                if lr < 1e-6:
                    break
                continue
            pts = step_pts
            lr *= 1.05
            # --- discrete worst-triplet refinement every few steps ---
            if rng.random() < 0.4:
                pts = triplet_refine(pts, iters=8)
        # final polish
        pts = triplet_refine(pts, iters=60)
        return pts

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
    sym = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.5, S3 / 2.0],
        [0.5, 0.0], [0.25, S3 / 4.0], [0.75, S3 / 4.0],
        [0.5, S3 / 18.0], [0.5, S3 / 4.0],
        [0.25, S3 / 12.0], [0.75, S3 / 12.0], [0.5, S3 / 3.0],
    ])
    starts.append(_project(sym))
    edge_pts = []
    for (P, Q, cnt) in [(verts[0], verts[1], 3), (verts[1], verts[2], 2), (verts[2], verts[0], 2)]:
        for tt in range(1, cnt + 1):
            edge_pts.append(P + (Q - P) * (tt / (cnt + 1)))
    edge_pts = np.array(edge_pts)
    starts.append(_project(np.vstack([verts, edge_pts, _init_points(rng, n - 3 - len(edge_pts))])))

    def score(pts):
        return float(np.min(np.abs(signed_areas(pts)) * norm))

    best_pts = None
    best_val = -1.0
    si = 0
    while time.time() - t0 < time_budget:
        if si < len(starts):
            p = starts[si].copy()
        else:
            p = _init_points(rng, n)
        si += 1
        p = optimize(p, min(time_budget, time_budget * 0.55 + t0 - time.time() * 0.0))
        v = score(p)
        if v > best_val:
            best_val, best_pts = v, p.copy()

    if best_pts is None:
        best_pts = _project(_init_points(np.random.default_rng(0), n))

    best_pts = _project(best_pts)
    if best_pts.shape != (11, 2) or not np.all(np.isfinite(best_pts)):
        return _project(_init_points(np.random.default_rng(0), n))
    return np.asarray(best_pts, dtype=float)


# EVOLVE-BLOCK-END