# EVOLVE-BLOCK-START
import numpy as np
import itertools

H = np.sqrt(3.0) / 2.0
SQ3 = np.sqrt(3.0)

# Precompute all 165 triplet index combinations once at module level.
_TRIPLETS = np.array(list(itertools.combinations(range(11), 3)), dtype=int)  # (165,3)
_TRI_AREA = np.sqrt(3.0) / 4.0


def _all_areas(pts):
    """Vectorized doubled areas of all 165 triplets."""
    p = pts[_TRIPLETS]  # (165, 3, 2)
    cross = (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - \
            (p[:, 1, 1] - p[:, 0, 1]) * (p[:, 2, 0] - p[:, 0, 0])
    return np.abs(cross)


def _min_area(pts):
    """Doubled area and index of the smallest triangle."""
    a = _all_areas(pts)
    k = int(np.argmin(a))
    return a[k], k


def _project_inside(pts):
    x, y = pts[:, 0], pts[:, 1]
    w1 = x - y / SQ3
    w2 = 2.0 * y / SQ3
    w0 = 1.0 - w1 - w2
    w = np.stack([w0, w1, w2], axis=1)
    w = np.clip(w, 0.0, None)
    s = w.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    w = w / s
    out = np.empty_like(pts)
    out[:, 0] = w[:, 1] + 0.5 * w[:, 2]
    out[:, 1] = H * w[:, 2]
    return out


def _softmin(pts, k=60.0):
    """Smoothed minimum over near-minimal doubled areas (soft lower bound)."""
    a = _all_areas(pts)
    m = a.min()
    sel = a <= m + 0.05 * _TRI_AREA * 2.0
    return m - np.log(np.exp(-k * (a[sel] - m)).sum()) / k


def _soft_objective(areas, frac=0.15, temp=200.0):
    kk = max(3, int(len(areas) * frac))
    smallest = np.partition(areas, kk - 1)[:kk]
    return -np.log(np.mean(np.exp(-smallest * temp)) + 1e-300) / temp


def _lattice_seed():
    pts = []
    rows = [4, 3, 2, 1, 1]
    nlev = len(rows)
    for r, cnt in enumerate(rows):
        y = H * (r + 0.5) / nlev
        xs = np.linspace(0.0, 1.0, cnt + 2)[1:-1]
        halfw = 0.5 * (1.0 - y / H)
        for x in xs:
            pts.append((0.5 + (x - 0.5) * (2 * halfw) * 0.98, y))
    return _project_inside(np.array(pts))


def _rings_seed():
    out = [(0.0, 0.0), (1.0, 0.0), (0.5, H),
           (0.5, 0.0), (0.25, H / 2.0), (0.75, H / 2.0)]
    cx, cy = 0.5, H / 3.0
    for s in (0.5, 0.25):
        for (vx, vy) in [(0.0, 0.0), (1.0, 0.0), (0.5, H)]:
            out.append((cx + s * (vx - cx), cy + s * (vy - cy)))
    out.append((cx, cy))
    return _project_inside(np.array(out[:11]))


def _hex_perturb_seed(seed=777):
    rng = np.random.default_rng(seed)
    pts = []
    rows = [5, 4, 2]
    nlev = len(rows)
    for r, cnt in enumerate(rows):
        y = H * (r + 0.5) / nlev
        xs = np.linspace(0.0, 1.0, cnt + 2)[1:-1]
        halfw = 0.5 * (1.0 - y / H) * 0.95
        for x in xs:
            px = 0.5 + (x - 0.5) * 2 * halfw + rng.normal(0, 0.02)
            py = y + rng.normal(0, 0.02)
            pts.append((px, py))
    return _project_inside(np.array(pts))


def _vertex_start():
    h = H
    return _project_inside(np.array([
        [0.0, 0.0], [1.0, 0.0], [0.5, h],
        [1.0 / 3.0, 0.0], [2.0 / 3.0, 0.0],
        [1.0 / 6.0, h / 3.0], [5.0 / 6.0, h / 3.0],
        [1.0 / 3.0, 2.0 * h / 3.0], [2.0 / 3.0, 2.0 * h / 3.0],
        [0.5, h / 3.0], [0.5, 0.0],
    ]))


def _random_start(rng):
    u, v = rng.random(11), rng.random(11)
    su = np.sqrt(u)
    w0, w1, w2 = 1 - su, su * (1 - v), su * v
    return _project_inside(np.stack([w1 + 0.5 * w2, H * w2], axis=1))


def _soft_search(pts, step, iters, rng):
    pts = _project_inside(pts.copy())
    best = _soft_objective(_all_areas(pts))
    n = len(pts)
    NC = 8
    for it in range(iters):
        improved = False
        for i in range(n):
            for d in range(2):
                base = pts[i, d]
                deltas = rng.normal(0.0, step, size=NC)
                cands = np.repeat(pts[None, :, :], NC, axis=0).copy()
                cands[:, i, d] = base + deltas
                cands = _project_inside(cands.reshape(-1, 2)).reshape(NC, n, 2)
                vals = np.empty(NC)
                for ci in range(NC):
                    vals[ci] = _soft_objective(_all_areas(cands[ci]))
                k = int(np.argmax(vals))
                if vals[k] > best + 1e-13:
                    pts = cands[k]
                    best = vals[k]
                    improved = True
        if not improved:
            step *= 0.5
            if step < 1e-6:
                break
    return pts


def _annealed_group_phase(pts, iters, rng):
    """Soft-min annealing with coordinated group moves on the smallest
    triangle's vertices; Metropolis acceptance allows escaping pinning
    configurations. Returns best hard-min configuration seen."""
    n = 11
    cur = _project_inside(pts.copy())
    best = cur.copy()
    best_val = _min_area(cur)[0]
    soft_cur = _softmin(cur)
    for it in range(iters):
        step = 0.05 * (1.0 - it / iters) + 1e-3
        trial = cur.copy()
        if rng.random() < 0.20:
            a_all = _all_areas(cur)
            t = int(np.argmin(a_all))
            i0, i1, i2 = _TRIPLETS[t]
            verts = cur[[i0, i1, i2]]
            cen = verts.mean(axis=0)
            shared = rng.normal(0.0, 0.6 * step, size=2)
            for vi, v in zip((i0, i1, i2), verts):
                d = v - cen
                nd = np.hypot(d[0], d[1])
                d = d / nd if nd > 1e-12 else np.array([0.0, 1.0])
                delta = shared + d * step * (0.5 + rng.random()) \
                        + rng.normal(0.0, 0.35 * step, size=2)
                trial[vi] = trial[vi] + delta
        else:
            i = rng.integers(n)
            trial[i] += rng.normal(0.0, step, size=2)
        trial = _project_inside(trial)
        s_new = _softmin(trial)
        if s_new >= soft_cur or rng.random() < np.exp((s_new - soft_cur) * 40.0):
            cur, soft_cur = trial, s_new
            v = _min_area(cur)[0]
            if v > best_val:
                best, best_val = cur.copy(), v
    return best


def _hard_refine(pts, step, rounds, rng):
    pts = _project_inside(pts.copy())
    best, kmin = _min_area(pts)
    n = len(pts)
    for r in range(rounds):
        improved = False
        tri = _TRIPLETS[kmin]
        order = list(tri) + [i for i in range(n) if i not in tri]
        for i in order:
            for d in range(2):
                for s in (+1, -1):
                    cand = pts.copy()
                    cand[i, d] += s * step
                    cand = _project_inside(cand)
                    val, _ = _min_area(cand)
                    if val > best + 1e-13:
                        pts, best = cand, val
                        improved = True
        for t in range(25):
            if rng.random() < 0.7:
                i = int(tri[rng.integers(0, 3)])
            else:
                i = int(rng.integers(0, n))
            cand = pts.copy()
            cand[i] += rng.normal(0.0, step, size=2)
            cand = _project_inside(cand)
            val, _ = _min_area(cand)
            if val > best + 1e-13:
                pts, best = cand, val
                improved = True
        _, kmin = _min_area(pts)
        if not improved:
            step *= 0.5
            if step < 1e-7:
                break
    return pts


def _directed_polish(pts, step, rounds, rng, k_tri=10):
    pts = _project_inside(pts.copy())
    best, _ = _min_area(pts)
    for r in range(rounds):
        improved = False
        areas = _all_areas(pts)
        order = np.argsort(areas)[:k_tri]
        for tk in order:
            i0, i1, i2 = _TRIPLETS[tk]
            verts = [i0, i1, i2]
            for vi in range(3):
                i = verts[vi]
                p = pts[verts[(vi + 1) % 3]] - pts[verts[(vi - 1) % 3]]
                plen = np.hypot(p[0], p[1])
                if plen < 1e-12:
                    continue
                nrm = np.array([p[1], -p[0]]) / plen
                if np.dot(nrm, pts[i] - pts[verts[(vi + 1) % 3]]) < 0.0:
                    nrm = -nrm
                for s in (1.0, 0.5, 0.25, -0.5, -0.25):
                    cand = pts.copy()
                    cand[i] = cand[i] + s * step * nrm
                    cand = _project_inside(cand)
                    val, _ = _min_area(cand)
                    if val > best + 1e-13:
                        pts, best, improved = cand, val, True
                        break
        for t in range(12):
            tk = order[rng.integers(0, len(order))]
            i = _TRIPLETS[tk][rng.integers(0, 3)]
            cand = pts.copy()
            cand[i] += rng.normal(0.0, step, size=2)
            cand = _project_inside(cand)
            val, _ = _min_area(cand)
            if val > best + 1e-13:
                pts, best, improved = cand, val, True
        if not improved:
            step *= 0.5
            if step < 1e-7:
                break
    return pts


def _vertex_binding_scores(pts, k=20):
    areas = _all_areas(pts)
    n = len(pts)
    scores = np.zeros(n)
    for i in range(n):
        mask = (_TRIPLETS[:, 0] == i) | (_TRIPLETS[:, 1] == i) | (_TRIPLETS[:, 2] == i)
        a = areas[mask]
        kk = min(k, len(a))
        scores[i] = np.mean(np.partition(a, kk - 1)[:kk])
    return scores


def _pair_exchange(pts, rng, attempts=30):
    pts = _project_inside(pts.copy())
    best_val, _ = _min_area(pts)
    n = len(pts)
    for _ in range(attempts):
        _, kmin = _min_area(pts)
        tri = _TRIPLETS[kmin]
        scores = _vertex_binding_scores(pts)
        bi = min(tri, key=lambda v: scores[v])
        cand = [i for i in range(n) if i not in tri]
        cand.sort(key=lambda v: -scores[v])
        ni = cand[rng.integers(0, min(5, len(cand)))]
        sw = pts.copy()
        sw[bi], sw[ni] = sw[ni].copy(), sw[bi].copy()
        sw[bi] = sw[bi] + rng.normal(0.0, 0.01, size=2)
        sw = _project_inside(sw)
        sw = _hard_refine(sw, 0.01, 15, rng)
        val, _ = _min_area(sw)
        if val > best_val + 1e-12:
            pts, best_val = sw, val
    return pts


def _optimize():
    rng = np.random.default_rng(20240517)
    starts = [_lattice_seed(), _rings_seed(), _hex_perturb_seed(777),
              _hex_perturb_seed(4242), _vertex_start()]
    starts.append(_random_start(rng))
    starts.append(_random_start(rng))
    best_pts, best_val = None, -1.0
    for s in starts:
        cand = _soft_search(s, 0.02, 40, rng)
        cand = _hard_refine(cand, 0.005, 50, rng)
        val, _ = _min_area(cand)
        if val > best_val:
            best_pts, best_val = cand, val
    # hard polish on the winner
    cand = _hard_refine(best_pts, 0.002, 40, rng)
    cand = _directed_polish(cand, 0.002, 30, rng)
    val, _ = _min_area(cand)
    if val > best_val:
        best_pts, best_val = cand, val
    # annealed group-move escape from pinning configurations
    cand = _annealed_group_phase(best_pts, 6000, rng)
    cand = _hard_refine(cand, 0.003, 30, rng)
    cand = _directed_polish(cand, 0.001, 25, rng)
    val, _ = _min_area(cand)
    if val > best_val:
        best_pts, best_val = cand, val
    # structured pair-exchange escape from the local basin
    cand = _pair_exchange(best_pts, rng)
    cand = _directed_polish(cand, 0.001, 25, rng)
    val, _ = _min_area(cand)
    if val > best_val:
        best_pts, best_val = cand, val
    return _project_inside(best_pts)


try:
    _POINTS = _optimize()
except Exception:
    _POINTS = _lattice_seed()


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 11.

    Returns:
        points: np.ndarray of shape (11,2) with the x,y coordinates of the points.
    """
    return _POINTS.copy()

# EVOLVE-BLOCK-END