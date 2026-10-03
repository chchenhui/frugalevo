# EVOLVE-BLOCK-START
import numpy as np

H = np.sqrt(3.0) / 2.0


def _all_triple_indices(n):
    idx = [(i, j, k) for i in range(n) for j in range(i + 1, n) for k in range(j + 1, n)]
    return np.array(idx)


def _all_areas(pts, triples):
    p = pts[triples]
    a = p[:, 1] - p[:, 0]
    b = p[:, 2] - p[:, 0]
    return 0.5 * np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])


def _min_area(pts, triples):
    areas = _all_areas(pts, triples)
    k = int(np.argmin(areas))
    return areas[k], k


def _project_inside(pts):
    x, y = pts[:, 0], pts[:, 1]
    w1 = x - y / np.sqrt(3.0)
    w2 = 2.0 * y / np.sqrt(3.0)
    w0 = 1.0 - w1 - w2
    w = np.stack([w0, w1, w2], axis=1)
    w = np.clip(w, 0.0, None)
    w = w / w.sum(axis=1, keepdims=True)
    out = np.empty_like(pts)
    out[:, 0] = w[:, 1] + 0.5 * w[:, 2]
    out[:, 1] = H * w[:, 2]
    return out


def _soft_objective(areas, frac=0.2):
    k = max(3, int(len(areas) * frac))
    smallest = np.partition(areas, k - 1)[:k]
    return -np.log(np.mean(np.exp(-smallest * 300.0)) + 1e-300) / 300.0


def _lattice_seed():
    # hexagonal-style rows inside the triangle: 4+3+2+1+1 = 11 points
    pts = []
    rows = [4, 3, 2, 1, 1]
    total = H
    ys = []
    # vertical positions spread evenly
    counts = rows
    nlev = len(counts)
    for r, cnt in enumerate(counts):
        y = H * (r + 0.5) / nlev
        xs = np.linspace(0.0, 1.0, cnt + 2)[1:-1]
        # shrink row to fit inside triangle at height y
        halfw = 0.5 * (1.0 - y / H)
        center = 0.5
        for x in xs:
            pts.append((center + (x - 0.5) * (2 * halfw) * 0.98, y))
    return _project_inside(np.array(pts))


def _random_start(rng):
    u, v = rng.random(11), rng.random(11)
    su = np.sqrt(u)
    w0, w1, w2 = 1 - su, su * (1 - v), su * v
    return np.stack([w1 + 0.5 * w2, H * w2], axis=1)


def _soft_search(pts, triples, step, iters):
    pts = _project_inside(pts.copy())
    best = _soft_objective(_all_areas(pts, triples))
    n = len(pts)
    for it in range(iters):
        improved = False
        for i in range(n):
            for d in range(2):
                for s in (+1, -1):
                    cand = pts.copy()
                    cand[i, d] += s * step
                    cand = _project_inside(cand)
                    val = _soft_objective(_all_areas(cand, triples))
                    if val > best + 1e-13:
                        pts, best = cand, val
                        improved = True
        if not improved:
            # pairwise move attempt before shrinking
            done = True
            for i in range(n):
                for j in range(i + 1, n):
                    cand = pts.copy()
                    cand[i, 0] += step
                    cand[j, 0] -= step
                    cand = _project_inside(cand)
                    val = _soft_objective(_all_areas(cand, triples))
                    if val > best + 1e-13:
                        pts, best = cand, val
                        done = False
                        break
                if not done:
                    break
            if done:
                step *= 0.5
                if step < 1e-6:
                    break
    return pts


def _hard_refine(pts, triples, step, rounds):
    pts = _project_inside(pts.copy())
    best, kmin = _min_area(pts, triples)
    n = len(pts)
    rng = np.random.default_rng(31337)
    for r in range(rounds):
        improved = False
        tri = triples[kmin]
        # move each point of the minimal triangle first (binding constraint)
        order = list(tri) + [i for i in range(n) if i not in tri]
        for i in order:
            for d in range(2):
                for s in (+1, -1):
                    cand = pts.copy()
                    cand[i, d] += s * step
                    cand = _project_inside(cand)
                    val, _ = _min_area(cand, triples)
                    if val > best + 1e-13:
                        pts, best, _ = cand, val, None
                        improved = True
        # targeted stochastic phase: bias toward argmin-triplet points
        for t in range(30):
            if rng.random() < 0.7:
                i = int(tri[rng.integers(0, 3)])
            else:
                i = int(rng.integers(0, n))
            delta = rng.normal(0.0, step, size=2)
            cand = pts.copy()
            cand[i] += delta
            cand = _project_inside(cand)
            val, _ = _min_area(cand, triples)
            if val > best + 1e-13:
                pts, best = cand, val
                improved = True
        _, kmin = _min_area(pts, triples)
        if not improved:
            step *= 0.5
            if step < 1e-7:
                break
    return pts


def _optimize():
    triples = _all_triple_indices(11)
    rng = np.random.default_rng(20240517)
    starts = [_lattice_seed()]
    for _ in range(10):
        starts.append(_random_start(rng))
    best_pts, best_val = None, -1.0
    for s in starts:
        cand = _soft_search(s, triples, 0.02, 50)
        cand = _hard_refine(cand, triples, 0.005, 60)
        val, _ = _min_area(cand, triples)
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