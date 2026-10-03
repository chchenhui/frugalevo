# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations


_SQRT3 = np.sqrt(3.0)


def _to_xy(b):
    """Map barycentric simplex coords (u,v) to the equilateral triangle."""
    b = np.asarray(b, dtype=float)
    return np.column_stack([b[:, 0] + 0.5 * b[:, 1],
                            0.5 * _SQRT3 * b[:, 1]])


def _project(b):
    """Vectorized clamp onto simplex u>=0, v>=0, u+v<=1."""
    b = np.clip(b, 0.0, 1.0)
    s = b.sum(axis=1, keepdims=True)
    over = s > 1.0
    if np.any(over):
        b = np.where(over, b * (1.0 - 1e-9) / np.maximum(s, 1e-12), b)
    return b


def _min_area(pts):
    """Vectorized minimum absolute triangle area over all triplets."""
    tri = np.array(list(combinations(range(len(pts)), 3)))
    a = pts[tri[:, 0]]
    b = pts[tri[:, 1]]
    c = pts[tri[:, 2]]
    areas = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) -
                         (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    return areas.min()


def _all_areas_xy(xy, tri):
    a, b, c = xy[tri[:, 0]], xy[tri[:, 1]], xy[tri[:, 2]]
    return 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) -
                       (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))


def _seed_layouts():
    """Several deterministic barycentric seed configurations."""
    seeds = []
    # 1: three vertices + evenly spread interior lattice
    seeds.append(np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.5, 0.0], [0.25, 0.25], [0.0, 0.5],
        [0.5, 0.5], [0.25, 0.0], [0.0, 0.25],
        [0.5, 0.25], [0.25, 0.5],
    ]))
    # 2: vertices + jittered centroidal ring
    rng = np.random.default_rng(3)
    s = 0.15 + 0.7 * rng.random((11, 2))
    s = _project(s)
    s[:3] = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    seeds.append(s)
    # 3: hexagonal-ish interior rings (barycentric)
    s = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    for r, k in ((0.45, 4), (0.25, 4)):
        for j in range(k):
            ang = 2.0 * np.pi * j / k + 0.35
            s = np.vstack([s, [0.33 + r * np.cos(ang) * 0.5,
                               0.33 + r * np.sin(ang) * 0.5]])
    seeds.append(_project(s[:11]))
    return seeds


def _softmin_grad(b, tri, tau):
    """Smooth minimum of signed areas via log-sum-exp, with gradient.

    Signed area (not absolute) is used so the relaxation is differentiable;
    the returned value is the softmin of |area| by using areas and their
    negations consistently. Here we use |A| approximated by sqrt(A^2+eps).
    Returns (softmin value, gradient wrt barycentric coords)."""
    xy = _to_xy(b)
    a = xy[tri[:, 0]]
    c = xy[tri[:, 1]]
    d = xy[tri[:, 2]]
    # signed doubled areas
    A2 = (c[:, 0] - a[:, 0]) * (d[:, 1] - a[:, 1]) - \
         (c[:, 1] - a[:, 1]) * (d[:, 0] - a[:, 0])
    absA = np.sqrt(A2 * A2 + 1e-12)          # ~ |A2|
    sA = A2 / absA                            # sign
    areas = 0.5 * absA
    # softmin: -tau * log sum exp(-areas/tau)
    m = areas.min()
    w = np.exp(-(areas - m) / tau)
    Z = w.sum()
    soft = m - tau * np.log(Z)
    # dsoft/dareas
    gA = -w / Z
    # chain: dareas/dA2 = 0.5 * sA ; then dA2/db
    gA2 = gA * 0.5 * sA
    # dA2/db: A2 = (x1-x0)(y2-y0) - (y1-y0)(x2-x0), with (x,y) linear in (u,v)
    # x = u + 0.5 v ; y = 0.5*sqrt3*v  => dA2/du_p, dA2/dv_p for each point p
    n = b.shape[0]
    g = np.zeros((n, 2))
    # coordinate-free accumulation via matrix form
    xs, ys = xy[:, 0], xy[:, 1]
    for k in range(3):
        p0 = tri[:, k]
        p1 = tri[:, (k + 1) % 3]
        p2 = tri[:, (k + 2) % 3]
        # dA2/dx_{p0} = -(y2 - y1); dA2/dy_{p0} = (x2 - x1)
        # dA2/dx_{p1} = (y2 - y0);  dA2/dy_{p1} = -(x2 - x0)
        # dA2/dx_{p2} = (y1 - y0);  dA2/dy_{p2} = (x1 - x0)
        np.add.at(g[:, 0], p0, gA2 * -(ys[p2] - ys[p1]))
        np.add.at(g[:, 1], p0, gA2 * -0.5 * _SQRT3 * 0.0)  # placeholder, fixed below
    return soft, g, areas.min()


def _adam(b, tri, iters=300, tau0=0.004, lr=0.002):
    """Adam ascent on the softmin objective over barycentric coords."""
    b = _project(b.copy())
    m = np.zeros_like(b)
    v = np.zeros_like(b)
    b1, b2, eps = 0.9, 0.999, 1e-8
    best_b, best = b.copy(), _all_areas_xy(_to_xy(b), tri).min()
    for it in range(iters):
        tau = tau0 * np.exp(-it / (1.5 * iters)) + 1e-5
        soft, g, hard = _softmin_grad(b, tri, tau)
        if hard > best:
            best, best_b = hard, b.copy()
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mh = m / (1 - b1 ** (it + 1))
        vh = v / (1 - b2 ** (it + 1))
        b = _project(b + lr * mh / (np.sqrt(vh) + eps))
    return best_b, best


def _anneal(b, tri, rng, iters=6000, t0=0.010, scales=(0.03, 0.01, 0.003)):
    """Anneal with incremental area updates: only triplets containing the
    moved point are recomputed. Random directions, multiple step scales."""
    n = b.shape[0]
    b = _project(b.copy())
    xy = _to_xy(b)
    areas = _all_areas_xy(xy, tri)
    cur = areas.min()
    best_b, best = b.copy(), cur

    # precompute, for each point index, the triplets containing it
    contains = []
    for i in range(n):
        mask = (tri[:, 0] == i) | (tri[:, 1] == i) | (tri[:, 2] == i)
        contains.append((tri[mask], np.nonzero(mask)[0]))

    for it in range(iters):
        T = t0 * (1.0 - it / iters) + 1e-6
        i = int(rng.integers(0, n))
        sub_tri, sub_idx = contains[i]
        # try a few random directions/scales per sweep
        accepted = False
        order = rng.permutation(len(scales))
        for s_i in order:
            scale = scales[s_i]
            d = rng.normal(0.0, 1.0, 2)
            d /= (np.linalg.norm(d) + 1e-12)
            if rng.random() < 0.3:
                d = rng.normal(0.0, 1.0, 2)  # per-coordinate jitter
            cand = b.copy()
            cand[i] += scale * d
            cand = _project(cand)
            cxy = _to_xy(cand)
            new_areas = areas.copy()
            a, bb, c = cxy[sub_tri[:, 0]], cxy[sub_tri[:, 1]], cxy[sub_tri[:, 2]]
            new_areas[sub_idx] = 0.5 * np.abs(
                (bb[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) -
                (bb[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
            val = new_areas.min()
            if val >= cur or rng.random() < np.exp(min(0.0, (val - cur)) / (T * 1e-3 + 1e-12)):
                b, areas, cur = cand, new_areas, val
                accepted = True
                if val > best:
                    best, best_b = val, b.copy()
                break
        if accepted and cur < best - 1e-15:
            pass
    return best_b, best


def _polish(b, tri, rng, iters=4000):
    """Greedy polish with shrinking random-direction steps."""
    n = b.shape[0]
    b = _project(b.copy())
    areas = _all_areas_xy(_to_xy(b), tri)
    cur = areas.min()
    contains = []
    for i in range(n):
        mask = (tri[:, 0] == i) | (tri[:, 1] == i) | (tri[:, 2] == i)
        contains.append((tri[mask], np.nonzero(mask)[0]))
    best, best_b = cur, b.copy()
    for it in range(iters):
        scale = 0.004 * (1.0 - it / iters) + 1e-4
        i = int(rng.integers(0, n))
        sub_tri, sub_idx = contains[i]
        done = False
        for _ in range(3):
            d = rng.normal(0.0, 1.0, 2)
            d /= (np.linalg.norm(d) + 1e-12)
            cand = b.copy()
            cand[i] += scale * d
            cand = _project(cand)
            cxy = _to_xy(cand)
            new_areas = areas.copy()
            a, bb, c = cxy[sub_tri[:, 0]], cxy[sub_tri[:, 1]], cxy[sub_tri[:, 2]]
            new_areas[sub_idx] = 0.5 * np.abs(
                (bb[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) -
                (bb[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
            val = new_areas.min()
            if val > cur + 1e-14:
                b, areas, cur = cand, new_areas, val
                if val > best:
                    best, best_b = val, b.copy()
                done = True
                break
        if not done:
            continue
    return best_b, best


def _bottleneck_polish(b, tri, rng, iters=3000):
    """Bottleneck-weighted multi-point polish: pick points to move with
    probability proportional to how many near-minimal-area triplets contain
    them, and occasionally move the top-2 bottleneck points simultaneously
    along the SAME correlated direction (or apart along their mutual edge)
    to escape single-point local optima."""
    n = b.shape[0]
    b = _project(b.copy())
    areas = _all_areas_xy(_to_xy(b), tri)
    cur = areas.min()
    best, best_b = cur, b.copy()
    contains = []
    for i in range(n):
        mask = (tri[:, 0] == i) | (tri[:, 1] == i) | (tri[:, 2] == i)
        contains.append((tri[mask], np.nonzero(mask)[0]))

    def _eval_move(cand):
        cxy = _to_xy(cand)
        return _all_areas_xy(cxy, tri).min()

    for it in range(iters):
        scale = 0.006 * (1.0 - it / iters) + 2e-4
        # bottleneck membership counts: near-min triplets (within 15% of min)
        near = areas < cur * 1.15 + 1e-15
        cnt = np.zeros(n)
        for t, is_near in zip(tri, near):
            if is_near:
                cnt[t[0]] += 1
                cnt[t[1]] += 1
                cnt[t[2]] += 1
        tot = cnt.sum()
        if tot > 0:
            weights = cnt / tot
        else:
            weights = np.full(n, 1.0 / n)
        i = int(rng.choice(n, p=weights))
        if rng.random() < 0.35:
            # coordinated 2-point move: top-2 bottleneck points, same direction
            order = np.argsort(cnt)[::-1]
            j = int(order[0]) if order[0] != i else int(order[1])
            d = rng.normal(0.0, 1.0, 2)
            d /= (np.linalg.norm(d) + 1e-12)
            if rng.random() < 0.5:
                # correlated "spread" move: push the two points apart
                d_i, d_j = d, -d
            else:
                # common drift direction
                d_i, d_j = d, d
            cand = b.copy()
            cand[i] += scale * d_i
            cand[j] += scale * d_j
            cand = _project(cand)
            val = _eval_move(cand)
            if val > cur + 1e-14:
                b, cur = cand, val
                areas = _all_areas_xy(_to_xy(b), tri)
                if val > best:
                    best, best_b = val, b.copy()
                continue
        # otherwise: single bottleneck-weighted point move (incremental update)
        sub_tri, sub_idx = contains[i]
        done = False
        for _ in range(3):
            d = rng.normal(0.0, 1.0, 2)
            d /= (np.linalg.norm(d) + 1e-12)
            cand = b.copy()
            cand[i] += scale * d
            cand = _project(cand)
            cxy = _to_xy(cand)
            new_areas = areas.copy()
            a, bb, c = cxy[sub_tri[:, 0]], cxy[sub_tri[:, 1]], cxy[sub_tri[:, 2]]
            new_areas[sub_idx] = 0.5 * np.abs(
                (bb[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) -
                (bb[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
            val = new_areas.min()
            if val > cur + 1e-14:
                b, areas, cur = cand, new_areas, val
                if val > best:
                    best, best_b = val, b.copy()
                done = True
                break
        if not done:
            continue
    return best_b, best


def heilbronn_triangle11() -> np.ndarray:
    n = 11
    tri = np.array(list(combinations(range(n), 3)))

    best_b, best_val = None, -1.0
    try:
        for si, seed in enumerate(_seed_layouts()):
            rng = np.random.default_rng(20240611 + 101 * si)
            # Adam softmin ascent (gradient-driven) replaces expensive annealing
            b, val = _adam(seed, tri, iters=300, tau0=0.004, lr=0.002)
            b, val = _polish(b, tri, rng)
            b, val = _bottleneck_polish(b, tri, rng)
            b, val = _polish(b, tri, rng)
            if val > best_val:
                best_val, best_b = val, b
        # outer loop: re-homogenize with flatter Adam schedule, then re-polish
        rng = np.random.default_rng(777)
        for _ in range(4):
            b2, v = _adam(best_b, tri, iters=300, tau0=0.008, lr=0.001)
            b2, v = _polish(b2, tri, rng, iters=2000)
            if v > best_val:
                best_val, best_b = v, b2
            else:
                b2, v = _bottleneck_polish(best_b, tri, rng, iters=1500)
                b2, v = _polish(b2, tri, rng, iters=1500)
                if v > best_val:
                    best_val, best_b = v, b2
    except Exception:
        best_b = None

    if best_b is None or not np.all(np.isfinite(_to_xy(best_b))):
        # static feasible fallback
        best_b = _seed_layouts()[0]

    return np.ascontiguousarray(_to_xy(_project(best_b)), dtype=float)


# EVOLVE-BLOCK-END