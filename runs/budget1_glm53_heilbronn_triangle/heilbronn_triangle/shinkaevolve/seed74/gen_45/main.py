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
    """Greedy polish with shrinking random-direction steps.
    Uses fully incremental updates: only the triplets containing the moved
    point are recomputed (in place on a scratch copy), so each candidate
    costs ~45 triangle areas instead of a full 165-triangle recompute."""
    n = b.shape[0]
    b = _project(b.copy())
    xy = _to_xy(b)
    areas = _all_areas_xy(xy, tri)
    cur = areas.min()
    contains = []
    for i in range(n):
        mask = (tri[:, 0] == i) | (tri[:, 1] == i) | (tri[:, 2] == i)
        contains.append((tri[mask], np.nonzero(mask)[0]))
    # fixed 12-direction star for deterministic candidate generation
    angles = 2.0 * np.pi * np.arange(12) / 12.0
    DIRS = np.column_stack([np.cos(angles), np.sin(angles)])
    # per-point triplet index cache reused above
    best, best_b = cur, b.copy()
    for it in range(iters):
        scale = 0.004 * (1.0 - it / iters) + 1e-4
        i = int(rng.integers(0, n))
        sub_tri, sub_idx = contains[i]
        pi = xy[i].copy()
        done = False
        for _ in range(3):
            d = rng.normal(0.0, 1.0, 2)
            d /= (np.linalg.norm(d) + 1e-12)
            cand = b.copy()
            cand[i] += scale * d
            cand = _project(cand)
            cxy = _to_xy(cand)
            # incremental: overwrite only affected entries, restore after eval
            saved = areas[sub_idx].copy()
            a, bb, c = cxy[sub_tri[:, 0]], cxy[sub_tri[:, 1]], cxy[sub_tri[:, 2]]
            areas[sub_idx] = 0.5 * np.abs(
                (bb[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) -
                (bb[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
            val = areas.min()
            if val > cur + 1e-14:
                b, xy, cur = cand, cxy, val
                if val > best:
                    best, best_b = val, b.copy()
                done = True
                break
            areas[sub_idx] = saved  # revert
        if not done:
            xy[i] = pi
        # ---- coordinated pair moves on the top-2 bottleneck points ----
        if it % 3 == 0:
            # bottleneck membership counts: near-min triplets (within 15% of min)
            near = areas < cur * 1.15 + 1e-15
            cnt = np.zeros(n)
            if np.any(near):
                nt = tri[near]
                for p in (0, 1, 2):
                    np.add.at(cnt, nt[:, p], 1.0)
            order = np.argsort(cnt)[::-1]
            p1, p2 = int(order[0]), int(order[1])
            st1, si1 = contains[p1]
            st2, si2 = contains[p2]
            union_idx = np.union1d(si1, si2)
            union_tri = tri[union_idx]
            # candidate moves: 12 dirs for p1 alone-paired with each of the
            # 12 opposite/same dirs of p2 -> 24 coordinated pair candidates
            for dd in DIRS:
                for mode in (-1.0, 1.0):
                    cand = b.copy()
                    cand[p1] += scale * dd
                    cand[p2] += scale * mode * dd
                    cand = _project(cand)
                    cxy = _to_xy(cand)
                    saved = areas[union_idx].copy()
                    a, bb, c = cxy[union_tri[:, 0]], cxy[union_tri[:, 1]], cxy[union_tri[:, 2]]
                    areas[union_idx] = 0.5 * np.abs(
                        (bb[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) -
                        (bb[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
                    val = areas.min()
                    if val > cur + 1e-14:
                        b, xy, cur = cand, cxy, val
                        if val > best:
                            best, best_b = val, b.copy()
                        done = True
                        break
                    areas[union_idx] = saved
                if done:
                    break
    return best_b, best


def heilbronn_triangle11() -> np.ndarray:
    n = 11
    tri = np.array(list(combinations(range(n), 3)))

    best_b, best_val = None, -1.0
    try:
        for si, seed in enumerate(_seed_layouts()):
            rng = np.random.default_rng(20240611 + 101 * si)
            b, val = _anneal(seed, tri, rng)
            b, val = _polish(b, tri, rng)
            if val > best_val:
                best_val, best_b = val, b
        # extra polish rounds on the winner with fresh randomness
        rng = np.random.default_rng(777)
        for _ in range(3):
            best_b, v = _polish(best_b, tri, rng)
            best_val = max(best_val, v)
    except Exception:
        best_b = None

    if best_b is None or not np.all(np.isfinite(_to_xy(best_b))):
        # static feasible fallback
        best_b = _seed_layouts()[0]

    return np.ascontiguousarray(_to_xy(_project(best_b)), dtype=float)


# EVOLVE-BLOCK-END