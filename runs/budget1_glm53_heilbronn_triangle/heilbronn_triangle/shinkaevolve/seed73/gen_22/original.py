# EVOLVE-BLOCK-START
import time
import numpy as np

_TRI_AREA = np.sqrt(3.0) / 4.0  # area of the unit equilateral triangle
from itertools import combinations
_TRIPLETS = np.array(list(combinations(range(11), 3)), dtype=int)


def _min_area(points: np.ndarray) -> float:
    """Vectorized area of the smallest triangle among all triplets (normalized)."""
    # Precompute all triplet index combinations once (cached module-level)
    idx = _TRIPLETS
    p = points[idx]  # (165, 3, 2)
    # Twice the signed area via cross product
    a = (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - \
        (p[:, 1, 1] - p[:, 0, 1]) * (p[:, 2, 0] - p[:, 0, 0])
    return np.abs(a).min() / (2.0 * _TRI_AREA * 2.0) * 2.0  # = |a|/2 / TRI_AREA


def _in_triangle(pt):
    """Return True if pt lies inside the unit equilateral triangle."""
    x, y = pt
    return x >= -1e-9 and y >= -1e-9 and y <= np.sqrt(3.0) * x + 1e-9 and \
        y <= -np.sqrt(3.0) * (x - 1.0) + 1e-9


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct an arrangement of 11 points on or inside the unit equilateral
    triangle maximizing the minimum triangle area (Heilbronn problem, n=11).

    Deterministic: fixed seed, fixed budget. Returns (11,2) array.
    """
    n = 11
    h = np.sqrt(3.0) / 2.0
    # Symmetric initial configuration: 3 vertices, points along medians/layers
    pts = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, h],
        [0.25, h / 2.0],
        [0.75, h / 2.0],
        [0.5, 0.0],
        [0.125, h / 4.0],
        [0.875, h / 4.0],
        [0.375, 3.0 * h / 4.0],
        [0.625, 3.0 * h / 4.0],
        [0.5, h / 4.0],
    ])

    def _min_area_and_triplet(p):
        idx = _TRIPLETS
        q = p[idx]
        a = (q[:, 1, 0] - q[:, 0, 0]) * (q[:, 2, 1] - q[:, 0, 1]) - \
            (q[:, 1, 1] - q[:, 0, 1]) * (q[:, 2, 0] - q[:, 0, 0])
        a = np.abs(a)
        m = a.argmin()
        return a[m] / (2.0 * _TRI_AREA), idx[m]

    def _project(p):
        x, y = p
        y = min(max(y, 0.0), h)
        s3 = np.sqrt(3.0)
        if y > s3 * x:
            x = y / s3
        if y > -s3 * (x - 1.0):
            x = 1.0 - y / s3
        x = min(max(x, 0.0), 1.0)
        return np.array([x, y])

    # Deterministic multi-restart: perturb seeds around the symmetric layout.
    rng = np.random.default_rng(12345)
    deadline = time.time() + 1.8
    best = pts.copy()
    best_val = _min_area_and_triplet(best)[0]

    seeds = [pts.copy()]
    for s in range(5):
        q = pts + rng.normal(0.0, 0.02, pts.shape)
        seeds.append(np.array([_project(p) for p in q]))

    for seed in seeds:
        if time.time() > deadline:
            break
        cur = seed.copy()
        cur_val, tri = _min_area_and_triplet(cur)
        step = 0.02
        while time.time() < deadline:
            improved = False
            i0, i1, i2 = tri
            cen = (cur[[i0, i1, i2]]).mean(axis=0)
            # candidate moves: push each vertex of min triangle away from its
            # centroid, plus lateral and random jitter moves
            moves = []
            for vi in (i0, i1, i2):
                d = cur[vi] - cen
                nrm = np.hypot(*d)
                if nrm > 1e-12:
                    d = d / nrm
                else:
                    d = np.array([0.0, 1.0])
                perp = np.array([-d[1], d[0]])
                for st in (step, 0.5 * step):
                    moves.append((vi, d * st))
                    moves.append((vi, d * st + perp * 0.3 * st))
                    moves.append((vi, d * st - perp * 0.3 * st))
                    moves.append((vi, -d * 0.3 * st))
            for r in range(6):
                vi = int(rng.integers(n))
                moves.append((vi, rng.normal(0.0, step, 2)))
            for vi, mv in moves:
                cand = cur.copy()
                cand[vi] = _project(cand[vi] + mv)
                v, t2 = _min_area_and_triplet(cand)
                if v > cur_val + 1e-12:
                    cur, cur_val, tri = cand, v, t2
                    improved = True
                    break
            if not improved:
                step *= 0.5
                if step < 1e-5:
                    break
            if cur_val > best_val:
                best, best_val = cur.copy(), cur_val
        if best_val >= 0.0365:
            break

    pts = best
    return pts


# EVOLVE-BLOCK-END