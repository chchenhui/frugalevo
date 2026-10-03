# EVOLVE-BLOCK-START
import time
import numpy as np

_TRIP = np.array([[i, j, k] for i in range(11) for j in range(i + 1, 11)
                  for k in range(j + 1, 11)], dtype=np.int64)


def _project_into_triangle(p):
    """Project a point back into the unit equilateral triangle (barycentric clamp)."""
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, np.sqrt(3.0) / 2.0])
    v0, v1, v2 = B - A, C - A, p - A
    d00 = v0 @ v0
    d01 = v0 @ v1
    d11 = v1 @ v1
    d20 = v2 @ v0
    d21 = v2 @ v1
    denom = d00 * d11 - d01 * d01
    u = (d11 * d20 - d01 * d21) / denom
    v = (d00 * d21 - d01 * d20) / denom
    w = 1.0 - u - v
    if u >= 0 and v >= 0 and w >= 0:
        return p
    bu, bv, bw = max(u, 0.0), max(v, 0.0), max(w, 0.0)
    s = bu + bv + bw
    bu, bv, bw = bu / s, bv / s, bw / s
    return bu * A + bv * B + bw * C


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct 11 points inside the unit equilateral triangle maximizing the
    smallest triangle area (Heilbronn problem, n=11).

    Approach: deterministic multi-start simulated annealing with incremental
    evaluation. All 165 triangle areas are cached; moving a point only
    invalidates the 45 triangles containing it (vectorized recompute).
    Structured boundary seed plus random seeds; annealing with shrinking
    Gaussian steps and a mild Metropolis acceptance, followed by greedy
    axis/diagonal and edge-snap polish of the best configuration.
    """
    n = 11
    rng = np.random.default_rng(20240917)
    sqrt3 = np.sqrt(3.0) / 2.0
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, sqrt3])

    rows_of = [np.where((_TRIP == p).any(axis=1))[0] for p in range(n)]

    def cross_all(pts):
        a = pts[_TRIP[:, 0]]
        b = pts[_TRIP[:, 1]]
        c = pts[_TRIP[:, 2]]
        return np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                      - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])) / 2.0

    # ---- seeds: structured boundary layout + random uniform ----
    seeds = []
    s = [A.copy(), B.copy(), C.copy()]
    for t in (1 / 3, 2 / 3):
        s.append(A + t * (B - A))
        s.append(A + t * (C - A))
        s.append(B + t * (C - B))
    for t in (1 / 4, 1 / 2):
        bary = np.array([t, (1 - t) / 2, (1 - t) / 2])
        s.append(bary[0] * A + bary[1] * B + bary[2] * C)
    seeds.append(np.array(s))

    for _ in range(8):
        u = rng.random(n)
        v = rng.random(n)
        keep = u + v <= 1.0
        u[~keep] = 1 - u[~keep]
        y = v / sqrt3
        y[~keep] = 1 - y[~keep]
        seeds.append(np.stack([u, y * sqrt3], axis=1))

    deadline = time.time() + 150.0
    best_pts = None
    best_val = -1.0

    for seed_pts in seeds:
        pts = np.array([_project_into_triangle(p) for p in seed_pts])
        vals = cross_all(pts)
        cur_val = vals.min()
        step = 0.05
        temp = 1e-3
        stall = 0
        while time.time() < deadline and step > 1e-6:
            idx = int(rng.integers(n))
            old = pts[idx].copy()
            new = _project_into_triangle(old + rng.normal(0.0, step, 2))
            if np.allclose(new, old):
                continue
            rows = rows_of[idx]
            saved = vals[rows].copy()
            pts[idx] = new
            t = _TRIP[rows]
            a = pts[t[:, 0]]
            b = pts[t[:, 1]]
            c = pts[t[:, 2]]
            vals[rows] = np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                                - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])) / 2.0
            new_min = vals.min()
            accept = False
            if new_min >= cur_val - 1e-15:
                accept = True
            elif new_min > 0.0 and cur_val - new_min < temp * cur_val:
                accept = rng.random() < 0.5
            if accept:
                stall = 0 if new_min > cur_val + 1e-15 else stall + 1
                cur_val = new_min
            else:
                pts[idx] = old
                vals[rows] = saved
                stall += 1
            if stall > 4000:
                step *= 0.7
                temp *= 0.6
                stall = 0
        if cur_val > best_val:
            best_val = cur_val
            best_pts = pts.copy()
        if time.time() > deadline:
            break

    if best_pts is None:
        # graceful fallback: structured seed
        best_pts = seeds[0].copy()

    # greedy polish with axis/diagonal moves and edge snapping
    pts = best_pts.copy()
    vals = cross_all(pts)
    cur_val = vals.min()
    step = 0.01
    while time.time() < deadline and step > 1e-7:
        improved = False
        for idx in range(n):
            for dx, dy in ((step, 0), (-step, 0), (0, step), (0, -step),
                           (step, step), (-step, step), (step, -step), (-step, -step)):
                old = pts[idx].copy()
                new = _project_into_triangle(old + np.array([dx, dy]))
                if np.allclose(new, old):
                    continue
                rows = rows_of[idx]
                saved = vals[rows].copy()
                pts[idx] = new
                t = _TRIP[rows]
                a = pts[t[:, 0]]
                b = pts[t[:, 1]]
                c = pts[t[:, 2]]
                vals[rows] = np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                                    - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])) / 2.0
                v = vals.min()
                if v > cur_val + 1e-15:
                    cur_val = v
                    improved = True
                else:
                    pts[idx] = old
                    vals[rows] = saved
            # edge-snap moves: good n=11 configs place many points on edges
            for org, e0 in ((A, B - A), (A, C - A), (B, C - B)):
                old = pts[idx].copy()
                v2 = old - org
                u = (v2 @ e0) / (e0 @ e0)
                if u < -1e-12 or u > 1.0 + 1e-12:
                    continue
                new = org + u * e0
                if np.allclose(new, old):
                    continue
                rows = rows_of[idx]
                saved = vals[rows].copy()
                pts[idx] = new
                t = _TRIP[rows]
                a = pts[t[:, 0]]
                b = pts[t[:, 1]]
                c = pts[t[:, 2]]
                vals[rows] = np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                                    - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])) / 2.0
                v = vals.min()
                if v > cur_val + 1e-15:
                    cur_val = v
                    improved = True
                else:
                    pts[idx] = old
                    vals[rows] = saved
        if not improved:
            step *= 0.5

    return pts


# EVOLVE-BLOCK-END
