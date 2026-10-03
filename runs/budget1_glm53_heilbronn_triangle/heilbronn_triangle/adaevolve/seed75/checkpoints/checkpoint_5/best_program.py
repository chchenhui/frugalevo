# EVOLVE-BLOCK-START
import time
import numpy as np


_TRIP = np.array([[i, j, k] for i in range(11) for j in range(i + 1, 11)
                  for k in range(j + 1, 11)], dtype=np.int64)


def _min_area(points):
    """Minimum absolute triangle area over all C(11,3)=165 triplets (fully vectorized)."""
    a = points[_TRIP[:, 0]]
    b = points[_TRIP[:, 1]]
    c = points[_TRIP[:, 2]]
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) \
        - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    return np.abs(cross).min() / 2.0


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
    # clamp barycentric coords
    bu, bv, bw = max(u, 0.0), max(v, 0.0), max(w, 0.0)
    s = bu + bv + bw
    bu, bv, bw = bu / s, bv / s, bw / s
    return bu * A + bv * B + bw * C


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct 11 points inside the unit equilateral triangle maximizing the
    smallest triangle area (Heilbronn problem, n=11).

    Approach: deterministic seeded multi-start simulated annealing.
    - Start from several structured seeds (boundary hexagonal-ish layouts,
      interior grid, random uniform) projected into the triangle.
    - Local annealing: repeatedly pick a random point, perturb it by a
      shrinking Gaussian step, keep the move if the minimum triangle area
      (computed incrementally over the 55 pairs against the moved point,
      plus the untouched pairs' cached minimum) does not decrease.
    - Best configuration across restarts is returned.

    Deterministic: fixed RNG seed, fixed iteration counts, wall-clock cap.
    """
    n = 11
    rng = np.random.default_rng(12345)
    sqrt3 = np.sqrt(3.0) / 2.0
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, sqrt3])

    def tri_area(pts):
        return _min_area(pts)

    # Precompute, for each point index, the rows of _TRIP (triplet rows)
    # that contain that point -> only these 45 triangles change per move.
    rows_of = [np.where((_TRIP == p).any(axis=1))[0] for p in range(n)]

    def cross_all(pts):
        a = pts[_TRIP[:, 0]]
        b = pts[_TRIP[:, 1]]
        c = pts[_TRIP[:, 2]]
        return np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                      - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])) / 2.0

    # ---- build seeds ----
    seeds = []

    # structured seed: 3 vertices + points along edges + interior
    s = []
    s.append(A.copy())
    s.append(B.copy())
    s.append(C.copy())
    # edge points
    for t in (1 / 3, 2 / 3):
        s.append(A + t * (B - A))
        s.append(A + t * (C - A))
        s.append(B + t * (C - B))
    # interior points (centroid-ish, perturbed)
    for t in (1 / 4, 1 / 2):
        bary = np.array([t, (1 - t) / 2, (1 - t) / 2])
        s.append(bary[0] * A + bary[1] * B + bary[2] * C)
    seeds.append(np.array(s))

    # random seeds
    for _ in range(6):
        u = rng.random(n)
        v = rng.random(n)
        keep = u + v <= 1.0
        u[~keep] = 1 - u[~keep]
        pts = np.stack([u, v * sqrt3 * 1.0], axis=1)
        # map unit right triangle coords to equilateral: (x, y*sqrt3/2) with y<=1-x
        y = pts[:, 1] / sqrt3
        pts = np.stack([pts[:, 0], y * sqrt3], axis=1)
        seeds.append(pts)

    deadline = time.time() + 160.0
    best_pts = None
    best_val = -1.0

    for seed_pts in seeds:
        pts = np.array([_project_into_triangle(p) for p in seed_pts])
        vals = cross_all(pts)          # cached area of all 165 triangles
        cur_val = vals.min()
        step = 0.05
        stall = 0
        while time.time() < deadline and step > 1e-6:
            idx = int(rng.integers(n))
            old = pts[idx].copy()
            new = _project_into_triangle(old + rng.normal(0.0, step, 2))
            if np.allclose(new, old):
                continue
            rows = rows_of[idx]        # only triangles touching idx change
            saved = vals[rows].copy()
            pts[idx] = new
            t = _TRIP[rows]
            a = pts[t[:, 0]]
            b = pts[t[:, 1]]
            c = pts[t[:, 2]]
            vals[rows] = np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                                - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])) / 2.0
            new_min = vals.min()
            if new_min >= cur_val - 1e-15:
                stall = 0 if new_min > cur_val + 1e-15 else stall + 1
                cur_val = new_min
            else:
                pts[idx] = old
                vals[rows] = saved
                stall += 1
            if stall > 4000:
                step *= 0.7
                stall = 0
        if cur_val > best_val:
            best_val = cur_val
            best_pts = pts.copy()
        if time.time() > deadline:
            break

    # final greedy polish from best (incremental, axis + diagonal moves)
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
        if not improved:
            step *= 0.5

    return pts


# EVOLVE-BLOCK-END
