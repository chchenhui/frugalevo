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

    # ---- TEMPLATE ENUMERATION ----
    # Breakthrough idea: enumerate combinatorial templates (k points per edge
    # with staggered parametrized positions + interior points on medians),
    # then refine each with a short annealing burst. This searches
    # low-dimensional structured families (5-10 effective parameters)
    # that the unconstrained 22-D annealer misses, while allowing
    # asymmetric edge distributions (e.g. 4+4+3, 4+3+2, ...).

    def edge_points(cnt, org, e0, off):
        """cnt points on edge org->org+e0 at staggered fractions."""
        return [org + ((i + 1.0 + off) / (cnt + 1.0 + 2.0 * off)) * e0
                for i in range(cnt)]

    def median_point(frac, vertex):
        """Interior point on the median from `vertex` toward the centroid."""
        g = np.array([0.5, sqrt3 / 3.0])
        return vertex + frac * (g - vertex)

    templates = []
    # edge-count splits (a on AB, b on AC, c on BC), vertices always included
    splits = []
    for a in range(1, 5):
        for b in range(1, 5):
            for c in range(1, 5):
                if 3 + a + b + c <= 11 and a + b + c >= 5:
                    splits.append((a, b, c))
    for (a, b, c) in splits:
        n_int = 11 - 3 - a - b - c
        for off in (0.0, 0.35):
            base = [A.copy(), B.copy(), C.copy()]
            base += edge_points(a, A, B - A, off)
            base += edge_points(b, A, C - A, off)
            base += edge_points(c, B, C - B, off)
            if n_int == 0:
                templates.append(np.array(base))
            elif n_int == 1:
                for fr in (0.3, 0.5, 0.7):
                    templates.append(np.array(base + [median_point(fr, A)]))
            elif n_int == 2:
                for fr1 in (0.3, 0.55):
                    for fr2 in (0.35, 0.6):
                        templates.append(np.array(
                            base + [median_point(fr1, A),
                                    median_point(fr2, B)]))
            elif n_int == 3:
                for fr in ((0.3, 0.3, 0.3), (0.5, 0.5, 0.5),
                           (0.35, 0.5, 0.65)):
                    templates.append(np.array(
                        base + [median_point(fr[0], A),
                                median_point(fr[1], B),
                                median_point(fr[2], C)]))
            elif n_int == 4:
                templates.append(np.array(
                    base + [median_point(0.3, A), median_point(0.6, A),
                            median_point(0.4, B), median_point(0.4, C)]))
            elif n_int == 5:
                templates.append(np.array(
                    base + [median_point(0.25, A), median_point(0.55, A),
                            median_point(0.35, B), median_point(0.6, B),
                            median_point(0.45, C)]))

    t0 = time.time()
    deadline = t0 + 150.0
    best_pts = None
    best_val = -1.0
    candidates = []

    def sa_run(pts0, t_end, step0=0.03, temp0=1e-3):
        """Short annealing burst from a template; returns (pts, min_area)."""
        pts = np.array([_project_into_triangle(p) for p in pts0])
        vals = cross_all(pts)
        cur_val = vals.min()
        step = step0
        temp = temp0
        stall = 0
        while time.time() < t_end and step > 1e-6:
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
            if stall > 3000:
                step *= 0.7
                temp *= 0.6
                stall = 0
        return pts, cur_val

    # allocate a fixed time slice per template, deterministic
    n_tpl = max(1, len(templates))
    tpl_budget = min(2.5, max(0.5, 95.0 / n_tpl))
    for tpl in templates:
        if time.time() > t0 + 100.0:
            break
        pts, v = sa_run(tpl, min(time.time() + tpl_budget, t0 + 100.0))
        candidates.append((v, pts))
        if v > best_val:
            best_val = v
            best_pts = pts.copy()

    # second, longer pass on the top templates
    candidates.sort(key=lambda cv: -cv[0])
    for v0, sp in candidates[:6]:
        if time.time() > t0 + 125.0:
            break
        pts, v = sa_run(sp, min(time.time() + 4.0, t0 + 125.0), step0=0.01)
        if v > best_val:
            best_val = v
            best_pts = pts.copy()

    if best_pts is None:
        # graceful fallback: simple structured seed
        s = [A.copy(), B.copy(), C.copy()]
        for t in (1 / 3, 2 / 3):
            s.append(A + t * (B - A))
            s.append(A + t * (C - A))
            s.append(B + t * (C - B))
        for t in (1 / 4, 1 / 2):
            bary = np.array([t, (1 - t) / 2, (1 - t) / 2])
            s.append(bary[0] * A + bary[1] * B + bary[2] * C)
        best_pts = np.array(s)

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
