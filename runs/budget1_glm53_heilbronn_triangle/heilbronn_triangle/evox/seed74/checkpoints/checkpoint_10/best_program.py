# EVOLVE-BLOCK-START
import numpy as np
import time


def _min_area_idx(points: np.ndarray):
    """Smallest triangle area and the indices (i,j,k) achieving it (triangle area normalized to 1)."""
    n = points.shape[0]
    best = np.inf
    bidx = (0, 1, 2)
    for i in range(n - 2):
        p = points[i]
        for j in range(i + 1, n - 1):
            q = points[j] - p
            r = points[j + 1:] - p
            areas = 0.5 * np.abs(q[0] * r[:, 1] - q[1] * r[:, 0])
            k = int(np.argmin(areas))
            if areas[k] < best:
                best = float(areas[k])
                bidx = (i, j, j + 1 + k)
    return best, bidx


def _in_triangle(pts: np.ndarray) -> np.ndarray:
    """Keep points inside the unit equilateral triangle (vertices (0,0),(1,0),(.5,sqrt3/2))."""
    x, y = pts[:, 0], pts[:, 1]
    # barycentric constraints: y>=0, y <= sqrt3*x, y <= sqrt3*(1-x)
    s = np.sqrt(3.0)
    y = np.clip(y, 0.0, None)
    # project back: clamp y under the two slanted edges
    ymax = np.minimum(s * x, s * (1.0 - x))
    y = np.minimum(y, np.maximum(ymax, 0.0))
    x = np.clip(x, 0.0, 1.0)
    # re-clamp y (x clip may loosen/ tighten edges)
    ymax = np.minimum(s * x, s * (1.0 - x))
    y = np.minimum(y, np.maximum(ymax, 0.0))
    return np.stack([x, y], axis=1)


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct 11 points in the unit equilateral triangle maximizing the minimum
    triangle area.

    Approach: deterministic multi-start local search with targeted moves.
    At each step the triangle achieving the current minimum area is located
    and its three vertices are perturbed preferentially (since only they can
    raise the minimum); all other points get occasional perturbations too.
    Both Gaussian and axis-aligned moves are tried. A mild annealing schedule
    (occasionally accepting slightly worse configurations) helps escape local
    optima; the best configuration ever seen is tracked and returned.
    Fixed RNG seed for reproducibility.
    """
    n = 11
    rng = np.random.default_rng(12345)
    deadline = time.time() + 200.0  # seconds budget

    s = np.sqrt(3.0) / 2.0
    tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, s]])

    def bary_to_xy(b):
        # b: (n,3) barycentric weights
        return b @ tri

    starts = []

    # Start 1: points on the three edges (subdivided) plus centroid-ish interior
    def edge_points(m):
        pts = []
        for k in range(1, m):
            pts.append([k / m, 0.0])
            pts.append([0.5 + (k / m - 0.5) * 0.5, s * (k / m)])
            pts.append([0.5 - (k / m - 0.5) * 0.5, s * (k / m)])
        return np.array(pts)

    e = edge_points(4)  # 9 edge points
    interior = np.array([[0.25, 0.25], [0.75, 0.25]])
    starts.append(np.vstack([e, interior]))

    # Start 2: uniform random barycentric
    for _ in range(3):
        b = rng.dirichlet(np.ones(3), size=n)
        starts.append(bary_to_xy(b))

    # Start 3: structured rows (like small triangular lattice)
    pts = []
    rows = [(0.0, [0.5]), (0.2, [0.2, 0.5, 0.8]), (0.45, [0.1, 0.35, 0.6, 0.85]),
            (0.65, [0.25, 0.5, 0.75])]
    for y, xs in rows:
        for x in xs:
            pts.append([x, y])
    starts.append(np.array(pts[:n]))

    best_pts, best_val = None, -1.0

    for pts0 in starts:
        pts = _in_triangle(np.array(pts0[:n], dtype=float))
        cur, _ = _min_area_idx(pts)
        run_best_pts, run_best_val = pts.copy(), cur
        scale = 0.05
        temp = 1e-4  # annealing tolerance for accepting near-equal/worse moves
        while scale > 1e-6 and time.time() < deadline:
            improved = False
            _, (bi, bj, bk) = _min_area_idx(pts)
            # prioritize the vertices of the minimal triangle, then all points
            order = list(rng.permutation(n))
            order = [bi, bj, bk] + [i for i in order if i not in (bi, bj, bk)]
            for i in order:
                base = pts[i].copy()
                for trial in range(10):
                    if trial % 2 == 0:
                        delta = rng.normal(0.0, scale, size=2)
                    else:
                        # axis-aligned move
                        delta = np.zeros(2)
                        delta[rng.integers(0, 2)] = rng.normal(0.0, scale * 1.5)
                    cand = pts.copy()
                    cand[i] = _in_triangle((base + delta)[None, :])[0]
                    v, _ = _min_area_idx(cand)
                    if v > cur - temp:
                        # accept improving or near-equal moves; track best ever
                        pts = cand
                        cur = v
                        if v > run_best_val:
                            run_best_val = v
                            run_best_pts = pts.copy()
                        improved = True
                        break
            if not improved:
                scale *= 0.6
                temp *= 0.5
        if run_best_val > best_val:
            best_val = run_best_val
            best_pts = run_best_pts.copy()

    if best_pts is None:
        # fallback: deterministic layout
        best_pts = starts[0]
    return np.asarray(best_pts, dtype=float)


# EVOLVE-BLOCK-END
