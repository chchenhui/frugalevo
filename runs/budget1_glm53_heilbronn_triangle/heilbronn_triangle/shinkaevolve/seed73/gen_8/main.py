# EVOLVE-BLOCK-START
import numpy as np


def _triangle_area(a, b, c):
    return 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _clamp_to_triangle(pts):
    """Project points back inside the unit equilateral triangle."""
    s3 = np.sqrt(3.0)
    x = np.clip(pts[:, 0], 0.0, 1.0)
    y = pts[:, 1]
    # constraints: y >= 0, y <= sqrt(3)*x, y <= sqrt(3)*(1-x)
    y = np.clip(y, 0.0, s3 * np.minimum(x, 1.0 - x))
    return np.stack([x, y], axis=1)


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct an arrangement of 11 points on or inside the unit equilateral triangle
    to maximize the smallest triangle area (Heilbronn problem, n = 11).

    Deterministic: hardcoded initial layout followed by a fixed-seed greedy local
    search that pushes apart the vertices of the currently-smallest triangle.

    Returns:
        points: np.ndarray of shape (11,2).
    """
    n = 11
    s3 = np.sqrt(3.0) / 2.0

    # Non-degenerate starts: quasi-uniform interior layouts (no collinear triples
    # on a single line), jittered deterministically to seed diversity.
    base = np.array([
        [0.02, 0.02], [0.98, 0.02], [0.50, 0.86],
        [0.26, 0.02], [0.74, 0.02], [0.13, 0.24], [0.87, 0.24],
        [0.38, 0.26], [0.62, 0.26], [0.25, 0.46], [0.75, 0.46],
    ])
    starts = [base.copy()]
    for s in range(3):
        rs = np.random.default_rng(1234 + s)
        starts.append(_clamp_to_triangle(base + rs.normal(0.0, 0.05, base.shape)))
=======

    # Precompute all point triples and pairwise-index arrays for vectorized area computation.
    tri = np.array([(i, j, k) for i in range(n) for j in range(i + 1, n)
                    for k in range(j + 1, n)])  # (165, 3)
    I = tri[:, 0]
    J = tri[:, 1]
    K = tri[:, 2]

    def min_area(pts):
        """Vectorized: minimum triangle area over all triples."""
        ax, ay = pts[I, 0], pts[I, 1]
        area2 = (pts[J, 0] - ax) * (pts[K, 1] - ay) - (pts[J, 1] - ay) * (pts[K, 0] - ax)
        return 0.5 * np.min(np.abs(area2))

    best_points = None
    best_area = -1.0

    # Deterministic multi-start annealing with targeted moves:
    # perturb a vertex of the currently-smallest triangle.
    n_iters = 4000
    for restart, start_pts in enumerate(starts):
        rng = np.random.default_rng(42 + restart)
        points = start_pts.copy()
        cur_area = min_area(points)
        if cur_area > best_area:
            best_area = cur_area
            best_points = points.copy()
        scale = 0.08
        for it in range(n_iters):
            if best_area >= 0.0368:
                break
            # identify the smallest triangle to target a vertex of it
            ax, ay = points[I, 0], points[I, 1]
            area2 = np.abs((points[J, 0] - ax) * (points[K, 1] - ay)
                           - (points[J, 1] - ay) * (points[K, 0] - ax))
            t_idx = int(np.argmin(area2))
            if rng.random() < 0.7:
                p_idx = int(tri[t_idx][rng.integers(3)])
            else:
                p_idx = int(rng.integers(n))
            cand = points.copy()
            cand[p_idx] += rng.normal(0.0, scale, 2)
            cand = _clamp_to_triangle(cand)
            cand_area = min_area(cand)

            temp = 0.004 * (1.0 - it / n_iters) + 1e-6
            if cand_area >= cur_area - 1e-15 or rng.random() < np.exp((cand_area - cur_area) / temp):
                points = cand
                cur_area = cand_area
                if cand_area > best_area:
                    best_area = cand_area
                    best_points = cand.copy()
            # geometric cooling
            scale *= 0.9997
        if best_area >= 0.0368:
            break

    # Final polish: small-step greedy refinement from the best found layout.
    rng = np.random.default_rng(7)
    points = best_points.copy()
    cur_area = min_area(points)
    for it in range(1500):
        cand = points.copy()
        p_idx = int(rng.integers(n))
        cand[p_idx] += rng.normal(0.0, 0.005, 2)
        cand = _clamp_to_triangle(cand)
        cand_area = min_area(cand)
        if cand_area > cur_area:
            points = cand
            cur_area = cand_area
            if cand_area > best_area:
                best_area = cand_area
                best_points = cand.copy()

    return best_points


# EVOLVE-BLOCK-END