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

    # Initial layout: points distributed on the boundary and along the midline.
    points = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, s3],
        [0.25, 0.0],
        [0.75, 0.0],
        [0.0, 0.5 * s3],
        [1.0, 0.5 * s3],
        [0.5, 0.0],
        [0.125, 0.25 * s3],
        [0.875, 0.25 * s3],
        [0.5, 0.5 * s3],
    ])
    points = _clamp_to_triangle(points)

    rng = np.random.default_rng(0)
    idx_triples = [(i, j, k) for i in range(n) for j in range(i + 1, n) for k in range(j + 1, n)]

    def min_area_and_idx(pts):
        best = np.inf
        best_t = idx_triples[0]
        for t in idx_triples:
            a = _triangle_area(pts[t[0]], pts[t[1]], pts[t[2]])
            if a < best:
                best = a
                best_t = t
        return best, best_t

    cur_area, _ = min_area_and_idx(points)

    # Greedy local search: perturb vertices of the smallest triangle to enlarge it.
    n_iters = 400
    for _ in range(n_iters):
        a, t = min_area_and_idx(points)
        if a >= 0.040:  # good enough, stop early
            break
        improved = False
        for p_idx in t:
            for step in (0.02, 0.005, -0.005, -0.02):
                for dim in (0, 1):
                    cand = points.copy()
                    cand[p_idx, dim] += step
                    cand = _clamp_to_triangle(cand)
                    # small random extra jitter for determinism-seeded exploration
                    cand[p_idx] += rng.normal(0, 0.003, 2)
                    cand = _clamp_to_triangle(cand)
                    cand_a, _ = min_area_and_idx(cand)
                    if cand_a > cur_area + 1e-12:
                        points = cand
                        cur_area = cand_a
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break

    return points


# EVOLVE-BLOCK-END