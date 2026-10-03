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

    rng = np.random.default_rng(0)
    cur_area = min_area(points)
    best_points = points.copy()
    best_area = cur_area

    # Fixed-seed randomized local search (simulated-annealing flavored).
    n_iters = 6000
    scale = 0.03
    for it in range(n_iters):
        if best_area >= 0.0375:  # excellent configuration reached, stop early
            break
        cand = points.copy()
        p_idx = rng.integers(n)
        cand[p_idx] += rng.normal(0.0, scale, 2)
        cand = _clamp_to_triangle(cand)
        cand_area = min_area(cand)

        # Accept improving moves always; worsening moves with small, decaying probability.
        temp = 0.002 * (1.0 - it / n_iters)
        if cand_area >= cur_area - 1e-15 or rng.random() < np.exp((cand_area - cur_area) / max(temp, 1e-9)):
            points = cand
            cur_area = cand_area
            if cand_area > best_area:
                best_area = cand_area
                best_points = cand.copy()
        # shrink step scale slowly to refine
        if it % 1000 == 999:
            scale *= 0.6

    return best_points


# EVOLVE-BLOCK-END