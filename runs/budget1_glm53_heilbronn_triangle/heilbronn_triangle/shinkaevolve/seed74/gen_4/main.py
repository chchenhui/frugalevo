# EVOLVE-BLOCK-START
import numpy as np

SQRT3_2 = np.sqrt(3.0) / 2.0


def _triangle_min_area(points: np.ndarray) -> float:
    """Vectorized minimum |area| over all C(n,3) triplets."""
    from itertools import combinations
    idx = np.array(list(combinations(range(len(points)), 3)))
    p = points[idx]  # (T,3,2)
    area2 = np.abs(
        (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
        - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1])
    )
    return 0.5 * area2.min()


def _project_into_triangle(points: np.ndarray) -> np.ndarray:
    """Project points back into the unit equilateral triangle."""
    p = points.copy()
    # triangle vertices
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, SQRT3_2])
    for i in range(len(p)):
        # simple clipping: clamp y>=0, and inside the two slanted edges
        x, y = p[i]
        if y < 0.0:
            y = 0.0
        # left edge: y <= sqrt3*x  (through (0,0) and (0.5,h))
        if y > SQRT3_2 * 2.0 * x:
            # project onto line y = sqrt3 * x
            # line: -sqrt3*x + y = 0
            d = (-SQRT3_2 * 2.0 * x + y) / (1.0 + (SQRT3_2 * 2.0) ** 2)
            x += SQRT3_2 * 2.0 * d
            y -= d
        # right edge: y <= -sqrt3*(x-1)
        if y > -SQRT3_2 * 2.0 * (x - 1.0):
            d = (SQRT3_2 * 2.0 * (x - 1.0) + y) / (1.0 + (SQRT3_2 * 2.0) ** 2)
            x -= SQRT3_2 * 2.0 * d
            y -= d
        # top: y <= h
        if y > SQRT3_2:
            y = SQRT3_2
            x = min(max(x, 0.0), 1.0)
        p[i] = (x, y)
    return p


def _initial_config() -> np.ndarray:
    """Structured initial arrangement: 3 vertices, 3 edge midpoints-region points,
    5 interior points on a perturbed lattice."""
    pts = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, SQRT3_2],
        [0.5, 0.0],
        [0.25, SQRT3_2 / 2.0],
        [0.75, SQRT3_2 / 2.0],
        [0.1, 0.09],
        [0.9, 0.09],
        [0.5, SQRT3_2 * 0.32],
        [0.3, SQRT3_2 * 0.72],
        [0.7, SQRT3_2 * 0.72],
    ])
    return pts


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct an arrangement of 11 points inside/on the unit equilateral triangle
    maximizing the minimum triangle area (Heilbronn problem, n=11).

    Returns:
        points: np.ndarray of shape (11,2).
    """
    try:
        base = _initial_config()
        best_pts = base
        best_val = _triangle_min_area(base)

        rng = np.random.default_rng(12345)
        pts = base.copy()
        # deterministic hill-climbing with shrinking steps
        step = 0.02
        cur_val = best_val
        while step > 1e-4:
            improved = False
            for i in range(len(pts)):
                for dx, dy in ((step, 0), (-step, 0), (0, step), (0, -step)):
                    cand = pts.copy()
                    cand[i, 0] += dx
                    cand[i, 1] += dy
                    cand = _project_into_triangle(cand)
                    v = _triangle_min_area(cand)
                    if v > cur_val + 1e-12:
                        pts = cand
                        cur_val = v
                        improved = True
                        if v > best_val:
                            best_val = v
                            best_pts = cand.copy()
                        break
            if not improved:
                step *= 0.5
        points = best_pts
        if points.shape != (11, 2) or not np.all(np.isfinite(points)):
            raise ValueError("optimization produced invalid configuration")
    except Exception:
        points = _initial_config()
    return np.asarray(points)


# EVOLVE-BLOCK-END