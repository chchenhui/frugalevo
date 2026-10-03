# EVOLVE-BLOCK-START
import numpy as np
import itertools


def _min_triangle_area(points: np.ndarray) -> float:
    """Exact smallest area among all C(13,3)=286 triangles (vectorized)."""
    n = points.shape[0]
    i, j, k = np.array(list(itertools.combinations(range(n), 3))).T
    ax, ay = points[i, 0], points[i, 1]
    bx, by = points[j, 0], points[j, 1]
    cx, cy = points[k, 0], points[k, 1]
    areas = 0.5 * np.abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay))
    return float(areas.min())


def heilbronn_convex13() -> np.ndarray:
    """
    Approach: start from a regular 13-gon inscribed in a unit-area circle
    (a strong, symmetric baseline), then apply a deterministic seeded
    local search (random perturbations with shrinking step size,
    accept-if-better) that directly maximizes the exact minimum triangle
    area over all 286 triples. Points stay inside the unit-area disk,
    which is convex, so all points remain in a convex region.

    Returns:
        points: np.ndarray of shape (13,2), deterministic.
    """
    n = 13
    # Circle with area 1 -> R = 1/sqrt(pi)
    R = 1.0 / np.sqrt(np.pi)
    angles = 2.0 * np.pi * np.arange(n) / n
    points = np.column_stack([R * np.cos(angles), R * np.sin(angles)])

    best = _min_triangle_area(points)
    rng = np.random.default_rng(seed=42)
    radius = 0.02 * R
    # Deterministic local search: fixed iteration budget, shrinking steps
    for round_idx in range(6):
        improved = True
        while improved:
            improved = False
            for trials in range(200):
                cand = points + rng.normal(0.0, radius, (n, 2))
                # keep points inside the unit-area disk (convex region)
                norms = np.hypot(cand[:, 0], cand[:, 1])
                scale = np.maximum(norms, 1e-12)
                cand = cand * np.minimum(1.0, R / scale)[:, None]
                a = _min_triangle_area(cand)
                if a > best + 1e-12:
                    points, best = cand, a
                    improved = True
        radius *= 0.5

    return points


# EVOLVE-BLOCK-END
