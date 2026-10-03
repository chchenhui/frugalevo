# EVOLVE-BLOCK-START
import numpy as np


def _min_triangle_area(pts: np.ndarray) -> float:
    """Vectorized minimum area over all C(n,3) triplets."""
    n = pts.shape[0]
    i, j, k = np.triple = np.array(
        [(a, b, c) for a in range(n) for b in range(a + 1, n) for c in range(b + 1, n)]
    ).T
    ax, ay = pts[i, 0], pts[i, 1]
    bx, by = pts[j, 0], pts[j, 1]
    cx, cy = pts[k, 0], pts[k, 1]
    areas = 0.5 * np.abs(
        (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)
    )
    return float(areas.min())


def _inside(pts: np.ndarray) -> np.ndarray:
    """Boolean mask: points inside/on the equilateral triangle (0,0),(1,0),(0.5,√3/2)."""
    h = np.sqrt(3.0) / 2.0
    x, y = pts[:, 0], pts[:, 1]
    return (
        (y >= -1e-12)
        & (y <= np.sqrt(3.0) * x + 1e-12)
        & (y <= -np.sqrt(3.0) * (x - 1.0) + 1e-12)
    )


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct an arrangement of 11 points inside/on the equilateral triangle
    (0,0),(1,0),(0.5,sqrt(3)/2) maximizing the minimum triangle area.

    Strategy: symmetric lattice-like seed (vertices, edge midpoints, coarse
    interior grid) refined by a deterministic seeded hill climb.

    Returns:
        points: np.ndarray of shape (11,2).
    """
    n = 11
    h = np.sqrt(3.0) / 2.0

    # --- Stage 1: hand-designed symmetric seed ---
    points = np.array(
        [
            [0.0, 0.0],          # vertex
            [1.0, 0.0],          # vertex
            [0.5, h],            # vertex
            [0.5, 0.0],          # bottom edge midpoint
            [0.25, h / 2.0],     # left edge midpoint
            [0.75, h / 2.0],     # right edge midpoint
            [0.5, h / 3.0],      # interior, lower center
            [0.5, 2.0 * h / 3.0],# interior, upper center
            [0.25, h / 6.0],     # interior, lower left
            [0.75, h / 6.0],     # interior, lower right
            [0.5, h / 6.0],      # interior, lower middle
        ]
    )

    best_area = _min_triangle_area(points)

    # --- Stage 2: deterministic seeded hill climb ---
    rng = np.random.default_rng(12345)
    scale = 0.02
    for round_idx in range(6):
        for i in range(n):
            improved = True
            local_tries = 0
            while improved and local_tries < 40:
                improved = False
                local_tries += 1
                cands = points[i] + rng.normal(0.0, scale, size=(12, 2))
                # clamp into bounding box first
                cands[:, 0] = np.clip(cands[:, 0], 0.0, 1.0)
                cands[:, 1] = np.clip(cands[:, 1], 0.0, h)
                best_cand = None
                for c in cands:
                    trial = points.copy()
                    trial[i] = c
                    if not _inside(trial[[i]])[0]:
                        continue
                    a = _min_triangle_area(trial)
                    if a > best_area + 1e-15:
                        best_area = a
                        best_cand = c
                        improved = True
                if best_cand is not None:
                    points[i] = best_cand
        scale *= 0.6  # anneal perturbation size

    # Safety: final check (should always hold)
    if not np.all(_inside(points)):
        # fall back to safe seed (should never trigger)
        points = np.array(
            [
                [0.0, 0.0], [1.0, 0.0], [0.5, h], [0.5, 0.0],
                [0.25, h / 2.0], [0.75, h / 2.0], [0.5, h / 3.0],
                [0.5, 2.0 * h / 3.0], [0.25, h / 6.0],
                [0.75, h / 6.0], [0.5, h / 6.0],
            ]
        )

    return np.asarray(points, dtype=float)


# EVOLVE-BLOCK-END