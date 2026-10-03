# EVOLVE-BLOCK-START
import numpy as np


def _min_triple_area(points: np.ndarray) -> float:
    """Area of the smallest triangle among all C(13,3) triples, via cross products."""
    n = len(points)
    best = np.inf
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            v1x = points[j, 0] - points[i, 0]
            v1y = points[j, 1] - points[i, 1]
            dx = points[j + 1:, 0] - points[i, 0]
            dy = points[j + 1:, 1] - points[i, 1]
            areas = 0.5 * np.abs(v1x * dy - v1y * dx)
            m = areas.min()
            if m < best:
                best = m
    return float(best)


def _hull_area(points: np.ndarray) -> float:
    """Area of convex hull via the shoelace formula on hull vertices."""
    pts = points[np.lexsort((points[:, 1], points[:, 0]))]
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = np.array(lower[:-1] + upper[:-1])
    x, y = hull[:, 0], hull[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _score(points: np.ndarray) -> float:
    ha = _hull_area(points)
    if ha <= 1e-12:
        return 0.0
    return _min_triple_area(points) / ha


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points maximizing the smallest triangle area
    (normalized by convex hull area). Deterministic coordinate-descent
    refinement starting from a symmetric circle+center configuration.
    """
    n = 13
    # Initialization: 12 points on a circle + center (symmetric, no
    # near-degenerate triples), then refine.
    ang = 2.0 * np.pi * np.arange(12) / 12.0
    points = np.zeros((n, 2))
    points[:12, 0] = np.cos(ang)
    points[:12, 1] = np.sin(ang)
    points[12] = 0.0  # center point

    dirs = np.array([[1, 0], [-1, 0], [0, 1], [0, -1],
                     [1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=float)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    step = 0.25
    best = _score(points)
    while step > 1e-5:
        improved = False
        for i in range(n):
            for d in dirs:
                for s in (step, 0.5 * step):
                    trial = points.copy()
                    trial[i] += s * d
                    sc = _score(trial)
                    if sc > best + 1e-12:
                        points = trial
                        best = sc
                        improved = True
        if not improved:
            step *= 0.5
    return points


# EVOLVE-BLOCK-END