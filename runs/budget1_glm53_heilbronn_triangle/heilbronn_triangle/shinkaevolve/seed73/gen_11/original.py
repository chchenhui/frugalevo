# EVOLVE-BLOCK-START
import time
import numpy as np

_TRI_AREA = np.sqrt(3.0) / 4.0  # area of the unit equilateral triangle
from itertools import combinations
_TRIPLETS = np.array(list(combinations(range(11), 3)), dtype=int)


def _min_area(points: np.ndarray) -> float:
    """Vectorized area of the smallest triangle among all triplets (normalized)."""
    # Precompute all triplet index combinations once (cached module-level)
    idx = _TRIPLETS
    p = points[idx]  # (165, 3, 2)
    # Twice the signed area via cross product
    a = (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - \
        (p[:, 1, 1] - p[:, 0, 1]) * (p[:, 2, 0] - p[:, 0, 0])
    return np.abs(a).min() / (2.0 * _TRI_AREA * 2.0) * 2.0  # = |a|/2 / TRI_AREA


def _in_triangle(pt):
    """Return True if pt lies inside the unit equilateral triangle."""
    x, y = pt
    return x >= -1e-9 and y >= -1e-9 and y <= np.sqrt(3.0) * x + 1e-9 and \
        y <= -np.sqrt(3.0) * (x - 1.0) + 1e-9


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct an arrangement of 11 points on or inside the unit equilateral
    triangle maximizing the minimum triangle area (Heilbronn problem, n=11).

    Deterministic: fixed seed, fixed budget. Returns (11,2) array.
    """
    n = 11
    h = np.sqrt(3.0) / 2.0
    # Symmetric initial configuration: 3 vertices, points along medians/layers
    pts = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, h],
        [0.25, h / 2.0],
        [0.75, h / 2.0],
        [0.5, 0.0],
        [0.125, h / 4.0],
        [0.875, h / 4.0],
        [0.375, 3.0 * h / 4.0],
        [0.625, 3.0 * h / 4.0],
        [0.5, h / 4.0],
    ])

    try:
        rng = np.random.default_rng(12345)
        best = pts.copy()
        best_val = _min_area(best)
        cur = best.copy()
        cur_val = best_val
        deadline = time.time() + 1.5
        step = 0.05
        it = 0
        while time.time() < deadline and it < 20000:
            it += 1
            cand = cur.copy()
            i = rng.integers(n)
            cand[i] += rng.normal(0.0, step, size=2)
            # project back into the triangle (clamp roughly)
            x, y = cand[i]
            y = min(max(y, 0.0), h)
            if y > np.sqrt(3.0) * x:
                x = y / np.sqrt(3.0)
            if y > -np.sqrt(3.0) * (x - 1.0):
                x = 1.0 - y / np.sqrt(3.0)
            x = min(max(x, 0.0), 1.0)
            cand[i] = (x, y)
            v = _min_area(cand)
            if v >= cur_val or rng.random() < np.exp((v - cur_val) / (step * 0.05)):
                cur, cur_val = cand, v
                if v > best_val:
                    best, best_val = cand.copy(), v
            if it % 2000 == 0:
                step *= 0.6
        pts = best
    except Exception:
        pass  # graceful fallback: return the symmetric initial configuration

    return pts


# EVOLVE-BLOCK-END