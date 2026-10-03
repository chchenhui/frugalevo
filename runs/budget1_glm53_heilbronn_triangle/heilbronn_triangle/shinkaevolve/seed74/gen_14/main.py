# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations

_H = np.sqrt(3.0) / 2.0
_VERTS = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, _H]])
_TRI = np.array(list(combinations(range(11), 3)), dtype=int)  # (165,3)


def _clip(pts: np.ndarray) -> np.ndarray:
    """Vectorized clip of all points into the equilateral triangle via barycentric clamping."""
    l2 = pts[:, 1] / _H
    l1 = pts[:, 0] - 0.5 * l2
    l0 = 1.0 - l1 - l2
    bc = np.stack([l0, l1, l2], axis=1)
    bc = np.clip(bc, 0.0, None)
    bc = bc / bc.sum(axis=1, keepdims=True)
    return bc @ _VERTS


def _areas(pts: np.ndarray) -> np.ndarray:
    """All 165 triangle areas at once, fully vectorized."""
    a = pts[_TRI[:, 0]]
    b = pts[_TRI[:, 1]]
    c = pts[_TRI[:, 2]]
    return 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                         - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))


def _min_area(pts: np.ndarray) -> float:
    return float(_areas(pts).min())


def _random_in_triangle(rng, n=11):
    """Uniform random points in the triangle via barycentric sampling."""
    u = rng.random((n, 2))
    su = np.sqrt(u[:, 0])
    bc = np.empty((n, 3))
    bc[:, 0] = 1.0 - su
    bc[:, 1] = su * (1.0 - u[:, 1])
    bc[:, 2] = su * u[:, 1]
    return bc @ _VERTS


def _refine(pts, rng, steps=(0.05, 0.02, 0.01, 0.004, 0.002, 8e-4, 3e-4, 1e-4)):
    """Greedy hill climbing with decaying steps; moves vertices of near-worst triangles."""
    pts = _clip(pts)
    cur = _min_area(pts)
    n = len(pts)
    for h in steps:
        improved = True
        it = 0
        while improved and it < 300:
            improved = False
            it += 1
            ar = _areas(pts)
            thr = ar.min() * 1.05
            bad = _TRI[ar <= thr]
            cand = pts.copy()
            movers = np.unique(bad.ravel())
            cand[movers] += rng.normal(0.0, h, (len(movers), 2))
            cand = _clip(cand)
            v = _min_area(cand)
            if v > cur + 1e-15:
                pts, cur, improved = cand, v, True
                continue
            # fallback: random single-point move
            cand = pts.copy()
            k = rng.integers(0, n)
            cand[k] += rng.normal(0.0, h, 2)
            cand = _clip(cand)
            v = _min_area(cand)
            if v > cur + 1e-15:
                pts, cur, improved = cand, v, True
    return pts, cur

def heilbronn_triangle11() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 11.

    Deterministic multi-start greedy local search with a fully vectorized objective.

    Returns:
        points: np.ndarray of shape (11,2) with the x,y coordinates of the points.
    """
    n = 11
    s = np.sqrt(3.0)
    rng = np.random.default_rng(12345)

    seeds = [np.array([
        [0.0, 0.0], [1.0, 0.0], [0.5, s / 2.0],
        [0.5, 0.0], [0.25, s / 8.0], [0.75, s / 8.0],
        [0.5, s / 4.0], [0.125, s / 16.0], [0.875, s / 16.0],
        [0.375, 3.0 * s / 16.0], [0.625, 3.0 * s / 16.0],
    ])]

    best, best_val = None, -np.inf
    try:
        # Multi-start: deterministic seed + 30 random barycentric restarts
        for t in range(31):
            start = seeds[0] if t == 0 else _random_in_triangle(rng, n)
            pts, val = _refine(start.copy(), rng)
            if val > best_val:
                best, best_val = pts.copy(), val

        # Final polish: coordinate descent on worst-triangle vertices
        P = best.copy()
        for _ in range(200):
            ar = _areas(P)
            worst = _TRI[np.argmin(ar)]
            improved = False
            for k in worst:
                for ax in range(2):
                    for sgn in (1.0, -1.0):
                        for h in (2e-3, 5e-4, 1e-4):
                            cand = P.copy()
                            cand[k, ax] += sgn * h
                            cand = _clip(cand)
                            v = _min_area(cand)
                            if v > best_val + 1e-15:
                                P, best_val, improved = cand, v, True
                                break
                        if improved:
                            break
                if improved:
                    break
            if not improved:
                break
        best = P
    except Exception:
        if best is None:
            best = seeds[0]
        best = _clip(best)

    return best

# EVOLVE-BLOCK-END