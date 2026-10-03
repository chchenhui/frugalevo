# EVOLVE-BLOCK-START
import numpy as np


_IDX_I, _IDX_J, _IDX_K = np.triu_indices(13, k=1)
_TRIPLES = np.array([(i, j, k)
                     for i in range(11)
                     for j in range(i + 1, 12)
                     for k in range(j + 1, 13)])
_TI = _TRIPLES[:, 0]
_TJ = _TRIPLES[:, 1]
_TK = _TRIPLES[:, 2]


def _min_triple_area(points: np.ndarray) -> float:
    """Area of the smallest triangle among all C(13,3) triples, vectorized."""
    ax, ay = points[_TI, 0], points[_TI, 1]
    bx, by = points[_TJ, 0], points[_TJ, 1]
    cx, cy = points[_TK, 0], points[_TK, 1]
    areas = 0.5 * np.abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
    return float(areas.min())


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


def _refine(points: np.ndarray, step: float = 0.25,
            min_step: float = 1e-5) -> np.ndarray:
    """Deterministic coordinate-descent refinement of a configuration."""
    dirs = np.array([[1, 0], [-1, 0], [0, 1], [0, -1],
                     [1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=float)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    points = points.copy()
    best = _score(points)
    while step > min_step:
        improved = False
        for i in range(13):
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


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points maximizing the smallest triangle area
    (normalized by convex hull area). Deterministic multi-start coordinate
    descent over several structured, symmetry-informed seed configurations.
    """
    n = 13
    seeds = []

    # Seed 1: regular 13-gon (all boundary points, no interior crowding).
    ang = 2.0 * np.pi * np.arange(n) / n
    reg = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    seeds.append(reg)

    # Seed 2: regular 12-gon + center.
    ang = 2.0 * np.pi * np.arange(12) / 12.0
    pc = np.zeros((n, 2))
    pc[:12, 0] = np.cos(ang)
    pc[:12, 1] = np.sin(ang)
    seeds.append(pc)

    # Seed 3: 3-fold symmetric configuration: three rotated arcs of 4 points
    # each plus one center-ish point pushed toward the boundary of a sector.
    base = np.array([[1.0, 0.0], [0.8, 0.35], [0.7, 0.7], [0.35, 0.8]])
    p3 = np.zeros((n, 2))
    for s in range(3):
        th = 2.0 * np.pi * s / 3.0
        R = np.array([[np.cos(th), -np.sin(th)],
                      [np.sin(th), np.cos(th)]])
        p3[4 * s:4 * s + 4] = base @ R.T
    p3[12] = [0.0, 0.0]
    seeds.append(p3)

    # Seed 4: slightly perturbed 13-gon (deterministic perturbation).
    rng = np.random.default_rng(seed=7)
    seeds.append(reg + 0.02 * rng.standard_normal((n, 2)))

    best_pts, best_sc = None, -np.inf
    for seed in seeds:
        pts = _refine(seed)
        sc = _score(pts)
        if sc > best_sc:
            best_sc, best_pts = sc, pts
    return best_pts


# EVOLVE-BLOCK-END