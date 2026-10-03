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


def _grad_ascent(points: np.ndarray, iters: int = 60) -> np.ndarray:
    """Finite-difference ascent on a soft-min objective (LogSumExp of the
    smallest few triangle areas), normalized by hull area."""
    n = len(points)
    # Precompute triple index list once
    trips = np.array([(i, j, k) for i in range(n - 2)
                       for j in range(i + 1, n - 1)
                       for k in range(j + 1, n)], dtype=int)
    pts = points.copy()
    alpha = 300.0
    h = 1e-5
    step = 0.02
    for _ in range(iters):
        ha = _hull_area(pts)
        if ha <= 1e-12:
            break
        a = pts[trips[:, 0]]
        b = pts[trips[:, 1]]
        c = pts[trips[:, 2]]
        areas = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) -
                             (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])) / ha
        soft = -np.log(np.sum(np.exp(-alpha * areas))) / alpha
        grad = np.zeros_like(pts)
        w = np.exp(-alpha * areas)
        w /= w.sum()
        for t in range(len(trips)):
            i, j, k = trips[t]
            ai, bi, ci = pts[i], pts[j], pts[k]
            cr = (bi[0] - ai[0]) * (ci[1] - ai[1]) - (bi[1] - ai[1]) * (ci[0] - ai[0])
            sgn = 1.0 if cr >= 0 else -1.0
            g_i = 0.5 * sgn * np.array([-(ci[1] - bi[1]), (ci[0] - bi[0])])
            g_j = 0.5 * sgn * np.array([(ci[1] - ai[1]), -(ci[0] - ai[0])])
            g_k = 0.5 * sgn * np.array([(bi[1] - ai[1]), -(bi[0] - ai[0])])
            grad[i] += w[t] * g_i
            grad[j] += w[t] * g_j
            grad[k] += w[t] * g_k
        grad /= ha
        pts = pts + step * grad
        step *= 0.97
    return pts


def _hill_climb(points: np.ndarray, max_iters: int = 4) -> np.ndarray:
    dirs = np.array([[1, 0], [-1, 0], [0, 1], [0, -1],
                     [1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=float)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    pts = points.copy()
    step = 0.02
    best = _score(pts)
    for _ in range(max_iters):
        improved = False
        for i in range(len(pts)):
            for d in dirs:
                for s in (step, 0.5 * step):
                    trial = pts.copy()
                    trial[i] += s * d
                    sc = _score(trial)
                    if sc > best + 1e-12:
                        pts = trial
                        best = sc
                        improved = True
        if not improved:
            step *= 0.5
            if step < 1e-5:
                break
    return pts


def _expand(base: np.ndarray, m: int) -> np.ndarray:
    """Expand a base configuration by m-fold rotations about the origin."""
    out = [base]
    for r in range(1, m):
        th = 2.0 * np.pi * r / m
        R = np.array([[np.cos(th), -np.sin(th)],
                      [np.sin(th), np.cos(th)]])
        out.append(base @ R.T)
    return np.vstack(out)


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points maximizing the smallest triangle area
    (normalized by convex hull area). Deterministic multi-seed search:
    symmetric seeds (3-fold, 4-fold rotation expansions, 13-gon, circle+center)
    are each refined by gradient ascent + hill climbing; best true score wins.
    """
    candidates = []

    # Seed A: 12-gon + center (classic baseline)
    ang = 2.0 * np.pi * np.arange(12) / 12.0
    pa = np.zeros((13, 2))
    pa[:12, 0] = np.cos(ang)
    pa[:12, 1] = np.sin(ang)
    candidates.append(pa)

    # Seed B: 3-fold expansion of a 4-point base + center
    base = np.array([[1.0, 0.0], [0.35, 0.55], [-0.45, 0.75], [0.0, -0.95]])
    pb = np.vstack([_expand(base, 3), [[0.0, 0.0]]])
    candidates.append(pb)

    # Seed C: 4-fold expansion of a 3-point base + center
    base = np.array([[1.0, 0.0], [0.45, 0.45], [0.6, -0.5]])
    pc = np.vstack([_expand(base, 4), [[0.0, 0.0]]])
    candidates.append(pc)

    # Seed D: irregular 13-gon on a circle (mild golden-angle perturbation)
    angd = 2.0 * np.pi * (np.arange(13) + 0.618 * (np.arange(13) % 3)) / 13.0
    pd = np.column_stack([np.cos(angd), np.sin(angd)])
    candidates.append(pd)

    # Seed E: 13-gon, one vertex pushed to center (12 ring + 1 interior)
    ange = 2.0 * np.pi * np.arange(12) / 12.0
    pe = np.zeros((13, 2))
    pe[:12, 0] = 0.9 * np.cos(ange)
    pe[:12, 1] = 0.9 * np.sin(ange)
    pe[12] = [0.15, 0.1]
    candidates.append(pe)

    best_pts = None
    best_sc = -1.0
    for cand in candidates:
        p = _grad_ascent(cand, iters=40)
        p = _hill_climb(p, max_iters=3)
        sc = _score(p)
        if sc > best_sc:
            best_sc = sc
            best_pts = p
    return best_pts


# EVOLVE-BLOCK-END