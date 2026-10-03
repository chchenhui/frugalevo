# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations


def _tri_areas(P: np.ndarray) -> np.ndarray:
    idx = np.array(list(combinations(range(len(P)), 3)))
    A = P[idx[:, 0]]
    B = P[idx[:, 1]]
    C = P[idx[:, 2]]
    return 0.5 * np.abs(
        (B[:, 0] - A[:, 0]) * (C[:, 1] - A[:, 1])
        - (B[:, 1] - A[:, 1]) * (C[:, 0] - A[:, 0])
    )


def _hull_area(P: np.ndarray) -> float:
    pts = P[np.lexsort((P[:, 1], P[:, 0]))]
    if len(pts) < 3:
        return 1e-12

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return 1e-12
    s = 0.0
    for i in range(len(hull)):
        x1, y1 = hull[i]
        x2, y2 = hull[(i + 1) % len(hull)]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def heilbronn_convex13() -> np.ndarray:
    n = 13
    rng = np.random.default_rng(seed=42)

    th = 2.0 * np.pi / 3.0
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s], [s, c]])       # rotate by +120 deg
    R2 = R @ R

    def expand(B: np.ndarray) -> np.ndarray:
        # 4 base points -> 12 points by 3-fold rotation, plus center at origin
        return np.vstack([B, B @ R.T, B @ R2.T, [[0.0, 0.0]]])

    def score(P: np.ndarray) -> float:
        a = _tri_areas(P)
        h = _hull_area(P)
        if h <= 1e-12:
            return 0.0
        # smooth-ish objective: blend min with mean of smallest few
        k = np.partition(a, 4)[:5]
        return (0.6 * a.min() + 0.4 * k.mean()) / h

    def true_score(P: np.ndarray) -> float:
        a = _tri_areas(P)
        h = _hull_area(P)
        return a.min() / h if h > 1e-12 else 0.0

    def hill_climb(P: np.ndarray, steps_max: int = 24) -> np.ndarray:
        step = 0.05
        best = score(P)
        for _ in range(steps_max):
            improved = False
            for i in range(len(P)):
                for j in range(2):
                    for d in (+1, -1):
                        Q = P.copy()
                        Q[i, j] += d * step
                        sc = score(Q)
                        if sc > best + 1e-12:
                            best = sc
                            P = Q
                            improved = True
            if not improved:
                step *= 0.5
                if step < 2e-4:
                    break
        return P

    # ---- Build seeds for the 4 base points (sector-free, rotation handles symmetry)
    seeds = []
    ang = np.pi / 2.0
    # vertex ring + inner rings, classic triangular-lattice-like starts
    for r4 in ([1.0, 0.72, 0.45, 0.2], [1.0, 0.8, 0.55, 0.3],
               [1.0, 0.85, 0.6, 0.35], [1.0, 0.75, 0.5, 0.25]):
        B = []
        for k, r in enumerate(r4):
            a = ang - k * np.pi / 9.0
            B.append([r * np.cos(a), r * np.sin(a)])
        seeds.append(np.array(B))
    # spread along one edge direction
    for spread in (0.9, 0.6, 0.3):
        B = np.array([[spread, 0.0],
                      [spread * 0.5, spread * 0.85],
                      [0.0, spread],
                      [spread * 0.4, spread * 0.3]])
        seeds.append(B)
    # random seeds
    for _ in range(24):
        r = rng.uniform(0.1, 1.0, 4)
        a = rng.uniform(0, 2 * np.pi, 4)
        seeds.append(np.column_stack([r * np.cos(a), r * np.sin(a)]))

    best_P, best_ts = None, -1.0
    for B0 in seeds:
        try:
            P = expand(B0)
            P = hill_climb(P)                    # symmetric search (12+1)
            P = hill_climb(P, steps_max=10)      # symmetry-breaking polish
            ts = true_score(P)
            if ts > best_ts:
                best_ts = ts
                best_P = P.copy()
        except Exception:
            continue

    if best_P is None:
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_P = np.column_stack([np.cos(t), np.sin(t)])

    # numerical safety: no duplicates
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(best_P[i] - best_P[j]) < 1e-9:
                best_P[j, 0] += 1e-6 * (j + 1)
    return best_P


# EVOLVE-BLOCK-END