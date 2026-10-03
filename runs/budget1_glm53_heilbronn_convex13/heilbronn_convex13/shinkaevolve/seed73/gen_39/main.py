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

    def true_score(P: np.ndarray) -> float:
        a = _tri_areas(P)
        h = _hull_area(P)
        return a.min() / h if h > 1e-12 else 0.0

    # Precompute all triangle index triples ONCE for vectorized gradients.
    idx = np.array(list(combinations(range(n), 3)))
    I, J, K = idx[:, 0], idx[:, 1], idx[:, 2]

    def grad_ascent(P0: np.ndarray, iters: int = 300, lr0: float = 0.08) -> np.ndarray:
        """Fast vectorized soft-min gradient ascent on min triangle area.
        Exact analytic gradient of LogSumExp-smoothed min of |signed areas|,
        with annealed sharpness p: 30 -> 600."""
        P = P0.copy()
        for it in range(iters):
            A, B, C = P[I], P[J], P[K]
            S = 0.5 * ((B[:, 0] - A[:, 0]) * (C[:, 1] - A[:, 1])
                       - (B[:, 1] - A[:, 1]) * (C[:, 0] - A[:, 0]))
            a = np.abs(S)
            m = a.min()
            p = 30.0 + 2.0 * it
            w = np.exp(-p * (a - m))
            w /= w.sum()
            wt = w * np.sign(S)
            X, Y = P[:, 0], P[:, 1]
            gx = 0.5 * (np.bincount(I, wt * (Y[J] - Y[K]), minlength=n)
                        + np.bincount(J, wt * (Y[K] - Y[I]), minlength=n)
                        + np.bincount(K, wt * (Y[I] - Y[J]), minlength=n))
            gy = 0.5 * (np.bincount(I, wt * (X[K] - X[J]), minlength=n)
                        + np.bincount(J, wt * (X[I] - X[K]), minlength=n)
                        + np.bincount(K, wt * (X[J] - X[I]), minlength=n))
            G = np.column_stack([gx, gy])
            nrm = np.linalg.norm(G)
            if nrm < 1e-16:
                break
            lr = lr0 * (0.5 * (1 + np.cos(np.pi * it / iters)))  # cosine decay
            P = P + (lr * G / nrm) * n
            P = np.clip(P, -1.5, 1.5)
        return P

    def hill_climb(P: np.ndarray, steps_max: int = 14, step0: float = 0.05) -> np.ndarray:
        """Exact coordinate descent on the TRUE objective (min area / hull area)."""
        step = step0

        def score(P):
            a = _tri_areas(P)
            h = _hull_area(P)
            if h <= 1e-12:
                return 0.0
            k = np.partition(a, 4)[:5]
            return (0.6 * a.min() + 0.4 * k.mean()) / h

        best = score(P)
        for _ in range(steps_max):
            improved = False
            for i in range(len(P)):
                for ax in (0, 1):
                    for d in (+1, -1):
                        for st in (step, 0.5 * step):
                            Q = P.copy()
                            Q[i, ax] += d * st
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

    # ---- Build seeds for the 4 base points (rotation handles symmetry)
    seeds = []
    ang = np.pi / 2.0
    # vertex ring + inner rings, classic triangular-lattice-like starts
    for r4 in ([1.0, 0.72, 0.45, 0.2], [1.0, 0.8, 0.55, 0.3],
               [1.0, 0.85, 0.6, 0.35], [1.0, 0.75, 0.5, 0.25],
               [1.0, 0.9, 0.7, 0.4], [1.0, 0.65, 0.4, 0.15]):
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
    # RANDOM seeds: broad multi-start (scaled up from 24 to 70)
    for _ in range(70):
        r = rng.uniform(0.05, 1.0, 4)
        a = rng.uniform(0, 2 * np.pi, 4)
        seeds.append(np.column_stack([r * np.cos(a), r * np.sin(a)]))

    best_P, best_ts = None, -1.0

    def keep(P: np.ndarray) -> None:
        nonlocal best_P, best_ts
        try:
            ts = true_score(P)
            if ts > best_ts:
                best_ts = ts
                best_P = P.copy()
        except Exception:
            pass

    # --- Structured full-configuration seeds (symmetry-free baselines)
    full_seeds = []
    t7 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    t5 = np.linspace(0, 2 * np.pi, 5, endpoint=False) + np.pi / 5
    full_seeds.append(np.vstack([np.column_stack([np.cos(t7), np.sin(t7)]),
                                 0.55 * np.column_stack([np.cos(t5), np.sin(t5)]),
                                 [[0.0, 0.0]]]))          # 7-ring + 5-ring + center
    for r2 in (0.5, 0.6):
        pts = [[0.0, 0.0]]
        for k in range(6):
            a = np.pi / 3 * k
            pts.append([np.cos(a), np.sin(a)])
            pts.append([r2 * np.cos(a + np.pi / 6), r2 * np.sin(a + np.pi / 6)])
        full_seeds.append(np.array(pts))
    for P0 in full_seeds:
        try:
            P = grad_ascent(P0)
            keep(P)
        except Exception:
            continue

    # --- Symmetric (3-fold) seeds; broken symmetry allowed during descent
    candidates = []
    for B0 in seeds:
        try:
            P = expand(B0)
            P = grad_ascent(P)                   # fast vectorized refinement
            ts = true_score(P)
            candidates.append((ts, P))
        except Exception:
            continue

    # Keep TOP 10 candidates for the exact-objective polish (scaled from 4)
    candidates.sort(key=lambda t: -t[0])
    for ts0, P0 in candidates[:10]:
        try:
            P = hill_climb(P0)                   # exact-objective polish
            keep(P)
        except Exception:
            continue

    # Basin hopping: deterministic perturbations around the best, then re-descend
    if best_P is not None:
        for k in range(6):
            try:
                amp = 0.025 * (k + 1)
                Q = best_P + amp * np.sin(np.arange(1, n + 1)[:, None]
                                          * np.array([1.7, 2.3])[None, :])
                Q = grad_ascent(Q, iters=200)
                Q = hill_climb(Q, steps_max=8, step0=0.03)
                keep(Q)
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