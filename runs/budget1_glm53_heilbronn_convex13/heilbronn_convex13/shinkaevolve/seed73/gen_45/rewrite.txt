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
    R = np.array([[c, -s], [s, c]])
    R2 = R @ R

    def expand(B: np.ndarray) -> np.ndarray:
        return np.vstack([B, B @ R.T, B @ R2.T, [[0.0, 0.0]]])

    def true_score(P: np.ndarray) -> float:
        a = _tri_areas(P)
        h = _hull_area(P)
        return a.min() / h if h > 1e-12 else 0.0

    idx = np.array(list(combinations(range(n), 3)))
    I, J, K = idx[:, 0], idx[:, 1], idx[:, 2]

    def grad_ascent(P0: np.ndarray, iters: int = 300, lr0: float = 0.08) -> np.ndarray:
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
            lr = lr0 * (0.5 * (1 + np.cos(np.pi * it / iters)))
            P = P + (lr * G / nrm) * n
            P = np.clip(P, -1.5, 1.5)
        return P

    dirs = np.array([[1, 0], [-1, 0], [0, 1], [0, -1],
                     [1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=float)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    def exact_climb(P: np.ndarray, step0: float = 0.02, steps_max: int = 3,
                    tol: float = 1e-12, min_step: float = 1e-5) -> np.ndarray:
        """Coordinate descent on the TRUE objective. steps_max limits sweeps;
        used both for cheap re-ranking (steps_max=3) and full polish."""
        P = P.copy()
        best = true_score(P)
        step = step0
        for _ in range(steps_max):
            improved = False
            for i in range(n):
                for d in dirs:
                    for st in (step, 0.5 * step):
                        Q = P.copy()
                        Q[i] += st * d
                        sc = true_score(Q)
                        if sc > best + tol:
                            best = sc
                            P = Q
                            improved = True
            if not improved:
                step *= 0.5
                if step < min_step:
                    break
        return P

    # ---------- Seed generation ----------
    seeds = []            # 3-fold symmetric base (4 pts -> 13)
    ang = np.pi / 2.0
    for r4 in ([1.0, 0.72, 0.45, 0.2], [1.0, 0.8, 0.55, 0.3],
               [1.0, 0.85, 0.6, 0.35], [1.0, 0.75, 0.5, 0.25],
               [1.0, 0.9, 0.7, 0.4], [1.0, 0.65, 0.4, 0.15]):
        B = []
        for k, r in enumerate(r4):
            a = ang - k * np.pi / 9.0
            B.append([r * np.cos(a), r * np.sin(a)])
        seeds.append(np.array(B))
    for spread in (0.9, 0.6, 0.3):
        seeds.append(np.array([[spread, 0.0],
                               [spread * 0.5, spread * 0.85],
                               [0.0, spread],
                               [spread * 0.4, spread * 0.3]]))
    for _ in range(30):
        r = rng.uniform(0.1, 1.0, 4)
        a = rng.uniform(0, 2 * np.pi, 4)
        seeds.append(np.column_stack([r * np.cos(a), r * np.sin(a)]))

    # ---------- Stage 1: softmin gradient ascent on symmetric seeds ----------
    candidates = []
    for B0 in seeds:
        try:
            P = grad_ascent(expand(B0))
            candidates.append(P)
        except Exception:
            continue

    # ---------- Stage 1b: full-configuration structural seeds ----------
    full_seeds = []
    t7 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    t5 = np.linspace(0, 2 * np.pi, 5, endpoint=False) + np.pi / 5
    for r2 in (0.55, 0.45):
        full_seeds.append(np.vstack([np.column_stack([np.cos(t7), np.sin(t7)]),
                                     r2 * np.column_stack([np.cos(t5), np.sin(t5)]),
                                     [[0.0, 0.0]]]))
    for r2 in (0.5, 0.6):
        pts = [[0.0, 0.0]]
        for k in range(6):
            a = np.pi / 3 * k
            pts.append([np.cos(a), np.sin(a)])
            pts.append([r2 * np.cos(a + np.pi / 6), r2 * np.sin(a + np.pi / 6)])
        full_seeds.append(np.array(pts))
    t13 = np.linspace(0, 2 * np.pi, 13, endpoint=False)
    full_seeds.append(np.column_stack([np.cos(t13), np.sin(t13)]))
    for _ in range(6):
        full_seeds.append(rng.uniform(-0.9, 0.9, (n, 2)))
    for P0 in full_seeds:
        try:
            candidates.append(grad_ascent(P0, iters=200))
        except Exception:
            continue

    # ---------- Stage 2: TWO-PASS RANKING ----------
    # Pass A: cheap 3-step exact mini-climb on ALL candidates to re-rank.
    reranked = []
    for P in candidates:
        try:
            Q = exact_climb(P, step0=0.02, steps_max=3)
            reranked.append((true_score(Q), Q))
        except Exception:
            continue
    reranked.sort(key=lambda t: -t[0])

    best_P, best_ts = None, -1.0

    # Pass B: full polish on top-10 of the re-ranked list.
    for _ts0, P0 in reranked[:10]:
        try:
            Q = exact_climb(P0, step0=0.02, steps_max=30, min_step=1e-5)
            ts = true_score(Q)
            if ts > best_ts:
                best_ts, best_P = ts, Q.copy()
        except Exception:
            continue

    # ---------- Stage 3: basin hopping around best ----------
    if best_P is not None:
        for k in range(5):
            try:
                amp = 0.02 * (k + 1)
                Q = best_P + amp * np.sin(np.arange(1, n + 1)[:, None]
                                           * np.array([1.7, 2.3])[None, :])
                Q = grad_ascent(Q, iters=150)
                Q = exact_climb(Q, step0=0.015, steps_max=10)
                ts = true_score(Q)
                if ts > best_ts:
                    best_ts, best_P = ts, Q.copy()
            except Exception:
                continue

    if best_P is None:
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_P = np.column_stack([np.cos(t), np.sin(t)])

    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(best_P[i] - best_P[j]) < 1e-9:
                best_P[j, 0] += 1e-6 * (j + 1)
    return best_P


# EVOLVE-BLOCK-END