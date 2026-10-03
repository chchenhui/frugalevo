# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations


def _hull_area(P: np.ndarray) -> float:
    pts = sorted(set(map(tuple, np.round(P, 12))))
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
    rng = np.random.default_rng(seed=12345)

    idx = np.array(list(combinations(range(n), 3)))
    I, J, K = idx[:, 0], idx[:, 1], idx[:, 2]
    NT = idx.shape[0]

    def signed_areas(P: np.ndarray) -> np.ndarray:
        A, B, C = P[I], P[J], P[K]
        return 0.5 * ((B[:, 0] - A[:, 0]) * (C[:, 1] - A[:, 1])
                      - (B[:, 1] - A[:, 1]) * (C[:, 0] - A[:, 0]))

    def optimize(P0: np.ndarray, iters: int = 400) -> np.ndarray:
        P = P0.copy()
        lr0 = 0.09
        for it in range(iters):
            S = signed_areas(P)
            a = np.abs(S)
            m = a.min()
            p = 40.0 + it  # annealed sharpness: 40 -> 440
            w = np.exp(-p * (a - m))
            w /= w.sum()
            wt = w * np.sign(S)
            X, Y = P[:, 0], P[:, 1]
            # Exact analytic gradient of soft-min of |triangle areas|.
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
            P = np.clip(P, -1.0, 1.0)
        return P

    # --- Deterministic, diverse seed set ---
    seeds = []
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    seeds.append(np.column_stack([np.cos(t), np.sin(t)]))          # 13-gon
    t12 = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    seeds.append(np.vstack([np.column_stack([np.cos(t12), np.sin(t12)]),
                             [[0.0, 0.0]]]))                        # 12-ring + center
    # Double-ring configurations (outer hull + inner rotated ring)
    for r_in in (0.35, 0.45, 0.55, 0.65):
        for n_in in (3, 4):
            outer = np.column_stack([np.cos(t12), np.sin(t12)])
            a_in = np.linspace(0, 2 * np.pi, n_in, endpoint=False) + np.pi / 12
            inner = r_in * np.column_stack([np.cos(a_in), np.sin(a_in)])
            cnt = 12 - n_in
            am = np.linspace(0, 2 * np.pi, cnt, endpoint=False) + np.pi / 24
            mid = 0.8 * np.column_stack([np.cos(am), np.sin(am)])
            seeds.append(np.vstack([outer[:6], inner, mid[:6 - n_in + 3][:12 - n_in - len(mid)]] if False else [outer[:12 - n_in], inner, mid]))
    # Hexagonal-cluster style seeds
    for r2 in (0.4, 0.5, 0.6):
        pts = []
        for k in range(6):
            ang = 2 * np.pi * k / 6
            pts.append([np.cos(ang), np.sin(ang)])
        for k in range(6):
            ang = 2 * np.pi * k / 6 + np.pi / 6
            pts.append([r2 * np.cos(ang), r2 * np.sin(ang)])
        pts.append([0.0, 0.0])
        seeds.append(np.array(pts) * 0.95)
    # Jittered lattice seeds
    for jx in (0.0, 0.15):
        gx_, gy_ = np.meshgrid(np.linspace(-0.8, 0.8, 4), np.linspace(-0.8, 0.8, 4))
        lat = np.column_stack([gx_.ravel(), gy_.ravel()])[:12]
        seeds.append(np.vstack([lat, [[jx, 0.25]]]))
    # Seeded random restarts
    for _ in range(12):
        seeds.append(rng.uniform(-0.95, 0.95, (n, 2)))

    best_P, best_score = None, -1.0
    for P0 in seeds:
        try:
            P = optimize(P0.copy())
            a = np.abs(signed_areas(P))
            ha = _hull_area(P)
            if ha <= 0:
                continue
            score = a.min() / ha
            if score > best_score:
                best_score = score
                best_P = P.copy()
        except Exception:
            continue

    if best_P is None:
        best_P = np.column_stack([np.cos(t), np.sin(t)])

    # Duplicate-point safety (deterministic offsets).
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(best_P[i] - best_P[j]) < 1e-9:
                best_P[j, 0] += 1e-6 * (j + 1)
    return best_P


# EVOLVE-BLOCK-END