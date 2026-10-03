# EVOLVE-BLOCK-START
import numpy as np

_N = 13
_idx = np.array([(i, j, k) for i in range(_N - 2)
                 for j in range(i + 1, _N - 1)
                 for k in range(j + 1, _N)], dtype=np.int64)
_I, _J, _K = _idx[:, 0], _idx[:, 1], _idx[:, 2]
_NT = _idx.shape[0]


def _min_tri(P):
    ax, ay = P[_I, 0], P[_I, 1]
    bx, by = P[_J, 0], P[_J, 1]
    cx, cy = P[_K, 0], P[_K, 1]
    return np.abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay)).min() * 0.5


def _hull_area(P):
    order = np.lexsort((P[:, 1], P[:, 0]))
    pts = P[order]

    def chain(pp):
        out = []
        for p in pp:
            while len(out) >= 2:
                o, a = out[-2], out[-1]
                if (a[0]-o[0])*(p[1]-o[1]) - (a[1]-o[1])*(p[0]-o[0]) <= 0:
                    out.pop()
                else:
                    break
            out.append(p)
        return out

    lo = chain(pts)
    up = chain(pts[::-1])
    hull = np.array(lo[:-1] + up[:-1])
    m = len(hull)
    a = 0.0
    for i in range(m):
        x1, y1 = hull[i]
        x2, y2 = hull[(i + 1) % m]
        a += x1 * y2 - x2 * y1
    return abs(a) * 0.5


def _ratio(P):
    ha = _hull_area(P)
    if ha < 1e-12:
        return 0.0
    return _min_tri(P) / ha


def _grad_ascent(P0, iters=300, lr0=0.09):
    """Fast softmin-gradient ascent on min |triangle area| (vectorized)."""
    P = P0.copy()
    for it in range(iters):
        A, B, C = P[_I], P[_J], P[_K]
        S = (B[:, 0]-A[:, 0])*(C[:, 1]-A[:, 1]) - (B[:, 1]-A[:, 1])*(C[:, 0]-A[:, 0])
        a = np.abs(S)
        m = a.min()
        p = 40.0 + 2.0 * it  # annealed sharpness
        w = np.exp(-p * (a - m))
        w /= w.sum()
        wt = w * np.sign(S)
        X, Y = P[:, 0], P[:, 1]
        gx = 0.5 * (np.bincount(_I, wt * (Y[_J]-Y[_K]), minlength=_N)
                    + np.bincount(_J, wt * (Y[_K]-Y[_I]), minlength=_N)
                    + np.bincount(_K, wt * (Y[_I]-Y[_J]), minlength=_N))
        gy = 0.5 * (np.bincount(_I, wt * (X[_K]-X[_J]), minlength=_N)
                    + np.bincount(_J, wt * (X[_I]-X[_K]), minlength=_N)
                    + np.bincount(_K, wt * (X[_J]-X[_I]), minlength=_N))
        G = np.column_stack([gx, gy])
        nrm = np.linalg.norm(G)
        if nrm < 1e-16:
            break
        lr = lr0 * (0.5 * (1 + np.cos(np.pi * it / iters)))
        P = P + (lr * G / nrm) * _N
        P = np.clip(P, 0.0, 1.0)
    return P


def _climb(P, rng, iters, amp0=0.05, amp_hi=0.15):
    """Exact stochastic hill climbing on the ratio; one point at a time."""
    P = P.copy()
    best = _ratio(P)
    best_P = P.copy()
    amp = amp0
    fails = 0
    for it in range(iters):
        i = rng.integers(_N)
        d = rng.normal(size=2) * amp
        old = P[i].copy()
        new = old + d
        if not (0.0 <= new[0] <= 1.0 and 0.0 <= new[1] <= 1.0):
            amp *= 0.98
            continue
        P[i] = new
        v = _ratio(P)
        if v > best + 1e-14:
            best = v
            best_P = P.copy()
            fails = 0
            amp = min(amp * 1.02, amp_hi)
        else:
            P[i] = old
            fails += 1
            if fails > 60:
                amp *= 0.85
                fails = 0
                if amp < 1e-4:
                    break
    return best_P, best


def _inits():
    """~72 diverse deterministic seeds in the unit box."""
    seeds = []
    t = np.linspace(0, 2 * np.pi, _N, endpoint=False)
    seeds.append(0.5 + 0.45 * np.column_stack([np.cos(t), np.sin(t)]))  # 13-gon
    t12 = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    seeds.append(np.vstack([0.5 + 0.45*np.column_stack([np.cos(t12), np.sin(t12)]),
                            [[0.5, 0.5]]]))  # 12-ring + center
    # Double rings: outer ring of m, inner ring of k at radius r, plus center
    for m in (7, 8, 9, 10, 11):
        for k in (3, 4, 5):
            if m + k + 1 != _N:
                continue
            for r_in in (0.3, 0.4, 0.5):
                outer = 0.5 + 0.45 * np.column_stack(
                    [np.cos(np.linspace(0, 2*np.pi, m, endpoint=False)),
                     np.sin(np.linspace(0, 2*np.pi, m, endpoint=False))])
                ain = np.linspace(0, 2*np.pi, k, endpoint=False) + np.pi/m
                inner = 0.5 + r_in * 0.45 * np.column_stack(
                    [np.cos(ain), np.sin(ain)])
                seeds.append(np.vstack([outer, inner, [[0.5, 0.5]]]))
    # 3-fold symmetric: center + 4 orbits of 3
    for base in [((0.25, 0.30), (0.40, 0.15), (0.30, 0.45), (0.45, 0.40)),
                 ((0.20, 0.25), (0.45, 0.25), (0.25, 0.45), (0.42, 0.42)),
                 ((0.30, 0.10), (0.48, 0.20), (0.15, 0.40), (0.40, 0.45)),
                 ((0.35, 0.35), (0.47, 0.12), (0.12, 0.47), (0.28, 0.28))]:
        P = np.full((_N, 2), 0.5)
        for oi, b in enumerate(base):
            for r in range(3):
                a = 2 * np.pi * r / 3
                R = np.array([[np.cos(a), -np.sin(a)],
                              [np.sin(a), np.cos(a)]])
                v = R @ (np.array(b) - 0.5)
                P[1 + 3 * oi + r] = 0.5 + v
        seeds.append(P)
    # Jittered lattices
    rng_lat = np.random.default_rng(7)
    for jig in (0.0, 0.02, 0.05):
        gx_, gy_ = np.meshgrid(np.linspace(0.08, 0.92, 5),
                               np.linspace(0.12, 0.88, 3))
        lat = np.column_stack([gx_.ravel(), gy_.ravel()])
        lat = lat + rng_lat.normal(scale=jig, size=lat.shape)
        seeds.append(np.clip(np.vstack([lat, [[0.5, 0.5]]]), 0, 1))
    # Seeded random restarts (deterministic)
    rr = np.random.default_rng(2024)
    for s in range(24):
        r2 = np.random.default_rng(1000 + s)
        seeds.append(r2.random((_N, 2)) * 0.9 + 0.05)
    return seeds


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points in a convex region maximizing
    the smallest triangle area (Heilbronn problem, n = 13).

    Returns:
        points: np.ndarray of shape (13,2) with x,y coordinates.
    """
    best_P = None
    best_val = -1.0

    # --- Stage 1: broad funnel — fast softmin grad ascent on all seeds
    cands = []
    for P0 in _inits():
        try:
            Q = _grad_ascent(P0, iters=300)
            v = _ratio(Q)
            cands.append((v, Q))
        except Exception:
            continue

    # --- Stage 2: exact polish on the top candidates
    cands.sort(key=lambda c: -c[0])
    polish_pool = cands[:12]
    polish_seed = 12345
    for rank, (v0, Q0) in enumerate(polish_pool):
        rng = np.random.default_rng(polish_seed + rank)
        # medium climb from the gradient-refined candidate
        P, v = _climb(Q0, rng, iters=5000, amp0=0.02)
        # short basin hop
        Qh = np.clip(P + rng.normal(scale=0.015, size=P.shape), 0.0, 1.0)
        Qh, vh = _climb(Qh, rng, iters=2000, amp0=0.01)
        if vh > v:
            P, v = Qh, vh
        if v > best_val:
            best_val = v
            best_P = P.copy()

    if best_P is None or best_val <= 0:
        t = np.linspace(0, 2 * np.pi, _N, endpoint=False)
        best_P = 0.5 + 0.45 * np.column_stack([np.cos(t), np.sin(t)])

    # --- Stage 3: final deterministic polish on the global best
    rng = np.random.default_rng(999)
    best_P, best_val = _climb(best_P, rng, iters=6000, amp0=0.01)

    # Rescale so the convex hull has unit area (ratio invariant under this)
    ha = _hull_area(best_P)
    if ha > 1e-12:
        c = best_P.mean(axis=0)
        best_P = (best_P - c) / np.sqrt(ha) + c
    else:
        t = np.linspace(0, 2 * np.pi, _N, endpoint=False)
        best_P = 0.5 + 0.45 * np.column_stack([np.cos(t), np.sin(t)])
        ha = _hull_area(best_P)
        c = best_P.mean(axis=0)
        best_P = (best_P - c) / np.sqrt(ha) + c

    assert np.all(np.isfinite(best_P)) and best_P.shape == (13, 2)
    return best_P


# EVOLVE-BLOCK-END
