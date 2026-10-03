# EVOLVE-BLOCK-START
import numpy as np

_N = 13
_idx = np.array([(i, j, k) for i in range(_N - 2)
                 for j in range(i + 1, _N - 1)
                 for k in range(j + 1, _N)], dtype=np.int64)
_I, _J, _K = _idx[:, 0], _idx[:, 1], _idx[:, 2]


def _min_tri(P):
    ax, ay = P[_I, 0], P[_I, 1]
    bx, by = P[_J, 0], P[_J, 1]
    cx, cy = P[_K, 0], P[_K, 1]
    return float(np.abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay)).min() * 0.5)


def _hull_area(P):
    order = np.lexsort((P[:, 1], P[:, 0]))
    pts = P[order]

    def chain(seq):
        out = []
        for p in seq:
            while len(out) >= 2:
                o, a = out[-2], out[-1]
                if (a[0] - o[0]) * (p[1] - o[1]) - (a[1] - o[1]) * (p[0] - o[0]) <= 0:
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


def _anneal(P, rng, iters, t0, amp0, shrink_prob=0.15):
    """Simulated annealing on the exact ratio with point moves AND
    global affine shrink moves (scale hull down -> denominator drops)."""
    P = P.copy()
    best = _ratio(P)
    best_P = P.copy()
    cur = best
    T = t0
    amp = amp0
    c = P.mean(axis=0)
    for it in range(iters):
        T = t0 * (1.0 - it / iters) + 1e-9
        if rng.random() < shrink_prob:
            # GLOBAL MOVE: affine contraction toward centroid
            s = 1.0 + rng.normal() * 0.02
            if s <= 0.05:
                continue
            trial = c + (P - c) * s
        else:
            trial = P.copy()
            i = rng.integers(_N)
            trial[i] += rng.normal(size=2) * amp
        v = _ratio(trial)
        if v > cur or rng.random() < np.exp((v - cur) / max(T, 1e-12)):
            P = trial
            cur = v
            if v > best:
                best = v
                best_P = P.copy()
        else:
            amp *= 0.999
            if amp < 1e-5:
                amp = 1e-5
    return best_P, best


def _greedy(P, rng, iters, amp):
    """Plain greedy hill climb on ratio (polish)."""
    P = P.copy()
    best = _ratio(P)
    for it in range(iters):
        i = rng.integers(_N)
        trial = P.copy()
        trial[i] += rng.normal(size=2) * amp
        v = _ratio(trial)
        if v > best + 1e-14:
            P = trial
            best = v
    return P, best


def _inits(rng):
    inits = []
    # regular 13-gon
    th = 2 * np.pi * np.arange(_N) / _N
    inits.append(np.stack([np.cos(th), np.sin(th)], axis=1))
    # 3-fold symmetric orbits
    P = np.zeros((_N, 2))
    base = [(.0, .0), (.3, .1), (.5, .2), (.2, .5), (.4, .55)]
    oi = 0
    for bi, b in enumerate(base):
        if bi == 0:
            P[0] = b
        else:
            for r in range(3):
                a = 2 * np.pi * r / 3
                R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
                P[1 + 3 * (bi - 1) + r] = R @ np.array(b)
    inits.append(P)
    # triangle-ish: 4+4+4 stacked rows + center
    P = np.zeros((_N, 2))
    P[0] = [0, 0]
    P[1:5] = [[.2, .5], [.4, .5], [.6, .5], [.8, .5]]
    P[5:9] = [[.3, 1.0], [.5, 1.0], [.7, 1.0], [.9, 1.0]]
    P[9:13] = [[.1, 1.4], [.3, 1.4], [.5, 1.4], [.7, 1.4]]
    inits.append(P)
    # jittered grid
    xs = np.linspace(0.1, 0.9, 5)
    ys = np.linspace(0.15, 0.85, 3)
    P = np.stack([[x, y] for y in ys for x in xs])
    inits.append(P + rng.normal(scale=0.03, size=P.shape))
    # random
    inits.append(rng.random((_N, 2)))
    return inits


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points in a convex region maximizing
    the smallest triangle area (Heilbronn problem, n = 13).

    Returns:
        points: np.ndarray of shape (13,2) with x,y coordinates.
    """
    best_P = None
    best_val = -1.0

    for seed in (42, 7, 123, 2024):
        rng = np.random.default_rng(seed)
        for init in _inits(rng):
            # Stage 1: annealing with shrink moves (ratio objective)
            P, v = _anneal(init, rng, iters=2500, t0=0.004, amp0=0.05)
            # Stage 2: greedy polish
            P, v = _greedy(P, rng, iters=1500, amp=0.01)
            P, v2 = _greedy(P, rng, iters=800, amp=0.003)
            v = max(v, v2)
            # Stage 3: explicit hull shrink sweep — contract toward centroid
            # and keep the best ratio along the sweep (ratio often peaks
            # below full scale, since hull area shrinks quadratically).
            c = P.mean(axis=0)
            for s in np.linspace(0.80, 1.05, 26):
                Q = c + (P - c) * s
                vr = _ratio(Q)
                if vr > v:
                    v = vr
                    P = Q
            if v > best_val:
                best_val = v
                best_P = P.copy()

    if best_P is None or best_val <= 0:
        th = 2 * np.pi * np.arange(_N) / _N
        best_P = np.stack([np.cos(th), np.sin(th)], axis=1)

    # Final deterministic anneal polish on the global best
    rng = np.random.default_rng(12345)
    best_P, best_val = _anneal(best_P, rng, iters=2000, t0=0.0005, amp0=0.008)
    best_P, v = _greedy(best_P, rng, iters=1500, amp=0.002)
    best_val = max(best_val, v)

    # Rescale so the convex hull has unit area (ratio invariant)
    ha = _hull_area(best_P)
    if ha > 1e-12:
        c = best_P.mean(axis=0)
        best_P = (best_P - c) / np.sqrt(ha) + c

    assert np.all(np.isfinite(best_P)) and best_P.shape == (13, 2)
    return best_P


# EVOLVE-BLOCK-END
