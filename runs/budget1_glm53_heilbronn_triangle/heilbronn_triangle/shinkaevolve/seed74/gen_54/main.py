# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations

_V0 = np.array([0.0, 0.0])
_V1 = np.array([1.0, 0.0])
_V2 = np.array([0.5, np.sqrt(3.0) / 2.0])
_V = np.array([_V0, _V1, _V2])
_H = np.sqrt(3.0) / 2.0
_TRI_AREA = 0.5 * _H
_TRIP = np.array(list(combinations(range(11), 3)), dtype=int)
_I, _J, _K = _TRIP[:, 0], _TRIP[:, 1], _TRIP[:, 2]
_PT_TRIPS = [np.where((_I == p) | (_J == p) | (_K == p))[0] for p in range(11)]
_PT_MASK = [np.zeros(len(_TRIP), dtype=bool) for _ in range(11)]
for _p in range(11):
    _PT_MASK[_p][_PT_TRIPS[_p]] = True


def _min_area(pts):
    a, b, c = pts[_I], pts[_J], pts[_K]
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    return np.min(np.abs(cross)) / (2.0 * _TRI_AREA)


def _clip(pts):
    l2 = pts[:, 1] / _H
    l1 = pts[:, 0] - 0.5 * l2
    l0 = 1.0 - l1 - l2
    L = np.stack([l0, l1, l2], axis=1)
    L = np.clip(L, 1e-9, 1.0)
    L = L / L.sum(axis=1, keepdims=True)
    return L @ _V


def _lattice(k=3):
    pts = []
    for i in range(k + 1):
        for j in range(k + 1 - i):
            l = k - i - j
            pts.append((i * _V0 + j * _V1 + l * _V2) / k)
    pts.append(np.array([0.5, _H / 3.0]))
    return np.array(pts)


def _initial(seed):
    rng = np.random.default_rng(seed)
    return _clip(_lattice() + rng.normal(0.0, 0.02, _lattice().shape))


def _golden_seed():
    g = (np.sqrt(5.0) - 1.0) / 2.0
    pts = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.5, _H])]
    for kk in range(3, 11):
        u = (kk * g) % 1.0
        v = (kk * g * g) % 1.0 * (1.0 - u)
        lam = np.array([1.0 - u - v, u, v])
        lam = np.clip(lam, 0.02, 1.0)
        lam /= lam.sum()
        pts.append(lam[0] * _V0 + lam[1] * _V1 + lam[2] * _V2)
    return _clip(np.array(pts))


def _repulsion_seed(seed, iters=50):
    rng = np.random.default_rng(seed)
    pts = _clip(_lattice() + rng.normal(0.0, 0.05, _lattice().shape))
    step0 = 0.004
    for t in range(iters):
        d = pts[:, None, :] - pts[None, :, :]
        dist2 = (d ** 2).sum(axis=2) + 1e-6
        np.fill_diagonal(dist2, np.inf)
        f = (d / (dist2 ** 2)[:, :, None]).sum(axis=1)
        pts = _clip(pts + step0 * (0.95 ** t) * f)
    return _clip(pts)


def _adam(pts, iters=400, lr=0.02):
    m = np.zeros_like(pts)
    v = np.zeros_like(pts)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for it in range(1, iters + 1):
        tau = max(0.0004, 0.02 * np.exp(-it / 150.0))
        x, y = pts[:, 0], pts[:, 1]
        xi, yi = x[_I], y[_I]
        xj, yj = x[_J], y[_J]
        xk, yk = x[_K], y[_K]
        c = (xj - xi) * (yk - yi) - (yj - yi) * (xk - xi)
        a = 0.5 * np.abs(c)
        s = np.sign(c)
        z = np.exp(-(a - a.min()) / tau)
        w = z / z.sum()
        coef = -0.5 * w * s
        g = np.zeros_like(pts)
        np.add.at(g, _I, np.stack([coef * (yj - yk), coef * (xk - xj)], axis=1))
        np.add.at(g, _J, np.stack([coef * (yk - yi), coef * (xi - xk)], axis=1))
        np.add.at(g, _K, np.stack([coef * (yi - yj), coef * (xj - xi)], axis=1))
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * (g * g)
        pts = _clip(pts - lr * (m / (1 - b1 ** it)) / (np.sqrt(v / (1 - b2 ** it)) + eps))
    return pts


def _polish(pts, steps=(0.01, 0.004, 0.001, 0.0003)):
    """Incremental directional polish: only triplets containing the moved
    point are recomputed (~90 rows); unaffected min is cached."""
    best = _clip(pts)
    p, q, r = best[_I], best[_J], best[_K]
    cross = np.abs((q[:, 0] - p[:, 0]) * (r[:, 1] - p[:, 1]) - (q[:, 1] - p[:, 1]) * (r[:, 0] - p[:, 0]))
    best_val = float(cross.min()) / (2.0 * _TRI_AREA)
    ang = 2.0 * np.pi * np.arange(24) / 24
    dirs = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    for s in steps:
        improved = True
        while improved:
            improved = False
            tight = cross <= cross.min() * 1.5
            cnt = np.zeros(11)
            np.add.at(cnt, _I[tight], 1)
            np.add.at(cnt, _J[tight], 1)
            np.add.at(cnt, _K[tight], 1)
            order = np.argsort(-cnt)
            for pt in order:
                if cnt[pt] <= 0:
                    continue
                rows = _PT_TRIPS[pt]
                ri, rj, rk = _I[rows], _J[rows], _K[rows]
                mask = _PT_MASK[pt]
                rest_min = float(np.min(cross[~mask]))
                for d in dirs:
                    cand = best.copy()
                    cand[pt] += s * d
                    cand = _clip(cand)
                    p2, q2, r2 = cand[ri], cand[rj], cand[rk]
                    newc = np.abs((q2[:, 0] - p2[:, 0]) * (r2[:, 1] - p2[:, 1]) - (q2[:, 1] - p2[:, 1]) * (r2[:, 0] - p2[:, 0]))
                    v = min(rest_min, float(newc.min())) / (2.0 * _TRI_AREA)
                    if v > best_val + 1e-14:
                        cross[rows] = newc
                        best_val, best, improved = v, cand, True
                        break
                if improved:
                    break
    return best, best_val


def _grad_polish(P, iters=200):
    """Adaptive-threshold gradient polish (from crossover parent): move
    vertices of tight triangles along signed-area ascent with backtracking."""
    P = P.copy()
    cur = _min_area(P)
    for it in range(iters):
        thresh = 3.0 - 1.8 * (it / iters)
        xi, yi = P[_I, 0], P[_I, 1]
        xj, yj = P[_J, 0], P[_J, 1]
        xk, yk = P[_K, 0], P[_K, 1]
        c = (xj - xi) * (yk - yi) - (yj - yi) * (xk - xi)
        a = 0.5 * np.abs(c)
        amin = a.min()
        tight = a <= thresh * amin
        if not np.any(tight):
            tight = a <= 1.2 * amin
        coef = np.where(tight, 0.5 * np.sign(c), 0.0)
        g = np.zeros_like(P)
        np.add.at(g, _I, np.stack([coef * (yj - yk), coef * (xk - xj)], axis=1))
        np.add.at(g, _J, np.stack([coef * (yk - yi), coef * (xi - xk)], axis=1))
        np.add.at(g, _K, np.stack([coef * (yi - yj), coef * (xj - xi)], axis=1))
        step = 0.01 * (1.0 - it / iters) + 0.0005
        gn = np.linalg.norm(g, axis=1, keepdims=True)
        gn[gn < 1e-15] = 1.0
        s_try = step
        moved = False
        for _ in range(6):
            cand = _clip(P + s_try * g / gn)
            v = _min_area(cand)
            if v > cur + 1e-14:
                P, cur, moved = cand, v, True
                break
            s_try *= 0.5
        if not moved and step < 1e-6:
            break
    return P, cur


def _seeds():
    s = [_initial(sd) for sd in (1234, 1235, 1236, 20240611, 7)]
    s.append(_golden_seed())
    s.extend(_repulsion_seed(rs) for rs in (11, 202))
    rng = np.random.default_rng(2024)
    base = _lattice()
    for sig in (0.05, 0.08):
        s.append(_clip(base + rng.normal(0.0, sig, base.shape)))
    return s


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct 11 points in the equilateral triangle maximizing the minimum
    triangle area. Hybrid: diverse deterministic seed portfolio (perturbed
    lattices, golden-ratio, repulsion) + Adam softmin ascent + incremental
    directional polish + gradient polish. Deterministic, feasible fallback.
    """
    n = 11
    try:
        best_pts, best_val = None, -1.0
        for init in _seeds():
            p = _adam(_clip(init.copy()))
            p, val = _polish(p)
            if val > best_val:
                p, val2 = _grad_polish(p)
                if val2 > val:
                    p, val = _polish(p, steps=(0.002, 0.0005, 0.0002))
                    val = max(val, _min_area(p))
                best_val, best_pts = val, p
        if best_pts is None or not np.all(np.isfinite(best_pts)) or best_val < 1e-6:
            raise RuntimeError("optimization failed")
        return np.ascontiguousarray(best_pts, dtype=float)
    except Exception:
        base = _initial(1234)
        pts, val = _polish(base)
        if val < 1e-6:
            pts = _clip(base)
        return np.ascontiguousarray(pts, dtype=float)


# EVOLVE-BLOCK-END