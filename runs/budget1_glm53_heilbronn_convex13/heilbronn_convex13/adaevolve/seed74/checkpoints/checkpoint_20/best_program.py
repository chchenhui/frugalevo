# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations
from scipy.optimize import linprog

_N = 13
_idx = np.array(list(combinations(range(_N), 3)), dtype=np.intp)
_I, _J, _K = _idx[:, 0].copy(), _idx[:, 1].copy(), _idx[:, 2].copy()


def _hull_area(P: np.ndarray) -> float:
    """Convex hull area (Andrew's monotone chain)."""
    pts = sorted(set(map(tuple, np.asarray(P, dtype=float).tolist())))
    if len(pts) < 3:
        return 0.0

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
        return 0.0
    s = 0.0
    m = len(hull)
    for i in range(m):
        x1, y1 = hull[i]
        x2, y2 = hull[(i + 1) % m]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def _tri_areas(P: np.ndarray) -> np.ndarray:
    """Vectorized areas of all C(13,3)=286 triangles."""
    x, y = P[:, 0], P[:, 1]
    return 0.5 * np.abs((x[_I] - x[_K]) * (y[_J] - y[_K]) -
                        (y[_I] - y[_K]) * (x[_J] - x[_K]))


def _obj(P: np.ndarray) -> float:
    """min triangle area / hull area (affine-invariant score)."""
    a = _tri_areas(P)
    h = _hull_area(P)
    return float(a.min() / h) if h > 1e-12 else 0.0


def _signed_areas(P: np.ndarray) -> np.ndarray:
    x, y = P[:, 0], P[:, 1]
    return 0.5 * ((x[_I] - x[_K]) * (y[_J] - y[_K]) -
                  (y[_I] - y[_K]) * (x[_J] - x[_K]))


def _area_grads(P: np.ndarray) -> np.ndarray:
    """Gradients (286, 26) of signed triangle areas w.r.t. all coordinates."""
    x, y = P[:, 0], P[:, 1]
    G = np.zeros((len(_I), 2 * _N))
    for col, (ii, jj, kk) in enumerate(zip(_I, _J, _K)):
        G[col, 2 * ii] = 0.5 * (y[jj] - y[kk])
        G[col, 2 * ii + 1] = 0.5 * (x[kk] - x[jj])
        G[col, 2 * jj] = 0.5 * (y[kk] - y[ii])
        G[col, 2 * jj + 1] = 0.5 * (x[ii] - x[kk])
        G[col, 2 * kk] = 0.5 * (y[ii] - y[jj])
        G[col, 2 * kk + 1] = 0.5 * (x[jj] - x[ii])
    return G


def _polish(P: np.ndarray, steps) -> tuple:
    """Greedy pattern search: 8 directions per point, shrinking steps."""
    P = P.copy()
    best = _obj(P)
    dirs = np.array([[1, 0], [-1, 0], [0, 1], [0, -1],
                     [1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=float) / np.sqrt(2)
    for st in steps:
        improved = True
        while improved:
            improved = False
            for j in range(_N):
                for d in dirs:
                    cand = P.copy()
                    cand[j] = P[j] + st * d
                    f2 = _obj(cand)
                    if f2 > best + 1e-12:
                        P, best = cand, f2
                        improved = True
    return best, P


def _lp_refine(P: np.ndarray, max_iters: int = 60, rel_tol: float = 0.02) -> tuple:
    """Active-set LP maximin ascent: at each step solve
        max t  s.t.  sgn_t * grad(S_t) . d >= t (active triangles), |d|<=1
    then backtrack on the true ratio objective. Handles many simultaneous
    active (minimal) triangles exactly, unlike smooth surrogates."""
    P = P.copy()
    best = _obj(P)
    step = 0.05
    fails = 0
    for _ in range(max_iters):
        S = _signed_areas(P)
        Aabs = np.abs(S)
        m = Aabs.min()
        if m <= 1e-15:
            break
        act = np.where(Aabs <= m * (1.0 + rel_tol))[0]
        if len(act) == 0:
            act = np.array([int(np.argmin(Aabs))])
        G = _area_grads(P)[act]
        sgn = np.sign(S[act])
        sgn[sgn == 0] = 1.0
        A_ub = np.hstack([-(sgn[:, None] * G), np.ones((len(act), 1))])
        c = np.zeros(2 * _N + 1)
        c[-1] = -1.0
        bounds = [(-1, 1)] * (2 * _N) + [(None, None)]
        res = linprog(c, A_ub=A_ub, b_ub=np.zeros(len(act)),
                      bounds=bounds, method='highs')
        if not res.success:
            fails += 1
            step *= 0.5
            if fails > 6:
                break
            continue
        d = res.x[:2 * _N]
        t = res.x[-1]
        if t <= 1e-14:
            fails += 1
            step *= 0.5
            rel_tol *= 1.5
            if fails > 6:
                break
            continue
        dn = np.linalg.norm(d)
        if dn < 1e-12:
            break
        d = d / dn
        accepted = False
        for s_try in [step, step * 0.5, step * 0.25, step * 0.1]:
            cand = P + s_try * d.reshape(_N, 2)
            f2 = _obj(cand)
            if f2 > best + 1e-14:
                P, best = cand, f2
                accepted = True
                step = min(step * 1.3, 0.05)
                break
        if not accepted:
            fails += 1
            step *= 0.4
            if fails > 8:
                break
        else:
            fails = 0
    return best, P


def _sa(P: np.ndarray, iters: int, t0: float, t1: float, rng) -> tuple:
    """Low-temperature simulated annealing on the hard min objective."""
    P = P.copy()
    f = _obj(P)
    best, bestP = f, P.copy()
    for it in range(iters):
        T = t0 * (t1 / t0) ** (it / max(iters - 1, 1))
        sig = 0.6 * np.sqrt(T) + 1e-5
        j = int(rng.integers(_N))
        old = P[j].copy()
        P[j] = old + sig * rng.standard_normal(2)
        f2 = _obj(P)
        if f2 >= f or rng.random() < np.exp(min((f2 - f) / T, 0.0)):
            f = f2
            if f2 > best:
                best, bestP = f2, P.copy()
        else:
            P[j] = old
    return best, bestP


def _sym_orbit(k: int, r: float, off: float) -> np.ndarray:
    """One orbit of k equally-spaced points at radius r, angular offset off."""
    ang = off + 2.0 * np.pi * np.arange(k) / k
    return np.stack([r * np.cos(ang), r * np.sin(ang)], axis=1)


def _sym_search():
    """Symmetry-reduced grid search over 3-fold symmetric configurations.

    Covers ring-orbit families (4,4,4)+1, (6,6)+1, (3,3,3,3)+1, (6,3,3)+1,
    plus TRIANGULAR-HULL families: 3 corners + edge orbits + center, which
    historically dominate Heilbronn-type normalized scores because a small
    hull vertex count maximizes area-per-point. Returns top-5 distinct
    (score, points) candidates for independent refinement."""
    pool = []

    def upd(P):
        pool.append((_obj(P), P.copy()))

    C = np.zeros((1, 2))
    # Family A: (4,4,4) + center
    for r2 in np.linspace(0.25, 0.95, 13):
        for r3 in np.linspace(0.12, 0.75, 11):
            for o2 in np.linspace(0, np.pi / 2, 4, endpoint=False):
                for o3 in np.linspace(0, np.pi / 2, 4, endpoint=False):
                    P = np.vstack([_sym_orbit(4, 1.0, 0.0),
                                   _sym_orbit(4, r2, o2),
                                   _sym_orbit(4, r3, o3), C])
                    upd(P)
    # Family B: (6,6) + center
    for r2 in np.linspace(0.25, 0.85, 11):
        for o1 in np.linspace(0, np.pi / 3, 3, endpoint=False):
            for o2 in np.linspace(0, np.pi / 3, 5, endpoint=False):
                P = np.vstack([_sym_orbit(6, 1.0, o1),
                               _sym_orbit(6, r2, o2), C])
                upd(P)
    # Family C: (3,3,3,3) + center
    for r2 in np.linspace(0.55, 0.95, 7):
        for r3 in np.linspace(0.30, 0.70, 6):
            for r4 in np.linspace(0.10, 0.50, 5):
                for o1 in np.linspace(0, 2 * np.pi / 3, 3, endpoint=False):
                    for o2 in np.linspace(0, 2 * np.pi / 3, 3, endpoint=False):
                        for o3 in np.linspace(0, 2 * np.pi / 3, 3, endpoint=False):
                            for o4 in np.linspace(0, 2 * np.pi / 3, 3, endpoint=False):
                                P = np.vstack([_sym_orbit(3, 1.0, o1),
                                               _sym_orbit(3, r2, o2),
                                               _sym_orbit(3, r3, o3),
                                               _sym_orbit(3, r4, o4), C])
                                upd(P)
    # Family D: (6,3,3) + center
    for r2 in np.linspace(0.30, 0.80, 9):
        for r3 in np.linspace(0.15, 0.60, 7):
            for o1 in np.linspace(0, np.pi / 3, 3, endpoint=False):
                for o2 in np.linspace(0, 2 * np.pi / 3, 3, endpoint=False):
                    for o3 in np.linspace(0, 2 * np.pi / 3, 3, endpoint=False):
                        P = np.vstack([_sym_orbit(6, 1.0, o1),
                                       _sym_orbit(3, r2, o2),
                                       _sym_orbit(3, r3, o3), C])
                        upd(P)
    # Family E: TRIANGULAR HULL — 3 corners + 3 edge orbits + center (3+9+1)
    ca = np.pi / 2 + 2.0 * np.pi * np.arange(3) / 3
    V = np.stack([np.cos(ca), np.sin(ca)], axis=1)
    for t1 in np.linspace(0.08, 0.92, 11):
        for t2 in np.linspace(0.08, 0.92, 11):
            for t3 in np.linspace(0.08, 0.92, 11):
                if not (t1 < t2 < t3):
                    continue
                E = []
                for t in (t1, t2, t3):
                    E.append((1 - t) * V + t * np.roll(V, -1, axis=0))
                P = np.vstack([V, np.array(E).reshape(-1, 2), C])
                upd(P)
    # Family F: triangular hull, 2 edge orbits (6) + inner triangle + center
    for t1 in np.linspace(0.10, 0.90, 8):
        for t2 in np.linspace(0.10, 0.90, 8):
            if t2 <= t1:
                continue
            E = []
            for t in (t1, t2):
                E.append((1 - t) * V + t * np.roll(V, -1, axis=0))
            for r in np.linspace(0.15, 0.55, 5):
                for off in np.linspace(0, 2 * np.pi / 3, 3, endpoint=False):
                    P = np.vstack([V, np.array(E).reshape(-1, 2),
                                   _sym_orbit(3, r, off), C])
                    upd(P)
    pool.sort(key=lambda t: -t[0])
    tops = []
    for f, P in pool:
        ok = True
        for _, Q in tops:
            if np.max(np.abs(np.sort(P, axis=0) - np.sort(Q, axis=0))) < 1e-6:
                ok = False
                break
        if ok:
            tops.append((f, P))
        if len(tops) >= 5:
            break
    return tops


def heilbronn_convex13() -> np.ndarray:
    """Construct 13 points maximizing min triangle area / convex hull area.

    Strategy: (1) symmetry-reduced deterministic grid search over ~50k
    3-fold-symmetric configurations (ring orbits and triangular-hull edge
    placements), (2) independent pattern-search + active-set LP maximin
    refinement of the top-5 distinct candidates, (3) seeded low-temperature
    annealing escapes from the incumbent followed by LP + pattern polish,
    (4) rescale so the hull has unit area. Fully deterministic."""
    best_f, best_P = -1.0, None
    # Stage 1+2: refine top-5 symmetric candidates independently
    for cand_f, cand_P in _sym_search():
        if cand_P is None:
            continue
        f, P = _polish(cand_P, [2e-2, 5e-3, 1e-3, 2e-4, 5e-5])
        if f > best_f:
            best_f, best_P = f, P
        f, P = _lp_refine(P)
        if f > best_f:
            best_f, best_P = f, P
    # Stage 3: annealing escapes + re-polish
    if best_P is not None:
        for r_i in range(3):
            rng = np.random.default_rng(777 + r_i)
            f, P = _sa(best_P, 20000, 8e-4, 1e-5, rng)
            f, P = _lp_refine(P)
            f, P = _polish(P, [1e-2, 3e-3, 7e-4, 1e-4, 2e-5, 5e-6])
            if f > best_f:
                best_f, best_P = f, P
    # Stage 4: deterministic perturb-and-refine multistart
    for r_i in range(6):
        rng = np.random.default_rng(4242 + r_i)
        amp = 0.005 + 0.02 * (r_i % 4)
        P0 = best_P + amp * rng.standard_normal((13, 2))
        f, P = _lp_refine(P0, max_iters=80, rel_tol=0.02)
        f, P = _polish(P, [1e-2, 3e-3, 7e-4, 1e-4, 2e-5, 5e-6])
        if f > best_f:
            best_f, best_P = f, P
    # Stage 5: final deep polish
    for _ in range(2):
        f, P = _lp_refine(best_P, max_iters=80, rel_tol=0.01)
        f, P = _polish(P, [3e-4, 7e-5, 2e-5, 5e-6, 2e-6])
        if f > best_f:
            best_f, best_P = f, P
    # guard against pathological results
    if best_P is None or not np.all(np.isfinite(best_P)) or best_f <= 0:
        ang = np.linspace(0, 2 * np.pi, 13, endpoint=False)
        best_P = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    # rescale so the convex hull has exactly unit area
    h = _hull_area(best_P)
    if h > 1e-12:
        best_P = best_P / np.sqrt(h)
    return np.ascontiguousarray(best_P, dtype=float)


# EVOLVE-BLOCK-END
