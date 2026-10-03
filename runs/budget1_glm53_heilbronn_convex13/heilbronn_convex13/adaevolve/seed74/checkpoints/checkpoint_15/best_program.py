# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations
from scipy.optimize import linprog

_N = 13
_idx = np.array(list(combinations(range(_N), 3)), dtype=np.intp)
_I, _J, _K = _idx[:, 0].copy(), _idx[:, 1].copy(), _idx[:, 2].copy()


def _hull_area(P: np.ndarray) -> float:
    """Convex hull area of the point set (Andrew's monotone chain)."""
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
    """Vectorized areas of all C(13,3) triangles."""
    x, y = P[:, 0], P[:, 1]
    return 0.5 * np.abs((x[_I] - x[_K]) * (y[_J] - y[_K]) -
                        (y[_I] - y[_K]) * (x[_J] - x[_K]))


def _obj(P: np.ndarray) -> float:
    """min triangle area / hull area (affine-invariant score)."""
    a = _tri_areas(P)
    h = _hull_area(P)
    return float(a.min() / h) if h > 1e-12 else 0.0


def _soft_obj(P: np.ndarray, beta: float = 200.0) -> float:
    """Smooth soft-min surrogate: -log(sum(exp(-beta*a)))/beta approximates min.

    Gives nonzero gradient information on the ~285 non-critical triangles,
    helping annealing escape flat basins of the hard min objective.
    """
    a = _tri_areas(P)
    h = _hull_area(P)
    if h <= 1e-12:
        return 0.0
    m = a.min()
    soft = m - np.log(np.sum(np.exp(-beta * (a - m)))) / beta
    return float(soft / h)


def _sa(P: np.ndarray, iters: int, t0: float, t1: float, rng,
        soft_frac: float = 0.5) -> tuple:
    """Simulated annealing on the 26 coordinates; returns (score, points).

    Continuation scheme: the first `soft_frac` of iterations use the smooth
    soft-min surrogate (beta annealed from 50 to 400) to explore, then switch
    to the hard min objective for final refinement. Best-so-far is always
    tracked with the hard objective.
    """
    P = P.copy()
    f = _obj(P)
    best, bestP = f, P.copy()
    n_soft = int(iters * soft_frac)
    for it in range(iters):
        T = t0 * (t1 / t0) ** (it / max(iters - 1, 1))
        sig = 0.6 * np.sqrt(T) + 1e-5
        j = int(rng.integers(_N))
        old = P[j].copy()
        P[j] = old + sig * rng.standard_normal(2)
        if it < n_soft:
            beta = 50.0 * (400.0 / 50.0) ** (it / max(n_soft - 1, 1))
            f2 = _soft_obj(P, beta)
            # accept on soft score, but track hard score for the record
            hard = _obj(P)
            if f2 >= f or rng.random() < np.exp(min((f2 - f) / T, 0.0)):
                f = f2
                if hard > best:
                    best, bestP = hard, P.copy()
            else:
                P[j] = old
        else:
            f2 = _obj(P)
            if f2 >= f or rng.random() < np.exp(min((f2 - f) / T, 0.0)):
                f = f2
                if f2 > best:
                    best, bestP = f2, P.copy()
            else:
                P[j] = old
    return best, bestP


def _polish(P: np.ndarray, steps) -> tuple:
    """Greedy pattern search: try 8 directions per point with shrinking steps."""
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


def _signed_areas(P: np.ndarray) -> np.ndarray:
    """Signed (orientation-preserving) areas of all 286 triangles."""
    x, y = P[:, 0], P[:, 1]
    return 0.5 * ((x[_I] - x[_K]) * (y[_J] - y[_K]) -
                  (y[_I] - y[_K]) * (x[_J] - x[_K]))


def _area_grads(P: np.ndarray) -> np.ndarray:
    """Gradients of signed triangle areas w.r.t. all 26 coordinates.

    Returns array of shape (286, 26): row t holds dS_t/d(x_0,y_0,...,x_12,y_12).
    For S = 0.5*(x_i(y_j-y_k) + x_j(y_k-y_i) + x_k(y_i-y_j)):
        dS/dx_i = 0.5*(y_j - y_k),  dS/dy_i = 0.5*(x_k - x_j),  etc.
    """
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


def _lp_refine(P: np.ndarray, max_iters: int = 60, rel_tol: float = 0.02) -> tuple:
    """Active-set LP maximin refinement of the exact nonsmooth objective.

    At each iteration, the active set A = {triangles with signed area within
    rel_tol of the minimum (by |area|)} is identified. A small LP in 27
    variables (26 displacement coords + margin t):

        max t  s.t.  sgn_t * grad(S_t) . d >= t  for t in A,   -1 <= d <= 1

    (scipy linprog: minimize -t) yields the steepest-ascent direction of the
    linearized minimum — a valid ascent direction even with many active
    triangles, unlike soft-min surrogates. A backtracking line search on the
    TRUE ratio objective (min |area| / hull area) accepts the step; the step
    scale shrinks geometrically on failure. Signs are re-derived from the
    current orientation each iteration, so |S| is always what we maximize.
    """
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
        # LP vars: [d(26), t]; maximize t -> minimize -t
        # constraints: -sgn*G @ d + t <= 0
        A_ub = np.hstack([-(sgn[:, None] * G), np.ones((len(act), 1))])
        b_ub = np.zeros(len(act))
        # bounds: d in [-1,1], t free (but bounded implicitly)
        bounds = [(-1, 1)] * (2 * _N) + [(None, None)]
        c = np.zeros(2 * _N + 1)
        c[-1] = -1.0
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
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
        # backtracking line search on the true objective
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


def _seed(kind: str, rng) -> np.ndarray:
    """Structured / random initial configurations."""
    if kind == 'gon13':
        ang = np.linspace(0, 2 * np.pi, 13, endpoint=False)
        return np.stack([np.cos(ang), np.sin(ang)], axis=1)
    if kind == 'gon12c':
        ang = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        ring = np.stack([np.cos(ang), np.sin(ang)], axis=1)
        return np.vstack([ring, [[0.0, 0.0]]])
    if kind == 'ring931':
        a1 = np.linspace(0, 2 * np.pi, 9, endpoint=False)
        outer = np.stack([np.cos(a1), np.sin(a1)], axis=1)
        a2 = np.linspace(0, 2 * np.pi, 3, endpoint=False) + np.pi / 9
        inner = 0.5 * np.stack([np.cos(a2), np.sin(a2)], axis=1)
        return np.vstack([outer, inner, [[0.0, 0.0]]])
    # random points in the unit disk
    r = np.sqrt(rng.random(_N))
    t = rng.random(_N) * 2 * np.pi
    return np.stack([r * np.cos(t), r * np.sin(t)], axis=1)


def _sym_orbit(k: int, r: float, off: float, mod: float = 1.0) -> np.ndarray:
    """One orbit of k equally-spaced points at radius r, offset off.

    `mod` alternates radii of even/odd points (radial modulation), giving
    reflection-symmetric 'flower' orbits in addition to plain rings.
    """
    ang = off + 2.0 * np.pi * np.arange(k) / k
    rad = np.ones(k)
    if k > 1:
        rad[1::2] = mod
    return np.stack([r * rad * np.cos(ang), r * rad * np.sin(ang)], axis=1)


def _sym_search():
    """Symmetry-reduced parametric grid search over 3-fold symmetric configs.

    Exploits 120-degree rotational symmetry: configurations are unions of
    orbits (sizes 3, 4, or 6) on concentric circles, optionally with radial
    modulation, plus a center point. This collapses the 26-dim search to
    4-8 parameters (radii + angular offsets) that can be gridded
    near-exhaustively. Families covered: (4,4,4)+1, (6,6)+1, (3,3,3,3)+1,
    (6,3,3)+1, and modulated variants. Returns (best_score, best_points).
    """
    best = [-1.0, None]

    def upd(P):
        f = _obj(P)
        if f > best[0]:
            best[0] = f
            best[1] = P.copy()

    C = np.zeros((1, 2))
    # Family A: (4,4,4) + center
    for r2 in np.linspace(0.25, 0.95, 15):
        for r3 in np.linspace(0.12, 0.75, 13):
            for o2 in np.linspace(0, np.pi / 2, 5, endpoint=False):
                for o3 in np.linspace(0, np.pi / 2, 5, endpoint=False):
                    P = np.vstack([_sym_orbit(4, 1.0, 0.0),
                                   _sym_orbit(4, r2, o2),
                                   _sym_orbit(4, r3, o3), C])
                    upd(P)
    # Family B: (6,6) + center
    for r2 in np.linspace(0.25, 0.85, 13):
        for o1 in np.linspace(0, np.pi / 3, 4, endpoint=False):
            for o2 in np.linspace(0, np.pi / 3, 6, endpoint=False):
                P = np.vstack([_sym_orbit(6, 1.0, o1),
                               _sym_orbit(6, r2, o2), C])
                upd(P)
    # Family C: (3,3,3,3) + center
    for r2 in np.linspace(0.55, 0.95, 9):
        for r3 in np.linspace(0.30, 0.70, 8):
            for r4 in np.linspace(0.10, 0.50, 7):
                for o1 in np.linspace(0, 2 * np.pi / 3, 4, endpoint=False):
                    for o2 in np.linspace(0, 2 * np.pi / 3, 3, endpoint=False):
                        for o3 in np.linspace(0, 2 * np.pi / 3, 3, endpoint=False):
                            for o4 in np.linspace(0, 2 * np.pi / 3, 3, endpoint=False):
                                P = np.vstack([_sym_orbit(3, 1.0, o1),
                                               _sym_orbit(3, r2, o2),
                                               _sym_orbit(3, r3, o3),
                                               _sym_orbit(3, r4, o4), C])
                                upd(P)
    # Family D: (6,3,3) + center
    for r2 in np.linspace(0.30, 0.80, 11):
        for r3 in np.linspace(0.15, 0.60, 9):
            for o1 in np.linspace(0, np.pi / 3, 3, endpoint=False):
                for o2 in np.linspace(0, 2 * np.pi / 3, 3, endpoint=False):
                    for o3 in np.linspace(0, 2 * np.pi / 3, 3, endpoint=False):
                        P = np.vstack([_sym_orbit(6, 1.0, o1),
                                       _sym_orbit(3, r2, o2),
                                       _sym_orbit(3, r3, o3), C])
                        upd(P)
    # Family E: modulated outer orbit, (4,4,4) + center
    for m in [0.75, 0.85, 0.95, 1.05, 1.15, 1.25]:
        for r2 in np.linspace(0.35, 0.90, 8):
            for r3 in np.linspace(0.15, 0.65, 6):
                for o2 in np.linspace(0, np.pi / 2, 3, endpoint=False):
                    for o3 in np.linspace(0, np.pi / 2, 3, endpoint=False):
                        P = np.vstack([_sym_orbit(4, 1.0, 0.0, m),
                                       _sym_orbit(4, r2, o2),
                                       _sym_orbit(4, r3, o3), C])
                        upd(P)
    return best[0], best[1]


def heilbronn_convex13() -> np.ndarray:
    """
    Construct 13 points maximizing the minimum triangle area normalized by
    the convex hull area.

    Primary strategy: symmetry-reduced parametric search. 3-fold rotationally
    symmetric configurations (unions of concentric-ring orbits under 120-degree
    rotation, with several orbit-size partitions and radial modulation) are
    parameterized by only 4-8 numbers, enabling a dense deterministic grid over
    ~100k symmetric configurations evaluated exactly via the vectorized
    286-triangle code. The best symmetric candidate is polished by pattern
    search and refined with low-temperature annealing (which may break
    symmetry if beneficial). A reduced asymmetric multi-start SA provides
    diversity. Fully deterministic (fixed seeds, fixed grids/iterations).
    """
    best_f, best_P = -1.0, None
    # --- Stage 1: symmetry-reduced exhaustive grid search ---
    f, P = _sym_search()
    if P is not None and f > best_f:
        best_f, best_P = f, P
    # --- Stage 2: polish + exact LP maximin refinement of the symmetric best ---
    if best_P is not None:
        f, P = _polish(best_P, [2e-2, 5e-3, 1e-3, 2e-4, 5e-5])
        if f > best_f:
            best_f, best_P = f, P
        # Active-set LP refinement: exact nonsmooth maximin ascent (KKT).
        f, P = _lp_refine(best_P)
        if f > best_f:
            best_f, best_P = f, P
        for r_i in range(2):
            rng = np.random.default_rng(777 + r_i)
            f, P = _sa(best_P, 30000, 8e-4, 1e-5, rng, soft_frac=0.0)
            f, P = _lp_refine(P)
            f, P = _polish(P, [1e-2, 3e-3, 7e-4, 1e-4, 2e-5, 5e-6])
            if f > best_f:
                best_f, best_P = f, P
    # --- Stage 3: reduced asymmetric multi-start for diversity ---
    for s_i, kind in enumerate(['gon12c', 'gon13', 'ring931', 'rand', 'rand']):
        rng = np.random.default_rng(1000 + s_i)
        P0 = _seed(kind, rng)
        if kind == 'rand':
            P0 = P0 + 0.03 * rng.standard_normal((13, 2))
        f, P = _sa(P0, 30000, 4e-3, 2e-5, rng, soft_frac=0.5)
        f, P = _lp_refine(P)
        f, P = _polish(P, [2e-2, 5e-3, 1e-3, 2e-4, 5e-5])
        if f > best_f:
            best_f, best_P = f, P
    # --- Stage 4: final deep LP maximin polish on the global best ---
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
