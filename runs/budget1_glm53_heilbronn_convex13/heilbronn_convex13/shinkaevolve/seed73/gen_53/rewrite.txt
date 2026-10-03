# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations

try:
    from scipy.spatial import ConvexHull as _ConvexHull
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

n = 13
_IDX = np.array(list(combinations(range(n), 3)))
_I0, _I1, _I2 = _IDX[:, 0], _IDX[:, 1], _IDX[:, 2]


def _tri_areas(P: np.ndarray) -> np.ndarray:
    a = P[_I0]
    b = P[_I1]
    c = P[_I2]
    return 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) -
                        (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))


def _hull_area(P: np.ndarray) -> float:
    if _HAVE_SCIPY and len(P) >= 3:
        try:
            return float(_ConvexHull(P).volume)
        except Exception:
            pass
    pts = P[np.lexsort((P[:, 1], P[:, 0]))]

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
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return 1e-12
    h = np.array(hull)
    x, y = h[:, 0], h[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _true_score(P: np.ndarray) -> float:
    ha = _hull_area(P)
    if ha <= 1e-12:
        return 0.0
    return float(_tri_areas(P).min()) / ha


# 8 search directions at 45-degree increments for the greedy polish.
_thetas = np.arange(8) * (np.pi / 4.0)
_DIRS = np.stack([np.cos(_thetas), np.sin(_thetas)], axis=1)


def _greedy_polish(P: np.ndarray, step: float = 0.02,
                   min_step: float = 2e-4) -> np.ndarray:
    """Coordinate-wise greedy pattern search on the exact true score."""
    best = _true_score(P)
    while step > min_step:
        improved = False
        for i in range(n):
            for d in range(8):
                trial = P.copy()
                trial[i] = P[i] + step * _DIRS[d]
                val = _true_score(trial)
                if val > best + 1e-15:
                    P = trial
                    best = val
                    improved = True
        if not improved:
            step *= 0.5
    return P


def _anneal(P: np.ndarray, deadline: float, rng: np.random.Generator,
            T0: float = 0.004, T1: float = 2e-5) -> np.ndarray:
    """Simulated annealing with point moves, affine maps, and global scaling.
    All moves scored with the exact min-triangle/hull-area ratio."""
    best_P = P.copy()
    cur = P.copy()
    cur_sc = _true_score(cur)
    best_sc = cur_sc
    t_start = time.perf_counter()
    budget = max(deadline - t_start, 0.05)
    # estimate iterations by temperature schedule fraction
    it = 0
    MAXIT = 6000
    while True:
        now = time.perf_counter()
        if now >= deadline or it >= MAXIT:
            break
        frac = min((now - t_start) / budget, 1.0)
        T = T0 * (T1 / T0) ** frac
        it += 1
        u = rng.random()
        Q = cur.copy()
        if u < 0.75:
            # single-point move
            i = int(rng.integers(0, n))
            amp = T * (0.5 + rng.random()) * 6.0
            ang = rng.uniform(0, 2 * np.pi)
            Q[i] = Q[i] + amp * np.array([np.cos(ang), np.sin(ang)])
        elif u < 0.90:
            # small affine transform about centroid
            cen = Q.mean(axis=0)
            M = np.eye(2) + T * 4.0 * (rng.standard_normal((2, 2)) * 0.5)
            Q = cen + (Q - cen) @ M.T
        else:
            # global scale about centroid
            cen = Q.mean(axis=0)
            f = 1.0 + rng.uniform(-1, 1) * T * 10.0
            Q = cen + f * (Q - cen)
        sc = _true_score(Q)
        if sc > cur_sc or rng.random() < np.exp(min((sc - cur_sc) / max(T, 1e-12), 0.0)):
            cur = Q
            cur_sc = sc
            if sc > best_sc + 1e-15:
                best_sc = sc
                best_P = Q.copy()
    return best_P


def _seed_configs(rng: np.random.Generator):
    seeds = []
    # 3-fold symmetric seeds (4 base points rotated -> 12 + center)
    th = 2.0 * np.pi / 3.0
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s], [s, c]])
    ang = np.pi / 2.0
    for r4 in ([1.0, 0.72, 0.45, 0.2], [1.0, 0.8, 0.55, 0.3],
               [0.95, 0.7, 0.42, 0.1]):
        B = np.array([[r * np.cos(ang - k * np.pi / 9.0),
                       r * np.sin(ang - k * np.pi / 9.0)] for k, r in enumerate(r4)])
        P = np.vstack([B, B @ R.T, B @ (R @ R).T, [[0.0, 0.0]]])
        seeds.append(P)
    # ring-like seeds with jittered radii
    for base_r in (0.95, 0.75):
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        P = np.column_stack([base_r * np.cos(t), base_r * np.sin(t)])
        seeds.append(P)
    # two-ring seeds: 6 outer + 6 inner + center
    for rin in (0.35, 0.5, 0.62):
        t1 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        t2 = t1 + np.pi / 6.0
        P = np.vstack([np.column_stack([np.cos(t1), np.sin(t1)]),
                       np.column_stack([rin * np.cos(t2), rin * np.sin(t2)]),
                       [[0.0, 0.0]]])
        seeds.append(P)
    # fully random seeds
    for _ in range(8):
        P = rng.uniform(-1, 1, (n, 2))
        seeds.append(P)
    return seeds


def heilbronn_convex13() -> np.ndarray:
    """
    Simulated-annealing construction for the 13-point Heilbronn problem.
    Multiple deterministic seeds are annealed (point moves + affine maps +
    global scaling, scored by the exact min-triangle/hull-area ratio) under
    a wall-clock budget, then greedily polished. Returns the best found
    configuration of exactly 13 points.
    """
    t0 = time.perf_counter()
    TOTAL = 6.0  # seconds of search budget
    rng = np.random.default_rng(seed=42)

    seeds = _seed_configs(rng)
    ns = len(seeds)
    # allocate budget: 60% annealing across seeds, rest for polish + round 2
    anneal_deadline_all = t0 + 0.55 * TOTAL
    per_seed = max((anneal_deadline_all - t0) / ns, 0.05)

    best_P, best_sc = None, -np.inf
    elites = []
    for si, P0 in enumerate(seeds):
        try:
            dl = min(t0 + (si + 1) * per_seed, anneal_deadline_all)
            if time.perf_counter() > dl:
                dl = time.perf_counter() + 0.03
            P = _anneal(np.array(P0, dtype=float), dl, rng)
            P = _greedy_polish(P)
            sc = _true_score(P)
            elites.append((sc, P))
            if sc > best_sc:
                best_sc, best_P = sc, P.copy()
        except Exception:
            continue

    # Round 2: re-anneal top-3 elites with hotter schedule for diversification
    try:
        elites.sort(key=lambda t: -t[0])
        for rank, (sc0, elite) in enumerate(elites[:3]):
            dl = min(t0 + 0.85 * TOTAL + rank * 0.2, t0 + TOTAL - 0.4)
            dl = max(dl, time.perf_counter() + 0.1)
            Q = _anneal(elite.copy(), dl, rng, T0=0.002)
            Q = _greedy_polish(Q)
            sc = _true_score(Q)
            if sc > best_sc + 1e-15:
                best_sc, best_P = sc, Q.copy()
    except Exception:
        pass

    if best_P is None:
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_P = np.column_stack([np.cos(t), np.sin(t)])

    # numerical safety: no duplicates
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(best_P[i] - best_P[j]) < 1e-9:
                best_P[j, 0] += 1e-6 * (j + 1)

    assert np.all(np.isfinite(best_P))
    assert best_P.shape == (n, 2)
    return best_P

# EVOLVE-BLOCK-END