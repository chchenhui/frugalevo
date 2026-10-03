# EVOLVE-BLOCK-START
import numpy as np

N, D = 14, 3
_IU = np.triu_indices(N, 1)
PI_, PJ = _IU


def _true_score(points):
    d2 = np.sum((points[PI_] - points[PJ]) ** 2, axis=1)
    dx2 = d2.max()
    if dx2 <= 0:
        return 0.0
    return d2.min() / dx2


def _norm_diameter(P):
    """Center and rescale so max pairwise distance == 1."""
    P = P - P.mean(axis=0)
    dm = np.sqrt(np.sum((P[PI_] - P[PJ]) ** 2, axis=1).max())
    if dm > 1e-15:
        P = P / dm
    return P


def _anneal(P0, iters=1200, seed=0, t0=0.06, t1=1e-4):
    """Force-directed simulated annealing on the unit-diameter sphere.

    Pairs closer than an annealed cutoff repel; configuration is
    re-projected to the unit sphere and rescaled to unit diameter
    each iteration, so the ratio dmin/dmax == dmin throughout.
    """
    rng = np.random.default_rng(seed)
    # project to unit sphere
    P = np.asarray(P0, dtype=float).copy()
    nr = np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)
    P = P / nr

    for it in range(iters):
        frac = it / max(iters - 1, 1)
        temp = t0 * (t1 / t0) ** frac  # exponential cooling
        diff = P[PI_] - P[PJ]
        d2 = np.sum(diff * diff, axis=1)
        d = np.sqrt(d2)

        # target: push all pairs up to current min plus slack
        dcur_min = d.min()
        cutoff = dcur_min + temp
        mask = d < cutoff
        if not mask.any():
            # nothing to fix: random thermal kick on a random point
            k = rng.integers(N)
            v = rng.standard_normal(D) * temp * 0.5
            P[k] += v
        else:
            # repulsive impulses proportional to shortfall
            shortfall = (cutoff - d[mask])[:, None] * diff[mask] / np.maximum(d[mask, None], 1e-12)
            imp = np.zeros((N, D))
            np.add.at(imp, PI_[mask], shortfall)
            np.add.at(imp, PJ_[mask], -shortfall)
            P += 0.5 * imp
            # thermal noise
            P += temp * 0.1 * rng.standard_normal((N, D))

        # project back to sphere
        nr = np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)
        P = P / nr
        # occasional re-normalization of diameter (spherical shell keeps
        # things bounded; final ratio computed on raw points anyway)
    return _norm_diameter(P)


def _polish(P, rounds=60):
    """Greedy steepest-ascent on the exact squared-ratio objective:
    move the two closest points apart along their connecting axis and
    pull the farthest pair slightly in; keep if the true score improves."""
    P = _norm_diameter(P)
    best_s = _true_score(P)
    step = 0.01
    for _ in range(rounds):
        d2 = np.sum((P[PI_] - P[PJ]) ** 2, axis=1)
        imin, imax = int(np.argmin(d2)), int(np.argmax(d2))
        i0, j0 = PI_[imin], PJ_[imin]
        i1, j1 = PI_[imax], PJ_[imax]
        u = (P[i0] - P[j0]); nu = np.linalg.norm(u)
        if nu < 1e-15:
            break
        u /= nu
        v = (P[i1] - P[j1]); nv = np.linalg.norm(v)
        v /= max(nv, 1e-15)
        Q = P.copy()
        Q[i0] += step * u; Q[j0] -= step * u
        Q[i1] -= step * v; Q[j1] += step * v
        Q = _norm_diameter(Q)
        s = _true_score(Q)
        if s > best_s:
            best_s = s
            P = Q
        else:
            step *= 0.7
            if step < 1e-6:
                break
    return P, best_s


def _structured_inits():
    inits = []
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ], dtype=float)
    ico /= np.linalg.norm(ico, axis=1, keepdims=True)
    inits.append(np.vstack([ico, [[0, 0, 1.3], [0, 0, -1.3]]]))
    inits.append(np.vstack([ico, [[0, 0, 1.0], [0, 0, -1.0]]]))
    cube = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], dtype=float)
    cube /= np.linalg.norm(cube, axis=1, keepdims=True)
    octa = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float)
    inits.append(np.vstack([cube, 1.05 * octa]))
    inits.append(np.vstack([1.05 * cube, octa]))
    t = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    r1 = np.stack([np.cos(t), np.sin(t), 0.5 * np.ones(7)], axis=1)
    r2 = np.stack([np.cos(t + np.pi / 7), np.sin(t + np.pi / 7), -0.5 * np.ones(7)], axis=1)
    inits.append(np.vstack([r1, r2]))
    t = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    ring1 = np.stack([np.cos(t), np.sin(t), 0.55 * np.ones(6)], axis=1)
    ring2 = np.stack([np.cos(t + np.pi / 6), np.sin(t + np.pi / 6), -0.55 * np.ones(6)], axis=1)
    inits.append(np.vstack([ring1, ring2, [[0, 0, 1.6], [0, 0, -1.6]]]))
    # Fibonacci sphere
    k = np.arange(N) + 0.5
    ph = np.arccos(1.0 - 2.0 * k / N)
    th = np.pi * (1.0 + 5.0 ** 0.5) * k
    inits.append(np.stack([np.cos(th) * np.sin(ph), np.sin(th) * np.sin(ph), np.cos(ph)], axis=1))
    # rotated variants of the best seed (rotation-invariant basins)
    rng = np.random.default_rng(99)
    for _ in range(3):
        Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        rot = np.vstack([ico, [[0, 0, 1.3], [0, 0, -1.3]]])
        rot = rot @ Q
        inits.append(rot)
    return inits


def min_max_dist_dim3_14() -> np.ndarray:
    target = 0.2400
    best_pts, best = None, -1.0

    starts = []
    for p in _structured_inits():
        nr = np.linalg.norm(p, axis=1, keepdims=True)
        starts.append(p / np.maximum(nr, 1e-12))
    for seed in range(24):
        v = np.random.RandomState(seed).randn(N, D)
        starts.append(v / np.linalg.norm(v, axis=1, keepdims=True))

    # Stage 1: quick annealing screen of all starts (short runs)
    screened = []
    for idx, P0 in enumerate(starts):
        P = _anneal(P0, iters=300, seed=1000 + idx)
        screened.append((_true_score(P), P, idx))
    screened.sort(key=lambda x: -x[0])

    # Stage 2: long annealing + polish on the top candidates
    for s0, P0, idx in screened[:8]:
        P = _anneal(P0, iters=2000, seed=2000 + idx, t0=0.04)
        P, s = _polish(P)
        if s > best:
            best, best_pts = s, P
        if best >= target:
            break

    # Stage 3: extra long anneals from the incumbent (reheat-and-cool)
    if best_pts is not None:
        rng = np.random.default_rng(5)
        for k in range(6):
            P0 = best_pts + 0.05 * rng.standard_normal((N, D))
            nr = np.maximum(np.linalg.norm(P0, axis=1, keepdims=True), 1e-12)
            P0 = P0 / nr
            P = _anneal(P0, iters=1500, seed=3000 + k, t0=0.05)
            P, s = _polish(P)
            if s > best:
                best, best_pts = s, P

    if best_pts is None:
        np.random.seed(42)
        best_pts = np.random.randn(N, D)

    best_pts = _norm_diameter(np.asarray(best_pts, dtype=float))
    return np.ascontiguousarray(best_pts, dtype=float)
# EVOLVE-BLOCK-END