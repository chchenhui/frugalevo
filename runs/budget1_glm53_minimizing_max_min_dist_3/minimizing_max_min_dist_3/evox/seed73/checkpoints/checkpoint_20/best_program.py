# EVOLVE-BLOCK-START
import numpy as np


def _ratio(pts):
    """(dmin/dmax)^2 of a point set."""
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    iu = np.triu_indices(len(pts), 1)
    dm = d[iu]
    return dm.min() / dm.max()


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Constructs 14 points in 3D maximizing (dmin/dmax)^2.

    Approach: relax structured antiprism seeds and many random starts on
    the unit sphere with repulsive ~1/d^p forces (Tammes-like spreading),
    keep a POOL of the best resulting configurations from distinct basins,
    then run coarse-to-fine simulated annealing on the true objective in
    unconstrained coordinates from the top candidates, ending with a cold
    greedy polish. The multi-basin annealing is what reached the best
    known ratio (~0.2388); the anneal accepts worsening moves with
    probability exp((r - cur)/T) using per-point adaptive step sizes.

    Returns
        points: np.ndarray of shape (14, 3)
    """
    n, dim = 14, 3
    rng = np.random.default_rng(12345)

    def relax(pts, steps=250, lr=0.02):
        for p in (1.0, 2.0, 4.0, 8.0):
            for _ in range(steps):
                diff = pts[:, None, :] - pts[None, :, :]
                dist = np.sqrt((diff ** 2).sum(-1) + 1e-12)
                inv = dist ** (-(p + 1))
                np.fill_diagonal(inv, 0.0)
                force = (diff * inv[:, :, None]).sum(axis=1)
                nrm = np.linalg.norm(force, axis=1, keepdims=True)
                nrm[nrm == 0] = 1.0
                pts = pts + lr * force / nrm
                pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        return pts

    # Pool of good (ratio, points) candidates from different relaxation
    # basins; annealing from several distinct starts escapes single-basin
    # local optima far better than annealing one seed.
    pool = []

    def consider(pts):
        pool.append((_ratio(pts), pts.copy()))

    # Structured seeds: antiprisms (two staggered m-gons at z = ±h, plus
    # optional poles) are natural near-optimal sphere configurations for
    # n = 14; sweeping the aspect ratio h covers the plausible optima.
    seeds = []
    for m, poles in ((6, True), (7, False)):
        for h in np.linspace(0.15, 0.65, 11):
            r = np.sqrt(max(1.0 - h * h, 1e-6))
            ang = 2.0 * np.pi / m
            pts = [[r * np.cos(i * ang), r * np.sin(i * ang), h]
                   for i in range(m)]
            pts += [[r * np.cos((i + 0.5) * ang),
                     r * np.sin((i + 0.5) * ang), -h]
                    for i in range(m)]
            if poles:
                pts = [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]] + pts
            pts = np.array(pts, dtype=float)
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
            seeds.append(pts)

    for pts0 in seeds:
        consider(relax(pts0.copy()))

    for _ in range(300):
        pts = rng.normal(size=(n, dim))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        consider(relax(pts))

    # Simulated annealing on the true objective (dmin/dmax)^2 over
    # UNCONSTRAINED coordinates (the ratio is scale/translation
    # invariant). Worsening moves are accepted with probability
    # exp((r - cur) / T) so the search can escape local optima;
    # per-point adaptive step sizes are kept. Track the best-seen state.
    def anneal(pts, iters, step0, t0, t1, seed):
        rng = np.random.default_rng(seed)
        pts = pts.copy()
        cur = _ratio(pts)
        loc_pts, loc = pts.copy(), cur
        steps = np.full(n, step0)
        for t in range(iters):
            frac = t / iters
            T = t0 * (t1 / t0) ** frac
            i = rng.integers(n)
            old = pts[i].copy()
            cand = old + steps[i] * rng.normal(size=dim)
            if np.linalg.norm(cand) < 1e-9:
                continue
            pts[i] = cand
            r = _ratio(pts)
            if r >= cur or rng.random() < np.exp((r - cur) / max(T, 1e-12)):
                cur = r
                steps[i] = min(steps[i] * 1.2, step0 * 4)
                if r > loc:
                    loc, loc_pts = r, pts.copy()
            else:
                pts[i] = old
                steps[i] = max(steps[i] * 0.8, step0 * 0.01)
        return loc, loc_pts

    # Anneal from the top few distinct seeds, then refine the overall
    # best with a long coarse-to-fine schedule and a final cold polish.
    pool.sort(key=lambda t: -t[0])
    best, best_pts = pool[0]

    seed = 777
    for _, pts0 in pool[:6]:
        for iters, s0, t0 in ((80000, 0.05, 0.004),
                              (80000, 0.02, 0.0015),
                              (60000, 0.005, 0.0004)):
            seed += 13
            cur, pts = anneal(pts0, iters, s0, t0, 1e-7, seed)
            if cur > best:
                best, best_pts = cur, pts.copy()
            pts0 = pts

    # Cold (T -> 0) greedy polish with tiny adaptive steps; several
    # independent restarts of the polish with progressively smaller
    # step sizes squeeze out the last digits of the ratio.
    for s0, sd in ((0.003, 424242), (0.0015, 11111),
                   (0.0008, 22222), (0.0004, 33333)):
        cur, pts = anneal(best_pts, 80000, s0, 1e-9, 1e-12, sd)
        if cur > best:
            best, best_pts = cur, pts.copy()

    return best_pts


# EVOLVE-BLOCK-END
