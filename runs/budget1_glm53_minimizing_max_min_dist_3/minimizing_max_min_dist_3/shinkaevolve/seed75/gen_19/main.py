# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y) coordinates of the 14 points.

    """

    n = 14
    d = 3

    np.random.seed(42)

    # Build a library of structured seeds, all known families for the
    # 14-point 3D max-min-ratio problem.
    phi = (1 + np.sqrt(5)) / 2
    base = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ], dtype=float)
    base /= np.linalg.norm(base, axis=1, keepdims=True)

    seeds = []
    # icosahedron + 2 poles (several pole heights)
    for pz in (1.0, 1.05, 1.15):
        seeds.append(np.vstack([base, [[0, 0, pz], [0, 0, -pz]]]))
    # D6 family: two staggered hexagonal rings + 2 poles (best of prior gens)
    def d6_points(p, h, r):
        t = 2 * np.pi * np.arange(6) / 6.0
        ring1 = np.stack([r * np.cos(t), r * np.sin(t), np.full(6, h)], axis=1)
        ring2 = np.stack([r * np.cos(t + np.pi / 6), r * np.sin(t + np.pi / 6),
                          np.full(6, -h)], axis=1)
        poles = np.array([[0.0, 0.0, p], [0.0, 0.0, -p]])
        return np.vstack([ring1, ring2, poles])
    for p, h, r in [(1.0, 0.35, 1.0), (0.9, 0.3, 0.9), (1.1, 0.4, 1.05)]:
        seeds.append(d6_points(p, h, r))
    # two twisted heptagon rings
    for twist in (0.0, np.pi / 7, 2 * np.pi / 7):
        t = 2 * np.pi * np.arange(7) / 7
        ring1 = np.stack([np.cos(t), np.sin(t), np.full(7, -0.5)], axis=1)
        ring2 = np.stack([np.cos(t + twist), np.sin(t + twist), np.full(7, 0.5)], axis=1)
        seeds.append(np.vstack([ring1, ring2]))
    # ring of 12 + 2 poles
    t = 2 * np.pi * np.arange(12) / 12
    seeds.append(np.vstack([
        np.stack([np.cos(t), np.sin(t), np.zeros(12)], axis=1),
        [[0, 0, 1.0], [0, 0, -1.0]]]))
    points = np.vstack([base, [[0.0, 0.0, 1.05], [0.1, -0.1, -1.05]]])
    seeds.append(points)

    def pairwise(pts):
        diff = pts[:, None, :] - pts[None, :, :]
        return np.sqrt(np.sum(diff * diff, axis=-1) + 1e-12)

    iu = np.triu_indices(n, 1)

    def objective(flat, tau):
        P = flat.reshape(n, d)
        D = pairwise(P)
        dv = D[iu]
        smin = -tau * np.log(np.sum(np.exp(-np.clip(dv / tau, -50, 50))))
        smax = tau * np.log(np.sum(np.exp(np.clip(dv / tau, -50, 50))))
        return -(smin / smax)

    # Coordinate descent over annealed temperatures with local random restarts
    best_pts = points.copy()
    def true_ratio(P):
        D = pairwise(P)[np.triu_indices(n, 1)]
        return D.min() / D.max()

    best_val = true_ratio(best_pts)
    rng = np.random.RandomState(0)

    # annealed tau ladder going much lower so the surrogate converges to
    # the true dmin/dmax ratio
    taus = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    try:
        from scipy.optimize import minimize
        have_scipy = True
    except Exception:
        have_scipy = False

    if have_scipy:
        try:
            for S in seeds:
                try:
                    P0 = S.astype(float).copy()
                    P0 -= P0.mean(axis=0)
                    for tau in taus:
                        res = minimize(objective, P0.ravel(), args=(tau,),
                                       method="L-BFGS-B",
                                       options={"maxiter": 400})
                        P0 = res.x.reshape(n, d)
                    v = true_ratio(P0)
                    if v > best_val:
                        best_val = v
                        best_pts = P0.copy()
                except Exception:
                    continue
            # perturb-and-reanneal restarts around the current best
            for trial in range(10):
                try:
                    P0 = best_pts + (0.03 + 0.02 * trial) * rng.randn(n, d)
                    for tau in taus:
                        res = minimize(objective, P0.ravel(), args=(tau,),
                                       method="L-BFGS-B",
                                       options={"maxiter": 300})
                        P0 = res.x.reshape(n, d)
                    v = true_ratio(P0)
                    if v > best_val:
                        best_val = v
                        best_pts = P0.copy()
                except Exception:
                    continue
        except Exception:
            have_scipy = False

    if not have_scipy:
        # scipy unavailable: centroid-based repulsion relaxation fallback
        try:
            for _ in range(300):
                D = pairwise(best_pts)
                dmin = D[iu].min()
                for i in range(n):
                    d_i = np.delete(D[i], i)
                    nearest = np.argmin(d_i)
                    others = np.delete(np.arange(n), i)
                    best_pts[i] += 0.02 * (best_pts[i] - best_pts[others[nearest]]) / max(dmin, 1e-9)
        except Exception:
            pass

    points = best_pts

    # Normalize: translate centroid to origin, scale so max distance = 1
    points = points - points.mean(axis=0)
    D = pairwise(points)[iu]
    points = points / max(D.max(), 1e-12)

    # final robustness guard: always return a valid finite 14x3 array
    if (not np.isfinite(points).all()) or points.shape != (n, d) or D.max() <= 0:
        np.random.seed(42)
        points = np.random.randn(n, d)

    return np.asarray(points, dtype=float)


# EVOLVE-BLOCK-END