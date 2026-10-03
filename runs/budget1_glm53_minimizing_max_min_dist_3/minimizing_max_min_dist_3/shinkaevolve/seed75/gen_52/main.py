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

    # Start from an icosahedron (12 well-spread vertices) + 2 extra points
    phi = (1 + np.sqrt(5)) / 2
    verts = []
    for a, b in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        pass
    base = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ], dtype=float)
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    # two extra points near opposite poles, slightly perturbed to break symmetry
    extra = np.array([[0.0, 0.0, 1.05], [0.1, -0.1, -1.05]])
    points = np.vstack([base, extra])

    def pairwise(pts):
        diff = pts[:, None, :] - pts[None, :, :]
        return np.sqrt(np.sum(diff * diff, axis=-1) + 1e-12)

    def objective(flat, tau):
        P = flat.reshape(n, d)
        D = pairwise(P)
        iu = np.triu_indices(n, 1)
        dv = D[iu]
        smin = -tau * np.log(np.sum(np.exp(-dv / tau)))
        smax = tau * np.log(np.sum(np.exp(dv / tau)))
        return -(smin / smax)

    # Coordinate descent over annealed temperatures with local random restarts
    best_pts = points.copy()
    def true_ratio(P):
        D = pairwise(P)[np.triu_indices(n, 1)]
        return D.min() / D.max()

    def normalize(P):
        P = P - P.mean(axis=0)
        Dm = pairwise(P)[np.triu_indices(n, 1)].max()
        return P / Dm

    best_pts = normalize(best_pts)
    best_val = true_ratio(best_pts)
    rng = np.random.RandomState(0)
    # second, different basin: jittered points on the unit sphere
    alt = rng.randn(n, d)
    alt /= np.linalg.norm(alt, axis=1, keepdims=True)
    try:
        from scipy.optimize import minimize
        for trial in range(10):
            if trial == 0:
                P0 = best_pts.copy()
            elif trial == 1:
                P0 = normalize(alt)
            else:
                P0 = normalize(best_pts + 0.08 * rng.randn(n, d))
            for tau in [0.1, 0.05, 0.02, 0.01, 0.005]:
                res = minimize(objective, P0.ravel(), args=(tau,),
                               method="L-BFGS-B",
                               options={"maxiter": 600})
                P0 = normalize(res.x.reshape(n, d))
            v = true_ratio(P0)
            if v > best_val:
                best_val = v
                best_pts = P0.copy()
            # occasionally try a strongly perturbed restart to explore other basins
            if trial in (3, 6, 9):
                P0 = normalize(best_pts + 0.25 * rng.randn(n, d))
                for tau in [0.05, 0.01, 0.005]:
                    res = minimize(objective, P0.ravel(), args=(tau,),
                                   method="L-BFGS-B",
                                   options={"maxiter": 600})
                    P0 = normalize(res.x.reshape(n, d))
                v = true_ratio(P0)
                if v > best_val:
                    best_val = v
                    best_pts = P0.copy()
    except Exception:
        # scipy unavailable: fall back to simple centroid-based relaxation
        for _ in range(200):
            D = pairwise(best_pts)
            iu = np.triu_indices(n, 1)
            dmin = D[iu].min()
            dmax = D[iu].max()
            for i in range(n):
                d_i = np.delete(D[i], i)
                nearest = np.argmin(np.delete(D[i], i))
                others = np.delete(np.arange(n), i)
                best_pts[i] += 0.02 * (best_pts[i] - best_pts[others[nearest]]) / max(dmin, 1e-9)

    points = best_pts

    # Greedy hill-climbing polish directly on the true ratio
    best_pts = normalize(best_pts)
    best_val = true_ratio(best_pts)
    step = 0.01
    while step > 1e-4:
        improved = False
        for i in range(n):
            for axis in range(d):
                for sgn in (1, -1):
                    cand = best_pts.copy()
                    cand[i, axis] += sgn * step
                    cand = normalize(cand)
                    v = true_ratio(cand)
                    if v > best_val + 1e-9:
                        best_val = v
                        best_pts = cand
                        improved = True
        if not improved:
            step *= 0.5
    points = best_pts

    # Final safety stage: targeted perturbation of binding-constraint points
    iu = np.triu_indices(n, 1)
    D = pairwise(best_pts)
    dv = D[iu]
    dmin = dv.min()
    # identify points involved in near-minimal distance pairs
    tight_idx = np.where(dv < dmin + 0.05 * (dv.max() - dmin) + 1e-9)[0]
    involved = np.unique(np.concatenate([iu[0][tight_idx], iu[1][tight_idx]]))
    rng2 = np.random.RandomState(123)
    guard_pts = best_pts.copy()
    guard_val = best_val
    for _ in range(8):
        cand = best_pts.copy()
        for i in involved:
            u = rng2.randn(d)
            u /= max(np.linalg.norm(u), 1e-12)
            cand[i] += 0.01 * u
        cand = normalize(cand)
        v = true_ratio(cand)
        if v > guard_val:
            guard_val = v
            guard_pts = cand
    best_pts = guard_pts
    best_val = guard_val
    points = best_pts

    # Normalize: translate centroid to origin, scale so max distance = 1
    points = points - points.mean(axis=0)
    D = pairwise(points)[np.triu_indices(n, 1)]
    points = points / D.max()

    return points


# EVOLVE-BLOCK-END