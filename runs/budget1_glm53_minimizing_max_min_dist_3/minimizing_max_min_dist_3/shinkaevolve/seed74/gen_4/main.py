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

    from scipy.optimize import minimize
    from itertools import combinations

    pairs = list(combinations(range(n), 2))
    ii = np.array([p[0] for p in pairs])
    jj = np.array([p[1] for p in pairs])

    def sq_dists(flat):
        pts = flat.reshape(n, d)
        return np.sum((pts[ii] - pts[jj]) ** 2, axis=1)

    def objective(flat, k=40.0):
        # soft-min of squared distances -> maximize min distance
        sd = sq_dists(flat)
        # smooth min: -(1/k) log sum exp(-k * sd)
        m = np.min(sd)
        softmin = m - np.log(np.sum(np.exp(-k * (sd - m)))) / k
        return -softmin

    def grad(flat, k=40.0):
        pts = flat.reshape(n, d)
        sd = sq_dists(flat)
        m = np.min(sd)
        w = np.exp(-k * (sd - m))
        w /= w.sum()
        g = np.zeros((n, d))
        for a, (p, q) in enumerate(pairs):
            diff = 2.0 * w[a] * (pts[p] - pts[q])
            g[p] += diff
            g[q] -= diff
        return g.ravel()

    def normalize(flat):
        pts = flat.reshape(n, d)
        sd = sq_dists(flat)
        pts = pts - pts.mean(axis=0)
        pts = pts / np.sqrt(sd.max())
        return pts.ravel()

    def true_score(flat):
        pts = flat.reshape(n, d)
        dm = np.sqrt(sq_dists(flat))
        return dm.min() / dm.max()

    # structured starts
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ])
    starts = []
    # icosahedron vertices + 2 poles
    starts.append(np.vstack([ico / np.linalg.norm(ico, axis=1, keepdims=True),
                             [[0, 0, 1.6], [0, 0, -1.6]]))
    # cuboctahedron (12) + 2 poles
    cub = np.array([[s1, s2, s3] for s1 in (-1, 1) for s2 in (-1, 1)
                    for s3 in (-1, 1)], dtype=float)
    cub = np.vstack([cub[np.abs(cub).sum(axis=1) == 1],
                     np.array([[1, 1, 0], [1, -1, 0], [-1, 1, 0],
                               [-1, -1, 0], [1, 0, 1], [1, 0, -1],
                               [-1, 0, 1], [-1, 0, -1], [0, 1, 1],
                               [0, 1, -1], [0, -1, 1], [0, -1, -1]],
                              dtype=float)])
    starts.append(np.vstack([cub, [[0, 0, 1.4], [0, 0, -1.4]]]))

    rng = np.random.default_rng(42)
    for _ in range(8):
        pts = rng.normal(size=(n, d))
        starts.append(pts.ravel())

    best_flat, best = None, -1.0
    for s in starts:
        x0 = normalize(np.asarray(s, dtype=float))
        try:
            res = minimize(objective, x0, jac=grad, method="L-BFGS-B",
                           options={"maxiter": 2000})
            cand = normalize(res.x)
        except Exception:
            cand = x0
        sc = true_score(cand)
        if sc > best:
            best, best_flat = sc, cand

    return best_flat.reshape(n, d)


# EVOLVE-BLOCK-END