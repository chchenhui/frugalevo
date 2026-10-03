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

    iu = np.triu_indices(n, 1)

    def score_sq(pts):
        dm = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        dists = dm[iu]
        dmax = dists.max()
        if dmax <= 0:
            return 0.0
        return (dists.min() / dmax) ** 2

    def sphere(p):
        p = p - p.mean(axis=0)
        r = np.linalg.norm(p, axis=1, keepdims=True)
        r[r == 0] = 1.0
        return p / r

    rng = np.random.default_rng(42)

    # ---- diverse structured + random seeds ----
    seeds = []
    for _ in range(8):
        seeds.append(sphere(rng.standard_normal((n, d))))
    # antipodal pairs (7 random directions)
    for _ in range(4):
        dirs = sphere(rng.standard_normal((n // 2, d)))
        seeds.append(np.vstack([dirs, -dirs]))
    # icosahedron (12 verts) + antipodal extras along coordinate axes
    phi = (1 + np.sqrt(5)) / 2
    ico = sphere(np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ], dtype=float))
    for extra in [np.array([[0, 0, 1.2], [0, 0, -1.2]]),
                  np.array([[1.2, 0, 0], [-1.2, 0, 0]]),
                  np.array([[0, 1.2, 0], [0, -1.2, 0]])]:
        seeds.append(np.vstack([ico, sphere(extra)]))
    # cube vertices + center + extras
    lat = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1)
                    for z in (-1, 1)], dtype=float)
    extra5 = sphere(rng.standard_normal((5, d))) * 0.5
    seeds.append(sphere(np.vstack([lat, [[0, 0, 0]], extra5])))

    # ---- fast stochastic hill climb on the true objective ----
    def hill_climb(pts, iters, step0):
        s = score_sq(pts)
        step = step0
        for it in range(iters):
            i = int(rng.integers(n))
            old = pts[i].copy()
            pert = rng.standard_normal(d)
            pert /= np.linalg.norm(pert)
            pts[i] = old + pert * step * rng.random()
            sn = score_sq(pts)
            if sn >= s:
                s = sn
            else:
                pts[i] = old
            if it % 400 == 399:
                step *= 0.7
        return pts, s

    candidates = []
    for seed_pts in seeds:
        pts, s = hill_climb(seed_pts.copy(), 700, 0.15)
        candidates.append((s, pts))

    candidates.sort(key=lambda c: -c[0])
    top = candidates[:6]

    # ---- SLSQP refinement: maximize t s.t. t <= |xi-xj|^2 <= 1 ----
    def obj(z):
        return -z[-1]

    def obj_grad(z):
        g = np.zeros_like(z)
        g[-1] = -1.0
        return g

    def make_constraints():
        cons = []
        for i in range(n):
            for j in range(i + 1, n):
                def fmin(z, i=i, j=j):
                    return np.sum((z[3*i:3*i+3] - z[3*j:3*j+3])**2) - z[-1]
                def gmin(z, i=i, j=j):
                    g = np.zeros(len(z))
                    di = 2.0 * (z[3*i:3*i+3] - z[3*j:3*j+3])
                    g[3*i:3*i+3] = di
                    g[3*j:3*j+3] = -di
                    g[-1] = -1.0
                    return g
                def fmax(z, i=i, j=j):
                    return 1.0 - np.sum((z[3*i:3*i+3] - z[3*j:3*j+3])**2)
                def gmax(z, i=i, j=j):
                    g = np.zeros(len(z))
                    di = 2.0 * (z[3*i:3*i+3] - z[3*j:3*j+3])
                    g[3*i:3*i+3] = -di
                    g[3*j:3*j+3] = di
                    return g
                cons.append({'type': 'ineq', 'fun': fmin, 'jac': gmin})
                cons.append({'type': 'ineq', 'fun': fmax, 'jac': gmax})
        return cons

    constraints = make_constraints()

    best_pts = None
    best_s = -1.0
    for s0, pts0 in top:
        # normalize so diameter^2 = 1 before SLSQP
        dm = np.linalg.norm(pts0[:, None, :] - pts0[None, :, :], axis=-1)
        p = pts0 / dm.max()
        dm = dm / dm.max()
        t0 = dm[iu].min() ** 2
        z0 = np.concatenate([p.ravel(), [t0]])
        try:
            res = minimize(obj, z0, jac=obj_grad,
                           constraints=constraints,
                           method='SLSQP',
                           options={'maxiter': 200, 'ftol': 1e-12})
            pts = res.x[:-1].reshape(n, d)
            s = score_sq(pts)
            if s > best_s:
                best_s = s
                best_pts = pts.copy()
        except Exception:
            pass
        # also keep the unrefined candidate in case SLSQP failed
        s = score_sq(p)
        if s > best_s:
            best_s = s
            best_pts = p.copy()

    # final quick polish on the best configuration
    best_pts, best_s = hill_climb(best_pts.copy(), 1500, 0.01)

    # normalize: scale so max pairwise distance equals 1 (objective invariant)
    dm = np.linalg.norm(best_pts[:, None, :] - best_pts[None, :, :], axis=-1)
    best_pts = best_pts / dm.max()

    return np.asarray(best_pts, dtype=float)


# EVOLVE-BLOCK-END