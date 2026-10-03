# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """Construct 14 points in 3D maximizing min pairwise / max pairwise distance."""
    from scipy.optimize import minimize
    from scipy.special import logsumexp

    n, d = 14, 3
    iu = np.triu_indices(n, 1)

    def dists(p):
        diff = p[:, None, :] - p[None, :, :]
        return np.sqrt(np.maximum((diff * diff).sum(-1), 1e-16))[iu]

    def ratio(p):
        D = dists(p)
        return D.min() / D.max()

    def norm_pts(p):
        p = p - p.mean(axis=0)
        s = np.linalg.norm(p, axis=1).max()
        return p / max(s, 1e-12)

    rng = np.random.default_rng(0)
    best_pts, best_r = None, -1.0

    # --- Seed set: dense D6 family scan + icosahedron + random ---
    def d6_seed(h, r, pz):
        t = 2 * np.pi * np.arange(6) / 6.0
        r1 = np.stack([r * np.cos(t), r * np.sin(t), np.full(6, h)], axis=1)
        r2 = np.stack([r * np.cos(t + np.pi / 6), r * np.sin(t + np.pi / 6),
                       np.full(6, -h)], axis=1)
        return np.vstack([r1, r2, [[0, 0, pz], [0, 0, -pz]]])

    seeds = [d6_seed(h, r, pz) for h in (0.25, 0.3, 0.35, 0.4, 0.45, 0.5)
             for r in (0.9, 1.0, 1.1) for pz in (1.0, 1.1)]
    # icosahedron + 2 poles
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]], dtype=float)
    ico /= np.linalg.norm(ico, axis=1, keepdims=True)
    for pz in (1.0, 1.15):
        seeds.append(np.vstack([ico, [[0, 0, pz], [0, 0, -pz]]]))
    for _ in range(4):
        p = rng.normal(size=(n, d))
        seeds.append(p / np.linalg.norm(p, axis=1, keepdims=True))

    T_LADDER = (0.06, 0.03, 0.012, 0.005, 0.002, 0.001)

    def anneal(pts, ladder=T_LADDER, maxit=400):
        for T in ladder:
            def loss(x, T=T):
                p = x.reshape(n, d)
                p = p - p.mean(axis=0)
                D = dists(p)
                D = D / max(D.max(), 1e-9)
                smin = -T * logsumexp(-D / T)
                smax = T * logsumexp(D / T)
                return -(smin / smax)
            res = minimize(loss, pts.ravel(), method="L-BFGS-B",
                           options={"maxiter": maxit, "maxfun": 4 * maxit})
            q = res.x.reshape(n, d)
            if np.isfinite(q).all():
                pts = norm_pts(q)
        return pts

    # --- Targeted pair-move polish: push closest pair apart, pull diameter in ---
    def pair_polish(pts, steps=(0.02, 0.01, 0.005, 0.002), rounds=40):
        pts = pts.copy()
        cur = ratio(pts)
        for _ in range(rounds):
            improved = False
            Dfull = np.sqrt(np.maximum(
                ((pts[:, None] - pts[None]) ** 2).sum(-1), 1e-16))
            Dp = Dfull[iu]
            imin, jmin = iu[0][np.argmin(Dp)], iu[1][np.argmin(Dp)]
            imax, jmax = iu[0][np.argmax(Dp)], iu[1][np.argmax(Dp)]
            for s in steps:
                Q = pts.copy()
                u = Q[imin] - Q[jmin]
                u /= (np.linalg.norm(u) + 1e-12)
                Q[imin] += s * u
                Q[jmin] -= s * u
                v = Q[imax] - Q[jmax]
                v /= (np.linalg.norm(v) + 1e-12)
                Q[imax] -= s * v
                Q[jmax] += s * v
                r = ratio(Q)
                if r > cur + 1e-12:
                    pts, cur, improved = Q, r, True
                    break
                # also try pure closest-pair move only
                Q2 = pts.copy()
                Q2[imin] += s * u
                Q2[jmin] -= s * u
                r2 = ratio(Q2)
                if r2 > cur + 1e-12:
                    pts, cur, improved = Q2, r2, True
                    break
            if not improved:
                break
        return pts

    # --- Main optimization over seeds ---
    for P0 in seeds:
        pts = anneal(norm_pts(np.asarray(P0, float)))
        r = ratio(pts)
        if r > best_r:
            best_r, best_pts = r, pts.copy()

    # --- Perturb-restart cycles with pair polish ---
    for trial in range(12):
        scale = 0.015 + 0.012 * trial
        P0 = best_pts + scale * rng.normal(size=(n, d)) / np.sqrt(3)
        pts = anneal(norm_pts(P0), ladder=(0.02, 0.008, 0.003, 0.001), maxit=300)
        pts = pair_polish(pts)
        r = ratio(pts)
        if r > best_r:
            best_r, best_pts = r, pts.copy()

    # Final full anneal + polish from the best
    pts = pair_polish(anneal(best_pts, ladder=(0.004, 0.001), maxit=600))
    if ratio(pts) > best_r:
        best_pts = pts

    points = np.asarray(best_pts, dtype=float)
    points = points - points.mean(axis=0)
    Dm = dists(points).max()
    if (not np.isfinite(points).all()) or points.shape != (n, d) or Dm <= 0:
        points = np.random.default_rng(1).normal(size=(n, d))
    else:
        points = points / Dm
    return points


# EVOLVE-BLOCK-END