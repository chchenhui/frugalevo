# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Construct 14 points in 3D maximizing dmin/dmax.

    Approach: the ratio objective is scale invariant, so we normalize
    dmax = 2 after every step but do NOT constrain points to a sphere --
    the optimal configuration for dmin/dmax is generally non-spherical
    (unlike Tammes packing).  We run many restarts (structured inits:
    staggered heptagon bi-layers at several latitudes, icosahedron+2,
    plus random) of soft-min (log-sum-exp) ascent on squared distances
    with annealed sharpness, rescaling each step so the diameter equals 2.
    The best candidate is polished with a hard ascent that pushes all
    near-minimal pairs apart AND pulls the diameter pairs together
    (shrinking dmax raises the ratio directly), tracking the best
    dmin^2/dmax^2 throughout.

    Returns:
        points: np.ndarray of shape (14, 3)
    """
    n = 14
    rng = np.random.default_rng(42)
    II, JJ = np.triu_indices(n, 1)

    def sq_dists(P):
        d = P[:, None, :] - P[None, :, :]
        return np.sum(d * d, axis=-1)

    def offdiag(P):
        return sq_dists(P)[II, JJ]

    def ratio_sq(P):
        off = offdiag(P)
        return off.min() / off.max()

    def normalize(P):
        # rescale so that the maximum pairwise distance equals 2
        mx = np.sqrt(offdiag(P).max())
        if mx > 0:
            P = P * (2.0 / mx)
        return P

    def grad_soft_min(P, t):
        sq = offdiag(P)
        m = sq.min() / t
        w = np.exp(-(sq / t - m))
        w /= w.sum()
        diff = P[II] - P[JJ]
        G = np.zeros_like(P)
        np.add.at(G, II, 2.0 * w[:, None] * diff)
        np.add.at(G, JJ, -2.0 * w[:, None] * diff)
        return G

    def structured_init(kind):
        if kind < 4:
            # two staggered regular heptagons on parallel latitudes,
            # varying the latitude (free radial optimization will move
            # them off the sphere if that helps the ratio)
            z = (0.35, 0.45, 0.55, 0.65)[kind]
            r = np.sqrt(max(1e-9, 1.0 - z * z))
            P = []
            for zz, off in ((z, 0.0), (-z, np.pi / 7)):
                for i in range(7):
                    a = 2 * np.pi * i / 7 + off
                    P.append([r * np.cos(a), r * np.sin(a), zz])
            return np.array(P, dtype=float)
        elif kind == 4:
            # icosahedron vertices + 2 random extras
            p = (1 + np.sqrt(5)) / 2
            V = np.array([[-1, 0, p], [1, 0, p], [-1, 0, -p], [1, 0, -p],
                          [0, p, -1], [0, p, 1], [0, -p, -1], [0, -p, 1],
                          [p, -1, 0], [p, 1, 0], [-p, -1, 0], [-p, 1, 0]],
                         dtype=float)
            V /= np.linalg.norm(V, axis=1, keepdims=True)
            extra = rng.standard_normal((2, 3))
            extra /= np.linalg.norm(extra, axis=1, keepdims=True)
            return np.vstack([V, extra])
        else:
            # cuboctahedron-like: 12 vertices of cuboctahedron + 2 extras
            a = np.array([[1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
                          [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
                          [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1]],
                         dtype=float)
            extra = rng.standard_normal((2, 3))
            return np.vstack([a, extra])

    best_P = None
    best_val = -np.inf

    inits = []
    for kind in range(6):
        for _ in range(8):
            inits.append(structured_init(kind) + 0.01 * rng.standard_normal((n, 3)))
    for _ in range(110):
        inits.append(rng.standard_normal((n, 3)))

    def anneal(X, t_list, iters, step):
        """Soft-min ascent; track best ratio seen along the trajectory."""
        nonlocal best_val, best_P
        X = normalize(X.copy())
        for t in t_list:
            for _ in range(iters):
                G = grad_soft_min(X, t)
                ng = np.linalg.norm(G)
                if ng < 1e-14:
                    break
                X = normalize(X + step * G / ng)
                val = ratio_sq(X)
                if val > best_val:
                    best_val = val
                    best_P = X.copy()
        return X

    for X0 in inits:
        anneal(X0, (0.2, 0.1, 0.05, 0.02, 0.01, 0.005), 150, 0.05)

    # polish: push near-minimal pairs apart, pull diameter pairs together
    X = best_P.copy()
    step = 0.02
    for it in range(4000):
        off = offdiag(X)
        mn = off.min()
        mx = off.max()
        mmask = off < mn * (1.0 + 5e-4)   # pairs at the minimum
        xmask = off > mx * (1.0 - 5e-4)  # pairs at the maximum
        G = np.zeros_like(X)
        dmin_diff = X[II[mmask]] - X[JJ[mmask]]
        np.add.at(G, II[mmask], dmin_diff)
        np.add.at(G, JJ[mmask], -dmin_diff)
        dmax_diff = X[II[xmask]] - X[JJ[xmask]]
        # shrink the diameter: move its endpoints toward each other
        np.add.at(G, II[xmask], -0.5 * dmax_diff)
        np.add.at(G, JJ[xmask], 0.5 * dmax_diff)
        ng = np.linalg.norm(G)
        if ng < 1e-14:
            break
        X = normalize(X + step * G / ng)
        if it % 400 == 399:
            step *= 0.75
        val = ratio_sq(X)
        if val > best_val:
            best_val = val
            best_P = X.copy()

    # basin hopping: perturb the incumbent, re-anneal at low sharpness,
    # re-polish; accept improvements.  Escapes flat/plateau local optima
    # where the soft-min gradient vanishes.
    for hop in range(150):
        Y = best_P + (0.006 + 0.006 * rng.random()) * rng.standard_normal((n, 3))
        Y = anneal(Y, (0.02, 0.01, 0.005, 0.002), 200, 0.04)
        # quick polish of the hopped candidate
        step = 0.02
        for it in range(2000):
            off = offdiag(Y)
            mn = off.min()
            mx = off.max()
            mmask = off < mn * (1.0 + 1e-3)
            xmask = off > mx * (1.0 - 1e-3)
            G = np.zeros_like(Y)
            dmin_diff = Y[II[mmask]] - Y[JJ[mmask]]
            np.add.at(G, II[mmask], dmin_diff)
            np.add.at(G, JJ[mmask], -dmin_diff)
            dmax_diff = Y[II[xmask]] - Y[JJ[xmask]]
            np.add.at(G, II[xmask], -0.5 * dmax_diff)
            np.add.at(G, JJ[xmask], 0.5 * dmax_diff)
            ng = np.linalg.norm(G)
            if ng < 1e-14:
                break
            Y = normalize(Y + step * G / ng)
            if it % 300 == 299:
                step *= 0.7
            val = ratio_sq(Y)
            if val > best_val:
                best_val = val
                best_P = Y.copy()

    # final ultra-fine polish: geometric step decay on the incumbent,
    # pushing minimal pairs apart and shrinking the diameter with very
    # tight active-pair masks. This extracts the last few digits of
    # dmin^2/dmax^2 without disturbing the configuration's structure.
    X = best_P.copy()
    step = 0.01
    for it in range(15000):
        off = offdiag(X)
        mn = off.min()
        mx = off.max()
        mmask = off < mn * (1.0 + 1e-5)
        xmask = off > mx * (1.0 - 1e-5)
        G = np.zeros_like(X)
        dmin_diff = X[II[mmask]] - X[JJ[mmask]]
        np.add.at(G, II[mmask], dmin_diff)
        np.add.at(G, JJ[mmask], -dmin_diff)
        dmax_diff = X[II[xmask]] - X[JJ[xmask]]
        np.add.at(G, II[xmask], -0.5 * dmax_diff)
        np.add.at(G, JJ[xmask], 0.5 * dmax_diff)
        ng = np.linalg.norm(G)
        if ng < 1e-14:
            break
        X = normalize(X + step * G / ng)
        step *= 0.99965
        val = ratio_sq(X)
        if val > best_val:
            best_val = val
            best_P = X.copy()

    return best_P.astype(float)


# EVOLVE-BLOCK-END
