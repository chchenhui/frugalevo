# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Constructs exactly 14 points in 3D maximizing (dmin/dmax)^2.

    Approach: Tammes-like spherical optimization.
      1) Deterministic starts: random unit-sphere configurations plus
         structured seeds built from a (perturbed) icosahedron plus extra
         points, which are excellent basins for n=14.
      2) Each start runs projected gradient ascent on a soft-minimum of
         pairwise distances (vectorized pair gradient, annealed temp).
      3) The top few candidates (not just the best) each get an
         adaptive-sigma random-perturbation polish on the sphere.
      4) The overall best then gets a free-space polish (points may leave
         the sphere, dmax renormalized to 1 each step).
    Finally points are rescaled so dmax = 1.
    """

    n = 14
    d = 3
    rng = np.random.default_rng(12345)
    iu = np.triu_indices(n, 1)
    ii, jj = iu

    def ratio_sq(P):
        D = np.sqrt(np.sum((P[:, None, :] - P[None, :, :]) ** 2, axis=-1) + 1e-18)
        dv = D[iu]
        return (dv.min() / dv.max()) ** 2

    def step_grad(P, temp):
        diff = P[:, None, :] - P[None, :, :]
        D = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-18)
        dv = D[iu]
        w = np.exp((1.0 / dv - 1.0 / dv.min()) / temp)
        w /= w.sum()
        w[w < 1e-6] = 0.0
        w /= w.sum()
        u = diff[ii, jj] / (dv[:, None] + 1e-12)
        G = np.zeros_like(P)
        np.add.at(G, ii, w[:, None] * u)
        np.add.at(G, jj, -w[:, None] * u)
        return G

    def project_sphere(P):
        return P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)

    def normalize(P):
        D = np.sqrt(np.sum((P[:, None, :] - P[None, :, :]) ** 2, axis=-1) + 1e-18)
        return P / D[iu].max()

    # Icosahedron vertices (12 points) for structured seeds
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ], dtype=float)
    ico /= np.linalg.norm(ico, axis=1, keepdims=True)

    def structured_seed(k):
        # Structured seed families (good basins for n=14):
        #  fam 0: 12 icosahedron vertices (jittered) + 2 random points
        #  fam 1: icosahedron + one antipodal pair along a random axis
        #  fam 2: 7 random antipodal pairs (dmax is a diameter by design)
        P = np.empty((n, d))
        m = min(12, n)
        fam = k % 3
        if fam == 0:
            P[:m] = ico[:m] + 0.02 * rng.standard_normal((m, d))
            P[m:] = rng.standard_normal((n - m, d))
        elif fam == 1:
            P[:m] = ico[:m] + 0.02 * rng.standard_normal((m, d))
            ax = rng.standard_normal(d)
            ax /= np.linalg.norm(ax)
            P[m] = ax
            if m + 1 < n:
                P[m + 1] = -ax
        else:
            V = rng.standard_normal((n // 2, d))
            V /= np.linalg.norm(V, axis=1, keepdims=True)
            P[0::2] = V
            P[1::2] = -V
            if n % 2:
                P[-1] = rng.standard_normal(d)
        return project_sphere(P)

    # ---- Stage 0: gradient ascent from many starts ----
    candidates = []
    n_random = 60
    n_struct = 30
    steps = 700
    for s in range(n_random + n_struct):
        P = structured_seed(s) if s >= n_random else project_sphere(
            rng.standard_normal((n, d)))
        lr = 0.02
        for it in range(steps):
            temp = 0.05 * (0.995 ** it) + 0.005
            P = project_sphere(P + lr * step_grad(P, temp))
        candidates.append((ratio_sq(P), P))

    candidates.sort(key=lambda t: -t[0])

    # ---- Stage 1: polish top candidates on the sphere ----
    rng2 = np.random.default_rng(7)

    def sphere_polish(P, iters):
        best = ratio_sq(P)
        sigma = 0.004
        for _ in range(iters):
            Q = project_sphere(P + sigma * rng2.standard_normal(P.shape))
            rq = ratio_sq(Q)
            if rq > best:
                best = rq
                P = Q
                sigma = min(sigma * 1.3, 0.02)
            else:
                sigma *= 0.9995
                if sigma < 1e-5:
                    sigma = 1e-5
        return best, P

    def free_polish(P, iters, seed):
        # Adaptive-sigma polish in free space: points may leave the sphere,
        # dmax renormalized to 1 each step (best 14-point sets are slightly
        # non-spherical, so this can beat the spherical Tammes optimum).
        r3 = np.random.default_rng(seed)
        P = normalize(P)
        best = ratio_sq(P)
        sigma = 0.004
        for _ in range(iters):
            Q = normalize(P + sigma * r3.standard_normal(P.shape))
            rq = ratio_sq(Q)
            if rq > best:
                best = rq
                P = Q
                sigma = min(sigma * 1.3, 0.02)
            else:
                sigma *= 0.9995
                if sigma < 1e-6:
                    sigma = 1e-6
        return best, P

    # ---- Stage 1+2: polish top candidates on sphere AND in free space ----
    # The spherical best is not necessarily the best basin for the true
    # (non-spherical) optimum, so free-polish every top candidate.
    best_ratio = -1.0
    best_P = None
    for idx, (r0, P0) in enumerate(candidates[:8]):
        r, P = sphere_polish(P0.copy(), 6000)
        if r > best_ratio:
            best_ratio = r
            best_P = P.copy()
        r2, P2 = free_polish(P, 6000, 1000 + idx)
        if r2 > best_ratio:
            best_ratio = r2
            best_P = P2.copy()

    # ---- Stage 3: long annealed free-space polish on overall best ----
    _, P = free_polish(best_P, 30000, 7)

    return np.asarray(normalize(P), dtype=float)


# EVOLVE-BLOCK-END
