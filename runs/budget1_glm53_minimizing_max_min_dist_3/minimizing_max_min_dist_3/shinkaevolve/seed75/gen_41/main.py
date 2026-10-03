# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

N = 14
D = 3
_IU = np.triu_indices(N, 1)
_IJ = np.array([(i, j) for i in range(N) for j in range(i + 1, N)])


def pair_dists(P):
    return np.linalg.norm(P[:, None] - P[None, :], axis=-1)[_IU]


def ratio(P):
    d = pair_dists(P)
    return d.min() / d.max()


def normalize(P):
    P = P - P.mean(axis=0)
    return P / max(pair_dists(P).max(), 1e-12)


def anneal(P0, taus=(0.05, 0.02, 0.01, 0.005, 0.002)):
    """Annealed soft-min/soft-max L-BFGS polish on squared distances."""
    x = normalize(P0).ravel()

    def obj(z, tau=0.01):
        X = z.reshape(N, D)
        diff = X[_IJ[:, 0]] - X[_IJ[:, 1]]
        d2 = np.sum(diff * diff, axis=1)
        return tau * (logsumexp(-d2 / tau) + logsumexp(d2 / tau))

    for tau in taus:
        res = minimize(lambda z, t=tau: obj(z, t), x, method="L-BFGS-B",
                       options={"maxiter": 400, "ftol": 1e-14, "gtol": 1e-12})
        if np.isfinite(res.x).all():
            x = res.x
    return normalize(x.reshape(N, D))


def d6_family(R, h, z0):
    th = np.arange(6) * (np.pi / 3.0)
    r1 = np.stack([R * np.cos(th), R * np.sin(th), np.full(6, h)], axis=1)
    r2 = np.stack([R * np.cos(th + np.pi / 6), R * np.sin(th + np.pi / 6),
                   np.full(6, -h)], axis=1)
    return np.vstack([r1, r2,
                      [[0.0, 0.0, z0], [0.0, 0.0, -z0]]])


def hillclimb(P0, iters=300, step0=0.01):
    P = normalize(P0.copy())
    r_best = ratio(P)
    step = step0
    for _ in range(iters):
        d = pair_dists(P)
        ia, ib = _IJ[d.argmin()]
        ja, jb = _IJ[d.argmax()]
        improved = False
        u_min = P[ia] - P[ib]
        nm = np.linalg.norm(u_min)
        u_min = u_min / nm if nm > 1e-12 else u_min
        u_max = P[ja] - P[jb]
        nm = np.linalg.norm(u_max)
        u_max = u_max / nm if nm > 1e-12 else u_max
        for mag in (step, step / 2, step / 4):
            if (ia, ib) != (ja, jb):
                Q = P.copy()
                Q[ia] += mag * u_min
                Q[ib] -= mag * u_min
                Q[ja] -= mag * u_max
                Q[jb] += mag * u_max
                Q = normalize(Q)
                r = ratio(Q)
                if r > r_best + 1e-14:
                    P, r_best, improved = Q, r, True
                    break
            else:
                Q = P.copy()
                Q[ia] += mag * u_min
                Q[ib] -= mag * u_min
                Q = normalize(Q)
                r = ratio(Q)
                if r > r_best + 1e-14:
                    P, r_best, improved = Q, r, True
                    break
        if not improved:
            step *= 0.6
            if step < 1e-8:
                break
    return P, r_best


def min_max_dist_dim3_14() -> np.ndarray:
    rng = np.random.default_rng(7)
    best = None
    best_r = -1.0

    def consider(Q):
        nonlocal best, best_r
        Q = normalize(Q)
        r = ratio(Q)
        if r > best_r:
            best_r, best = r, Q.copy()
            return True
        return False

    # --- Structured seed 1: D6 family (staggered hex rings + poles), grid + refine ---
    bp, br = (1.0, 0.6, 1.6), -1.0
    for h in np.linspace(0.15, 1.4, 25):
        for z0 in np.linspace(max(h, 0.2) + 0.05, 2.6, 40):
            r = ratio(d6_family(1.0, h, z0))
            if r > br:
                br, bp = r, (1.0, h, z0)

    def soft_obj(z, tau=0.002):
        R, h, z0 = z
        if R <= 0 or h <= 0 or z0 <= h:
            return 1e6
        d = pair_dists(d6_family(R, h, z0))
        return tau * (logsumexp(-d / tau) + logsumexp(d / tau))

    res = minimize(soft_obj, np.array(bp), method="Nelder-Mead",
                   options={"xatol": 1e-10, "fatol": 1e-14, "maxiter": 3000})
    if np.isfinite(res.x).all() and res.x[0] > 0 and res.x[1] > 0 and res.x[2] > res.x[1]:
        if ratio(d6_family(*res.x)) > br:
            bp = tuple(res.x)
    consider(d6_family(*bp))

    # --- Structured seed 2: icosahedron + 2 poles ---
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ], dtype=float)
    ico /= np.linalg.norm(ico, axis=1, keepdims=True)
    consider(np.vstack([ico, [[0.0, 0.0, 1.05], [0.1, -0.1, -1.05]]]))

    # --- Annealing trials: incumbent-centered + random sphere starts ---
    for trial in range(8):
        if trial == 0:
            P0 = best.copy()
        elif trial == 1:
            P0 = rng.normal(size=(N, D))
            P0 /= np.linalg.norm(P0, axis=1, keepdims=True)
        else:
            P0 = best + 0.08 * rng.normal(size=(N, D))
        Q = anneal(P0)
        consider(Q)

    # --- Key stage: perturb-and-reanneal restarts around the incumbent ---
    for k in range(10):
        Q = anneal(best + 0.08 * rng.normal(size=(N, D)),
                   taus=(0.06, 0.02, 0.01, 0.005, 0.002))
        consider(Q)

    # --- Fine-tau final polish of incumbent ---
    Q = anneal(best, taus=(0.002, 0.001, 0.0003))
    consider(Q)

    # --- Targeted pair hill-climb on min/max pairs ---
    for k in range(5):
        Q, r = hillclimb(best, iters=250, step0=0.01 * (0.7 ** k))
        if r > best_r + 1e-14:
            best, best_r = Q, r
        else:
            break

    points = np.asarray(best, dtype=float)
    assert points.shape == (N, D) and np.isfinite(points).all()
    assert pair_dists(points).max() > 0
    return points
# EVOLVE-BLOCK-END