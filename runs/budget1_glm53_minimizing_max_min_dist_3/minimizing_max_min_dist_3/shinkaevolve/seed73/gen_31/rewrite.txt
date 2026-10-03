# EVOLVE-BLOCK-START
import numpy as np

try:
    from scipy.optimize import minimize
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


def min_max_dist_dim3_14() -> np.ndarray:
    n, d = 14, 3
    iu = np.triu_indices(n, 1)
    ii, jj = iu

    def true_score(P):
        d2 = np.sum((P[:, None, :] - P[None, :, :]) ** 2, axis=-1)
        dd = d2[iu]
        return dd.min() / dd.max()

    def make_obj(p):
        """maximize (softmin d^2 / softmax d^2) -> minimize negative."""
        def obj(flat):
            P = flat.reshape(n, d)
            diff = P[:, None, :] - P[None, :, :]
            d2 = np.sum(diff * diff, axis=-1)
            dd = d2[iu]                    # pairwise squared dists
            dij = diff[iu]                 # (m,3)
            # p-norm soft min (harmonic-like) and soft max
            inv = np.maximum(dd, 1e-12) ** (-p)
            soft_min = (np.sum(inv)) ** (-1.0 / p)
            soft_max = (np.sum(dd ** p)) ** (1.0 / p)
            F = soft_min / soft_max
            # gradient
            g_min = (-1.0 / p) * soft_min * inv / np.maximum(dd, 1e-12)
            g_max = (1.0 / p) * soft_max * dd ** (p - 1.0)
            g_dd = -g_min / soft_max + F * (-g_max) / soft_max
            grad = np.zeros((n, d))
            contrib = 2.0 * g_dd[:, None] * dij
            np.add.at(grad, ii, contrib)
            np.add.at(grad, jj, -contrib)
            return -F, grad.ravel()
        return obj

    t = (1.0 + 5.0 ** 0.5) / 2.0
    ico = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], dtype=float)
    ico /= np.linalg.norm(ico, axis=1, keepdims=True)

    cube = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1)
                     for z in (-1, 1)], dtype=float)

    def fibonacci(k):
        i = np.arange(k) + 0.5
        phi = np.pi * (1.0 + 5.0 ** 0.5) * i
        z = 1.0 - 2.0 * i / k
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)

    poles = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    seeds = []
    # symmetric icosahedron+poles family, random rotations + small noise
    for k in range(10):
        rng = np.random.RandomState(1000 + k)
        Q, _ = np.linalg.qr(rng.randn(d, d))
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        base = np.vstack([ico @ Q.T, poles @ Q.T])
        seeds.append(base + 0.03 * rng.randn(n, d))
    # antipodal cube 7+7 family
    for k in range(4):
        rng = np.random.RandomState(2000 + k)
        seeds.append(np.vstack([cube[:7], -cube[:7]]) + 0.02 * rng.randn(n, d))
    seeds.append(fibonacci(n))
    # random seeds
    for s in range(10):
        rng = np.random.RandomState(s)
        P0 = rng.randn(n, d)
        P0 /= np.linalg.norm(P0, axis=1, keepdims=True)
        seeds.append(P0)

    def run(P, p, maxiter):
        if not HAVE_SCIPY:
            return P
        obj = make_obj(p)
        res = minimize(obj, P.ravel(), jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "maxfun": 3 * maxiter})
        return res.x.reshape(n, d)

    def anneal(P):
        for p, it in [(4, 400), (16, 400), (40, 300)]:
            P = run(P, p, it)
        return P

    cands = []
    for P0 in seeds:
        P = anneal(P0)
        cands.append((true_score(P), P))
    cands.sort(key=lambda c: -c[0])

    best_s, best_P = cands[0]
    # polish top candidates with perturbed restarts
    for s0, P0 in cands[:6]:
        rng = np.random.RandomState(777)
        for trial in range(4):
            start = P0 if trial == 0 else P0 + 0.01 * rng.randn(n, d)
            P = anneal(start)
            s = true_score(P)
            if s > best_s:
                best_s, best_P = s, P

    # antipodal symmetrization polish
    def symmetrize(P):
        # match each point to best antipodal partner
        D = np.linalg.norm(P[:, None, :] + P[None, :, :], axis=-1)
        Ps = P.copy()
        used = set()
        for i in range(n):
            if i in used:
                continue
            j = int(np.argmin(D[i]))
            if j in used or j == i:
                continue
            mid = 0.5 * (P[i] - P[j])   # antipodal pair (P[i] ~ -P[j])
            Ps[i] = mid
            Ps[j] = -mid
            used.update([i, j])
        return Ps

    for _ in range(3):
        Psym = symmetrize(best_P)
        P = run(Psym, 60, 300)
        if HAVE_SCIPY:
            obj = make_obj(60)
            res = minimize(obj, P.ravel(), jac=True, method="SLSQP",
                           options={"maxiter": 200, "ftol": 1e-16})
            P = res.x.reshape(n, d)
        s = true_score(P)
        if s > best_s:
            best_s, best_P = s, P
        else:
            # try one more polish from current best with tight tolerance
            P = run(best_P, 60, 500)
            s = true_score(P)
            if s > best_s:
                best_s, best_P = s, P
            break

    best_P = best_P - best_P.mean(axis=0)
    scale = np.abs(best_P).max()
    if scale > 0:
        best_P = best_P / scale
    return np.asarray(best_P, dtype=float)


# EVOLVE-BLOCK-END