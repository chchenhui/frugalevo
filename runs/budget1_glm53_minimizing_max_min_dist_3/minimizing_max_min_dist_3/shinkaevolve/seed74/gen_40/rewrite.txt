# EVOLVE-BLOCK-START
import numpy as np

try:
    from scipy.optimize import minimize
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


def _ratio_sq(pts):
    pts = np.asarray(pts, dtype=float)
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    iu = np.triu_indices(len(pts), 1)
    dm = d[iu].min()
    dx = d[iu].max()
    if dx <= 1e-15:
        return 0.0
    return (dm / dx) ** 2


def _normalize(P):
    P = np.asarray(P, dtype=float)
    P = P - P.mean(axis=0)
    iu = np.triu_indices(len(P), 1)
    dmax = np.linalg.norm(P[iu[0]] - P[iu[1]], axis=1).max()
    if dmax > 1e-15:
        P = P / dmax
    return P


def _force_relax(P0, iters=1200, seed=0):
    """Spring/force balance: repel short pairs, mildly attract diameter
    pairs, decay jitter, renormalize to dmax=1 each step."""
    rng = np.random.default_rng(seed)
    n = len(P0)
    P = _normalize(np.asarray(P0, dtype=float).copy())
    I, J = np.triu_indices(n, 1)
    best = P.copy()
    best_v = _ratio_sq(P)
    for it in range(iters):
        f = 1.0 - it / iters
        diff = P[I] - P[J]
        dd = np.linalg.norm(diff, axis=1)
        dd = np.maximum(dd, 1e-12)
        m = dd.min()
        # repulsion for pairs near/below the min distance
        w_rep = np.exp(-8.0 * (dd - m))
        w_rep /= max(w_rep.sum(), 1e-12)
        # weak attraction for pairs at the diameter
        w_att = np.exp(20.0 * (dd - dd.max()))
        w_att /= max(w_att.sum(), 1e-12)
        u = diff / dd[:, None]
        F = np.zeros_like(P)
        np.add.at(F, I, (w_rep - 0.05 * w_att)[:, None] * u)
        np.add.at(F, J, -(w_rep - 0.05 * w_att)[:, None] * u)
        step = 0.05 * f + 1e-4
        P = _normalize(P + step * F + rng.normal(0, 0.01 * f, P.shape))
        v = _ratio_sq(P)
        if v > best_v:
            best_v = v
            best = P.copy()
    return best, best_v


def _slsqp_polish(P, n=14, rng=None):
    """Exact constrained polish: maximize t s.t. t^2 <= d_ij^2 <= 1."""
    if not HAVE_SCIPY:
        return P, _ratio_sq(P)
    P = _normalize(P)
    if _ratio_sq(P) <= 0:
        return P, 0.0
    I, J = np.triu_indices(n, 1)
    m = len(I)
    t0 = np.sqrt(_ratio_sq(P))
    x0 = np.concatenate([P.ravel(), [t0]])

    def fobj(x):
        return -x[-1]

    def fjack(x):
        g = np.zeros(3 * n + 1)
        g[-1] = -1.0
        return g

    def cons(x):
        P2 = x[:3 * n].reshape(n, 3)
        dv = np.sum((P2[I] - P2[J]) ** 2, axis=1)
        return np.concatenate([1.0 - dv, dv - x[-1] ** 2])

    def cjack(x):
        P2 = x[:3 * n].reshape(n, 3)
        diff = P2[I] - P2[J]
        Jc = np.zeros((2 * m, 3 * n + 1))
        idx_a = (3 * I[:, None] + np.arange(3))
        idx_b = (3 * J[:, None] + np.arange(3))
        Jc[:m, idx_a] = -2.0 * diff
        Jc[:m, idx_b] = 2.0 * diff
        Jc[m:, idx_a] = 2.0 * diff
        Jc[m:, idx_b] = -2.0 * diff
        Jc[m:, -1] = -2.0 * x[-1]
        return Jc

    try:
        res = minimize(fobj, x0, method="SLSQP",
                       constraints=[{"type": "ineq", "fun": cons, "jac": cjack}],
                       options={"maxiter": 300, "ftol": 1e-14})
        P2 = res.x[:3 * n].reshape(n, 3)
        if np.all(np.isfinite(P2)):
            P2 = P2 - P2.mean(axis=0)
            return P2, _ratio_sq(P2)
    except Exception:
        pass
    return P, _ratio_sq(P)


def _seeds(n=14):
    rng = np.random.default_rng(7)
    seeds = []
    phi = (1 + 5 ** 0.5) / 2
    ico = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]],
        dtype=float) / np.sqrt(1 + phi ** 2)
    for pz in (1.0, 1.2, 1.4):
        for ax in range(3):
            p1 = np.zeros(3); p1[ax] = pz
            p2 = np.zeros(3); p2[ax] = -pz
            seeds.append(np.vstack([ico, p1, p2]))
    # two staggered heptagon rings (twisted prism family)
    for h in (0.5, 0.7, 0.9, 1.1):
        for twist in (0.0, np.pi / 7):
            pts = []
            for k in range(7):
                a = 2 * np.pi * k / 7 + twist
                pts.append([np.cos(a), np.sin(a), h])
                pts.append([np.cos(a), np.sin(a), -h])
            seeds.append(np.array(pts))
    # cuboctahedron + poles
    cubo = np.array(
        [[x, y, 0] for x in (-1, 1) for y in (-1, 1)] +
        [[x, 0, z] for x in (-1, 1) for z in (-1, 1)] +
        [[0, y, z] for y in (-1, 1) for z in (-1, 1)], dtype=float)
    seeds.append(np.vstack([cubo, [[0, 0, 1.2], [0, 0, -1.2]]]))
    # Fibonacci sphere
    kk = np.arange(n) + 0.5
    ga = np.pi * (3 - np.sqrt(5))
    fib = np.zeros((n, 3))
    fib[:, 2] = 1 - 2 * kk / n
    r_ = np.sqrt(np.maximum(0.0, 1 - fib[:, 2] ** 2))
    fib[:, 0] = r_ * np.cos(ga * kk)
    fib[:, 1] = r_ * np.sin(ga * kk)
    seeds.append(fib.copy())
    # jittered variants
    for base in (seeds[0], fib, np.vstack([cubo, [[0, 0, 1.2], [0, 0, -1.2]]])):
        for sc in (0.05, 0.15):
            seeds.append(base + sc * rng.standard_normal((n, 3)))
    # random
    for _ in range(8):
        seeds.append(rng.standard_normal((n, 3)))
    return seeds


def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    rng = np.random.default_rng(42)
    best_pts = None
    best_val = -1.0
    results = []

    for si, s in enumerate(_seeds(n)):
        s = np.asarray(s, dtype=float)
        assert s.shape == (n, 3), "bad seed shape"
        if not np.all(np.isfinite(s)):
            continue
        P, v = _force_relax(s, iters=900, seed=si)
        Pp, vp = _slsqp_polish(P, n)
        if vp > v:
            P, v = Pp, vp
        results.append((v, P))
        if v > best_val:
            best_val = v
            best_pts = P.copy()

    # basin dedup: keep top distinct candidates and polish harder
    results.sort(key=lambda t: -t[0])
    seen = []
    for v, P in results:
        dup = False
        for vs, Ps in seen:
            if abs(vs - v) < 1e-9:
                dup = True
                break
        if not dup:
            seen.append((v, P))
        if len(seen) >= 4:
            break

    for v0, P0 in seen:
        for scale in (0.03, 0.08, 0.15):
            for rep in range(2):
                Pc = _normalize(P0 + scale * rng.standard_normal((n, 3)))
                Pc, vc = _slsqp_polish(Pc, n)
                if vc > best_val:
                    best_val = vc
                    best_pts = Pc.copy()

    if best_pts is None or not np.all(np.isfinite(best_pts)):
        best_pts = np.zeros((n, 3))
        best_pts[:, 0] = np.arange(n)
    best_pts = np.asarray(best_pts, dtype=float)
    assert best_pts.shape == (n, 3)
    # final normalization (scale/translation invariant convenience)
    best_pts = _normalize(best_pts)
    if not np.all(np.isfinite(best_pts)) or _ratio_sq(best_pts) <= 0:
        best_pts = np.zeros((n, 3))
        best_pts[:, 0] = np.arange(n)
        best_pts = _normalize(best_pts)
    return best_pts


# EVOLVE-BLOCK-END