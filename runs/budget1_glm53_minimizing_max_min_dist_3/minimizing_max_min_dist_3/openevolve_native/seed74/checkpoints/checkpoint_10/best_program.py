# EVOLVE-BLOCK-START
import numpy as np


def _pairwise(pts):
    """Pairwise difference vectors and pairwise distances."""
    diff = pts[:, None, :] - pts[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-18)
    return diff, d


def _ratio_sq(pts):
    """Exact objective: (dmin / dmax)^2 over all pairwise distances."""
    _, d = _pairwise(pts)
    iu = np.triu_indices(len(pts), k=1)
    dm = d[iu]
    dmax = dm.max()
    if dmax <= 0:
        return 0.0
    return (dm.min() / dmax) ** 2


def _relax_on_sphere(pts, iters=350, step0=0.05):
    """
    Tammes-style repulsion relaxation of points constrained to the
    unit sphere: inverse-square magnitude forces (diff / d^3) push
    points apart, maximizing the minimum chord distance. The best
    configuration seen (by minimum pairwise distance) is returned.
    """
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    pts = pts / norms
    n = len(pts)
    iu = np.triu_indices(n, k=1)
    best = pts.copy()
    best_min = 0.0
    for it in range(iters):
        diff, d = _pairwise(pts)
        f = (diff / (d ** 3)[:, :, None]).sum(axis=1)
        # remove radial component so motion stays tangent to the sphere
        r = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        f = f - np.sum(f * r, axis=1, keepdims=True) * r
        step = step0 * (1.0 - it / iters) + 1e-4
        pts = pts + step * f
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        cur_min = d[iu].min()
        if cur_min > best_min:
            best_min = cur_min
            best = pts.copy()
    return best


def _normalize(pts):
    """Re-center and rescale so the maximum pairwise distance equals 1
    (the objective is invariant under translation and uniform scaling)."""
    pts = np.asarray(pts, dtype=float).copy()
    pts = pts - pts.mean(axis=0)
    iu = np.triu_indices(len(pts), k=1)
    _, d = _pairwise(pts)
    mx = d[iu].max()
    if mx > 0:
        pts = pts / mx
    return pts


def _soft_ascent(pts, iters=500, tau0=0.05, tau1=0.002, step0=0.03):
    """
    Smooth gradient ascent on (dmin/dmax)^2 using log-sum-exp softmin and
    softmax of the squared pairwise distances, so every near-minimal and
    near-maximal pair contributes to the gradient (avoids the single-pair
    subgradient pathology at tied distances). Re-centered and rescaled to
    dmax = 1 each step; coordinates are free in 3D because the N=14
    optimum is non-spherical.
    """
    n = len(pts)
    iu = np.triu_indices(n, k=1)
    I, J = iu[0], iu[1]
    pts = _normalize(pts)
    best = pts.copy()
    best_val = _ratio_sq(pts)
    cur_val = best_val
    step = step0
    for it in range(iters):
        frac = it / max(1, iters - 1)
        tau = tau0 * (tau1 / tau0) ** frac
        dv = pts[I] - pts[J]
        sq = np.sum(dv * dv, axis=1)
        smin, smax = sq.min(), sq.max()
        emin = smin - tau * np.log(np.sum(np.exp(-(sq - smin) / tau)))
        emax = smax + tau * np.log(np.sum(np.exp((sq - smax) / tau)))
        wmin = np.exp(-(sq - smin) / tau)
        wmin /= wmin.sum()
        wmax = np.exp((sq - smax) / tau)
        wmax /= wmax.sum()
        dr = (wmin * emax - emin * wmax) / (emax ** 2 + 1e-18)
        g = np.zeros_like(pts)
        contrib = (2.0 * dr)[:, None] * dv
        np.add.at(g, I, contrib)
        np.add.at(g, J, -contrib)
        gn = np.linalg.norm(g)
        if gn <= 0:
            break
        trial = _normalize(pts + step * g / gn)
        v = _ratio_sq(trial)
        if v > cur_val:
            pts, cur_val = trial, v
            if v > best_val:
                best_val, best = v, trial.copy()
            step = min(step * 1.2, 0.08)
        else:
            step *= 0.5
            if step < 1e-7:
                break
    return best


def _polish_slsqp(pts, rounds=3, seed=0):
    """
    Exact epigraph polish: maximize t subject to d_ij^2 >= t and
    d_ij^2 <= 1 for all pairs (42 coordinates + t, analytic Jacobians).
    After rescaling to dmax = 1 this is precisely max (dmin/dmax)^2.
    Falls back to the normalized input if scipy is unavailable.
    """
    try:
        from scipy.optimize import minimize
    except Exception:
        return _normalize(pts), _ratio_sq(pts)
    rng = np.random.default_rng(seed)
    n = len(pts)
    iu = np.triu_indices(n, k=1)
    I, J = iu[0], iu[1]
    P = len(I)
    cvar = 3 * n
    rows = np.arange(P)

    def cons_f(z):
        p = z[:cvar].reshape(n, 3)
        dv = p[I] - p[J]
        s = np.sum(dv * dv, axis=1)
        return np.concatenate([s - z[-1], 1.0 - s])

    def cons_j(z):
        p = z[:cvar].reshape(n, 3)
        dv = p[I] - p[J]
        Jc = np.zeros((2 * P, cvar + 1))
        Jc[rows, -1] = -1.0
        for k in range(P):
            a, b = 3 * I[k], 3 * J[k]
            u = 2.0 * dv[k]
            Jc[k, a:a + 3] = u
            Jc[k, b:b + 3] = -u
            Jc[P + k, a:a + 3] = -u
            Jc[P + k, b:b + 3] = u
        return Jc

    def obj(z):
        return -z[-1]

    def obj_jac(z):
        g = np.zeros(cvar + 1)
        g[-1] = -1.0
        return g

    best = _normalize(pts)
    best_val = _ratio_sq(best)
    for r in range(rounds):
        start = best if r == 0 else _normalize(
            best + 0.02 * rng.standard_normal((n, 3)))
        _, d = _pairwise(start)
        t0 = 0.999 * d[iu].min() ** 2
        z0 = np.concatenate([start.ravel(), [t0]])
        try:
            res = minimize(obj, z0, jac=obj_jac,
                           constraints={'type': 'ineq',
                                        'fun': cons_f, 'jac': cons_j},
                           method='SLSQP',
                           options={'maxiter': 800, 'ftol': 1e-14})
            cand = _normalize(res.x[:cvar].reshape(n, 3))
            v = _ratio_sq(cand)
            if v > best_val:
                best_val, best = v, cand
        except Exception:
            continue
    return best, best_val


def _icosahedron():
    """12 icosahedron vertices on the unit sphere (edge ~ 1.0515)."""
    phi = (1 + 5 ** 0.5) / 2
    v = []
    for a in (-1, 1):
        for b in (-phi, phi):
            v.append([0, a, b])
            v.append([a, b, 0])
            v.append([b, 0, a])
    v = np.array(v, dtype=float)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _icosahedron_plus_two():
    """12 icosahedron vertices plus 2 poles as a structured seed."""
    return np.vstack([_icosahedron(), [[0, 0, 1.0], [0, 0, -1.0]]])


def _cuboctahedron():
    """12 cuboctahedron vertices on the unit sphere (edge ~ 1.4142)."""
    v = []
    for s1 in (-1, 1):
        for s2 in (-1, 1):
            v.append([s1, s2, 0.0])
            v.append([s1, 0.0, s2])
            v.append([0.0, s1, s2])
    v = np.array(v, dtype=float)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _cuboctahedron_plus_two():
    """12 cuboctahedron vertices plus 2 axial points as a structured seed."""
    return np.vstack([_cuboctahedron(), [[0, 0, 1.5], [0, 0, -1.5]]])


def _cubocta_center_one():
    """Cuboctahedron shell + center + one interior point. A shell+center
    reaches squared ratio 0.25 for 13 points, so this family is a strong
    seed for the (non-spherical) 14-point optimum."""
    v = _cuboctahedron() * 1.41421356
    return np.vstack([v, [[0.0, 0.0, 0.0], [0.6, 0.6, 0.6]]])


def _cubocta_two_interior():
    """Cuboctahedron shell + two interior axial points."""
    v = _cuboctahedron() * 1.41421356
    return np.vstack([v, [[0.0, 0.0, 0.55], [0.0, 0.0, -0.55]]])


def _twisted_shells():
    """Two staggered cuboctahedral shells: outer shell of 8, inner shell
    of 6 rotated 45 degrees about z at a smaller radius. A distinct
    non-spherical topology candidate for the 14-point optimum."""
    outer = []
    for s1 in (-1, 1):
        for s2 in (-1, 1):
            for s3 in (-1, 1):
                outer.append([s1, s2, s3])
    outer = np.array(outer, dtype=float)
    outer = outer / np.linalg.norm(outer, axis=1, keepdims=True)
    c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    inner = _cuboctahedron()[:6] @ Rz.T
    return np.vstack([outer, 0.55 * inner])


def _icosa_center_one():
    """Icosahedron shell + center + one interior point (another 13-point
    0.25 squared-ratio family, seeded for 14 points)."""
    v = _icosahedron()
    return np.vstack([v, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5]]])


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Construct 14 points in 3D maximizing (dmin / dmax)^2.

    Pipeline:
      1. Multi-restart Tammes-style repulsion on the unit sphere produces
         well-spread seeds.
      2. Free-coordinate smooth (log-sum-exp) gradient ascent from those
         seeds plus structured non-spherical seeds (cuboctahedron /
         icosahedron shells with interior points) — the N=14 optimum is
         non-spherical, so interior-point families are essential.
      3. Exact epigraph SLSQP polish: maximize dmin^2 subject to dmax^2 <= 1,
         which is exactly the target objective after rescaling.
      4. Decreasing-jitter restarts of stages 2-3 around the incumbent.
    """
    n = 14
    rng = np.random.default_rng(12345)

    # Stage 1: sphere relaxation for spread-out seeds.
    sphere_seeds = [_icosahedron_plus_two(), _cuboctahedron_plus_two()]
    for _ in range(6):
        sphere_seeds.append(_icosahedron_plus_two()
                            + 0.1 * rng.standard_normal((n, 3)))
        sphere_seeds.append(_cuboctahedron_plus_two()
                            + 0.1 * rng.standard_normal((n, 3)))
    for _ in range(160):
        sphere_seeds.append(rng.standard_normal((n, 3)))
    relaxed = sorted(((_ratio_sq(_relax_on_sphere(c, iters=450)), c)
                     for c in sphere_seeds), key=lambda t: -t[0])
    top_sphere = [p for _, p in relaxed[:10]]

    # Stage 2: free-coordinate smooth ascent from diverse seeds.
    free_seeds = list(top_sphere)
    for f in (_cubocta_center_one, _cubocta_two_interior, _icosa_center_one,
              _twisted_shells):
        free_seeds.append(f())
        for _ in range(6):
            free_seeds.append(f() + 0.05 * rng.standard_normal((n, 3)))
    for _ in range(30):
        free_seeds.append(rng.standard_normal((n, 3)))
    soft = sorted(((_ratio_sq(_soft_ascent(s, iters=700)), s)
                   for s in free_seeds), key=lambda t: -t[0])

    # Stage 3: exact epigraph SLSQP polish of the best smooth results.
    best_pts = _normalize(soft[0][1])
    best_val = soft[0][0]
    for _, p in soft[:14]:
        q, v = _polish_slsqp(p, rounds=4, seed=1)
        if v > best_val:
            best_val, best_pts = v, q

    # Stage 4: decreasing-jitter restarts around the incumbent, with a
    # finer cascade of perturbation scales and more restarts per scale.
    for scale in (0.05, 0.03, 0.02, 0.015, 0.01, 0.008, 0.005, 0.004,
                  0.003, 0.002, 0.0015, 0.001, 0.0005, 0.0003, 0.0002):
        for _ in range(14):
            q = _soft_ascent(best_pts + scale * rng.standard_normal((n, 3)),
                             iters=400)
            q, v = _polish_slsqp(q, rounds=4, seed=2)
            if v > best_val:
                best_val, best_pts = v, q

    # Stage 5: final high-precision polish of the incumbent (no jitter),
    # repeated to let SLSQP converge fully from the best-known point.
    for _ in range(5):
        q, v = _polish_slsqp(best_pts, rounds=3, seed=3)
        if v > best_val:
            best_val, best_pts = v, q
        else:
            break

    return np.asarray(best_pts, dtype=float)


# EVOLVE-BLOCK-END
