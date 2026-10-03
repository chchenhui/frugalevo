# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations

_N = 11
_S3 = np.sqrt(3) / 2.0
_V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, _S3]])
_TRI_IDX = np.array(list(combinations(range(_N), 3)))
_REGION_AREA = np.sqrt(3) / 4.0


def _min_area(P):
    """Vectorized smallest normalized triangle area over all triplets."""
    T = P[_TRI_IDX]
    a2 = np.abs((T[:, 1, 0] - T[:, 0, 0]) * (T[:, 2, 1] - T[:, 0, 1])
               - (T[:, 2, 0] - T[:, 0, 0]) * (T[:, 1, 1] - T[:, 0, 1]))
    i = int(a2.argmin())
    return 0.5 * a2[i] / _REGION_AREA, _TRI_IDX[i]


def _clip(P):
    """Barycentric clamp all points into the triangle."""
    A, B, C = _V
    v0, v1 = B - A, C - A
    d = v0[0] * v1[1] - v0[1] * v1[0]
    w = np.empty((len(P), 3))
    w[:, 0] = ((P[:, 0] - A[0]) * v1[1] - (P[:, 1] - A[1]) * v1[0]) / d
    w[:, 1] = (v0[0] * (P[:, 1] - A[1]) - v0[1] * (P[:, 0] - A[0])) / d
    w[:, 2] = 1.0 - w[:, 0] - w[:, 1]
    w = np.maximum(w, 0.0)
    w /= w.sum(axis=1, keepdims=True)
    return w[:, :1] * A + w[:, 1:2] * B + w[:, 2:3] * C


def _areas(P):
    """Vectorized double-areas (unnormalized) for all triplets."""
    T = P[_TRI_IDX]
    return np.abs((T[:, 1, 0] - T[:, 0, 0]) * (T[:, 2, 1] - T[:, 0, 1])
                  - (T[:, 2, 0] - T[:, 0, 0]) * (T[:, 1, 1] - T[:, 0, 1]))


def _soft_score(a2, k=6):
    """Soft-min: mean of the k smallest double-areas (smooth guidance)."""
    return float(np.partition(a2, k)[:k].mean())


def _joint_polish(P, rounds=12, tau0=0.01):
    """Joint maximin nonlinear program via SLSQP: variables are all 22
    point coordinates plus a slack t; constraints s_ijk*D_ijk >= 2t for
    all 165 triplets (orientation signs locked at the current iterate)
    plus triangle-membership constraints. Signs are re-locked and the
    solve repeated until stable. Preceded by a short log-sum-exp soft-min
    continuation to smooth the landscape. Falls back to input on any
    solver failure."""
    try:
        from scipy.optimize import minimize
    except Exception:
        return P, _min_area(P)[0]
    P = _clip(P.copy())
    best = _min_area(P)[0]
    best_P = P.copy()
    s3 = _S3

    def signed(P):
        T = P[_TRI_IDX]
        return ((T[:, 1, 0] - T[:, 0, 0]) * (T[:, 2, 1] - T[:, 0, 1])
                - (T[:, 2, 0] - T[:, 0, 0]) * (T[:, 1, 1] - T[:, 0, 1]))

    # --- soft-min continuation (log-sum-exp), tau shrinking ---
    for tau in (tau0, tau0 * 0.35, tau0 * 0.12):
        def fobj(z):
            Q = z.reshape(_N, 2)
            d = signed(Q)
            # soft-min of signed areas (signs from current P)
            s = np.where(d >= 0.0, 1.0, -1.0)
            v = s * d
            m = v.min()
            e = np.exp((m - v) / tau)
            val = m - tau * np.log(e.sum() / len(v))
            return -val

        def fgrad(z):
            Q = z.reshape(_N, 2)
            d = signed(Q)
            s = np.where(d >= 0.0, 1.0, -1.0)
            v = s * d
            m = v.min()
            e = np.exp((m - v) / tau)
            w = e / e.sum()
            g = np.zeros((_N, 2))
            # dD/dx_i = y_j - y_k ; dD/dy_i = x_k - x_j per triplet
            i0, j0, k0 = _TRI_IDX[:, 0], _TRI_IDX[:, 1], _TRI_IDX[:, 2]
            gx = (Q[j0, 1] - Q[k0, 1]) * w
            gy = (Q[k0, 0] - Q[j0, 0]) * w
            np.add.at(g[:, 0], i0, gx)
            np.add.at(g[:, 1], i0, gy)
            # chain through sign s (s constant near nonzero d)
            return -(g * s * s).ravel()

        cons = [{'type': 'ineq',
                 'fun': lambda z: np.concatenate([
                     z[1::2],
                     s3 * z[0::2] - z[1::2],
                     s3 * (1.0 - z[0::2]) - z[1::2]])}]
        try:
            r = minimize(fobj, P.ravel(), jac=fgrad, method='SLSQP',
                         constraints=cons,
                         options={'maxiter': 200, 'ftol': 1e-12})
            Q = _clip(r.x.reshape(_N, 2))
            a = _min_area(Q)[0]
            if a > best - 1e9:  # always track soft progress
                P = Q
                if a > best:
                    best, best_P = a, Q.copy()
        except Exception:
            break

    # --- locked-sign joint maximin LP-like NLP, re-lock until stable ---
    for _ in range(rounds):
        d = signed(P)
        s = np.where(d >= 0.0, 1.0, -1.0)
        i0, j0, k0 = _TRI_IDX[:, 0], _TRI_IDX[:, 1], _TRI_IDX[:, 2]

        def obj(z):
            return -z[-1]

        def jac(z):
            g = np.zeros(2 * _N + 1)
            g[-1] = -1.0
            return g

        def cons_f(z):
            Q = z[:2 * _N].reshape(_N, 2)
            dQ = signed(Q)
            x, y, t = z[0::2], z[1::2], z[-1]
            return np.concatenate([
                s * dQ - 2.0 * t,
                y,
                s3 * x - y,
                s3 * (1.0 - x) - y])

        def cons_j(z):
            Q = z[:2 * _N].reshape(_N, 2)
            x, y = z[0::2], z[1::2]
            m = len(_TRI_IDX)
            J = np.zeros((m + 3 * _N, 2 * _N + 1))
            # dD/dx_i = y_j - y_k ; dD/dy_i = x_k - x_j
            J[:m, 2 * i0] = s * (y[j0] - y[k0])
            J[:m, 2 * i0 + 1] = s * (x[k0] - x[j0])
            J[:m, 2 * j0] = s * (y[k0] - y[i0])
            J[:m, 2 * j0 + 1] = s * (x[i0] - x[k0])
            J[:m, 2 * k0] = s * (y[i0] - y[j0])
            J[:m, 2 * k0 + 1] = s * (x[j0] - x[i0])
            J[:m, -1] = -2.0
            for i in range(_N):
                J[m + i, 2 * i + 1] = 1.0
                J[m + _N + i, 2 * i] = s3
                J[m + _N + i, 2 * i + 1] = -1.0
                J[m + 2 * _N + i, 2 * i] = -s3
                J[m + 2 * _N + i, 2 * i + 1] = -1.0
            return J

        z0 = np.concatenate([P.ravel(), [max(best, 1e-6)]])
        try:
            r = minimize(obj, z0, jac=jac, method='SLSQP',
                         constraints=[{'type': 'ineq',
                                      'fun': cons_f, 'jac': cons_j}],
                         options={'maxiter': 300, 'ftol': 1e-12})
            Q = _clip(r.x[:2 * _N].reshape(_N, 2))
            a = _min_area(Q)[0]
            if a > best + 1e-12:
                best, best_P = a, Q.copy()
                P = Q
            else:
                break
        except Exception:
            break
    return best_P, _min_area(best_P)[0]


def _greedy_polish(P, step0=0.02, rounds=60):
    """Deterministic coordinate descent: for each point, try moves in 8
    directions at two magnitudes; accept only strict improvements of the
    true minimum area. Step shrinks on stagnation."""
    P = _clip(P.copy())
    best = _min_area(P)[0]
    ang = np.linspace(0.0, 2.0 * np.pi, 13)[:-1]
    dirs = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    step = step0
    for _ in range(rounds):
        improved = False
        for j in range(_N):
            for mag in (1.0, 0.4, 0.15):
                for d in dirs:
                    Q = P.copy()
                    Q[j] += step * mag * d
                    Q = _clip(Q)
                    a = _min_area(Q)[0]
                    if a > best + 1e-12:
                        best, P, improved = a, Q, True
        if not improved:
            step *= 0.5
            if step < 1e-5:
                break
    return P, _min_area(P)[0]


def _lp_polish(P, sweeps=80):
    """Exact alternating linear-program polish: for each point in turn,
    solve the 3-variable LP (x, y, t) that places it to maximize the
    minimum signed area over all triplets containing it, subject to
    staying inside the triangle. With the other points and orientation
    signs fixed, each such area is linear in the moving point, so the LP
    is exact. Iterated sweeps converge to a configuration that no
    single-point move can improve. No-op if scipy is unavailable."""
    try:
        from scipy.optimize import linprog
    except Exception:
        return P, _min_area(P)[0]
    P = _clip(P.copy())
    s3 = _S3
    best = _min_area(P)[0]
    for _ in range(sweeps):
        improved = False
        for i in range(_N):
            rows, rhs = [], []
            for a_ in range(_N):
                if a_ == i:
                    continue
                for b_ in range(a_ + 1, _N):
                    if b_ == i:
                        continue
                    xj, yj = P[a_]
                    xk, yk = P[b_]
                    ca = yj - yk
                    cb = xk - xj
                    cc = xj * yk - xk * yj
                    D = cc + ca * P[i, 0] + cb * P[i, 1]
                    s = 1.0 if D >= 0.0 else -1.0
                    # s*(ca*x + cb*y + cc) >= t  <=>  -s*ca*x - s*cb*y + t <= s*cc
                    rows.append([-s * ca, -s * cb, 1.0])
                    rhs.append(s * cc)
            # triangle membership: y >= 0, y <= s3*x, y <= s3*(1-x)
            rows.append([0.0, -1.0, 0.0]); rhs.append(0.0)
            rows.append([-s3, 1.0, 0.0]); rhs.append(0.0)
            rows.append([s3, 1.0, 0.0]); rhs.append(s3)
            try:
                r = linprog([0.0, 0.0, -1.0], A_ub=np.asarray(rows),
                            b_ub=np.asarray(rhs), method='highs')
            except Exception:
                continue
            if not r.success:
                continue
            Q = P.copy()
            Q[i, 0], Q[i, 1] = r.x[0], r.x[1]
            a = _min_area(Q)[0]
            if a > best + 1e-12:
                best, P, improved = a, Q, True
        if not improved:
            break
    return P, _min_area(P)[0]


def _targeted_polish(P, rounds=200, step0=0.01):
    """Deterministic gradient-directed polish: repeatedly find the current
    worst triangle and try moving each of its vertices along the
    perpendicular to the opposite side (the exact gradient direction for
    increasing that triangle's area), at several magnitudes including a
    small negative one (sometimes shrinking frees room for others)."""
    P = _clip(P.copy())
    best = _min_area(P)[0]
    step = step0
    for _ in range(rounds):
        a2 = _areas(P)
        tri = _TRI_IDX[int(a2.argmin())]
        T = P[tri]
        cen = T.mean(axis=0)
        improved = False
        for k in range(3):
            o1, o2 = tri[(k + 1) % 3], tri[(k + 2) % 3]
            seg = P[o2] - P[o1]
            nrm = np.hypot(seg[0], seg[1])
            if nrm < 1e-12:
                continue
            nvec = np.array([-seg[1], seg[0]]) / nrm
            if np.dot(nvec, P[tri[k]] - cen) < 0.0:
                nvec = -nvec
            for mag in (1.0, 0.5, 0.25, -0.5):
                Q = P.copy()
                Q[tri[k]] += nvec * step * mag
                Q = _clip(Q)
                a = _min_area(Q)[0]
                if a > best + 1e-12:
                    best, P, improved = a, Q, True
        if not improved:
            step *= 0.6
            if step < 1e-5:
                break
    return P, _min_area(P)[0]


def _optimize(P, rng, iters, step0):
    """Annealed local search attacking near-worst triangles; acceptance on
    a soft-min criterion. Moves mix (a) random exploration perturbations,
    (b) perpendicular lifts that directly increase the attacked triangle's
    area (the exact gradient direction), and (c) radial pushes from the
    triangle centroid."""
    P = _clip(P.copy())
    a2 = _areas(P)
    best_a = _min_area(P)[0]
    best_P = P.copy()
    step = step0
    for it in range(iters):
        Q = P.copy()
        r = rng.random()
        if r < 0.20:
            j = int(rng.integers(0, _N))
            Q[j] += rng.normal(0.0, step, size=2)
        else:
            # attack one of the 6 currently worst triangles
            cand = np.argsort(a2)[:6]
            tri = _TRI_IDX[int(rng.choice(cand))]
            T = P[tri]
            if r < 0.55:
                # perpendicular lift: move one vertex of the attacked
                # triangle along the normal to the opposite side.
                k = int(rng.integers(0, 3))
                o1, o2 = tri[(k + 1) % 3], tri[(k + 2) % 3]
                seg = P[o2] - P[o1]
                nrm = np.hypot(seg[0], seg[1])
                if nrm > 1e-12:
                    nvec = np.array([-seg[1], seg[0]]) / nrm
                    cen = T.mean(axis=0)
                    if np.dot(nvec, P[tri[k]] - cen) < 0.0:
                        nvec = -nvec
                    Q[tri[k]] += nvec * step * rng.uniform(0.5, 2.0)
                else:
                    Q[tri[k]] += rng.normal(0.0, step, size=2)
            else:
                cen = T.mean(axis=0)
                for k in range(3):
                    d = T[k] - cen
                    nrm = np.linalg.norm(d)
                    if nrm > 1e-12:
                        Q[tri[k]] += (d / nrm) * step * rng.uniform(0.5, 1.5)
                    else:
                        Q[tri[k]] += rng.normal(0.0, step, size=2)
        Q = _clip(Q)
        a2q = _areas(Q)
        a = _min_area(Q)[0]
        # accept on soft-min improvement (with rare uphill moves)
        if _soft_score(a2q) >= _soft_score(a2) or rng.random() < 0.01:
            P, a2 = Q, a2q
            if a > best_a:
                best_a, best_P = a, Q.copy()
        if it % 500 == 499:
            step *= 0.85
            P = best_P.copy()
            a2 = _areas(P)
    return best_P, _min_area(best_P)[0]


def heilbronn_triangle11() -> np.ndarray:
    """
    Arrange 11 points in the unit equilateral triangle to maximize the
    minimum triangle area over all triplets.

    Approach: deterministic multi-start annealed local search. Several
    structured seeds (vertices + edge points + interior points, with
    seeded jitter) are optimized by perturbing the points of the current
    worst triangle; the best result is returned. Falls back to a fixed
    structured configuration on any failure.
    """
    try:
        rng = np.random.default_rng(20240607)
        seeds = []
        # Seed 1: 3 vertices, 2 per edge, 2 interior
        p = [v.copy() for v in _V]
        for t in (1.0 / 3.0, 2.0 / 3.0):
            p.append(_V[0] + t * (_V[1] - _V[0]))
            p.append(_V[1] + t * (_V[2] - _V[1]))
            p.append(_V[2] + t * (_V[0] - _V[2]))
        p.append(np.array([0.5, _S3 / 3.0]))
        p.append(np.array([0.5, 2.0 * _S3 / 3.0]))
        seeds.append(np.array(p))
        # Seed 2: jittered seed 1
        seeds.append(_clip(seeds[0] + rng.normal(0.0, 0.02, size=(_N, 2))))
        # Seed 3: uniform random
        w = rng.dirichlet(np.ones(3), size=_N)
        seeds.append(w[:, :1] * _V[0] + w[:, 1:2] * _V[1] + w[:, 2:3] * _V[2])
        # Seed 4: stronger jitter
        seeds.append(_clip(seeds[0] + rng.normal(0.0, 0.05, size=(_N, 2))))
        # Seed 5: edge-ring — all 11 points on the boundary at staggered
        # arc-length positions (boundary-heavy layouts often do well).
        ts = (np.arange(_N) + 0.5) / _N
        pts = []
        for t in ts:
            u = 3.0 * t
            e, f = int(u), u - int(u)
            pts.append(_V[e % 3] + f * (_V[(e + 1) % 3] - _V[e % 3]))
        seeds.append(np.array(pts))
        # Seeds 6-17: uniform random interior draws for basin diversity
        for _ in range(12):
            w = rng.dirichlet(np.ones(3), size=_N)
            seeds.append(w[:, :1] * _V[0] + w[:, 1:2] * _V[1]
                         + w[:, 2:3] * _V[2])

        import time
        deadline = time.time() + 220.0
        best_P, best_a = None, -1.0
        for si, S in enumerate(seeds):
            R = np.random.default_rng(1000 + si)
            P, a = _optimize(S, R, 5000, 0.04)
            P, a = _lp_polish(P)
            P, a = _joint_polish(P)
            if a > best_a:
                best_a, best_P = a, P
            if time.time() > deadline:
                break
        # Basin hopping: kick the incumbent at cycling scales, re-anneal
        # briefly, then apply the exact LP polish; keep strict improvements.
        # The LP polish makes each visited basin far deeper than plain
        # hill-climbing could, so kicks reliably explore better optima.
        rng2 = np.random.default_rng(31337)
        scales = [0.02, 0.01, 0.005, 0.015, 0.008, 0.003, 0.05, 0.012,
                 0.006, 0.004, 0.02, 0.007]
        for ki in range(48):
            if time.time() > deadline:
                break
            sc = scales[ki % len(scales)]
            S = _clip(best_P + rng2.normal(0.0, sc, size=(_N, 2)))
            R = np.random.default_rng(5000 + ki)
            P, a = _optimize(S, R, 3000, 0.5 * sc + 0.002)
            P, a = _lp_polish(P)
            P, a = _joint_polish(P)
            if a > best_a:
                best_a, best_P = a, P
        # Fine anneal + exact LP + targeted gradient + greedy micro-polish
        for st, sd in ((0.006, 2001), (0.002, 3001)):
            if time.time() > deadline:
                break
            R = np.random.default_rng(sd)
            P, a = _optimize(best_P, R, 4000, st)
            P, a = _lp_polish(P)
            if a > best_a:
                best_a, best_P = a, P
            P, a = _targeted_polish(best_P)
            if a > best_a:
                best_a, best_P = a, P
            P, a = _greedy_polish(best_P, step0=0.004, rounds=30)
            if a > best_a:
                best_a, best_P = a, P
        P, a = _lp_polish(best_P)
        if a > best_a:
            best_a, best_P = a, P
        return np.asarray(best_P, dtype=float)
    except Exception:
        # Deterministic fallback: vertices + edge points + interior
        p = [v.copy() for v in _V]
        for t in (1.0 / 3.0, 2.0 / 3.0):
            p.append(_V[0] + t * (_V[1] - _V[0]))
            p.append(_V[1] + t * (_V[2] - _V[1]))
            p.append(_V[2] + t * (_V[0] - _V[2]))
        p.append(np.array([0.5, _S3 / 3.0]))
        p.append(np.array([0.5, 2.0 * _S3 / 3.0]))
        return np.array(p)


# EVOLVE-BLOCK-END
