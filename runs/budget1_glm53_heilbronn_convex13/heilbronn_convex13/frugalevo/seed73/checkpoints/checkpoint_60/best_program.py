# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

_N = 13


def _triples():
    return np.array(list(combinations(range(_N), 3)), dtype=np.int64)


_TRIPLES = _triples()


def _areas(points, tri):
    a = points[tri[:, 0]]
    b = points[tri[:, 1]]
    c = points[tri[:, 2]]
    return np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                  - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])) * 0.5


def _area_grad(points, tri):
    """Analytic gradient of each triangle's (unsigned) area w.r.t. all points."""
    m = tri.shape[0]
    grad = np.zeros((m, _N, 2))
    for k in range(3):
        i = tri[:, k]
        j = tri[:, (k + 1) % 3]
        l = tri[:, (k + 2) % 3]
        cross = (points[j, 0] - points[i, 0]) * (points[l, 1] - points[i, 1]) - \
                (points[j, 1] - points[i, 1]) * (points[l, 0] - points[i, 0])
        s = np.sign(cross)
        s[cross == 0] = 1.0
        gx = 0.5 * s * (points[l, 1] - points[i, 1])
        gy = -0.5 * s * (points[l, 0] - points[i, 0])
        # d cross / d p_i
        grad[:, i, 0] += 0.5 * s * (points[j, 1] - points[l, 1])
        grad[:, i, 1] += -0.5 * s * (points[j, 0] - points[l, 0])
        # d cross / d p_j
        grad[:, j, 0] += gx
        grad[:, j, 1] += gy
        # d cross / d p_l
        grad[:, l, 0] += -gx
        grad[:, l, 1] += -gy
    return grad


def _softmin_val_grad(x, tri, T):
    points = x.reshape(_N, 2)
    ar = _areas(points, tri)
    w = np.exp(-(ar - ar.min()) / T)
    z = w.sum()
    val = -T * np.log(z) + ar.min()  # shifted log-sum-exp soft-min
    sw = w / z
    grad = _area_grad(points, tri)
    g = np.einsum('i,ijk->jk', sw, grad)
    return val, g.ravel()


def _exact_min(points, tri):
    return _areas(points, tri).min()


def _hull_area(points):
    try:
        return ConvexHull(points).volume
    except Exception:
        return 0.0


def _normalize(points):
    """Re-center to centroid and scale hull area to 1."""
    points = points - points.mean(axis=0)
    ha = _hull_area(points)
    if ha > 1e-12:
        points = points / np.sqrt(ha)
    return points


def _ring_points(params):
    """Center point at origin + 12-gon ring with Fourier radius profile.

    params = [phi, a1, b1, a2, b2, a3, b3]:
        r(theta) = 1 + sum_k a_k cos(6k theta) + b_k sin(6k theta),
    radii clipped to >= 0.2 so all ring points are distinct and star-shaped
    around the center; the hull is the ring (non-degenerate).
    """
    phi = float(params[0])
    k = np.arange(1, 13, dtype=np.float64)
    th = 2.0 * np.pi * (k - 1) / 12.0 + phi
    r = np.ones_like(th)
    for m in range(3):
        r = r + params[1 + 2 * m] * np.cos(6.0 * (m + 1) * th) \
              + params[2 + 2 * m] * np.sin(6.0 * (m + 1) * th)
    r = np.maximum(r, 0.2)
    pts = np.zeros((13, 2), dtype=np.float64)
    pts[0] = (0.0, 0.0)
    pts[1:, 0] = r * np.cos(th)
    pts[1:, 1] = r * np.sin(th)
    return pts


def _valid_candidate(pts, floor=1e-6):
    """Validity guard: finite coords, non-degenerate hull, sane normalized min area."""
    if pts is None or not np.all(np.isfinite(pts)):
        return -1.0
    ha = _hull_area(pts)
    if ha <= 1e-9:
        return -1.0
    em = _exact_min(pts, _TRIPLES)
    v = em / ha
    return v if v > floor else -1.0


def _seeds():
    """Deterministic seed configurations spanning hull/interior topologies."""
    seeds = []
    ang = 2.0 * np.pi * np.arange(_N) / _N
    seeds.append(np.stack([np.cos(ang), np.sin(ang)], axis=1))
    # 12-gon + center
    ang12 = 2.0 * np.pi * np.arange(12) / 12.0
    s = np.zeros((_N, 2))
    s[1:, 0] = np.cos(ang12); s[1:, 1] = np.sin(ang12)
    seeds.append(s)
    # 11-gon + 2 interior on a diameter
    ang11 = 2.0 * np.pi * np.arange(11) / 11.0
    s = np.zeros((_N, 2))
    s[2:, 0] = np.cos(ang11); s[2:, 1] = np.sin(ang11)
    s[0] = (0.3, 0.0); s[1] = (-0.3, 0.0)
    seeds.append(s)
    # 10-gon + 3 interior
    ang10 = 2.0 * np.pi * np.arange(10) / 10.0
    s = np.zeros((_N, 2))
    s[3:, 0] = np.cos(ang10); s[3:, 1] = np.sin(ang10)
    s[0] = (0.3, 0.1); s[1] = (-0.3, 0.0); s[2] = (0.0, -0.3)
    seeds.append(s)
    # Two fixed-seed quasi-random Sobol sets inside the unit disk
    try:
        from scipy.stats import qmc
        sob = qmc.Sobol(d=2, scramble=True, seed=7).random_base2(4)[:_N]
        seeds.append(2.0 * (sob - 0.5))
        sob = qmc.Sobol(d=2, scramble=True, seed=13).random_base2(4)[:_N]
        seeds.append(2.0 * (sob - 0.5))
    except Exception:
        pass
    return seeds


def heilbronn_convex13() -> np.ndarray:
    """
    Exact-objective multi-seed hill-climbing over all 26 coordinates with
    topological escape moves (hull vertex <-> interior role exchange).
    Six deterministic seeds (regular 13-gon; 12-gon + center; 11-gon + 2
    interior; 10-gon + 3 interior; two Sobol sets) each undergo <= 3000
    single-point Gaussian perturbation trials on the exact normalized
    min-triangle-area, sigma decaying 0.997 per rejection with periodic
    resets, and an escape move every 600 trials that swaps a hull vertex
    with an interior role to cross contact topologies. Global best valid
    incumbent is retained; regular 13-gon is the guaranteed fallback.
    """
    tri = _TRIPLES
    ang0 = 2.0 * np.pi * np.arange(_N) / _N
    fallback = np.stack([np.cos(ang0), np.sin(ang0)], axis=1)
    best = fallback.copy()
    best_val = _valid_candidate(best.copy())
    if best_val < 0.0:
        best_val = -np.inf

    rng = np.random.default_rng(12345)

    def score(pts):
        return _valid_candidate(pts.copy(), floor=0.0)

    try:
        for si, seed in enumerate(_seeds()):
            cur = np.ascontiguousarray(seed, dtype=np.float64).copy()
            cur_val = score(cur)
            if cur_val < 0.0:
                continue
            if cur_val > best_val:
                best_val, best = cur_val, cur.copy()
            sigma = 0.05
            trials = 0
            rejections = 0
            # Full deterministic hill-climb budget: this is the incumbent
            # producer and must not be reduced (the earlier 20k cut caused
            # a regression); the LP phase adds only a few seconds on top.
            while trials < 30000:
                trials += 1
                # Escape move every 2000 trials: role exchange
                if trials % 2000 == 0:
                    try:
                        hull = ConvexHull(cur)
                        hull_idx = list(hull.vertices)
                        inside = [i for i in range(_N) if i not in hull_idx]
                        ar = _areas(cur, tri)
                        m0 = ar.min()
                        active = np.where(ar <= m0 + 1e-12)[0]
                        cnt = {}
                        for t in _TRIPLES[active]:
                            for k in t:
                                cnt[int(k)] = cnt.get(int(k), 0) + 1
                        esc = cur.copy()
                        if inside:
                            # move the most-active hull vertex to interior
                            hv = max(hull_idx, key=lambda i: cnt.get(i, 0))
                            cen = cur[inside].mean(axis=0)
                            esc[hv] = cen * 0.5
                        elif hull_idx:
                            # move a random interior-active point to hull edge
                            iv = rng.choice(_N)
                            j = rng.choice(len(hull_idx), size=2, replace=False)
                            mid = 0.5 * (cur[hull_idx[j[0]]] + cur[hull_idx[j[1]]])
                            esc[iv] = mid
                        v_esc = score(esc)
                        improved = v_esc > cur_val
                        if improved:
                            cur, cur_val = esc, v_esc
                        # follow-up micro-perturbations
                        ref = cur_val if improved else cur_val
                        for _ in range(150):
                            mp = cur.copy()
                            mp += rng.normal(0.0, 0.01, mp.shape)
                            v_m = score(mp)
                            if v_m > ref:
                                cur, cur_val = mp, v_m
                                ref = v_m
                        if cur_val > best_val:
                            best_val, best = cur_val, cur.copy()
                        sigma = 0.01
                        continue
                    except Exception:
                        pass
                # Mixed move set: single-point kick (50%), coordinated move
                # of points participating in minimum-area (active) triangles
                # (30%), and a small coherent all-point move (20%).  The
                # active-set move targets exactly the degrees of freedom that
                # gate the exact objective, so mixed hull/interior topologies
                # created by escape moves converge much faster.  Sigma has a
                # floor so the search never freezes before its trial budget.
                u = rng.random()
                cand = cur.copy()
                if u < 0.5:
                    idx = int(rng.integers(_N))
                    cand[idx] += rng.normal(0.0, sigma, 2)
                elif u < 0.8:
                    ar = _areas(cur, tri)
                    m0 = ar.min()
                    act = np.where(ar <= m0 + 1e-12)[0]
                    pts_in = np.unique(_TRIPLES[act].ravel())
                    if pts_in.size == 0:
                        pts_in = np.arange(_N)
                    cand[pts_in] += rng.normal(0.0, sigma,
                                               (pts_in.size, 2))
                else:
                    cand += rng.normal(0.0, sigma * 0.5, cand.shape)
                v = score(cand)
                if v > cur_val:
                    cur, cur_val = cand, v
                    if cur_val > best_val:
                        best_val, best = cur_val, cur.copy()
                else:
                    rejections += 1
                    sigma = max(sigma * 0.997, 1e-5)
                    if rejections % 1500 == 0:
                        sigma = 0.02
    except Exception:
        pass

    # Exact epigraph-LP active-set refinement (mechanism family: epigraph-LP-
    # active-set), replacing the superseded soft-min L-BFGS-B polish.  For the
    # current point, linearize signed_area_k(x + dx) = s_k + G_k·dx >= t for
    # ALL 286 triples with signs s_k re-derived each iteration, and maximize t
    # subject to ||dx||_inf <= delta via HiGHS (27 vars, 286 rows, ~1 ms).
    # Steps are accepted when the exact normalized min-area (via
    # _valid_candidate) improves; delta grows on success, halves on failure.
    # Because signs are re-derived (not frozen, no flip guard), the identity of
    # the active triangles can change across iterations, so the solver
    # traverses contact topologies rather than terminating at a zero step.
    # Starts: the global incumbent plus up to 5 deterministic topology variants
    # (hull-vertex inward/outward displacements and a sheared 12-gon+center
    # seed), each refined by <= 30 LP iterations.  Budget <= 6 x 30 LP solves,
    # well under 5 s.  The untouched incumbent is returned on any failure.
    from scipy.optimize import linprog

    def _signed(pts):
        a = pts[tri[:, 0]]; b = pts[tri[:, 1]]; c = pts[tri[:, 2]]
        return 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                      - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))

    def _lp_seq(p0, max_iter=45):
        pts = _normalize(np.asarray(p0, dtype=np.float64).copy())
        cur_val = _valid_candidate(pts.copy(), floor=0.0)
        if cur_val <= 0.0:
            return pts, cur_val
        delta = 0.08
        nv = 2 * _N + 1
        for _ in range(max_iter):
            s0 = _signed(pts)
            sgn = np.where(s0 >= 0.0, 1.0, -1.0)
            A = np.zeros((tri.shape[0], nv))
            b = np.zeros(tri.shape[0])
            # signed area constraint: sgn*(s0 + G·dx) >= t
            #   <=>  t - sgn*G·dx <= sgn*s0
            A[:, 0] = 1.0
            b = sgn * s0
            i3, j3, l3 = tri[:, 0], tri[:, 1], tri[:, 2]
            Gxi = 0.5 * (pts[j3, 1] - pts[l3, 1])
            Gyi = -0.5 * (pts[j3, 0] - pts[l3, 0])
            Gxj = 0.5 * (pts[l3, 1] - pts[i3, 1])
            Gyj = -0.5 * (pts[l3, 0] - pts[i3, 0])
            Gxl = 0.5 * (pts[i3, 1] - pts[j3, 1])
            Gyl = -0.5 * (pts[i3, 0] - pts[j3, 0])
            np.add.at(A, (np.arange(tri.shape[0]), 1 + 2 * i3), -sgn * Gxi)
            np.add.at(A, (np.arange(tri.shape[0]), 2 + 2 * i3), -sgn * Gyi)
            np.add.at(A, (np.arange(tri.shape[0]), 1 + 2 * j3), -sgn * Gxj)
            np.add.at(A, (np.arange(tri.shape[0]), 2 + 2 * j3), -sgn * Gyj)
            np.add.at(A, (np.arange(tri.shape[0]), 1 + 2 * l3), -sgn * Gxl)
            np.add.at(A, (np.arange(tri.shape[0]), 2 + 2 * l3), -sgn * Gyl)
            c_obj = np.zeros(nv); c_obj[0] = -1.0
            bounds = [(None, None)] + [(-delta, delta)] * (2 * _N)
            res = linprog(c_obj, A_ub=A, b_ub=b, bounds=bounds, method='highs')
            if not res.success or not np.all(np.isfinite(res.x)):
                delta *= 0.5
                if delta < 1e-8:
                    break
                continue
            cand = pts + res.x[1:].reshape(_N, 2)
            v = _valid_candidate(_normalize(cand.copy()), floor=0.0)
            if v > cur_val + 1e-13:
                pts = _normalize(cand.copy())
                cur_val = v
                delta = min(delta * 1.5, 0.25)
            else:
                delta *= 0.5
                if delta < 1e-8:
                    break
        return pts, cur_val

    try:
        starts = []
        if best_val > -np.inf and best is not None:
            base = _normalize(np.asarray(best, dtype=np.float64).copy())
            starts.append(base)
            try:
                hull = ConvexHull(base)
                cen = base.mean(axis=0)
                hvs = list(hull.vertices)
                step = max(1, len(hvs) // 2)
                for hv in hvs[::step][:2]:
                    pi = base.copy(); pi[hv] = cen + 0.85 * (base[hv] - cen)
                    starts.append(_normalize(pi))
                    po = base.copy()
                    po[hv] = base[hv] + 0.12 * (base[hv] - cen)
                    starts.append(_normalize(po))
            except Exception:
                pass
        for seed in _seeds():
            if len(starts) >= 6:
                break
            sp = np.asarray(seed, dtype=np.float64).copy()
            sp += np.outer(sp[:, 1], [0.03, -0.02])
            if _valid_candidate(sp.copy(), floor=0.0) > 0.0:
                starts.append(_normalize(sp))
        # Ratio-objective-analytic-descent stage: SLSQP on the true scored
        # objective f(x) = min_k |area_k(x)| / hull_area(x) with the exact
        # analytic gradient (numerator: signed-area derivative of the argmin
        # triple; denominator: closed-form hull-edge area gradient).  The
        # hull-area term gives an inward pull on hull vertices that the
        # absolute-area epigraph LP structurally cannot produce, so this
        # generates ascent directions unavailable to _lp_seq.  A nonlinear
        # constraint keeps the active triple's signed area positive (no sign
        # flips).  Each SLSQP result is polished by one _lp_seq run and
        # accepted only on strict exact improvement of _valid_candidate, so
        # the incumbent is monotone.  Budget: <= 6 starts x <= 250 SLSQP
        # iterations (26 vars, 1 constraint) + one <=45-solve LP polish each.
        def _ratio_obj_grad(pts):
            ar = _areas(pts, tri)
            k = int(np.argmin(ar))
            i, j, l = (int(tri[k, 0]), int(tri[k, 1]), int(tri[k, 2]))
            ax, ay = pts[i]; bx, by = pts[j]; cx, cy = pts[l]
            C = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
            sgn = 1.0 if C >= 0.0 else -1.0
            A = 0.5 * abs(C)
            gA = np.zeros((_N, 2))
            gA[i, 0] += 0.5 * sgn * (ay - cy); gA[i, 1] += 0.5 * sgn * (cx - ax)
            gA[j, 0] += 0.5 * sgn * (cy - ay); gA[j, 1] += 0.5 * sgn * (ax - cx)
            gA[l, 0] += 0.5 * sgn * (ay - by); gA[l, 1] += 0.5 * sgn * (bx - ax)
            try:
                hull = ConvexHull(pts)
                hv = list(hull.vertices)
            except Exception:
                return 0.0, np.zeros((_N, 2)), 0.0, None, (i, j, l)
            m = len(hv)
            H = 0.0
            gH = np.zeros((_N, 2))
            for t in range(m):
                p0 = pts[hv[(t - 1) % m]]
                p1 = pts[hv[t]]
                p2 = pts[hv[(t + 1) % m]]
                H += p1[0] * p2[1] - p2[0] * p1[1]
                gH[hv[t], 0] += 0.5 * (p2[1] - p0[1])
                gH[hv[t], 1] += 0.5 * (p0[0] - p2[0])
            H *= 0.5
            grad = (gA * H - A * gH) / (H * H)
            return A / H, grad, H, np.array(hv, dtype=np.int64), (i, j, l)

        def _slsqp_ratio(p0, max_iter=250):
            pts = _normalize(np.asarray(p0, dtype=np.float64).copy())
            if _valid_candidate(pts.copy(), floor=0.0) <= 0.0:
                return pts, -1.0
            ar = _areas(pts, tri)
            kact = int(np.argmin(ar))
            ktup = (int(tri[kact, 0]), int(tri[kact, 1]),
                    int(tri[kact, 2]))

            def fun(x):
                P = x.reshape(_N, 2)
                v, _, _, _, _ = _ratio_obj_grad(P)
                return v

            def jac(x):
                P = x.reshape(_N, 2)
                _, g, _, _, _ = _ratio_obj_grad(P)
                return g.ravel()

            def con_f(x):
                P = x.reshape(_N, 2)
                a = P[ktup[0]]; b = P[ktup[1]]; c = P[ktup[2]]
                return 0.5 * ((b[0] - a[0]) * (c[1] - a[1])
                              - (b[1] - a[1]) * (c[0] - a[0]))

            def con_j(x):
                P = x.reshape(_N, 2)
                g = np.zeros((_N, 2))
                a = P[ktup[0]]; b = P[ktup[1]]; c = P[ktup[2]]
                g[ktup[0], 0] = 0.5 * (a[1] - c[1])
                g[ktup[0], 1] = 0.5 * (c[0] - a[0])
                g[ktup[1], 0] = 0.5 * (c[1] - a[1])
                g[ktup[1], 1] = 0.5 * (a[0] - c[0])
                g[ktup[2], 0] = 0.5 * (a[1] - b[1])
                g[ktup[2], 1] = 0.5 * (b[0] - a[0])
                return g.ravel()

            res = minimize(fun, pts.ravel(), jac=jac, method='SLSQP',
                           constraints=[{'type': 'ineq', 'fun': con_f,
                                         'jac': con_j}],
                           options={'maxiter': max_iter, 'ftol': 1e-12})
            out = np.asarray(res.x, dtype=np.float64).reshape(_N, 2)
            if np.all(np.isfinite(out)):
                out = _normalize(out.copy())
                if _valid_candidate(out.copy(), floor=0.0) > 0.0:
                    return out, _valid_candidate(out.copy(), floor=0.0)
            return pts, -1.0

        def _dual_ascent(p0, rounds=20):
            """LP-dual-weighted smooth ascent (mechanism family:
            lp-dual-weighted-area-ascent). Applied ONLY to LP-converged
            points, where the epigraph LP's dual multipliers lambda_k >= 0
            certify which triples (including 2nd/3rd-smallest, invisible to
            the exact-min active set) actually bind the trust region. Each
            round: (1) solve max t s.t. sgn_k*(s0_k + G_k.dx) >= t,
            ||dx||_inf <= delta and read normalized duals lambda; (2) run one
            L-BFGS-B (<=150 iters, 26 vars) on the smooth surrogate
            sum_k lambda_k * area_k(x) / hull_area(x) with exact analytic
            gradients (signed-area gradient via _area_grad, exact shoelace
            hull gradient); (3) accept only strict exact improvement of
            _valid_candidate, re-deriving lambda at the new point. No nudge
            drift: on stall the round simply ends (delta unchanged). A final
            _lp_seq polish re-maximizes the true min into created slack."""
            pts = _normalize(np.asarray(p0, dtype=np.float64).copy())
            cur_val = _valid_candidate(pts.copy(), floor=0.0)
            if cur_val <= 0.0:
                return pts, cur_val
            nv = 2 * _N + 1
            rows = np.arange(tri.shape[0])
            i3, j3, l3 = tri[:, 0], tri[:, 1], tri[:, 2]
            delta = 0.04
            for _r in range(rounds):
                s0 = _signed(pts)
                sgn = np.where(s0 >= 0.0, 1.0, -1.0)
                A = np.zeros((tri.shape[0], nv))
                b = sgn * s0
                A[:, 0] = 1.0
                Gxi = 0.5 * (pts[j3, 1] - pts[l3, 1])
                Gyi = -0.5 * (pts[j3, 0] - pts[l3, 0])
                Gxj = 0.5 * (pts[l3, 1] - pts[i3, 1])
                Gyj = -0.5 * (pts[l3, 0] - pts[i3, 0])
                Gxl = 0.5 * (pts[i3, 1] - pts[j3, 1])
                Gyl = -0.5 * (pts[i3, 0] - pts[j3, 0])
                np.add.at(A, (rows, 1 + 2 * i3), -sgn * Gxi)
                np.add.at(A, (rows, 2 + 2 * i3), -sgn * Gyi)
                np.add.at(A, (rows, 1 + 2 * j3), -sgn * Gxj)
                np.add.at(A, (rows, 2 + 2 * j3), -sgn * Gyj)
                np.add.at(A, (rows, 1 + 2 * l3), -sgn * Gxl)
                np.add.at(A, (rows, 2 + 2 * l3), -sgn * Gyl)
                c_obj = np.zeros(nv)
                c_obj[0] = -1.0
                bounds = [(None, None)] + [(-delta, delta)] * (2 * _N)
                res = linprog(c_obj, A_ub=A, b_ub=b, bounds=bounds,
                              method='highs')
                if not res.success or getattr(res, 'ineqlin', None) is None:
                    break
                lam = np.maximum(-np.asarray(res.ineqlin.marginals,
                                             dtype=np.float64), 0.0)
                tot = float(lam.sum())
                if not np.isfinite(tot) or tot <= 1e-12:
                    break
                lam = lam / tot

                def _hull_grad(P):
                    try:
                        hv = np.asarray(list(ConvexHull(P).vertices),
                                        dtype=np.int64)
                    except Exception:
                        return 0.0, None
                    m = hv.size
                    H = 0.0
                    gH = np.zeros((_N, 2))
                    for t in range(m):
                        pp = P[hv[(t - 1) % m]]
                        p1 = P[hv[t]]
                        pn = P[hv[(t + 1) % m]]
                        H += p1[0] * pn[1] - pn[0] * p1[1]
                        gH[hv[t], 0] += 0.5 * (pn[1] - pp[1])
                        gH[hv[t], 1] += 0.5 * (pp[0] - pn[0])
                    return 0.5 * H, gH

                def fun(x):
                    P = np.asarray(x, dtype=np.float64).reshape(_N, 2)
                    H = _hull_area(P)
                    if H <= 1e-12:
                        return 0.0
                    return float(_areas(P, tri).dot(lam)) / H

                def jac(x):
                    P = np.asarray(x, dtype=np.float64).reshape(_N, 2)
                    H, gH = _hull_grad(P)
                    if gH is None or H <= 1e-12:
                        return np.zeros(2 * _N)
                    gA = _area_grad(P, tri)
                    gW = np.einsum('k,kij->ij', lam, gA)
                    Abar = float(_areas(P, tri).dot(lam))
                    g = (gW * H - Abar * gH) / (H * H)
                    return g.ravel()

                r2 = minimize(fun, pts.ravel(), jac=jac,
                              method='L-BFGS-B',
                              bounds=[(-3.0, 3.0)] * (2 * _N),
                              options={'maxiter': 150, 'ftol': 1e-14})
                cand = np.asarray(r2.x, dtype=np.float64).reshape(_N, 2)
                if np.all(np.isfinite(cand)):
                    v = _valid_candidate(_normalize(cand.copy()), floor=0.0)
                    if v > cur_val + 1e-14:
                        pts = _normalize(cand.copy())
                        cur_val = v
                        delta = min(delta * 1.3, 0.12)
                        continue
                break
            return pts, cur_val

        for st in starts[:6]:
            p_s, v_s = _slsqp_ratio(st)
            p, v = _lp_seq(p_s)
            if v > best_val:
                best_val, best = v, np.ascontiguousarray(p.copy())
            # Dual-weighted ascent on the LP-converged point: the duals here
            # certify the true binding triples (incl. 2nd/3rd-smallest), and
            # the smooth weighted ascent trades min-area loss against gains
            # on the binding neighbors; the follow-up LP re-maximizes the
            # true min into the created slack. Strict improvement only.
            pd, vd = _dual_ascent(np.asarray(p, dtype=np.float64).copy())
            if vd > best_val:
                best_val, best = vd, np.ascontiguousarray(pd.copy())
            p2, v2 = _lp_seq(pd, max_iter=30)
            if v2 > best_val:
                best_val, best = v2, np.ascontiguousarray(p2.copy())
        # Deterministic LP topology-hop (extended budget): perturb the
        # LP-converged incumbent exactly on the points of its ACTIVE
        # (minimum-area) triangles, plus a deterministic rotation sweep, and
        # re-run the sequential LP. Rotation of the incumbent changes the LP
        # trajectory (which is not rotation-invariant in finitely many
        # steps), providing a symmetric-breaking mechanism that reaches
        # different contact topologies. All results are accepted only on
        # strict exact improvement, so the valid incumbent can never
        # degrade. Budget: 6 rounds x 3 reps x <=60 LP solves (<=1 ms each)
        # plus 4 rotation LP runs keeps added time under ~4 s.
        for rnd in range(18):
            base = _normalize(np.asarray(best, dtype=np.float64).copy())
            ar = _areas(base, tri)
            m0 = ar.min()
            act = np.unique(_TRIPLES[np.where(ar <= m0 + 1e-12)[0]].ravel())
            if act.size == 0:
                act = np.arange(_N)
            rng_lp = np.random.default_rng(900 + rnd)
            for rep in range(4):
                p0 = base.copy()
                p0[act] += rng_lp.normal(0.0, 0.004 + 0.002 * rnd,
                                         (act.size, 2))
                if _valid_candidate(_normalize(p0.copy()), floor=0.0) <= 0.0:
                    continue
                p, v = _lp_seq(p0, max_iter=60)
                if v > best_val:
                    best_val, best = v, np.ascontiguousarray(p.copy())
        # Deterministic radial-scaling variants of the current incumbent:
        # scaling hull vertices toward/away from the centroid changes which
        # triples are active at LP start, so the sequential LP (whose finite
        # trajectory depends on the starting active set) can land on
        # different contact topologies. Each result is accepted only on
        # strict exact improvement of _valid_candidate, so the incumbent is
        # monotone. Budget: 4 variants x <=45 LP solves (<=1 ms each).
        for fac in (0.92, 0.94, 0.96, 0.97, 1.03, 1.04, 1.06, 1.08):
            ps = _normalize(np.asarray(best, dtype=np.float64).copy())
            cen = ps.mean(axis=0)
            try:
                hvs = list(ConvexHull(ps).vertices)
            except Exception:
                break
            ps[hvs] = cen + fac * (ps[hvs] - cen)
            p, v = _lp_seq(ps, max_iter=45)
            if v > best_val:
                best_val, best = v, np.ascontiguousarray(p.copy())
        # Rotation sweep: rotate the incumbent by four deterministic angles
        # and refine each with the sequential LP (<=30 solves each). Because
        # the exact normalized objective is rotation-invariant only in the
        # exact objective, the LP finite-step trajectory differs per angle;
        # accept strictly-improving results.
        for ai in range(1, 19):
            th = 0.5 * np.pi * ai / 18.0
            ct, st_ = np.cos(th), np.sin(th)
            R = np.array([[ct, -st_], [st_, ct]], dtype=np.float64)
            p0 = _normalize(np.asarray(best, dtype=np.float64).copy()) @ R.T
            p, v = _lp_seq(p0, max_iter=30)
            if v > best_val:
                best_val, best = v, np.ascontiguousarray(p.copy())
        # Cross-topology convex-combination hops: blend the refined incumbent
        # with each of the diverse seed configurations (regular 13-gon,
        # 12-gon+center, 11-gon+2-interior, etc.). These basins lie in
        # genuinely different hull/interior topologies, so the sequential LP
        # starts from contact sets unreachable by perturbing the incumbent
        # alone. Weights 0.35/0.65 keep the point close enough to the
        # incumbent's quality while relocating the active set. Strict
        # improvement acceptance keeps the incumbent monotone. Budget:
        # <= 6 seeds x 2 weights x <=45 LP solves (~1 ms each).
        inc = _normalize(np.asarray(best, dtype=np.float64).copy())
        for seed in _seeds():
            sp = _normalize(np.asarray(seed, dtype=np.float64).copy())
            if _valid_candidate(sp.copy(), floor=0.0) <= 0.0:
                continue
            for w in (0.35, 0.65):
                mix = _normalize((1.0 - w) * inc + w * sp)
                if _valid_candidate(mix.copy(), floor=0.0) <= 0.0:
                    continue
                p, v = _lp_seq(mix, max_iter=45)
                if v > best_val:
                    best_val, best = v, np.ascontiguousarray(p.copy())
                    inc = _normalize(np.asarray(best,
                                                dtype=np.float64).copy())
        # Second rotation sweep from the (possibly updated) incumbent: since
        # the incumbent has changed, these LP trajectories are new relative
        # to the first sweep. Budget: 12 angles x <=30 LP solves.
        inc = _normalize(np.asarray(best, dtype=np.float64).copy())
        for ai in range(1, 25, 2):
            th = 2.0 * np.pi * ai / 48.0
            ct, st_ = np.cos(th), np.sin(th)
            R = np.array([[ct, -st_], [st_, ct]], dtype=np.float64)
            p0 = inc @ R.T
            p, v = _lp_seq(p0, max_iter=30)
            if v > best_val:
                best_val, best = v, np.ascontiguousarray(p.copy())
        # Deterministic shear hop sweep: the normalized objective is
        # invariant under rotations/scalings but NOT under shear, so small
        # anisotropic linear maps K = [[1, s], [s, 1]] present the sequential
        # LP with contact geometries unreachable by any previously probed hop
        # family (rotations are pure symmetries, radial scaling is isotropic,
        # noise is not a global linear deformation). Each sheared start is
        # LP-refined and accepted only on strict exact improvement of
        # _valid_candidate, so the incumbent is monotone. Budget:
        # 2 sign/scale families x 5 shears x <=45 LP solves (~1 ms each)
        # => added time under ~1 s.
        base = _normalize(np.asarray(best, dtype=np.float64).copy())
        for s in (0.04, -0.04, 0.09, -0.09, 0.15):
            for fam in (0, 1):
                K = (np.array([[1.0, s], [s, 1.0]]) if fam == 0
                     else np.array([[1.0, s], [-s, 1.0]]))
                p0 = base @ K.T
                if _valid_candidate(_normalize(p0.copy()), floor=0.0) <= 0.0:
                    continue
                p, v = _lp_seq(p0, max_iter=45)
                if v > best_val:
                    best_val, best = v, np.ascontiguousarray(p.copy())
                    base = _normalize(np.asarray(best,
                                                 dtype=np.float64).copy())
        # Permutation-continuation stage (new mechanism family): the
        # sequential LP's finite-step trajectory depends on the incumbent's
        # point ordering because each point maps to a fixed pair of LP
        # variables; cyclic relabelings of the same geometry therefore yield
        # different LP refinement paths, reaching contact topologies
        # unreachable by the spent rotation/shear/radial/noise hops (those
        # change coordinates, not the variable-to-vertex assignment for a
        # rotated copy is equivalent only under exact symmetry). For each of
        # 12 deterministic cyclic shifts (and their reversals), the
        # relabeled incumbent is LP-refined with max_iter=40, un-permuted,
        # and accepted only on strict exact improvement, so the incumbent is
        # monotone. Budget: 24 starts x <=40 LP solves (~1 ms each) < 1 s.
        perm_base = _normalize(np.asarray(best, dtype=np.float64).copy())
        nperm = _N
        for shift in range(1, 13):
            idx = [(shift + k) % _N for k in range(_N)]
            for rev in (False, True):
                order = idx[::-1] if rev else idx
                p0 = perm_base[order].copy()
                if _valid_candidate(_normalize(p0.copy()), floor=0.0) <= 0.0:
                    continue
                p, v = _lp_seq(p0, max_iter=40)
                if v > best_val:
                    best_val, best = v, np.ascontiguousarray(p.copy())
                    perm_base = _normalize(np.asarray(best,
                                                      dtype=np.float64).copy())
        # Hull-denominator-shrink with EXACT linearized hull area. The prior
        # variants minimized an inward radial proxy and stalled after a single
        # trust-region acceptance. The shoelace hull area H(x) = 0.5*sum
        # cross(p_t, p_{t+1}) over consecutive hull vertices is an EXACT
        # linear function of the vertex coordinates while the hull vertex
        # order is fixed, so the LP objective can be dH itself (gradient
        # 0.5*rot90(p_{t+1}-p_{t-1}) per hull vertex, zero for interior
        # points), giving a true steepest hull-area-descent step subject to
        # all 286 linearized signed areas >= t0 (signs re-derived per
        # iteration, exactly as in _lp_seq) and ||dx||_inf <= delta. The
        # stage alternates with the parent's _lp_seq (numerator
        # re-maximization into the slack created by the shrink) over 16
        # rounds; when a round yields no gain, the base is re-parameterized
        # by a deterministic rotation+shear pair (rotation/shear change the
        # LP trajectory and the hull topology presented to the shrink LP)
        # rather than Gaussian noise, keeping the loop deterministic and
        # monotone (accept only strict exact improvement via
        # _valid_candidate). Budget: 16 rounds x (<=20 shrink + <=45 remax)
        # LP solves (~1 ms each) => < 2 s added.
        def _shrink_exact(p0, max_iter=20):
            pts = _normalize(np.asarray(p0, dtype=np.float64).copy())
            cur_val = _valid_candidate(pts.copy(), floor=0.0)
            if cur_val <= 0.0:
                return pts, cur_val
            delta = 0.06
            nv = 2 * _N + 1
            rows = np.arange(tri.shape[0])
            for _ in range(max_iter):
                try:
                    hull = ConvexHull(pts)
                    hv = np.asarray(list(hull.vertices), dtype=np.int64)
                except Exception:
                    break
                if hv.size < 3:
                    break
                s0 = _signed(pts)
                t0 = s0.min() - 1e-12
                sgn = np.where(s0 >= 0.0, 1.0, -1.0)
                A = np.zeros((tri.shape[0] + 1, nv))
                b = np.zeros(tri.shape[0] + 1)
                A[:tri.shape[0], 0] = 1.0
                b[:tri.shape[0]] = sgn * s0
                i3, j3, l3 = tri[:, 0], tri[:, 1], tri[:, 2]
                Gxi = 0.5 * (pts[j3, 1] - pts[l3, 1])
                Gyi = -0.5 * (pts[j3, 0] - pts[l3, 0])
                Gxj = 0.5 * (pts[l3, 1] - pts[i3, 1])
                Gyj = -0.5 * (pts[l3, 0] - pts[i3, 0])
                Gxl = 0.5 * (pts[i3, 1] - pts[j3, 1])
                Gyl = -0.5 * (pts[i3, 0] - pts[j3, 0])
                np.add.at(A, (rows, 1 + 2 * i3), -sgn * Gxi)
                np.add.at(A, (rows, 2 + 2 * i3), -sgn * Gyi)
                np.add.at(A, (rows, 1 + 2 * j3), -sgn * Gxj)
                np.add.at(A, (rows, 2 + 2 * j3), -sgn * Gyj)
                np.add.at(A, (rows, 1 + 2 * l3), -sgn * Gxl)
                np.add.at(A, (rows, 2 + 2 * l3), -sgn * Gyl)
                A[tri.shape[0], 0] = -1.0
                b[tri.shape[0]] = -t0
                # exact dH: for hull vertex at index t with neighbors
                # p_prev, p_next, dH/dp_t = 0.5 * rot90(p_next - p_prev)
                c_obj = np.zeros(nv)
                m = hv.size
                for t in range(m):
                    pp = pts[hv[(t - 1) % m]]
                    pn = pts[hv[(t + 1) % m]]
                    c_obj[1 + 2 * hv[t]] = 0.5 * (pn[1] - pp[1])
                    c_obj[2 + 2 * hv[t]] = -0.5 * (pn[0] - pp[0])
                bounds = [(None, None)] + [(-delta, delta)] * (2 * _N)
                res = linprog(c_obj, A_ub=A, b_ub=b, bounds=bounds,
                              method='highs')
                if not res.success or not np.all(np.isfinite(res.x)):
                    delta *= 0.5
                    if delta < 1e-8:
                        break
                    continue
                cand = pts + res.x[1:].reshape(_N, 2)
                v = _valid_candidate(_normalize(cand.copy()), floor=0.0)
                if v > cur_val + 1e-13:
                    pts = _normalize(cand.copy())
                    cur_val = v
                    delta = min(delta * 1.5, 0.25)
                else:
                    delta *= 0.5
                    if delta < 1e-8:
                        break
            return pts, cur_val

        alt_base = _normalize(np.asarray(best, dtype=np.float64).copy())
        for rnd in range(16):
            ps, _ = _shrink_exact(alt_base, max_iter=20)
            p, v = _lp_seq(ps, max_iter=45)
            if v > best_val:
                best_val, best = v, np.ascontiguousarray(p.copy())
                alt_base = _normalize(np.asarray(best,
                                                 dtype=np.float64).copy())
            else:
                # deterministic rotation+shear re-parameterization to
                # re-expose denominator slack from a different geometry
                th = 2.0 * np.pi * (2 * rnd + 1) / 33.0
                s = 0.05 + 0.02 * rnd
                ct, st_ = np.cos(th), np.sin(th)
                K = np.array([[ct, -st_], [st_, ct]]) @ \
                    np.array([[1.0, s], [-s, 1.0]])
                pb = _normalize(np.asarray(best,
                                           dtype=np.float64).copy()) @ K.T
                if _valid_candidate(_normalize(pb.copy()), floor=0.0) > 0.0:
                    alt_base = _normalize(pb)
                else:
                    break
        # Second-order trust-region active-set refinement (Gauss-Newton /
        # Levenberg). At a kink where several triples are simultaneously
        # active, the first-order epigraph LP zig-zags and stalls; the GN
        # step instead solves the exact least-squares stationary direction
        # of the equalized active set jointly over all 26 coordinates:
        #   r_k = (t0 - area_k)/ha        for area_k <= t0 + 1e-9*ha
        #   r_H = (0.999999*ha - ha)/ha = -1e-6   (exact shoelace gradient
        #                                          row: shrink denominator)
        # and solves min ||J dx - r||^2 + lam ||dx||^2 via one damped
        # least-squares (26x26 normal equations, <0.1 ms per step). lam
        # halves on accepted steps, doubles on rejected ones. Each step is
        # accepted only when the exact normalized min-area improves.
        # Budget: <= 4 starts (incumbent + two shear variants + one
        # LP-polished variant) x <= 60 steps.
        def _gn_refine(p0, steps=60):
            pts = _normalize(np.asarray(p0, dtype=np.float64).copy())
            cur_val = _valid_candidate(pts.copy(), floor=0.0)
            if cur_val <= 0.0:
                return pts, cur_val
            lam = 1e-8
            eye = np.eye(2 * _N, dtype=np.float64)
            for _ in range(steps):
                ha = _hull_area(pts)
                if ha <= 1e-12:
                    break
                ar = _areas(pts, tri)
                t0 = ar.min()
                act = np.where(ar <= t0 + 1e-9 * ha)[0]
                if act.size == 0:
                    break
                J = _area_grad(pts, tri[act]).reshape(act.size, 2 * _N) / ha
                r = (t0 - ar[act]) / ha
                # exact shoelace hull-area gradient row (denominator shrink)
                try:
                    hvs = np.asarray(list(ConvexHull(pts).vertices),
                                     dtype=np.int64)
                except Exception:
                    break
                gh = np.zeros(2 * _N, dtype=np.float64)
                m = hvs.size
                for t in range(m):
                    pp = pts[hvs[(t - 1) % m]]
                    pn = pts[hvs[(t + 1) % m]]
                    gh[2 * hvs[t]] += 0.5 * (pn[1] - pp[1])
                    gh[2 * hvs[t] + 1] += -0.5 * (pn[0] - pp[0])
                J = np.vstack([J, gh.reshape(1, -1) / ha])
                r = np.concatenate([r, [-1e-6]])
                sq = np.sqrt(lam)
                A_aug = np.vstack([J, sq * eye])
                b_aug = np.concatenate([r, np.zeros(2 * _N)])
                try:
                    dx, *_ = np.linalg.lstsq(A_aug, b_aug, rcond=None)
                except Exception:
                    break
                if not np.all(np.isfinite(dx)):
                    lam = min(lam * 10.0, 1.0)
                    continue
                cand = pts + dx.reshape(_N, 2)
                v = _valid_candidate(_normalize(cand.copy()), floor=0.0)
                if v > cur_val + 1e-14:
                    pts = _normalize(cand.copy())
                    cur_val = v
                    lam = max(lam * 0.5, 1e-12)
                else:
                    lam = min(lam * 2.0, 1.0)
                    if lam >= 1.0:
                        break
            return pts, cur_val

        # Exact 1D breakpoint coordinate descent (mechanism family:
        # exact-1d-breakpoint-coordinate-descent), replacing the superseded
        # second-order trust-region (_gn_refine) and one-shot hull-shrink
        # polish. Along one coordinate x (a single component of a single
        # point), every triangle's signed area is exactly linear:
        # S_k(x) = a_k*x + b_k with a_k the analytic partial derivative and
        # b_k = S_k(x0) - a_k*x0. With the hull vertex order fixed (re-derived
        # per coordinate), the shoelace hull area is likewise exactly
        # H(x) = aH*x + bH. The normalized objective
        # g(x) = min_k |a_k x + b_k| / (aH x + bH) is therefore a ratio of a
        # concave piecewise-linear function to a positive linear function on
        # each kink-free interval; its maximizer over [x0-R, x0+R] lies at a
        # kink (zero of some a_k x + b_k), an interval endpoint, x0, or an
        # intersection of two near-active lines. All such candidates are
        # enumerated exactly (no trust region, no damping, no LP), evaluated
        # vectorized, and the best is accepted only if _valid_candidate
        # strictly improves the exact normalized min area. Sweeps stop when a
        # full pass over all 26 coordinates yields no improvement. Budget:
        # <= 26 coords x 8 sweeps x O(300) candidate evaluations with one
        # ConvexHull per coordinate — well under 0.5 s, less than the
        # removed GN/shrink polish.
        def _coord_descent(p0, sweeps=8, R=0.3):
            pts = _normalize(np.asarray(p0, dtype=np.float64).copy())
            cur_val = _valid_candidate(pts.copy(), floor=0.0)
            if cur_val <= 0.0:
                return pts, cur_val
            i3, j3, l3 = tri[:, 0], tri[:, 1], tri[:, 2]
            for _sweep in range(sweeps):
                any_impr = False
                for c in range(2 * _N):
                    p_i, comp = c // 2, c % 2
                    x0 = float(pts[p_i, comp])
                    s0 = _signed(pts)
                    mi = tri[:, 0] == p_i
                    mj = tri[:, 1] == p_i
                    ml = tri[:, 2] == p_i
                    ax_, ay_ = pts[i3, 0], pts[i3, 1]
                    bx_, by_ = pts[j3, 0], pts[j3, 1]
                    cx_, cy_ = pts[l3, 0], pts[l3, 1]
                    if comp == 0:
                        d_i = 0.5 * (ay_ - cy_)
                        d_j = 0.5 * (cy_ - ay_)
                        d_l = 0.5 * (ay_ - by_)
                    else:
                        d_i = 0.5 * (cx_ - ax_)
                        d_j = 0.5 * (ax_ - cx_)
                        d_l = 0.5 * (bx_ - ax_)
                    a = (np.where(mi, d_i, 0.0) + np.where(mj, d_j, 0.0)
                         + np.where(ml, d_l, 0.0))
                    b = s0 - a * x0
                    # Exact hull area as a linear function of x with the
                    # current hull vertex order fixed.
                    try:
                        hv = np.asarray(list(ConvexHull(pts).vertices),
                                        dtype=np.int64)
                    except Exception:
                        break
                    mH = hv.size
                    H0 = 0.0
                    aH = 0.0
                    for t in range(mH):
                        pp = pts[hv[(t - 1) % mH]]
                        p1 = pts[hv[t]]
                        pn = pts[hv[(t + 1) % mH]]
                        H0 += p1[0] * pn[1] - pn[0] * p1[1]
                        if hv[t] == p_i:
                            if comp == 0:
                                aH = 0.5 * (pn[1] - pp[1])
                            else:
                                aH = 0.5 * (pp[0] - pn[0])
                    H0 *= 0.5
                    bH = H0 - aH * x0
                    lo, hi = x0 - R, x0 + R
                    cand = [lo, hi, x0]
                    nz = np.abs(a) > 1e-14
                    zk = -b[nz] / a[nz]
                    zk = zk[(zk > lo) & (zk < hi)]
                    cand.extend(zk.tolist())
                    # Pairwise intersections of the 8 near-active lines:
                    # the concave lower envelope can peak where two of its
                    # supporting lines cross inside an interval.
                    order = np.argsort(np.abs(a * x0 + b))[:8]
                    for u in range(order.size):
                        for w in range(u + 1, order.size):
                            da = a[order[u]] - a[order[w]]
                            if abs(da) > 1e-14:
                                xi = (b[order[w]] - b[order[u]]) / da
                                if lo < xi < hi:
                                    cand.append(float(xi))
                    cand = np.asarray(cand, dtype=np.float64)
                    Hs = aH * cand + bH
                    ok = Hs > 1e-9
                    if not np.any(ok):
                        continue
                    g = np.full(cand.shape, -np.inf)
                    g[ok] = np.min(np.abs(np.outer(cand[ok], a) + b),
                                   axis=1) / Hs[ok]
                    xb = float(cand[int(np.argmax(g))])
                    if abs(xb - x0) < 1e-13:
                        continue
                    trial = pts.copy()
                    trial[p_i, comp] = xb
                    tn = _normalize(trial)
                    v = _valid_candidate(tn.copy(), floor=0.0)
                    if v > cur_val + 1e-14:
                        pts, cur_val = tn, v
                        any_impr = True
                if not any_impr:
                    break
            return pts, cur_val

        # Dinkelbach-ratio LP stage (mechanism family:
        # dinkelbach-ratio-lp-bisection). The incumbent's stages optimize
        # the ratio decomposed: _lp_seq maximizes the numerator with the
        # denominator frozen; _coord_descent moves one coordinate at a
        # time. A direction that raises the numerator while simultaneously
        # lowering the shoelace hull area is invisible to each decomposed
        # stage individually. The Dinkelbach LP expresses it in one step:
        # for a parameter rho, maximize t subject to, for ALL 286 triples,
        #   sgn_k*(s0_k + G_k.dx) - rho*(H0 + gH.dx) >= t,
        # ||dx||_inf <= delta, with signs and hull vertex order re-derived
        # each round exactly as in _lp_seq. The largest rho whose LP attains
        # t >= 0 is found by 8-round deterministic bisection; the
        # corresponding dx is accepted only on strict exact improvement of
        # _valid_candidate, with delta halving on failure (same discipline
        # as _lp_seq). Budget: incumbent + 5 deterministic shear variants,
        # each <= 4 alternating Dinkelbach/coord-descent rounds, each round
        # <= 12 bisection LPs + one coord-descent sweep (~1 ms per LP)
        # => well under 2 s added.
        def _dinkelbach(p0, rounds=12):
            """Joint-ratio LP: bisect rho over the linearized epigraph
            min_k sgn_k*(s0_k + G_k.dx) / (H0 + gH.dx) >= rho and return
            the dx that attains the best achievable linearized ratio."""
            pts = _normalize(np.asarray(p0, dtype=np.float64).copy())
            cur_val = _valid_candidate(pts.copy(), floor=0.0)
            if cur_val <= 0.0:
                return pts, cur_val
            delta = 0.08
            nv = 2 * _N + 1
            rows = np.arange(tri.shape[0])
            i3, j3, l3 = tri[:, 0], tri[:, 1], tri[:, 2]
            for _ in range(rounds):
                try:
                    hv = np.asarray(list(ConvexHull(pts).vertices),
                                    dtype=np.int64)
                except Exception:
                    break
                if hv.size < 3:
                    break
                s0 = _signed(pts)
                H0 = _hull_area(pts)
                if H0 <= 1e-12:
                    break
                sgn = np.where(s0 >= 0.0, 1.0, -1.0)
                # exact shoelace hull-area gradient (zero for interior pts)
                gh = np.zeros((2 * _N,), dtype=np.float64)
                m = hv.size
                for t in range(m):
                    pp = pts[hv[(t - 1) % m]]
                    pn = pts[hv[(t + 1) % m]]
                    gh[2 * hv[t]] += 0.5 * (pn[1] - pp[1])
                    gh[2 * hv[t] + 1] += -0.5 * (pn[0] - pp[0])
                Gxi = 0.5 * (pts[j3, 1] - pts[l3, 1])
                Gyi = -0.5 * (pts[j3, 0] - pts[l3, 0])
                Gxj = 0.5 * (pts[l3, 1] - pts[i3, 1])
                Gyj = -0.5 * (pts[l3, 0] - pts[i3, 0])
                Gxl = 0.5 * (pts[i3, 1] - pts[j3, 1])
                Gyl = -0.5 * (pts[i3, 0] - pts[j3, 0])
                base_A = np.zeros((tri.shape[0], nv))
                base_b = sgn * s0
                np.add.at(base_A, (rows, 1 + 2 * i3), -sgn * Gxi)
                np.add.at(base_A, (rows, 2 + 2 * i3), -sgn * Gyi)
                np.add.at(base_A, (rows, 1 + 2 * j3), -sgn * Gxj)
                np.add.at(base_A, (rows, 2 + 2 * j3), -sgn * Gyj)
                np.add.at(base_A, (rows, 1 + 2 * l3), -sgn * Gxl)
                np.add.at(base_A, (rows, 2 + 2 * l3), -sgn * Gyl)
                c_obj = np.zeros(nv)
                c_obj[0] = -1.0
                bounds = [(None, None)] + [(-delta, delta)] * (2 * _N)
                # bisection on rho: largest rho with LP optimum t >= 0
                lo, hi = 0.0, max(2.0 * cur_val, 0.05)
                best_dx = None
                for _bis in range(8):
                    rho = 0.5 * (lo + hi)
                    A = base_A.copy()
                    b = base_b - rho * H0
                    A[:, 1:] += rho * gh.reshape(1, -1)
                    res = linprog(c_obj, A_ub=A, b_ub=b, bounds=bounds,
                                  method='highs')
                    if res.success and np.all(np.isfinite(res.x)) \
                            and res.x[0] >= 0.0:
                        lo = rho
                        best_dx = res.x[1:].copy()
                    else:
                        hi = rho
                if best_dx is not None:
                    cand = pts + best_dx.reshape(_N, 2)
                    v = _valid_candidate(_normalize(cand.copy()), floor=0.0)
                    if v > cur_val + 1e-13:
                        pts = _normalize(cand.copy())
                        cur_val = v
                        delta = min(delta * 1.5, 0.25)
                        continue
                delta *= 0.5
                if delta < 1e-8:
                    break
            return pts, cur_val

        pc, vc = _coord_descent(np.asarray(best, dtype=np.float64).copy())
        if vc > best_val:
            best_val, best = vc, np.ascontiguousarray(pc.copy())
        # Dinkelbach <-> coord-descent alternation on the incumbent and 5
        # deterministic shear variants: each accepted Dinkelbach step
        # changes the active set, and the follow-up exact coordinate
        # descent re-polishes into the created slack, so the next joint
        # step is linearized at a fresh point. All acceptance is strict
        # exact improvement of _valid_candidate (monotone incumbent).
        dk_base = _normalize(np.asarray(best, dtype=np.float64).copy())
        dk_starts = [dk_base]
        for vi in range(5):
            s_dk = (0.05, -0.05, 0.09, -0.09, 0.15)[vi]
            K_dk = np.array([[1.0, s_dk], [-s_dk, 1.0]])
            pv_dk = dk_base @ K_dk.T
            if _valid_candidate(_normalize(pv_dk.copy()), floor=0.0) > 0.0:
                dk_starts.append(_normalize(pv_dk))
        for st_dk in dk_starts[:6]:
            pd, vd = _dinkelbach(st_dk, rounds=12)
            if vd > best_val:
                best_val, best = vd, np.ascontiguousarray(pd.copy())
            for _alt in range(4):
                pcd, vcd = _coord_descent(
                    np.asarray(best, dtype=np.float64).copy(), sweeps=3)
                if vcd > best_val:
                    best_val, best = vcd, np.ascontiguousarray(pcd.copy())
                pd, vd = _dinkelbach(
                    np.asarray(best, dtype=np.float64).copy(), rounds=10)
                if vd > best_val:
                    best_val, best = vd, np.ascontiguousarray(pd.copy())
    except Exception:
        pass

    if best is None or not np.all(np.isfinite(best)):
        best = fallback.copy()
    return np.ascontiguousarray(best, dtype=np.float64)


# EVOLVE-BLOCK-END
