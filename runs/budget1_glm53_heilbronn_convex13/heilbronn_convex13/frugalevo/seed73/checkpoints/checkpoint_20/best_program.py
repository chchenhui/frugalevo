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
        for st in starts[:6]:
            p, v = _lp_seq(st)
            if v > best_val:
                best_val, best = v, np.ascontiguousarray(p.copy())
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
        for rnd in range(6):
            base = _normalize(np.asarray(best, dtype=np.float64).copy())
            ar = _areas(base, tri)
            m0 = ar.min()
            act = np.unique(_TRIPLES[np.where(ar <= m0 + 1e-12)[0]].ravel())
            if act.size == 0:
                act = np.arange(_N)
            rng_lp = np.random.default_rng(900 + rnd)
            for rep in range(3):
                p0 = base.copy()
                p0[act] += rng_lp.normal(0.0, 0.004 + 0.002 * rnd,
                                         (act.size, 2))
                if _valid_candidate(_normalize(p0.copy()), floor=0.0) <= 0.0:
                    continue
                p, v = _lp_seq(p0, max_iter=60)
                if v > best_val:
                    best_val, best = v, np.ascontiguousarray(p.copy())
        # Rotation sweep: rotate the incumbent by four deterministic angles
        # and refine each with the sequential LP (<=30 solves each). Because
        # the exact normalized objective is rotation-invariant, only the LP
        # finite-step trajectory differs; accept strictly-improving results.
        for ai in range(1, 5):
            th = 0.5 * np.pi * ai / 5.0
            ct, st_ = np.cos(th), np.sin(th)
            R = np.array([[ct, -st_], [st_, ct]], dtype=np.float64)
            p0 = _normalize(np.asarray(best, dtype=np.float64).copy()) @ R.T
            p, v = _lp_seq(p0, max_iter=30)
            if v > best_val:
                best_val, best = v, np.ascontiguousarray(p.copy())
    except Exception:
        pass

    if best is None or not np.all(np.isfinite(best)):
        best = fallback.copy()
    return np.ascontiguousarray(best, dtype=np.float64)


# EVOLVE-BLOCK-END
