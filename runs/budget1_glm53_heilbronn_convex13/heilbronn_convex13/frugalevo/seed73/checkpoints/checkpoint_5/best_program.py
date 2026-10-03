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
            while trials < 3000:
                trials += 1
                # Escape move every 600 trials: role exchange
                if trials % 600 == 0:
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
                        for _ in range(50):
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
                # standard single-point Gaussian perturbation
                cand = cur.copy()
                idx = int(rng.integers(_N))
                cand[idx] += rng.normal(0.0, sigma, 2)
                v = score(cand)
                if v > cur_val:
                    cur, cur_val = cand, v
                    if cur_val > best_val:
                        best_val, best = cur_val, cur.copy()
                else:
                    rejections += 1
                    sigma *= 0.997
                    if rejections % 500 == 0:
                        sigma = 0.02
    except Exception:
        pass

    if best is None or not np.all(np.isfinite(best)):
        best = fallback.copy()
    return np.ascontiguousarray(best, dtype=np.float64)


# EVOLVE-BLOCK-END
