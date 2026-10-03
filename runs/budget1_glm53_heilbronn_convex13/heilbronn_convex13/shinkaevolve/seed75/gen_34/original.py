# EVOLVE-BLOCK-START
import time
import numpy as np

_N = 13
_IDX = np.array([(i, j, k) for i in range(_N) for j in range(i + 1, _N)
                  for k in range(j + 1, _N)])
_TRIANGLES_WITH = []
for i in range(_N):
    rows = np.where((_IDX == i).any(axis=1))[0]
    _TRIANGLES_WITH.append(rows)


def _full_areas(points):
    a = points[_IDX[:, 0]]
    b = points[_IDX[:, 1]]
    c = points[_IDX[:, 2]]
    return 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) -
                        (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))


def _areas_for_point(points, i):
    """Areas of the 66 triangles containing point i (closed form, vectorized)."""
    others = np.arange(_N)
    others = others[others != i]
    jj, kk = np.triu_indices(len(others), 1)
    pj = points[others[jj]]
    pk = points[others[kk]]
    pi = points[i]
    return 0.5 * np.abs((pj[:, 0] - pi[0]) * (pk[:, 1] - pi[1]) -
                        (pj[:, 1] - pi[1]) * (pk[:, 0] - pi[0]))


def _hull_area(points):
    pts = sorted(map(tuple, points))
    if len(pts) <= 2:
        return 1e-9
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    h = np.asarray(hull)
    if len(h) < 3:
        return 1e-9
    x, y = h[:, 0], h[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def _clip(p):
    return np.clip(p, 0.0, 1.0)


def heilbronn_convex13() -> np.ndarray:
    """
    Hybrid optimizer: fast incremental greedy descent (one point move ->
    only its 66 triangles change) combined with coordinated moves on the
    bottleneck triangle's vertices and a soft (mean of k smallest) objective
    for uphill escape, followed by a strict bottleneck-focused polish.
    Deterministic (fixed seeds) and time-budgeted.
    """
    n = _N
    rng = np.random.default_rng(seed=20240713)
    t0 = time.time()
    budget = 7.0

    def ring(nc, rad, phase=0.0, cx=0.5, cy=0.5, jitter=0.0):
        ang = phase + 2 * np.pi * np.arange(nc) / nc
        pts = np.stack([cx + rad * np.cos(ang) + rng.normal(0, jitter, nc),
                        cy + rad * np.sin(ang) + rng.normal(0, jitter, nc)],
                       axis=1)
        return _clip(pts)

    inits = []
    inits.append(np.vstack([ring(12, 0.47, 0.13), [[0.5, 0.5]]]))
    inits.append(ring(13, 0.47, 0.05))
    inits.append(np.vstack([ring(7, 0.47, 0.0), ring(6, 0.25, 0.5)]))
    inits.append(np.vstack([ring(8, 0.47, 0.2), ring(4, 0.22, 0.9),
                            [[0.5, 0.5]]]))
    inits.append(_clip(np.vstack([ring(12, 0.46, 0.7, jitter=0.03),
                                  [[0.5, 0.5]]])))
    inits.append(ring(13, 0.46, 0.4, jitter=0.03))
    inits.append(_clip(np.vstack([ring(7, 0.47, 0.3, jitter=0.03),
                                  ring(6, 0.24, 0.8, jitter=0.03)])))
    ang = rng.uniform(0, 2 * np.pi, n)
    r = 0.47 * np.sqrt(rng.uniform(0.2, 1.0, n))
    inits.append(_clip(np.stack([0.5 + r * np.cos(ang),
                                 0.5 + r * np.sin(ang)], axis=1)))

    best_pts, best_val = None, -1.0
    restart_budget = budget / len(inits)

    for ri, x0 in enumerate(inits):
        tr = t0 + restart_budget * (ri + 1)
        x = _clip(np.asarray(x0, dtype=float).copy())
        areas = _full_areas(x)
        ha = _hull_area(x)
        cur = areas.min() / ha if ha > 1e-9 else 0.0
        local_best, local_val = x.copy(), cur
        # soft objective: mean of smallest k areas (for uphill escape)
        k = 10
        cur_soft = np.partition(areas, k - 1)[:k].mean() / ha

        sigma = 0.02
        temp = 2e-4
        rrng = np.random.default_rng(seed=99000 + 137 * ri)

        def try_accept(cand, cand_areas, cand_ha, strict_only):
            nonlocal x, areas, ha, cur, cur_soft, local_best, local_val
            cand_min = cand_areas.min()
            cand_val = cand_min / cand_ha if cand_ha > 1e-9 else 0.0
            cand_soft = np.partition(cand_areas, k - 1)[:k].mean() / cand_ha
            if cand_val > cur + 1e-14:
                x, areas, ha, cur, cur_soft = (cand, cand_areas, cand_ha,
                                               cand_val, cand_soft)
                if cand_val > local_val:
                    local_val = cand_val
                    local_best = x.copy()
                return True
            if (not strict_only and
                    cand_soft > cur_soft - temp and cand_val > cur - temp):
                x, areas, ha, cur, cur_soft = (cand, cand_areas, cand_ha,
                                               cand_val, cand_soft)
                return True
            return False

        while time.time() - t0 < budget:
            if time.time() > tr:
                break
            improved = False
            for _ in range(300):
                if time.time() - t0 > budget:
                    break
                u = rrng.random()
                cand = x.copy()
                if u < 0.25:
                    # coordinated move on bottleneck triangle's 3 vertices
                    tri = _IDX[np.argmin(areas)]
                    for vi in tri:
                        cand[vi] = _clip(cand[vi] + rrng.normal(0, sigma, 2))
                    cand_areas = _full_areas(cand)
                    cand_ha = _hull_area(cand)
                    ok = try_accept(cand, cand_areas, cand_ha, True)
                else:
                    # single random point: incremental update
                    vi = rrng.integers(n)
                    cand[vi] = _clip(cand[vi] + rrng.normal(0, sigma * 2, 2))
                    if np.allclose(cand[vi], x[vi]):
                        continue
                    rows = _TRIANGLES_WITH[vi]
                    old_vals = areas[rows].copy()
                    new_tri = _areas_for_point(cand, vi)
                    areas[rows] = new_tri
                    # hull changes only if vi is a hull vertex
                    if abs(_hull_area(points=x) - _hull_area(x)) < 1e-12:
                        pass
                    is_hull = abs(_hull_area(x[[j for j in range(_N)
                                               if j != vi]]) - _hull_area(x)) > 1e-12
                    cand_ha = _hull_area(cand) if is_hull else ha
                    cand_areas = areas  # already updated in place
                    ok = try_accept(cand, cand_areas, cand_ha, u >= 0.7)
                    if not ok:
                        areas[rows] = old_vals
                if ok:
                    improved = True
            if not improved:
                sigma *= 0.6
                temp *= 0.7
                k = max(3, k - 1)
                if sigma < 3e-6:
                    break
            else:
                sigma = min(sigma * 1.12, 0.02)

        # strict bottleneck-focused polish of the local best
        pol = local_best.copy()
        pa = _full_areas(pol)
        pha = _hull_area(pol)
        pv = pa.min() / pha
        sig = 2e-3
        while sig > 1e-6 and time.time() - t0 < budget:
            improved = False
            for _ in range(100):
                if time.time() - t0 > budget:
                    break
                tri = _IDX[np.argmin(pa)]
                cand = pol.copy()
                if rrng.random() < 0.5:
                    for vi in tri:
                        cand[vi] = _clip(cand[vi] + rrng.normal(0, sig, 2))
                else:
                    for _ in range(3):
                        vi = rrng.integers(n)
                        cand[vi] = _clip(cand[vi] + rrng.normal(0, sig, 2))
                ca = _full_areas(cand)
                cha = _hull_area(cand)
                cv = ca.min() / cha if cha > 1e-9 else 0.0
                if cv > pv + 1e-13:
                    pol, pv, pa, pha = cand, cv, ca, cha
                    improved = True
            if not improved:
                sig *= 0.5
        if pv > best_val:
            best_val = pv
            best_pts = pol.copy()

    if best_pts is None:
        best_pts = np.vstack([ring(12, 0.47), [[0.5, 0.5]]])
    if not np.all(np.isfinite(best_pts)):
        best_pts = _clip(np.nan_to_num(best_pts))
    return best_pts


# EVOLVE-BLOCK-END