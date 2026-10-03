# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def _largest_empty_center(centers):
    """Return the point in the unit square maximizing the distance to the
    nearest existing circle center or wall (greedy Apollonian insertion
    point), found by a coarse grid scan plus local refinement."""
    xs = np.linspace(0.0, 1.0, 161)
    X, Y = np.meshgrid(xs, xs)
    P = np.stack([X.ravel(), Y.ravel()], axis=1)
    d = np.sqrt(((P - 0.5) ** 2).sum(axis=1)) * 0.0 + 0.5  # wall distance cap
    d = np.minimum(d, 0.5 - np.abs(P[:, 0] - 0.5))
    d = np.minimum(d, 0.5 - np.abs(P[:, 1] - 0.5))
    for c in centers:
        d = np.minimum(d, np.sqrt(((P - c) ** 2).sum(axis=1)))
    k = int(np.argmax(d))
    bx, by = P[k]
    bd = float(d[k])
    step = 1.0 / 160
    for _ in range(40):
        improved = False
        for dx in (-step, 0.0, step):
            for dy in (-step, 0.0, step):
                qx = min(max(bx + dx, 0.0), 1.0)
                qy = min(max(by + dy, 0.0), 1.0)
                qd = min(0.5 - abs(qx - 0.5), 0.5 - abs(qy - 0.5))
                if len(centers):
                    qd = min(qd, np.sqrt(((centers - [qx, qy]) ** 2).sum(axis=1)).min())
                if qd > bd + 1e-12:
                    bx, by, bd = qx, qy, qd
                    improved = True
        if not improved:
            step *= 0.5
            if step < 1e-9:
                break
    return np.array([bx, by]), bd


def _polish(centers, radii, maxiter=200, height=1.0):
    """SLSQP refinement of the 78 variables (52 center coords + 26 radii)
    maximizing the sum of radii subject to wall and pairwise-distance
    constraints in a 1 x height box (height=1 is the unit square).
    Returns refined centers, radii, or the inputs on failure."""
    try:
        from scipy.optimize import minimize
    except Exception:
        return centers, radii
    n = len(centers)
    x0 = np.concatenate([centers.ravel(), radii])

    def unpack(v):
        return v[: 2 * n].reshape(n, 2), v[2 * n:]

    def obj(v):
        return -np.sum(v[2 * n:])

    def grad(v):
        g = np.zeros_like(v)
        g[2 * n:] = -1.0
        return g

    def cons_wall(v):
        c, r = unpack(v)
        return np.concatenate([c[:, 0] - r, c[:, 1] - r,
                               1 - c[:, 0] - r, height - c[:, 1] - r])

    def jac_wall(v):
        c, r = unpack(v)
        J = np.zeros((4 * n, 3 * n))
        for k in range(n):
            J[k, 2 * k] = 1.0
            J[k, 2 * n + k] = -1.0
            J[n + k, 2 * k + 1] = 1.0
            J[n + k, 2 * n + k] = -1.0
            J[2 * n + k, 2 * k] = -1.0
            J[2 * n + k, 2 * n + k] = -1.0
            J[3 * n + k, 2 * k + 1] = -1.0
            J[3 * n + k, 2 * n + k] = -1.0
        return J

    def cons_pair(v):
        c, r = unpack(v)
        out = []
        for i in range(n):
            for j in range(i + 1, n):
                dd = np.sqrt(np.sum((c[i] - c[j]) ** 2))
                out.append(dd - r[i] - r[j])
        return np.array(out)

    def jac_pair(v):
        c, r = unpack(v)
        m = n * (n - 1) // 2
        J = np.zeros((m, 3 * n))
        row = 0
        for i in range(n):
            for j in range(i + 1, n):
                diff = c[i] - c[j]
                dd = max(np.sqrt(np.sum(diff ** 2)), 1e-12)
                J[row, 2 * i] = diff[0] / dd
                J[row, 2 * i + 1] = diff[1] / dd
                J[row, 2 * j] = -diff[0] / dd
                J[row, 2 * j + 1] = -diff[1] / dd
                J[row, 2 * n + i] = -1.0
                J[row, 2 * n + j] = -1.0
                row += 1
        return J

    res = minimize(obj, x0, jac=grad, method="SLSQP",
                   constraints=[{"type": "ineq", "fun": cons_wall,
                                 "jac": jac_wall},
                                {"type": "ineq", "fun": cons_pair,
                                 "jac": jac_pair}],
                   options={"maxiter": maxiter, "ftol": 1e-10})
    c, r = unpack(res.x)
    if not np.all(np.isfinite(c)) or not np.all(np.isfinite(r)):
        return centers, radii
    return c, np.maximum(r, 1e-9)


def _tighten(centers, radii, rounds=12, grow=1.004):
    """Active-set contact-graph tightening via grow-and-reproject:
    detect touching pairs (d_ij ~ r_i+r_j) and wall contacts, then
    repeatedly inflate all radii by a small factor and re-solve the
    equality system (pair distances = r_i+r_j, wall coords = r / 1-r)
    with bounded damped least-squares so centers shift to restore exact
    simultaneous contact. A shrink guard enforces strict feasibility;
    result accepted only if it improves sum(radii)."""
    n = len(radii)
    c0 = np.asarray(centers, dtype=float).copy()
    r0 = np.asarray(radii, dtype=float).copy()
    pairs, walls = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(c0[i] - c0[j]) - r0[i] - r0[j] < 5e-4:
                pairs.append((i, j))
    for i in range(n):
        for a in range(2):
            v = c0[i, a]
            if v - r0[i] < 5e-4:
                walls.append((i, a, 0))
            if 1.0 - v - r0[i] < 5e-4:
                walls.append((i, a, 1))

    def F(x):
        cc = x[: 2 * n].reshape(n, 2)
        rr = x[2 * n:]
        out = []
        for i, j in pairs:
            out.append(np.sqrt(np.sum((cc[i] - cc[j]) ** 2)) - rr[i] - rr[j])
        for i, a, s in walls:
            out.append((cc[i, a] - rr[i]) if s == 0
                       else (1.0 - cc[i, a] - rr[i]))
        return np.asarray(out)

    def guard(cc, rr):
        rr = np.maximum(rr, 1e-9)
        for i in range(n):
            lim = min(cc[i, 0], cc[i, 1], 1 - cc[i, 0], 1 - cc[i, 1])
            if rr[i] > lim:
                rr[i] = max(lim, 1e-9)
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt(np.sum((cc[i] - cc[j]) ** 2))
                if rr[i] + rr[j] > d:
                    sc = d / (rr[i] + rr[j])
                    rr[i] *= sc
                    rr[j] *= sc
        return cc, np.maximum(rr - 1e-9, 1e-9)

    x = np.concatenate([c0.ravel(), r0])
    try:
        from scipy.optimize import least_squares
        for _ in range(rounds):
            x[2 * n:] *= grow  # inflate, then reproject to exact contact
            sol = least_squares(F, x, method="lm", max_nfev=80,
                                xtol=1e-12, ftol=1e-12)
            x = sol.x
    except Exception:
        pass
    cc = np.asarray(x[: 2 * n]).reshape(n, 2)
    rr = np.asarray(x[2 * n:], dtype=float)
    cc, rr = guard(cc, rr)
    if np.all(np.isfinite(cc)) and np.all(np.isfinite(rr)) \
            and np.sum(rr) > np.sum(radii):
        return cc, rr
    return np.asarray(centers, dtype=float), np.asarray(radii, dtype=float)


def _destroy_repair(centers, radii, k, rng):
    """k-destroy-gap-repair: delete the k smallest-radius circles to expose
    a cavity, then rebuild k circles inside it by iterated largest-gap
    completion using exact closed-form solves (3-circle Descartes, 2-circle
    +wall quadratic, 1-circle+2-perpendicular-walls quadratic) over the
    cavity's boundary contact set; largest valid solution per insertion,
    greedy largest-empty fallback. Returns (new_centers, new_radii)."""
    n = len(radii)
    c = np.asarray(centers, dtype=float).copy()
    r = np.asarray(radii, dtype=float).copy()
    k = int(min(max(k, 1), n // 4))
    order = np.argsort(r)
    remove = set(int(i) for i in order[:k])
    keep = [i for i in range(n) if i not in remove]
    pts = [c[i].tolist() for i in keep]
    rad = [float(r[i]) for i in keep]

    # adaptive tangency tolerance from the removed circles' contact slacks
    slack = []
    for i in sorted(remove):
        for j in keep:
            slack.append(abs(np.hypot(c[i, 0] - c[j, 0], c[i, 1] - c[j, 1])
                             - r[i] - r[j]))
        slack.append(abs(min(c[i, 0], c[i, 1], 1 - c[i, 0], 1 - c[i, 1])
                         - r[i]))
    tol = float(np.clip(np.percentile(slack, 60) * 2.0 if slack else 5e-3,
                        3e-3, 1.5e-2))

    def gap_3c(a, b, d):
        b1, b2, b3 = 1.0 / rad[a], 1.0 / rad[b], 1.0 / rad[d]
        disc = b1 * b2 + b2 * b3 + b3 * b1
        if disc <= 0:
            return None
        b4 = b1 + b2 + b3 - 2.0 * np.sqrt(disc)
        if b4 <= 1e-9:
            return None
        z1, z2, z3 = complex(*pts[a]), complex(*pts[b]), complex(*pts[d])
        s2 = 2.0 * np.sqrt(b1 * b2 * z1 * z2 + b2 * b3 * z2 * z3
                           + b3 * b1 * z3 * z1)
        z4 = (b1 * z1 + b2 * z2 + b3 * z3 - s2) / b4
        rr = 1.0 / b4
        x, y = float(z4.real), float(z4.imag)
        if rr <= 1e-6 or not (1e-4 < x < 1 - 1e-4 and 1e-4 < y < 1 - 1e-4):
            return None
        return ([x, y], rr)

    def gap_2c1w(ci, cj, ax, sd):
        u1, u2 = pts[ci][ax], pts[cj][ax]
        v1 = pts[ci][1 - ax]
        r1, r2 = rad[ci], rad[cj]
        if abs(u1 - u2) < 1e-9:
            return None
        A = (r1 * r1 - r2 * r2 - u1 * u1 + u2 * u2) / (2.0 * (u2 - u1))
        B = (r1 - r2) / (u2 - u1)
        p = A - u1
        a_q = B * B
        b_q = 2.0 * p * B - 2.0 * v1 - 2.0 * r1
        c_q = p * p + v1 * v1 - r1 * r1
        disc = b_q * b_q - 4.0 * a_q * c_q
        if disc < 0:
            return None
        sq = np.sqrt(disc)
        sols = ([-c_q / b_q] if a_q < 1e-14
                else [(-b_q - sq) / (2.0 * a_q), (-b_q + sq) / (2.0 * a_q)])
        best = None
        for rr in sols:
            if rr <= 1e-5:
                continue
            x = A + B * rr
            v = rr if sd == 0 else 1.0 - rr
            cen = [x, v] if ax == 0 else [v, x]
            if not (1e-4 < cen[0] < 1 - 1e-4 and 1e-4 < cen[1] < 1 - 1e-4):
                continue
            if best is None or rr > best[1]:
                best = (cen, rr)
        return best

    def gap_1c2w(ci, a1, s1, a2, s2_):
        if a1 == a2:
            return None
        cx, cy = pts[ci]
        r1 = rad[ci]
        o1, e1 = (0.0, 1.0) if s1 == 0 else (1.0, -1.0)
        o2, e2 = (0.0, 1.0) if s2_ == 0 else (1.0, -1.0)
        b_q = -2.0 * (e1 * (cx - o1) + e2 * (cy - o2) + r1)
        c_q = (o1 - cx) ** 2 + (o2 - cy) ** 2 - r1 * r1
        disc = b_q * b_q - 8.0 * c_q
        if disc < 0:
            return None
        sq = np.sqrt(disc)
        best = None
        for s in [(-b_q - sq) / 4.0, (-b_q + sq) / 4.0]:
            if s <= 1e-5:
                continue
            p0 = o1 + e1 * s
            p1 = o2 + e2 * s
            cen = [p0, p1] if a1 == 0 else [p1, p0]
            if not (1e-4 < p0 < 1 - 1e-4 and 1e-4 < p1 < 1 - 1e-4):
                continue
            if best is None or s > best[1]:
                best = (cen, s)
        return best

    for _ in range(k):
        m = len(pts)
        T = set()
        for i in range(m):
            for j in range(i + 1, m):
                d = np.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])
                if abs(d - rad[i] - rad[j]) < tol:
                    T.add((i, j))
        TW = set()
        for i in range(m):
            for ax in range(2):
                for sd in range(2):
                    v = pts[i][ax]
                    if abs((v - rad[i]) if sd == 0
                           else (1.0 - v - rad[i])) < tol:
                        TW.add((i, ax, sd))
        best = None
        for a in range(m):
            for b in range(a + 1, m):
                if (a, b) not in T:
                    continue
                for d in range(b + 1, m):
                    if (a, d) not in T or (b, d) not in T:
                        continue
                    res = gap_3c(a, b, d)
                    if res is not None and (best is None or res[1] > best[1]):
                        best = res
        for (ci, ax, sd) in TW:
            for cj in range(m):
                if cj <= ci or (cj, ax, sd) not in TW or (ci, cj) not in T:
                    continue
                res = gap_2c1w(ci, cj, ax, sd)
                if res is not None and (best is None or res[1] > best[1]):
                    best = res
        for (ci, a1, s1) in TW:
            for (cj, a2, s2_) in TW:
                if cj != ci or a2 <= a1:
                    continue
                res = gap_1c2w(ci, a1, s1, a2, s2_)
                if res is not None and (best is None or res[1] > best[1]):
                    best = res
        if best is None:
            P = np.stack(np.meshgrid(np.linspace(0, 1, 101),
                                     np.linspace(0, 1, 101)), axis=-1
                         ).reshape(-1, 2)
            d = np.minimum(np.minimum(P[:, 0], 1 - P[:, 0]),
                           np.minimum(P[:, 1], 1 - P[:, 1]))
            for i in range(m):
                d = np.minimum(d, np.hypot(P[:, 0] - pts[i][0],
                                           P[:, 1] - pts[i][1]) - rad[i])
            kk = int(np.argmax(d))
            best = ([float(P[kk, 0]), float(P[kk, 1])], float(d[kk]) * 0.995)
        pts.append([float(best[0][0]), float(best[0][1])])
        rad.append(float(best[1]))
    return np.array(pts, dtype=float), np.array(rad, dtype=float)


def _diag_sym_candidate(rng, deadline, inc_c=None, inc_r=None, max_starts=40):
    """Diagonal-symmetry-reduction candidate generator (attempt 3).

    Primary start: SYMMETRIZED INCUMBENT — pair each incumbent circle with
    its nearest reflection across the diagonal x=y, average each pair to a
    single point in the triangle {y <= x}, and shrink radii to satisfy the
    mirror-clearance constraint r <= (x-y)/sqrt(2). This regularizes the
    incumbent onto its reflection-symmetric slice, then a 39-variable
    SLSQP (13 centers + 13 radii, diagonal as a hard wall, mirror
    clearance enforced) re-optimizes at half dimension. Secondary starts:
    a triangular hex patch and a greedy Apollonian-in-triangle chain.
    The best subproblem solve is mirrored to 26 circles and refined with
    _polish/_tighten/_final_guard (symmetry may break). Returns
    (centers, radii, sum) or None.
    """
    import time as _time
    m = 13
    tri = np.sqrt(2.0)

    def sub_polish(c0, r0, maxiter=200):
        x0 = np.concatenate([c0.ravel(), r0])

        def unpack(v):
            return v[:2 * m].reshape(m, 2), v[2 * m:]

        def obj(v):
            return -np.sum(v[2 * m:])

        def grad(v):
            g = np.zeros_like(v)
            g[2 * m:] = -1.0
            return g

        def cons(v):
            c, r = unpack(v)
            return np.concatenate([
                c[:, 0] - r,
                1.0 - c[:, 0] - r,
                c[:, 1] - r,
                (c[:, 0] - c[:, 1]) / tri - r,
                c[:, 0] - c[:, 1],
            ])

        def jac(v):
            c, r = unpack(v)
            J = np.zeros((5 * m, 3 * m))
            for k in range(m):
                J[k, 2 * k] = 1.0
                J[k, 2 * m + k] = -1.0
                J[m + k, 2 * k] = -1.0
                J[m + k, 2 * m + k] = -1.0
                J[2 * m + k, 2 * k + 1] = 1.0
                J[2 * m + k, 2 * m + k] = -1.0
                J[3 * m + k, 2 * k] = 1.0 / tri
                J[3 * m + k, 2 * k + 1] = -1.0 / tri
                J[3 * m + k, 2 * m + k] = -1.0
                J[4 * m + k, 2 * k] = 1.0
                J[4 * m + k, 2 * k + 1] = -1.0
            return J

        def cons_pair(v):
            c, r = unpack(v)
            out = []
            for i in range(m):
                for j in range(i + 1, m):
                    out.append(np.sqrt(np.sum((c[i] - c[j]) ** 2))
                               - r[i] - r[j])
            return np.array(out)

        def jac_pair(v):
            c, r = unpack(v)
            mm = m * (m - 1) // 2
            J = np.zeros((mm, 3 * m))
            row = 0
            for i in range(m):
                for j in range(i + 1, m):
                    diff = c[i] - c[j]
                    dd = max(np.sqrt(np.sum(diff ** 2)), 1e-12)
                    J[row, 2 * i] = diff[0] / dd
                    J[row, 2 * i + 1] = diff[1] / dd
                    J[row, 2 * j] = -diff[0] / dd
                    J[row, 2 * j + 1] = -diff[1] / dd
                    J[row, 2 * m + i] = -1.0
                    J[row, 2 * m + j] = -1.0
                    row += 1
            return J

        try:
            from scipy.optimize import minimize
            res = minimize(obj, x0, jac=grad, method="SLSQP",
                           constraints=[
                               {"type": "ineq", "fun": cons, "jac": jac},
                               {"type": "ineq", "fun": cons_pair,
                                "jac": jac_pair}],
                           options={"maxiter": maxiter, "ftol": 1e-10})
            c, r = unpack(res.x)
            if np.all(np.isfinite(c)) and np.all(np.isfinite(r)):
                return c, np.maximum(r, 1e-9)
        except Exception:
            pass
        return None

    def mirror_to_26(cs, rs):
        pts, rads = [], []
        for i in range(len(rs)):
            x, y = cs[i]
            if x - y < 1e-7:
                continue
            pts.append([x, y])
            rads.append(rs[i])
        if not pts:
            return None
        mir = [[p[1], p[0]] for p in pts]
        return (np.array(pts + mir, dtype=float),
                np.array(rads + rads, dtype=float))

    seeds = []
    # Primary: symmetrized incumbent (pair each circle with its diagonal
    # reflection, average to the triangle, enforce clearance).
    if inc_c is not None and inc_r is not None:
        ic = np.asarray(inc_c, dtype=float)
        ir = np.asarray(inc_r, dtype=float)
        used = np.zeros(len(ic), dtype=bool)
        sym = []
        for i in range(len(ic)):
            if used[i]:
                continue
            used[i] = True
            # nearest unused mirror partner
            best_j, best_d = -1, np.inf
            for j in range(len(ic)):
                if used[j]:
                    continue
                d = np.hypot(ic[i, 0] - ic[j, 1], ic[i, 1] - ic[j, 0])
                if d < best_d:
                    best_d, best_j = d, j
            if best_j >= 0:
                used[best_j] = True
                px = 0.5 * (ic[i, 0] + ic[best_j, 0])
                py = 0.5 * (ic[i, 1] + ic[best_j, 1])
            else:
                px, py = ic[i]
            px, py = 0.5 * (px + py + abs(px - py)), 0.5 * (px + py - abs(px - py))
            sym.append([min(max(px, 0.03), 0.97),
                        min(max(py, 0.02), px - 0.01)])
        sym = np.array(sym[:m], dtype=float)
        while len(sym) < m:
            t = 0.15 + 0.7 * rng.random()
            d = 0.03 + 0.15 * rng.random()
            sym = np.vstack([sym, [min(t + d, 0.95), max(t - d, 0.05)]])
        seeds.append(sym)
    # Secondary: triangular hex patch in the triangle
    tri_pts = []
    rows = [(4, 0), (4, 1), (3, 2)]
    s = 0.22
    for cnt, dy in rows:
        width = (cnt - 1) * s
        y = 0.14 + dy * s * 0.85
        for ci in range(cnt):
            x = 0.28 - width / 2.0 + ci * s + dy * s * 0.45
            if x >= y:
                tri_pts.append([min(max(x, 0.06), 0.94),
                                min(max(y, 0.04), x - 0.02)])
    while len(tri_pts) < m:
        t = 0.15 + 0.7 * rng.random()
        d = 0.03 + 0.15 * rng.random()
        tri_pts.append([min(t + d, 0.95), max(t - d, 0.05)])
    seeds.append(np.array(tri_pts[:m], dtype=float))

    best_c, best_r, best_s = None, None, -np.inf
    for sd in seeds:
        c0 = np.asarray(sd, dtype=float).copy()
        r0 = np.minimum(c0[:, 0] - c0[:, 1], c0[:, 1]) / tri * 0.95
        r0 = np.minimum(r0, np.minimum(c0[:, 0], 1.0 - c0[:, 0]) * 0.95)
        r0 = np.maximum(r0, 1e-4)
        starts = 0
        while starts < max_starts and _time.time() < deadline:
            starts += 1
            out = sub_polish(c0.copy(), r0.copy(), maxiter=200)
            if out is not None:
                cs, rs = out
                sc = float(np.sum(rs))
                if sc > best_s:
                    best_s = sc
                    best_c, best_r = cs.copy(), rs.copy()
            c0 = c0 + rng.normal(0.0, 0.02, size=c0.shape)
            c0[:, 1] = np.minimum(c0[:, 1], c0[:, 0] - 1e-3)
            c0 = np.clip(c0, 0.02, 0.98)
            r0 = np.minimum(np.minimum(c0[:, 0] - c0[:, 1],
                                       c0[:, 1]) / tri,
                            np.minimum(c0[:, 0], 1.0 - c0[:, 0])) * 0.95
            r0 = np.maximum(r0 * rng.uniform(0.9, 1.05, size=m), 1e-4)
    if best_c is None:
        return None
    pair = mirror_to_26(best_c, best_r)
    if pair is None:
        return None
    c26, r26 = pair
    c26, r26 = _polish(c26.copy(), r26.copy(), maxiter=250)
    c26, r26 = _tighten(c26, r26, rounds=14, grow=1.003)
    c26, r26 = _final_guard(c26, r26)
    if not np.all(np.isfinite(c26)) or not np.all(np.isfinite(r26)):
        return None
    return c26, r26, float(np.sum(r26))


def _final_guard(centers, radii):
    """Clip centers into the unit square and shrink any radius violating
    a wall or pairwise contact. Only shrinks; returns valid arrays."""
    n = len(radii)
    centers = np.clip(np.asarray(centers, dtype=float), 1e-9, 1 - 1e-9)
    radii = np.maximum(np.asarray(radii, dtype=float), 1e-9)
    for i in range(n):
        lim = min(centers[i, 0], centers[i, 1],
                  1 - centers[i, 0], 1 - centers[i, 1])
        if radii[i] > lim:
            radii[i] = max(lim, 1e-9)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if radii[i] + radii[j] > d:
                sc = d / (radii[i] + radii[j])
                radii[i] *= sc
                radii[j] *= sc
    return centers, radii


def _seed_hex_layers(n=26):
    """Deterministic hexagonal layered seed: rows of circles on a
    triangular lattice with exact tangency spacing s = 1/(2*(rows-1)+sqrt3)
    so adjacent rows touch; row lengths chosen as 5-4-5-4-5-3 to fill the
    square, centered per row."""
    rows = [5, 4, 5, 4, 5, 3]
    rows_total = 6
    s = 1.0 / (2.0 * (rows_total - 1) + np.sqrt(3.0))
    pts = []
    for ri, cnt in enumerate(rows):
        y = 0.5 + (ri - (rows_total - 1) / 2.0) * s * np.sqrt(3.0) / 2.0 * 2.0
        y = 0.5 + (ri - (rows_total - 1) / 2.0) * s * np.sqrt(3.0)
        width = (cnt - 1) * s
        for ci in range(cnt):
            x = 0.5 - width / 2.0 + ci * s
            pts.append([x, y])
    return np.array(pts[:n], dtype=float)


def _seed_grid_center(n=26):
    """Deterministic 5x5 square-grid seed (spacing 0.1, offset 0.1) plus
    one central circle; symmetry is broken later by _polish."""
    pts = [[0.1 + 0.2 * i, 0.1 + 0.2 * j] for i in range(5) for j in range(5)]
    pts.append([0.5, 0.5])
    return np.array(pts[:n], dtype=float)


def _seed_corner_shell(n=26):
    """Deterministic corner-anchored three-shell seed. Shell 1: four equal
    corner circles of radius r_c = (1 - 1/sqrt2)/2 tangent to two walls each.
    Shell 2: 3 wall-tangent circles per side (12 total) filling the free wall
    span between adjacent corner tangency points, tangent to the wall and to
    their wall neighbours. Shell 3: the remaining 10 circles on a 4-3-3
    triangular (hexagonal) patch centred at (0.5, 0.5). Constants are
    approximate; compute_max_radii shrinks to exact feasibility before
    polishing, so only basin proximity matters."""
    pts = []
    # Shell 1: corner circles
    rc = 0.5 - 0.5 / np.sqrt(2.0)  # ~0.1464
    pts += [[rc, rc], [1 - rc, rc], [rc, 1 - rc], [1 - rc, 1 - rc]]
    # Shell 2: per side, 3 circles tangent to the wall. Free wall span between
    # corner tangency points has length L = 1 - 2*rc*sqrt(2); edge radius
    # re = L / (2*3), circles tangent to wall and to each other and to the
    # corner circles' tangency projections.
    L = 1.0 - 2.0 * rc * np.sqrt(2.0)
    k = 3
    re = L / (2.0 * k)
    for i in range(k):
        u = rc * np.sqrt(2.0) + re + i * 2.0 * re
        pts += [[u, re], [u, 1.0 - re], [re, u], [1.0 - re, u]]
    # Shell 3: interior 10 circles as stacked rows 4-4-2 centred at
    # (0.5, 0.5). Inner free half-width after the edge shell is
    # 1/2 - 2*re (edge-circle inner envelope at 2*re); the widest
    # (4-circle) row spans 3 intervals, so use the tangency-matched
    # spacing s = (0.5 - 2*re)/1.5. Rows sit at 0, +sqrt3/2*s, and
    # 2*sqrt3/2*s so diagonal neighbours are near-tangent too, giving a
    # third structurally distinct interior basin (vs 4-3-3 and 3-4-3).
    s = (0.5 - 2.0 * re) / 1.5
    rows = [(4, 0.0), (4, 1.0), (2, 2.0)]
    for cnt, dy in rows:
        width = (cnt - 1) * s
        y = 0.5 + (dy - 1.0) * s * np.sqrt(3.0) / 2.0
        for ci in range(cnt):
            pts.append([0.5 - width / 2.0 + ci * s, y])
    return np.array(pts[:n], dtype=float)


def _seed_ring(n=26):
    """Deterministic Soddy/Descartes gap-completion seed with EXACT
    closed-form gap solves (no iterative root-finding, no line-geometry
    approximations). Shell 1: four corner circles r_c=(1-1/sqrt2)/2,
    tangent to two walls. Shell 2: two wall-tangent circles per side
    splitting the free wall span into three equal gaps. Shell 3: the
    remaining circles by iterative gap completion over curvilinear
    triangles of mutually tangent objects:
      - 3 circles: complex Descartes theorem (inner/minus branch).
      - 2 circles + 1 wall: wall coordinate fixed at y=r (etc.),
        subtracting the two circle equations gives x as a linear
        function of r; substituting back yields a quadratic in r.
      - 1 circle + 2 perpendicular walls: center at (s,s) from the
        shared corner; tangency to the circle gives a quadratic in s.
    At each round the largest valid interior solution is inserted; a
    greedy largest-empty insertion is the fallback. Produces a
    graded-curvature, wall-anchored topology no lattice/ring seed has.
    """
    pts, radii = [], []
    # Shell 1: corner circles
    rc = 0.5 - 0.5 / np.sqrt(2.0)
    for cx, cy in [(rc, rc), (1 - rc, rc), (rc, 1 - rc), (1 - rc, 1 - rc)]:
        pts.append([cx, cy])
        radii.append(rc)
    # Shell 2: two wall circles per side, tangent to wall and nearest corner
    u_t = rc * np.sqrt(2.0)
    re = (1.0 - 2.0 * u_t) / 6.0
    for u in (u_t + re, 1.0 - u_t - re):
        for cx, cy in [(u, re), (u, 1.0 - re), (re, u), (1.0 - re, u)]:
            pts.append([cx, cy])
            radii.append(re)
    NW = len(pts)
    walls = [(0, 0), (0, 1), (1, 0), (1, 1)]  # (axis, side); side0: coord=r

    def wall_coord(w):
        ax, sd = walls[w]
        return sd  # side 0 -> coord equals r, side 1 -> coord equals 1 - r

    def wall_slack(i, w):
        ax, sd = walls[w]
        v = pts[i][ax]
        return (v - radii[i]) if sd == 0 else (1.0 - v - radii[i])

    def gap_2c1w(ci, cj, w):
        """Circle tangent to wall w and to circles ci, cj (closed form)."""
        ax, sd = walls[w]
        x1, y1 = pts[ci]
        x2, y2 = pts[cj]
        r1, r2 = radii[ci], radii[cj]
        # work in rotated frame: wall at coordinate t=r, other coord u
        u1, v1 = (x1, y1) if ax == 0 else (y1, x1)
        u2, v2 = (x2, y2) if ax == 0 else (y2, x1)
        u2, v2 = (x2, y2) if ax == 0 else (y2, x2)
        if abs(u1 - u2) < 1e-9:
            return None
        # x(r) = A + B*r from subtracting the two circle equations
        A = (r1 * r1 - r2 * r2 - u1 * u1 + u2 * u2) / (2.0 * (u2 - u1))
        B = (r1 - r2) / (u2 - u1)
        # (A + B r - u1)^2 + (r - v1)^2 = (r1 + r)^2
        p = A - u1
        # p^2 + 2pB r + B^2 r^2 + r^2 - 2 v1 r + v1^2 = r1^2 + 2 r1 r + r^2
        a_q = B * B
        b_q = 2.0 * p * B - 2.0 * v1 - 2.0 * r1
        c_q = p * p + v1 * v1 - r1 * r1
        disc = b_q * b_q - 4.0 * a_q * c_q
        if a_q < 1e-14:
            # linear: 2 p B r + ... degenerate; skip (rare)
            if abs(b_q) < 1e-14:
                return None
            r = -c_q / b_q
            sols = [r]
        else:
            if disc < 0:
                return None
            sq = np.sqrt(disc)
            sols = [(-b_q - sq) / (2.0 * a_q), (-b_q + sq) / (2.0 * a_q)]
        best = None
        for r in sols:
            if r <= 1e-5:
                continue
            x = A + B * r
            cen = [x, r] if ax == 0 else [r, x]
            if not (1e-6 < cen[0] < 1 - 1e-6 and 1e-6 < cen[1] < 1 - 1e-6):
                continue
            # wall side 1: coordinate must be 1 - r
            if sd == 1:
                cen = [cen[0], cen[1]]
                if ax == 0:
                    cen[1] = 1.0 - cen[1]
                else:
                    cen[0] = 1.0 - cen[0]
            if best is None or r > best[1]:
                best = (cen, r)
        return best

    def gap_1c2w(ci, w1, w2):
        """Circle tangent to two perpendicular walls (sharing a corner)
        and to circle ci (closed form)."""
        if walls[w1][0] == walls[w2][0]:
            return None
        ax1, sd1 = walls[w1]
        ax2, sd2 = walls[w2]
        cx, cy = pts[ci]
        r1 = radii[ci]
        # center (s0, s1) with s_k = s (side 0) or 1 - s (side 1)
        # tangency: (s0-cx)^2 + (s1-cy)^2 = (r1+s)^2, quadratic in s
        # s0 = o1 + e1*s, s1 = o2 + e2*s with o=0/1, e=+1/-1
        o1, e1 = (0.0, 1.0) if sd1 == 0 else (1.0, -1.0)
        o2, e2 = (0.0, 1.0) if sd2 == 0 else (1.0, -1.0)
        # expand: e1^2=e2^2=1
        a_q = 2.0
        b_q = -2.0 * (e1 * (cx - o1) + e2 * (cy - o2) + r1)
        c_q = (o1 - cx) ** 2 + (o2 - cy) ** 2 - r1 * r1
        disc = b_q * b_q - 4.0 * a_q * c_q
        if disc < 0:
            return None
        sq = np.sqrt(disc)
        best = None
        for s in [(-b_q - sq) / 4.0, (-b_q + sq) / 4.0]:
            if s <= 1e-5:
                continue
            s0 = o1 + e1 * s
            s1 = o2 + e2 * s
            if not (1e-6 < s0 < 1 - 1e-6 and 1e-6 < s1 < 1 - 1e-6):
                continue
            if best is None or s > best[1]:
                best = ([s0, s1], s)
        return best

    def gap_3c(a, b, c):
        """Inner Soddy circle of three mutually tangent circles."""
        b1, b2, b3 = 1.0 / radii[a], 1.0 / radii[b], 1.0 / radii[c]
        disc = b1 * b2 + b2 * b3 + b3 * b1
        if disc <= 0:
            return None
        b4 = b1 + b2 + b3 - 2.0 * np.sqrt(disc)
        if b4 <= 1e-9:
            return None
        z1, z2, z3 = (complex(*pts[a]), complex(*pts[b]), complex(*pts[c]))
        s2 = 2.0 * np.sqrt(b1 * b2 * z1 * z2 + b2 * b3 * z2 * z3
                           + b3 * b1 * z3 * z1)
        z4 = (b1 * z1 + b2 * z2 + b3 * z3 - s2) / b4
        r = 1.0 / b4
        x, y = float(z4.real), float(z4.imag)
        if not (1e-6 < x < 1 - 1e-6 and 1e-6 < y < 1 - 1e-6):
            return None
        return ([x, y], r)

    for _ in range(n - NW):
        m = len(pts)
        T = set()
        for i in range(m):
            for j in range(i + 1, m):
                d = np.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])
                if abs(d - radii[i] - radii[j]) < 1e-6:
                    T.add((i, j))
        TW = set()
        for i in range(m):
            for w in range(4):
                if abs(wall_slack(i, w)) < 1e-6:
                    TW.add((i, w))
        best = None
        # 3-circle gaps
        for a in range(m):
            for b in range(a + 1, m):
                if (a, b) not in T:
                    continue
                for c in range(b + 1, m):
                    if (a, c) not in T or (b, c) not in T:
                        continue
                    res = gap_3c(a, b, c)
                    if res is not None and (best is None or res[1] > best[1]):
                        best = res
        # 2 circles + 1 wall
        for (ci, w) in TW:
            for cj in range(m):
                if cj <= ci or (cj, w) not in TW or (ci, cj) not in T:
                    continue
                res = gap_2c1w(ci, cj, w)
                if res is not None and (best is None or res[1] > best[1]):
                    best = res
        # 1 circle + 2 walls (perpendicular pair, both tangent to ci)
        for ci in range(m):
            for w1 in range(4):
                if (ci, w1) not in TW:
                    continue
                for w2 in range(w1 + 1, 4):
                    if (ci, w2) not in TW:
                        continue
                    res = gap_1c2w(ci, w1, w2)
                    if res is not None and (best is None or res[1] > best[1]):
                        best = res
        if best is None:
            P = np.stack(np.meshgrid(np.linspace(0, 1, 121),
                                     np.linspace(0, 1, 121)), axis=-1
                         ).reshape(-1, 2)
            d = np.minimum(np.minimum(P[:, 0], 1 - P[:, 0]),
                           np.minimum(P[:, 1], 1 - P[:, 1]))
            for i in range(m):
                d = np.minimum(d, np.hypot(P[:, 0] - pts[i][0],
                                           P[:, 1] - pts[i][1]) - radii[i])
            k = int(np.argmax(d))
            best = ([float(P[k, 0]), float(P[k, 1])], float(d[k]))
        pts.append([float(best[0][0]), float(best[0][1])])
        radii.append(float(best[1]))

    return np.array(pts[:n], dtype=float)


def _aspect_continuation(centers, radii, deadline):
    """Height-relaxation homotopy with n=26 fixed: for each container
    aspect ratio a in {1.05,1.10,1.15,0.95,0.90,0.85}, stretch the
    incumbent's y-coordinates into a 1 x a box, run SLSQP re-solves there,
    then continue back to height 1 in 5 equal steps, re-polishing at each
    step so the active contact set deforms continuously. The final unit-
    square state is tightened, guarded, and kept only on strict improvement
    over the best result seen so far (incumbent is the fallback). Hard
    deadline and per-solve maxiter bound total compute."""
    import time as _time
    c_best = np.array(centers, dtype=float).copy()
    r_best = np.array(radii, dtype=float).copy()
    s_best = float(np.sum(r_best))
    for a in (1.05, 1.10, 1.15, 0.95, 0.90, 0.85):
        if _time.time() > deadline:
            break
        c0 = np.array(centers, dtype=float).copy()
        c0[:, 1] = c0[:, 1] * a
        r0 = np.array(radii, dtype=float).copy()
        # shrink radii to fit the stretched box (y-wall limits)
        r0 = np.minimum(r0, np.minimum(c0[:, 1], a - c0[:, 1])) * 0.999
        r0 = np.maximum(r0, 1e-9)
        c, r = c0, r0
        for h in np.linspace(a, 1.0, 6):
            if _time.time() > deadline:
                break
            c, r = _polish(c.copy(), r.copy(), maxiter=300, height=float(h))
            r = np.maximum(np.asarray(r, dtype=float), 1e-9)
            c[:, 1] = np.clip(c[:, 1], r + 1e-9, h - r - 1e-9)
        tc, tr = _tighten(c, r)
        tc, tr = _final_guard(tc, tr)
        if np.all(np.isfinite(tc)) and np.all(np.isfinite(tr)) \
                and np.all(tr > 1e-9):
            s = float(np.sum(tr))
            if s > s_best:
                s_best = s
                c_best, r_best = tc, tr
    return c_best, r_best, s_best


def _lloyd_core(pts, iters=20, grid=55):
    """Lloyd fixed-point iteration on a grid-clipped Voronoi diagram:
    move each site to the damped centroid of its nearest-assignment cell.
    Fully vectorized; the result is only a seed — _eval_seed guarantees
    feasibility downstream."""
    ax_g = np.linspace(0.0, 1.0, grid)
    GX, GY = np.meshgrid(ax_g, ax_g)
    G = np.stack([GX.ravel(), GY.ravel()], axis=1)
    for _ in range(iters):
        d2 = ((G[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2)
        lab = np.argmin(d2, axis=1)
        new = pts.copy()
        cnts = np.bincount(lab, minlength=len(pts))
        sums = np.zeros_like(pts)
        np.add.at(sums, lab, G)
        ok = cnts > 0
        cen = sums / np.maximum(cnts, 1)[:, None]
        new[ok] = 0.5 * pts[ok] + 0.5 * cen[ok]
        pts = np.clip(new, 0.01, 0.99)
    return pts.astype(float)


def _seed_lloyd(n=26, iters=20, grid=55, seed=7):
    """Refined weighted-Lloyd seed generator: run the Lloyd dynamical
    system from two structurally distinct starts (jittered 5x5+1 grid and
    the hexagonal layered lattice), then pick the start whose Lloyd
    fixed point admits the larger max-feasible radius sum via the cheap
    `compute_max_radii` screen. Only the winner is returned, so the
    `_eval_seed` polish pipeline runs once — keeping the compute budget
    flat while probing two basins of the Lloyd attractor."""
    rng = np.random.default_rng(seed)
    grid_start = np.array([[0.1 + 0.2 * i, 0.1 + 0.2 * j]
                           for i in range(5) for j in range(5)],
                          dtype=float)
    grid_start = np.vstack([grid_start, [[0.5, 0.5]]])
    grid_start += rng.normal(0.0, 0.02, size=grid_start.shape)
    grid_start = np.clip(grid_start, 0.02, 0.98)
    hex_start = _seed_hex_layers(n)
    best_pts, best_s = None, -np.inf
    for start in (grid_start, hex_start):
        pts = _lloyd_core(start.copy(), iters=iters, grid=grid)[:n]
        s = float(np.sum(compute_max_radii(pts)))
        if np.isfinite(s) and s > best_s:
            best_s, best_pts = s, pts
    return best_pts if best_pts is not None else grid_start[:n]


def _eval_seed(seed_pts, budget_note=""):
    """Turn raw seed points into a polished, tightened, guarded solution;
    return (centers, radii, sum) with guards against invalid results."""
    pts = np.asarray(seed_pts, dtype=float).copy()
    if pts.shape[0] < 26:
        return None
    radii = compute_max_radii(pts)
    if not np.all(np.isfinite(radii)) or np.sum(radii) <= 1e-6:
        return None
    c, r = _polish(pts.copy(), radii.copy())
    c, r = _tighten(c, r)
    c, r = _final_guard(c, r)
    if not np.all(np.isfinite(c)) or not np.all(np.isfinite(r)):
        return None
    return c, r, float(np.sum(r))


def _best_gap(pts, rad, T, TW):
    """Closed-form largest-gap insertion among frozen circles: try the exact
    Descartes 3-circle solve, the 2-circle+wall quadratic, and the
    1-circle+2-perpendicular-walls quadratic; fall back to a coarse grid
    max-radius scan. Returns ([x, y], r) or None."""
    m = len(pts)
    best = None

    def consider(cand):
        nonlocal best
        if cand is not None and (best is None or cand[1] > best[1]):
            best = cand

    def gap_3c(a, b, d):
        b1, b2, b3 = 1.0 / rad[a], 1.0 / rad[b], 1.0 / rad[d]
        disc = b1 * b2 + b2 * b3 + b3 * b1
        if disc <= 0:
            return None
        b4 = b1 + b2 + b3 - 2.0 * np.sqrt(disc)
        if b4 <= 1e-9:
            return None
        z1, z2, z3 = complex(*pts[a]), complex(*pts[b]), complex(*pts[d])
        s2 = 2.0 * np.sqrt(b1 * b2 * z1 * z2 + b2 * b3 * z2 * z3
                           + b3 * b1 * z3 * z1)
        z4 = (b1 * z1 + b2 * z2 + b3 * z3 - s2) / b4
        rr = 1.0 / b4
        x, y = float(z4.real), float(z4.imag)
        if rr <= 1e-6 or not (1e-4 < x < 1 - 1e-4 and 1e-4 < y < 1 - 1e-4):
            return None
        return ([x, y], rr)

    def gap_2c1w(ci, cj, ax, sd):
        u1, u2 = pts[ci][ax], pts[cj][ax]
        v1 = pts[ci][1 - ax]
        r1, r2 = rad[ci], rad[cj]
        if abs(u1 - u2) < 1e-9:
            return None
        A = (r1 * r1 - r2 * r2 - u1 * u1 + u2 * u2) / (2.0 * (u2 - u1))
        B = (r1 - r2) / (u2 - u1)
        p = A - u1
        a_q = B * B
        b_q = 2.0 * p * B - 2.0 * v1 - 2.0 * r1
        c_q = p * p + v1 * v1 - r1 * r1
        disc = b_q * b_q - 4.0 * a_q * c_q
        if disc < 0:
            return None
        sq = np.sqrt(disc)
        sols = ([-c_q / b_q] if a_q < 1e-14
                else [(-b_q - sq) / (2.0 * a_q), (-b_q + sq) / (2.0 * a_q)])
        loc = None
        for rr in sols:
            if rr <= 1e-5:
                continue
            x = A + B * rr
            v = rr if sd == 0 else 1.0 - rr
            cen = [x, v] if ax == 0 else [v, x]
            if not (1e-4 < cen[0] < 1 - 1e-4 and 1e-4 < cen[1] < 1 - 1e-4):
                continue
            if loc is None or rr > loc[1]:
                loc = (cen, rr)
        return loc

    def gap_1c2w(ci, a1, s1, a2, s2_):
        if a1 == a2:
            return None
        cx, cy = pts[ci]
        r1 = rad[ci]
        o1, e1 = (0.0, 1.0) if s1 == 0 else (1.0, -1.0)
        o2, e2 = (0.0, 1.0) if s2_ == 0 else (1.0, -1.0)
        b_q = -2.0 * (e1 * (cx - o1) + e2 * (cy - o2) + r1)
        c_q = (o1 - cx) ** 2 + (o2 - cy) ** 2 - r1 * r1
        disc = b_q * b_q - 8.0 * c_q
        if disc < 0:
            return None
        sq = np.sqrt(disc)
        loc = None
        for s in [(-b_q - sq) / 4.0, (-b_q + sq) / 4.0]:
            if s <= 1e-5:
                continue
            p0 = o1 + e1 * s
            p1 = o2 + e2 * s
            if not (1e-4 < p0 < 1 - 1e-4 and 1e-4 < p1 < 1 - 1e-4):
                continue
            cen = [p0, p1] if a1 == 0 else [p1, p0]
            if loc is None or s > loc[1]:
                loc = (cen, s)
        return loc

    for a in range(m):
        for b in range(a + 1, m):
            if (a, b) not in T:
                continue
            for d in range(b + 1, m):
                if (a, d) not in T or (b, d) not in T:
                    continue
                consider(gap_3c(a, b, d))
    for (ci, ax, sd) in TW:
        for cj in range(m):
            if cj <= ci or (cj, ax, sd) not in TW or (ci, cj) not in T:
                continue
            consider(gap_2c1w(ci, cj, ax, sd))
    for (ci, a1, s1) in TW:
        for (cj, a2, s2_) in TW:
            if cj != ci or a2 <= a1:
                continue
            consider(gap_1c2w(ci, a1, s1, a2, s2_))
    if best is None:
        P = np.stack(np.meshgrid(np.linspace(0, 1, 81),
                                 np.linspace(0, 1, 81)), axis=-1).reshape(-1, 2)
        dd = np.minimum(np.minimum(P[:, 0], 1 - P[:, 0]),
                        np.minimum(P[:, 1], 1 - P[:, 1]))
        for i in range(m):
            dd = np.minimum(dd, np.hypot(P[:, 0] - pts[i][0],
                                         P[:, 1] - pts[i][1]) - rad[i])
        k = int(np.argmax(dd))
        best = ([float(P[k, 0]), float(P[k, 1])], float(dd[k]) * 0.999)
    return best


def _gs_regrowth(centers, radii, max_sweeps=4):
    """Gauss-Seidel single-circle maximal regrowth: in each sweep, visit
    circles in order of increasing radius; shrink circle i to zero against
    the 25 frozen circles/walls, then regrow it to its exact maximal radius
    at its current center OR reinsert it via closed-form Soddy gap solves
    (whichever is larger). Each step can retangent circle i to a different
    contact triple, changing the active set one vertex at a time — a move
    the joint 78-variable SLSQP cannot make from a KKT point. Returns the
    best guarded configuration seen across sweeps."""
    c = np.asarray(centers, dtype=float).copy()
    r = np.asarray(radii, dtype=float).copy()
    n = len(r)
    s_best = float(np.sum(r))
    cb, rb = c.copy(), r.copy()
    for _ in range(max_sweeps):
        moved = False
        for i in np.argsort(r):
            idx = [j for j in range(n) if j != i]
            pts = [c[j].tolist() for j in idx]
            rad = [float(r[j]) for j in idx]
            # Option 1: maximal radius at the current center vs frozen set
            lim = min(c[i, 0], c[i, 1], 1 - c[i, 0], 1 - c[i, 1])
            for p, rr in zip(pts, rad):
                lim = min(lim, np.hypot(c[i, 0] - p[0], c[i, 1] - p[1]) - rr)
            best = (c[i].copy(), float(lim))
            # Option 2: largest closed-form gap insertion in the cavity
            m = len(pts)
            T, TW = set(), set()
            for a in range(m):
                for b in range(a + 1, m):
                    if abs(np.hypot(pts[a][0] - pts[b][0],
                                    pts[a][1] - pts[b][1])
                           - rad[a] - rad[b]) < 1e-6:
                        T.add((a, b))
                for ax in range(2):
                    for sd in range(2):
                        v = pts[a][ax]
                        if abs((v - rad[a]) if sd == 0
                               else (1.0 - v - rad[a])) < 1e-6:
                            TW.add((a, ax, sd))
            g = _best_gap(pts, rad, T, TW)
            if g is not None and g[1] > best[1]:
                best = (np.array(g[0], dtype=float), float(g[1]))
            if best[1] > r[i] + 1e-12:
                c[i], r[i] = best[0], best[1]
                moved = True
        c, r = _final_guard(c, r)
        s = float(np.sum(r))
        if s > s_best + 1e-12:
            s_best = s
            cb, rb = c.copy(), r.copy()
        if not moved:
            break
    return cb, rb


def _penalty_anneal(centers, radii, deadline, stages=8, maxiter=150):
    """Annealed soft-penalty restart: L-BFGS-B on -sum(r)+mu*viol^2 (analytic
    grad), mu from 50 to ~1e4; transient overlap crosses infeasible ridges
    to a new contact graph; hard tail _polish/_tighten/_final_guard; accept
    only if strictly better, else return inputs."""
    import time as _time
    from scipy.optimize import minimize
    n = len(radii)
    c0 = np.asarray(centers, dtype=float).copy()
    r0 = np.asarray(radii, dtype=float).copy()
    s_in = float(np.sum(r0))

    def unpack(v):
        return v[:2 * n].reshape(n, 2), v[2 * n:]

    def viol_parts(c, r):
        diff = c[:, None, :] - c[None, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=2))
        np.fill_diagonal(dist, np.inf)
        vp = np.maximum(0.0, r[:, None] + r[None, :] - dist)
        vw = np.stack([r - c[:, 0], r - c[:, 1],
                       r - (1 - c[:, 0]), r - (1 - c[:, 1])], axis=1)
        return vp, vw, dist, diff

    def obj_grad(v, mu):
        c, r = unpack(v)
        vp, vw, dist, diff = viol_parts(c, r)
        val = -np.sum(r) + mu * ((vp ** 2).sum() + (vw ** 2).sum())
        g = np.zeros_like(v)
        g[2 * n:] = -1.0
        act = vp > 0
        coef = 2.0 * mu * vp * act
        safe = np.where(act, dist, 1.0)
        gvec = (coef[:, :, None] * diff / safe[:, :, None]).sum(axis=1)
        g[:2 * n].reshape(n, 2)[:] = gvec
        g[2 * n:] += 2.0 * mu * (vp * act).sum(axis=1)
        signs = np.array([-1.0, -1.0, 1.0, 1.0])
        aw = vw > 0
        wterm = 2.0 * mu * vw * aw
        g[:2 * n].reshape(n, 2)[:, 0] += (wterm * signs).sum(axis=1)
        g[:2 * n].reshape(n, 2)[:, 1] += (wterm * signs).sum(axis=1)
        g[2 * n:] += wterm.sum(axis=1)
        return val, g

    x = np.concatenate([c0.ravel(), r0])
    mu = 50.0
    for _ in range(stages):
        if _time.time() > deadline:
            break
        res = minimize(obj_grad, x, args=(mu,), jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-12,
                                "gtol": 1e-10})
        x = res.x
        mu *= 3.5
    c, r = unpack(x)
    r = np.maximum(r, 1e-9)
    if not (np.all(np.isfinite(c)) and np.all(np.isfinite(r))):
        return np.asarray(centers, dtype=float), np.asarray(radii, dtype=float)
    c, r = _polish(c.copy(), r.copy(), maxiter=200)
    c, r = _tighten(c, r, rounds=12, grow=1.003)
    c, r = _final_guard(c, r)
    if np.all(np.isfinite(c)) and np.all(np.isfinite(r)) \
            and float(np.sum(r)) > s_in:
        return c, r
    return np.asarray(centers, dtype=float), np.asarray(radii, dtype=float)


def construct_packing():
    """Multiseed basin-hopping seeds, then time-budgeted restarts: a
    penalty-continuation arm (annealed soft constraints crossing infeasible
    ridges into new contact graphs) plus the jitter/topology ladder, keeping
    strictly improving valid results. Monotone non-decreasing overall."""
    import time
    rng = np.random.default_rng(0)
    n = 26

    # Seed (d): greedy Apollonian control
    placed = []
    apo = np.zeros((n, 2))
    for k in range(n):
        p, _ = _largest_empty_center(np.array(placed))
        apo[k] = p
        placed.append(p)

    seeds = [_seed_corner_shell(n), _seed_hex_layers(n),
             _seed_grid_center(n), _seed_ring(n), _seed_lloyd(n), apo]
    best = None
    best_sum = -np.inf
    for sd in seeds:
        out = _eval_seed(sd)
        if out is None:
            continue
        c, r, s = out
        if s > best_sum:
            best_sum = s
            best = (c, r)
    if best is None:
        centers, radii = apo, compute_max_radii(apo)
        centers, radii = _final_guard(centers, radii)
        best_sum = float(np.sum(radii))
    else:
        centers, radii = best

    # Aspect-ratio continuation slice: deform the container height and
    # re-squash through re-solves to escape the incumbent's active-set
    # basin (bounded to ~120 s); incumbent is the fallback.
    t0 = time.time()
    try:
        hc, hr, hs = _aspect_continuation(centers, radii, t0 + 120.0)
        if np.all(np.isfinite(hc)) and np.all(np.isfinite(hr)) \
                and hs > best_sum:
            best_sum = hs
            centers, radii = hc, hr
    except Exception:
        pass

    # Diagonal-symmetry-reduction operator (attempt 3): symmetrize the
    # INCUMBENT across the diagonal x=y, re-solve the half-dimension
    # 13-circle subproblem (diagonal wall + mirror clearance), mirror back
    # to 26 and polish. Applied to the incumbent both before the restart
    # loop (as an alternative basin entry) and on a short second pass, so
    # the reflection-symmetric regularized slice directly feeds the
    # incumbent rather than running as an isolated one-shot.
    try:
        out = _diag_sym_candidate(rng, time.time() + 55.0,
                                  inc_c=centers, inc_r=radii,
                                  max_starts=40)
        if out is not None:
            dc, dr, ds = out
            if np.all(np.isfinite(dc)) and np.all(np.isfinite(dr)) \
                    and ds > best_sum:
                best_sum = ds
                centers, radii = dc, dr
    except Exception:
        pass

    # Persistent lineage candidate for compounding destroy-repair rewrites
    lin_c, lin_r = None, None
    t0 = time.time()
    # Budget leaves room for the two diagonal-symmetry passes (~55 s each)
    # and the penalty-continuation arm (~50 s) inside the 360 s limit.
    budget = 250.0
    for it in range(500):
        if time.time() - t0 > budget:
            break
        # Hybrid ladder: the parent's four-way jitter ladder remains the
        # dominant arm (it produced the 2.6293 incumbent), augmented with
        # SPARSE discrete contact-topology mutations — the only moves that
        # can change the combinatorial contact graph, since continuous
        # jitter is confined to one basin. Every 9th restart alternates
        # between (a) SWAP: exchange the radii of the largest circle and a
        # random below-median circle, forcing centers to rearrange into a
        # new contact graph during re-polish; and (b) REMOVE-REINSERT:
        # relocate the smallest circle to the largest-empty-space point
        # (reusing _largest_empty_center) with a mild 0.995 shrink.
        # The restart immediately after a topology move (it % 9 == 4)
        # uses a medium-jitter re-solve to exploit the shaken topology;
        # all other restarts run the original ladder unchanged.
        if it % 9 == 0:
            # Penalty-continuation restart: annealed soft constraints let
            # circles transiently overlap by O(1/mu) so centers migrate
            # across an infeasible ridge into a different contact graph;
            # the hard-feasibility tail re-projects onto the far side.
            try:
                pc, pr = _penalty_anneal(centers, radii,
                                         time.time() + 50.0,
                                         stages=8, maxiter=150)
            except Exception:
                pc, pr = centers.copy(), radii.copy()
            pr = pr * rng.uniform(0.995, 1.0, size=pr.shape)
            pc = pc + rng.normal(0.0, 0.002, size=pc.shape)
        elif it % 9 == 2:
            # k-DESTROY-GAP-REPAIR with persistent lineage: repair runs
            # from the lineage candidate if one exists (so successive
            # k-vertex topology rewrites compound), otherwise from the
            # incumbent. The repaired state is jittered and shrunk before
            # polish so SLSQP explores the new topology instead of
            # relaxing back onto the tangent-exact insertion. The lineage
            # best is retained even when worse than the incumbent and
            # periodically promoted for an exploitation re-polish.
            src_c, src_r = (lin_c if lin_c is not None
                            else centers, lin_r if lin_c is not None
                            else radii)
            k = 2 + (it // 9) % 3
            try:
                pc, pr = _destroy_repair(src_c, src_r, k, rng)
            except Exception:
                pc, pr = src_c.copy(), src_r.copy()
            pc = pc + rng.normal(0.0, 0.007, size=pc.shape)
            pr = pr * rng.uniform(0.98, 1.0, size=pr.shape)
            lin_c, lin_r = pc.copy(), pr.copy()
        elif it % 9 == 5 and lin_c is not None:
            # Gauss-Seidel maximal-regrowth on the retained lineage best:
            # the second GS arm probes a different contact-graph starting
            # point than the incumbent arm, increasing the chance a single
            # vertex relocation lands in a better active set.
            try:
                pc, pr = _gs_regrowth(lin_c, lin_r, max_sweeps=4)
            except Exception:
                pc, pr = lin_c.copy(), lin_r.copy()
            pr = pr * rng.uniform(0.995, 1.0, size=pr.shape)
            pc = pc + rng.normal(0.0, 0.002, size=pc.shape)
        elif it % 10 == 8:
            # Gauss-Seidel maximal-regrowth restart: cyclically shrink and
            # regrow one circle at a time to its exact maximal radius (or a
            # Soddy reinsertion into a different tangency triple), moving
            # single vertices of the contact graph out of the SLSQP KKT
            # basin. Bounded to 4 sweeps (~few seconds); result is then
            # tightened and accepted only if strictly better below.
            try:
                pc, pr = _gs_regrowth(centers, radii, max_sweeps=4)
            except Exception:
                pc, pr = centers.copy(), radii.copy()
            pr = pr * rng.uniform(0.995, 1.0, size=pr.shape)
            pc = pc + rng.normal(0.0, 0.002, size=pc.shape)
        else:
            # Parent jitter ladder; medium sigma after topology moves.
            if it % 9 == 4:
                sig = 0.012
            else:
                sig = 0.004 if (it % 4) else (0.012 if (it % 8 == 4) else 0.028)
            pc = centers + rng.normal(0.0, sig, size=centers.shape)
            pr = radii * rng.uniform(0.97, 1.03, size=radii.shape)
        pc, pr = _polish(pc.copy(), pr.copy(), maxiter=300)
        pc, pr = _tighten(pc, pr, rounds=18, grow=1.002)
        pc, pr = _final_guard(pc, pr)
        s = float(np.sum(pr))
        if np.all(np.isfinite(pc)) and np.all(np.isfinite(pr)) and s > best_sum:
            best_sum = s
            centers, radii = pc, pr
            # One extra gentle re-tighten on the newly accepted incumbent
            # to squeeze residual slack before the next restart.
            qc, qr = _tighten(centers.copy(), radii.copy(),
                              rounds=10, grow=1.0015)
            qc, qr = _final_guard(qc, qr)
            qs = float(np.sum(qr))
            if np.all(np.isfinite(qc)) and np.all(np.isfinite(qr)) and qs > best_sum:
                best_sum = qs
                centers, radii = qc, qr

    # Second diagonal-symmetry pass on the refined incumbent: the restart
    # loop may have moved the packing into a configuration whose symmetric
    # regularization differs from the pre-loop one; a short bounded pass
    # (20 starts, ~20 s) probes that slice before returning.
    try:
        out = _diag_sym_candidate(rng, time.time() + 25.0,
                                  inc_c=centers, inc_r=radii,
                                  max_starts=20)
        if out is not None:
            dc, dr, ds = out
            if np.all(np.isfinite(dc)) and np.all(np.isfinite(dr)) \
                    and ds > best_sum:
                best_sum = ds
                centers, radii = dc, dr
    except Exception:
        pass

    return centers, radii, best_sum


def compute_max_radii(centers):
    """
    Compute the maximum possible radii for each circle position
    such that they don't overlap and stay within the unit square.

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates

    Returns:
        np.array of shape (n) with radius of each circle
    """
    n = centers.shape[0]
    radii = np.ones(n)

    # First, limit by distance to square borders
    for i in range(n):
        x, y = centers[i]
        # Distance to borders
        radii[i] = min(x, y, 1 - x, 1 - y)

    # Then, limit by distance to other circles
    # Each pair of circles with centers at distance d can have
    # sum of radii at most d to avoid overlap
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))

            # If current radii would cause overlap
            if radii[i] + radii[j] > dist:
                # Scale both radii proportionally
                scale = dist / (radii[i] + radii[j])
                radii[i] *= scale
                radii[j] *= scale

    return radii


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii


def visualize(centers, radii):
    """
    Visualize the circle packing

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radius of each circle
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw unit square
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True)

    # Draw circles
    for i, (center, radius) in enumerate(zip(centers, radii)):
        circle = Circle(center, radius, alpha=0.5)
        ax.add_patch(circle)
        ax.text(center[0], center[1], str(i), ha="center", va="center")

    plt.title(f"Circle Packing (n={len(centers)}, sum={sum(radii):.6f})")
    plt.show()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii}")
    # AlphaEvolve improved this to 2.635

    # Uncomment to visualize:
    visualize(centers, radii)
