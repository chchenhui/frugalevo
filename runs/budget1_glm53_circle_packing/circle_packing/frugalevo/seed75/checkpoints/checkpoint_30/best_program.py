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


def _polish(centers, radii, maxiter=200):
    """SLSQP refinement of the 78 variables (52 center coords + 26 radii)
    maximizing the sum of radii subject to wall and pairwise-distance
    constraints. Returns refined centers, radii, or the inputs on failure."""
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
                               1 - c[:, 0] - r, 1 - c[:, 1] - r])

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


def construct_packing():
    """
    Multiseed basin-hopping: build four structurally distinct deterministic
    seeds (hexagonal layers, 5x5 grid + center, corner-ring, and the greedy
    Apollonian control), polish and tighten each once, and promote the best
    valid guarded result to the incumbent. Then run time-budgeted restarts
    that perturb the incumbent (Gaussian jitter on centers, multiplicative
    jitter on radii), re-polish and re-tighten, keeping strictly improving
    valid results. Monotone non-decreasing overall.
    """
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
             _seed_grid_center(n), _seed_ring(n), apo]
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

    t0 = time.time()
    # Scale effort with the available wall budget: analytic constraint
    # Jacobians make each restart much cheaper, so a longer budget (~300 s)
    # now fits many more restarts inside the 360 s limit.
    budget = 295.0
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
            # SWAP radii of largest and a random below-median circle.
            pc = centers.copy()
            pr = radii.copy()
            big = int(np.argmax(pr))
            small_pool = np.flatnonzero(pr <= np.median(pr))
            small_pool = small_pool[small_pool != big]
            if small_pool.size:
                small = int(rng.choice(small_pool))
                pr[big], pr[small] = float(pr[small]), float(pr[big])
        elif it % 9 == 2:
            # REMOVE-REINSERT: move the smallest circle to the largest
            # empty-space point (discrete topology change).
            pc = centers.copy()
            pr = radii * 0.995
            small = int(np.argmin(pr))
            p, _ = _largest_empty_center(
                np.delete(pc, small, axis=0))
            pc[small] = p
        elif it % 10 == 8:
            # Radius-only squeeze-regrow restart (parent arm kept).
            pc = centers.copy()
            pr = radii * rng.uniform(0.90, 0.99, size=radii.shape)
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
