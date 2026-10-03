# EVOLVE-BLOCK-START
"""Hex-row constructor: greedy radii, anneal, SLSQP polish, tangency-graph LP saturation."""
import numpy as np
from scipy.optimize import minimize, linprog

SQ3 = np.sqrt(3.0)


def construct_packing():
    """
    Build a staggered hexagonal-row layout of 26 circles, then grow radii
    individually via deterministic water-filling, keeping the best of
    several row-count patterns.

    Approach:
      1. Candidate layouts: rows of circles with counts summing to 26
         (e.g. 5,4,5,4,5,3), vertically spaced by sqrt(3)*r and
         horizontally by 2*r (hexagonal lattice), rows staggered by r.
      2. Uniform radius = tightest constraint over borders & neighbors.
      3. Water-filling passes: r_i <- min(border_i, min_j(d_ij - r_j)),
         letting edge/corner circles expand into leftover space.
      4. Final uniform safety scaling guarantees strict validity.

    Returns:
        Tuple of (centers, radii, sum_of_radii)
    """
    # Exhaustively enumerate ordered row-count partitions of 26 into
    # R rows (each row 2..6 circles) for R in 5..8, then rank by the
    # achievable uniform radius and water-fill only the best ones.
    patterns = []
    for R in range(5, 9):
        _gen_patterns(26, R, 2, 6, [], patterns)
    # Build all layouts, rank by uniform-radius sum (cheap proxy)
    cands = []
    for counts in patterns:
        centers, r0 = _build_layout(counts)
        if centers is not None:
            cands.append((len(centers) * r0, centers, r0))
    cands.sort(key=lambda t: -t[0])
    # Score top layouts with three radius strategies (uniform, water-fill,
    # greedy largest-first); greedy varied radii fill corners/edges best.
    scored = []
    for _, centers, r0 in cands[:40]:
        for radii in (_scale_safe(centers, np.full(len(centers), r0)),
                      _grow(centers, r0, iters=200),
                      _greedy_max(centers)):
            s = float(np.sum(radii))
            scored.append((s, centers, radii))
    scored.sort(key=lambda t: -t[0])
    best = scored[0]
    # Annealed local search on centers: nudge circles off the rigid
    # lattice to exploit square corners/edges (edge effects matter).
    for seed in (12345, 777, 2024, 42):
        rng = np.random.default_rng(seed)
        for _, centers, _ in scored[:5]:
            c, r, s = _anneal(centers, rng, iters=300, step=0.04)
            for rr in (r, _grow(c, float(np.min(r)), iters=150)):
                ss = float(np.sum(rr))
                if ss > best[0]:
                    best = (ss, c, rr)
            if s > best[0]:
                best = (s, c, r)
    # SLSQP continuous refinement: jointly optimize centers and radii with
    # smooth constraints; restarts from perturbed variants of the incumbent.
    rng = np.random.default_rng(31337)
    pool = [(best[1], best[2])]
    for _, c0, r0 in scored[:4]:
        pool.append((c0, r0))
    for _ in range(3):
        c0 = np.clip(best[1] + rng.normal(0, 0.01, best[1].shape), 0.02, 0.98)
        pool.append((c0, best[2]))
    best_s = best[0]
    for c0, r0 in pool:
        c, r, s = _slsqp_refine(c0, r0)
        if s > best_s + 1e-9:
            best_s, best = s, (s, c, r)
    # Tangency-graph LP saturation: fix contact normals from the current
    # packing, solve the linearized max-sum-radii LP exactly (HiGHS),
    # refresh normals, repeat. Inner approximation -> always valid.
    c, r, s = _lp_refine(best[1], best[2], rounds=14)
    if s > best_s + 1e-9:
        best_s, best = s, (s, c, r)
    # Explore nearby contact structures: perturb, re-saturate via LP.
    for k in range(3):
        c0 = np.clip(best[1] + rng.normal(0, 0.006 + 0.004 * k, best[1].shape), 0.02, 0.98)
        c2, r2, s2 = _lp_refine(c0, best[2], rounds=8)
        if s2 > best_s + 1e-9:
            best_s, best = s2, (s2, c2, r2)
    # Final nonlinear polish, then one last LP saturation pass.
    c, r, s = _slsqp_refine(best[1], best[2], maxiter=200)
    if s > best_s + 1e-9:
        best_s, best = s, (s, c, r)
    c, r, s = _lp_refine(best[1], best[2], rounds=8)
    if s > best_s + 1e-9:
        best_s, best = s, (s, c, r)
    return best[1], best[2], best[0]


def _slsqp_refine(centers, radii, maxiter=200):
    """Jointly refine centers and radii with scipy SLSQP.

    Variables: [x_0..x_n-1, y_0..y_n-1, r_0..r_n-1] (3n total).
    Objective: minimize -sum(r). Constraints (all >= 0):
      d_ij - r_i - r_j >= eps for each pair,
      x_i - r_i - eps >= 0, 1 - x_i - r_i - eps >= 0 (same for y).
    Final result is safety-scaled with _scale_safe for strict validity.
    Returns (centers, radii, sum_radii).
    """
    n = len(centers)
    eps = 1e-9
    pairs = np.array([(i, j) for i in range(n) for j in range(i + 1, n)])
    ii, jj = pairs[:, 0], pairs[:, 1]

    def unpack(z):
        return z[:n], z[n:2 * n], z[2 * n:]

    def obj(z):
        return -np.sum(z[2 * n:])

    def obj_grad(z):
        g = np.zeros(3 * n)
        g[2 * n:] = -1.0
        return g

    def cons(z):
        x, y, r = unpack(z)
        dx = x[ii] - x[jj]
        dy = y[ii] - y[jj]
        d = np.sqrt(dx * dx + dy * dy)
        pair = d - r[ii] - r[jj]
        border = np.concatenate([x - r, 1 - x - r, y - r, 1 - y - r])
        return np.concatenate([pair, border - eps])

    def cons_jac(z):
        x, y, r = unpack(z)
        dx = x[ii] - x[jj]
        dy = y[ii] - y[jj]
        d = np.sqrt(dx * dx + dy * dy) + 1e-300
        P = len(ii)
        J = np.zeros((P + 4 * n, 3 * n))
        rows = np.arange(P)
        J[rows, ii] = dx / d
        J[rows, jj] = -dx / d
        J[rows, n + ii] = dy / d
        J[rows, n + jj] = -dy / d
        J[rows, 2 * n + ii] = -1.0
        J[rows, 2 * n + jj] = -1.0
        idx = np.arange(n)
        J[P + idx, idx] = 1.0
        J[P + idx, 2 * n + idx] = -1.0
        J[P + n + idx, idx] = -1.0
        J[P + n + idx, 2 * n + idx] = -1.0
        J[P + 2 * n + idx, n + idx] = 1.0
        J[P + 2 * n + idx, 2 * n + idx] = -1.0
        J[P + 3 * n + idx, n + idx] = -1.0
        J[P + 3 * n + idx, 2 * n + idx] = -1.0
        return J

    z0 = np.concatenate([centers[:, 0], centers[:, 1], radii])
    z0[2 * n:] *= 0.98
    bounds = [(0.0, 1.0)] * (2 * n) + [(1e-6, 0.5)] * n
    res = minimize(obj, z0, jac=obj_grad, bounds=bounds,
                  constraints=[{'type': 'ineq', 'fun': cons, 'jac': cons_jac}],
                  method='SLSQP',
                  options={'maxiter': maxiter, 'ftol': 1e-12})
    x, y, r = unpack(res.x)
    c = np.column_stack([x, y])
    r = _scale_safe(c, np.maximum(r, 1e-12))
    return c, r, float(np.sum(r))


def _lp_refine(centers, radii, rounds=12):
    """Sequential-LP tangency-graph saturation.

    Fix contact directions (unit normals u_ij) from current geometry and
    solve the LP over (x_i, y_i, r_i):
        maximize sum(r_i)
        s.t. u_ij . (p_i - p_j) >= r_i + r_j  for all pairs,
             x_i - r_i >= 0, 1 - x_i - r_i >= 0 (same for y_i).
    Since ||p_i - p_j|| >= u_ij . (p_i - p_j), every LP-feasible point is
    a genuinely valid packing (inner approximation); wall constraints are
    exact. Refresh normals each round; all iterates are safety-scaled.
    Returns (centers, radii, sum_radii) of the best valid iterate.
    """
    n = len(centers)
    c = centers.copy()
    r = _scale_safe(centers, radii)
    best_s = float(np.sum(r))
    best_c, best_r = c.copy(), r.copy()
    iu, ju = np.triu_indices(n, 1)
    P = len(iu)
    ncon = P + 4 * n
    cost = np.zeros(3 * n)
    cost[2 * n:] = -1.0
    bounds = [(0.0, 1.0)] * (2 * n) + [(1e-9, 0.5)] * n
    idx = np.arange(n)
    for _ in range(rounds):
        x, y = c[:, 0], c[:, 1]
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.hypot(dx, dy)
        dsafe = np.where(d > 1e-12, d, 1.0)
        ux = dx / dsafe
        uy = dy / dsafe
        A = np.zeros((ncon, 3 * n))
        b = np.zeros(ncon)
        rows = np.arange(P)
        A[rows, iu] = -ux[iu, ju]
        A[rows, ju] = ux[iu, ju]
        A[rows, n + iu] = -uy[iu, ju]
        A[rows, n + ju] = uy[iu, ju]
        A[rows, 2 * n + iu] = 1.0
        A[rows, 2 * n + ju] = 1.0
        A[P + idx, idx] = -1.0
        A[P + idx, 2 * n + idx] = 1.0
        A[P + n + idx, idx] = 1.0
        A[P + n + idx, 2 * n + idx] = 1.0
        b[P + n + idx] = 1.0
        A[P + 2 * n + idx, n + idx] = -1.0
        A[P + 2 * n + idx, 2 * n + idx] = 1.0
        A[P + 3 * n + idx, n + idx] = 1.0
        A[P + 3 * n + idx, 2 * n + idx] = 1.0
        b[P + 3 * n + idx] = 1.0
        res = linprog(cost, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        if not res.success or res.x is None:
            break
        z = res.x
        nc = np.column_stack([z[:n], z[n:2 * n]])
        nr = _scale_safe(nc, np.maximum(z[2 * n:], 1e-12))
        ns = float(np.sum(nr))
        if ns > best_s + 1e-7:
            best_s, best_c, best_r = ns, nc, nr
            c, r = nc.copy(), nr.copy()
        else:
            break
    return best_c, best_r, best_s


def _greedy_max(centers):
    """Greedy largest-first radius assignment: process circles in order of
    increasing border slack; each radius takes the max feasible value given
    already-assigned neighbors. Yields a maximal, valid radius vector."""
    n = len(centers)
    x, y = centers[:, 0], centers[:, 1]
    border = np.minimum(np.minimum(x, y), np.minimum(1 - x, 1 - y))
    d = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    np.fill_diagonal(d, np.inf)
    r = np.zeros(n)
    assigned = np.zeros(n, dtype=bool)
    for i in np.argsort(border):
        ub = border[i]
        if assigned.any():
            ub = min(ub, float(np.min(d[i, assigned] - r[assigned])))
        r[i] = max(ub, 1e-12)
        assigned[i] = True
    return _scale_safe(centers, r)


def _candidate_moves(centers, i, rng, step=0.03, k=2):
    """Candidate moves for circle i: toward the nearest walls, away from
    its tightest neighbor, plus random jitter. Returns list of arrays."""
    out = []
    x, y = centers[i]
    walls = [(0, -1, y), (0, 1, 1 - y), (-1, 0, x), (1, 0, 1 - x)]
    walls.sort(key=lambda w: w[2])
    for dx, dy, _ in walls[:2]:
        p = centers.copy()
        p[i] = np.clip([x + dx * step, y + dy * step], 0.02, 0.98)
        out.append(p)
    d = np.hypot(centers[:, 0] - x, centers[:, 1] - y)
    d[i] = np.inf
    j = int(np.argmin(d))
    v = centers[i] - centers[j]
    nrm = np.hypot(*v)
    if nrm > 1e-12:
        p = centers.copy()
        p[i] = np.clip(centers[i] + (v / nrm) * step, 0.02, 0.98)
        out.append(p)
    for _ in range(k):
        p = centers.copy()
        p[i] = np.clip(centers[i] + rng.normal(0, step, 2), 0.02, 0.98)
        out.append(p)
    return out


def _anneal(centers, rng, iters=500, step=0.04):
    """Hill-climb with directed moves: each step picks a circle, tries all
    proposals, accepts the best if the greedy radius sum improves; step
    size anneals geometrically. Returns best (centers, radii, sum)."""
    cur = centers.copy()
    cur_r = _greedy_max(cur)
    cur_s = float(np.sum(cur_r))
    bc, br, bs = cur.copy(), cur_r, cur_s
    size, n = step, len(cur)
    for _ in range(iters):
        i = int(rng.integers(n))
        best_prop, best_ps, best_pr = None, cur_s, None
        for prop in _candidate_moves(cur, i, rng, step=size):
            pr = _greedy_max(prop)
            ps = float(np.sum(pr))
            if ps > best_ps:
                best_prop, best_ps, best_pr = prop, ps, pr
        if best_prop is not None:
            cur, cur_r, cur_s = best_prop, best_pr, best_ps
            if cur_s > bs:
                bc, br, bs = cur.copy(), cur_r, cur_s
        size *= 0.996
    return bc, br, bs


def _gen_patterns(total, rows, lo, hi, cur, out):
    """Recursively generate ordered row counts summing to total."""
    if rows == 0:
        if total == 0:
            out.append(list(cur))
        return
    for m in range(lo, min(hi, total) + 1):
        cur.append(m)
        _gen_patterns(total - m, rows - 1, lo, hi, cur, out)
        cur.pop()


def _build_layout(counts):
    """Place circles in staggered rows; return centers and uniform radius."""
    R = len(counts)
    if R < 2 or sum(counts) != 26:
        return None, None
    # Vertical: 2r + sqrt(3)*r*(R-1) <= 1
    r = 1.0 / (2.0 + SQ3 * (R - 1))
    # Horizontal per row (staggered rows shifted by r)
    for k, m in enumerate(counts):
        if m <= 1:
            continue
        if k % 2 == 0:  # unshifted row: 0.5-(m-1)r >= r
            r = min(r, 0.5 / m)
        else:           # shifted row: 0.5-(m-1)r + r >= r
            r = min(r, 0.5 / (m - 1))
    if r <= 0:
        return None, None
    dy = SQ3 * r
    total = 2 * r + dy * (R - 1)
    y0 = (1.0 - total) / 2.0 + r
    centers = []
    for k, m in enumerate(counts):
        y = y0 + k * dy
        off = r if k % 2 == 1 else 0.0
        x0 = 0.5 - (m - 1) * r + off
        for j in range(m):
            centers.append([x0 + 2 * r * j, y])
    return np.array(centers), r


def _grow(centers, r0, iters=400):
    """Water-filling: each radius expands to its maximal feasible value."""
    n = len(centers)
    x = centers[:, 0]
    y = centers[:, 1]
    border = np.minimum(np.minimum(x, y), np.minimum(1 - x, 1 - y))
    d = np.sqrt((x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2)
    np.fill_diagonal(d, np.inf)
    r = np.full(n, r0)
    for _ in range(iters):
        # upper bound for each circle given current radii of others
        ub = np.minimum(border, np.min(d - r[None, :], axis=1))
        r = np.maximum(r * 0.0 + ub, 1e-12)  # candidate radii
        # keep only improvements that stay jointly feasible via scaling
        r = _scale_safe(centers, r)
    return r


def _scale_safe(centers, radii):
    """Uniformly scale radii down so packing is strictly valid."""
    n = len(centers)
    x = centers[:, 0]
    y = centers[:, 1]
    border = np.minimum(np.minimum(x, y), np.minimum(1 - x, 1 - y))
    s = 1.0
    for i in range(n):
        if radii[i] > border[i]:
            s = min(s, border[i] / radii[i])
    for i in range(n):
        for j in range(i + 1, n):
            dij = np.hypot(x[i] - x[j], y[i] - y[j])
            need = radii[i] + radii[j]
            if need > dij:
                s = min(s, dij / need)
    return radii * s * 0.999999


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
