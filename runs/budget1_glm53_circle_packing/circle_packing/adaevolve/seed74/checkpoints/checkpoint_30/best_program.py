# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """
    Sweep staggered hexagonal-row layouts (26 circles over ~6 rows),
    solve the exact radius-maximization LP for each layout, keep best.

    Returns:
        Tuple of (centers, radii, sum_of_radii)
        centers: np.array of shape (26, 2) with (x, y) coordinates
        radii: np.array of shape (26) with radius of each circle
        sum_of_radii: Sum of all radii
    """
    """Approach: sweep staggered hex-row and grid-gap candidate layouts,
    pick the best by exact LP radius maximization, then refine only the
    single best seed with a deterministic (iteration-capped) hill-climb
    so runtime is bounded and reproducible."""
    candidates = []
    row_patterns = [
        [5, 4, 5, 4, 5, 3],
        [4, 5, 4, 5, 4, 4],
        [5, 5, 4, 5, 4, 3],
        [5, 4, 5, 4, 4, 4],
        [4, 4, 5, 4, 5, 4],
        [5, 4, 4, 5, 4, 4],
        [3, 5, 4, 5, 4, 5],
        [4, 4, 4, 5, 4, 5],
        [4, 5, 5, 4, 5, 3],
        [5, 4, 4, 5, 5, 3],
        [5, 3, 5, 4, 5, 4],
        [6, 4, 5, 4, 5, 2],
        [4, 4, 5, 5, 4, 4],
        [4, 5, 4, 4, 5, 4],
    ]
    for counts in row_patterns:
        for r0 in np.linspace(0.075, 0.125, 26):
            for stagger in (0.0, 0.5, 1.0):
                centers = build_layout(counts, r0, stagger)
                if centers is None:
                    continue
                radii = compute_max_radii(centers)
                candidates.append((centers, radii, float(np.sum(radii))))
    # Grid family: 5x5 grid at pitch 0.2 (radius 0.1 each, sum 2.5) plus a
    # 26th circle inserted at an interior gap or edge midpoint.
    for gap in [(0.2, 0.2), (0.4, 0.2), (0.2, 0.4), (0.4, 0.4),
                (0.5, 0.5), (0.1, 0.3), (0.3, 0.1), (0.5, 0.1)]:
        centers = build_grid_layout(gap)
        radii = compute_max_radii(centers)
        candidates.append((centers, radii, float(np.sum(radii))))

    if not candidates:
        # Safe fallback: plain 5x5 grid plus center circle.
        centers = build_grid_layout((0.5, 0.5))
        radii = compute_max_radii(centers)
        return centers, radii, float(np.sum(radii))

    # Pick up to 3 diverse top seeds (different contact graphs), then for
    # each: short hill-climb, joint SLSQP over all 78 variables, and
    # contact-graph rigidity refinement.
    candidates.sort(key=lambda c: -c[2])
    seeds = []
    for cand in candidates:
        if len(seeds) >= 3:
            break
        if all(np.linalg.norm((cand[0] - s0[0]).ravel()) > 0.5
               for s0 in seeds):
            seeds.append(cand)
    if not seeds:
        seeds = [candidates[0]]

    overall = seeds[0]
    for c0, r0_, s0 in seeds:
        # Cycle the three refiners repeatedly: hill-climb escapes LP
        # flatness, SLSQP makes joint moves, rigidity refinement works
        # the contact graph; each unlocks further gains for the next.
        cur_c, cur_r, cur_s = c0.copy(), r0_.copy(), s0
        for _ in range(4):
            prev = cur_s
            c1, r1, s1 = refine(cur_c.copy(), cur_r, cur_s, max_passes=6)
            if s1 > cur_s:
                cur_c, cur_r, cur_s = c1, r1, s1
            c2, r2, s2 = slsqp_refine(cur_c, cur_r, cur_s)
            if s2 > cur_s:
                cur_c, cur_r, cur_s = c2, r2, s2
            c3, r3, s3 = rigidity_refine(cur_c, cur_r, cur_s)
            if s3 > cur_s:
                cur_c, cur_r, cur_s = c3, r3, s3
            if cur_s > overall[2]:
                overall = (cur_c.copy(), cur_r.copy(), cur_s)
            if cur_s - prev < 1e-6:
                break
    # Perturbation restarts: kick the best solution's centers slightly,
    # re-run the refinement cycle, keep any improvement. This escapes
    # shallow local minima of the piecewise-linear LP landscape.
    rng = np.random.default_rng(1)
    bc, br, bs = overall
    for trial in range(6):
        amp = 0.01 * (1.0 - trial / 6.0)
        kick = bc + rng.uniform(-amp, amp, bc.shape)
        kick[:, 0] = np.clip(kick[:, 0], 0.02, 0.98)
        kick[:, 1] = np.clip(kick[:, 1], 0.02, 0.98)
        kr = compute_max_radii(kick)
        ks = float(np.sum(kr))
        c1, r1, s1 = refine(kick, kr, ks, max_passes=6)
        c2, r2, s2 = slsqp_refine(c1, r1, s1)
        if s2 > bs:
            bc, br, bs = c2, r2, s2
    return bc, br, bs


def slsqp_refine(centers, radii, sum_radii, maxiter=200):
    """Joint nonlinear optimization of centers AND radii via SLSQP.

    All 78 variables (26*2 center coords + 26 radii) are optimized
    simultaneously to maximize sum(r_i), subject to pairwise non-overlap
    (only pairs near contact in the seed are constrained -- far pairs
    can never become active) and wall containment. Analytic Jacobians
    supplied for speed. Final radii recomputed by the exact LP on the
    final centers, guaranteeing validity and maximality.
    """
    from scipy.optimize import minimize

    n = centers.shape[0]
    d = np.sqrt(((centers[:, None, :] - centers[None, :, :]) ** 2).sum(-1))
    thresh = (radii[:, None] + radii[None, :]) + 0.12
    ii, jj = np.triu_indices(n, k=1)
    keep = (d[ii, jj] < thresh[ii, jj])
    pi, pj = ii[keep], jj[keep]
    if len(pi) == 0:
        return centers, radii, sum_radii

    c0 = centers.copy()
    r0 = radii.copy() * 0.98  # strictly feasible start

    def unpack(z):
        return z[:n], z[n:2 * n], z[2 * n:]

    def neg_sum(z):
        return -np.sum(z[2 * n:])

    def neg_sum_jac(z):
        g = np.zeros(3 * n)
        g[2 * n:] = -1.0
        return g

    cons = []

    def pair_fun(z):
        x, y, r = unpack(z)
        dx = x[pi] - x[pj]
        dy = y[pi] - y[pj]
        return np.sqrt(dx * dx + dy * dy) - r[pi] - r[pj]

    def pair_jac(z):
        x, y, r = unpack(z)
        dx = x[pi] - x[pj]
        dy = y[pi] - y[pj]
        dist = np.maximum(np.sqrt(dx * dx + dy * dy), 1e-12)
        J = np.zeros((len(pi), 3 * n))
        rows = np.arange(len(pi))
        J[rows, pi] = dx / dist
        J[rows, pj] = -dx / dist
        J[rows, n + pi] = dy / dist
        J[rows, n + pj] = -dy / dist
        J[rows, 2 * n + pi] = -1.0
        J[rows, 2 * n + pj] = -1.0
        return J

    cons.append({'type': 'ineq', 'fun': pair_fun, 'jac': pair_jac})

    def wall_fun(z):
        x, y, r = unpack(z)
        return np.concatenate([x - r, 1.0 - x - r, y - r, 1.0 - y - r])

    def wall_jac(z):
        J = np.zeros((4 * n, 3 * n))
        idx = np.arange(n)
        J[idx, idx] = 1.0
        J[idx, 2 * n + idx] = -1.0
        J[n + idx, idx] = -1.0
        J[n + idx, 2 * n + idx] = -1.0
        J[2 * n + idx, n + idx] = 1.0
        J[2 * n + idx, 2 * n + idx] = -1.0
        J[3 * n + idx, n + idx] = -1.0
        J[3 * n + idx, 2 * n + idx] = -1.0
        return J

    cons.append({'type': 'ineq', 'fun': wall_fun, 'jac': wall_jac})

    bounds = ([(1e-4, 1 - 1e-4)] * n +
              [(1e-4, 1 - 1e-4)] * n + [(1e-6, 0.5)] * n)
    z0 = np.concatenate([c0[:, 0], c0[:, 1], r0])
    try:
        res = minimize(neg_sum, z0, jac=neg_sum_jac, method='SLSQP',
                       bounds=bounds, constraints=cons,
                       options={'maxiter': maxiter, 'ftol': 1e-12})
        z = res.x
    except Exception:
        return centers, radii, sum_radii

    new_c = np.column_stack([z[:n], z[n:2 * n]])
    new_r = compute_max_radii(new_c)
    new_s = float(np.sum(new_r))
    if new_s > sum_radii:
        return new_c, new_r, new_s
    return centers, radii, sum_radii


def rigidity_refine(centers, radii, sum_radii, rounds=6, max_nfev=700):
    """Contact-graph refinement via scipy least_squares.

    Alternates the exact radius LP with a trust-region least-squares
    solve over centers AND radii whose residuals are the tangency
    equations of the currently ACTIVE constraints (circle-circle pairs
    and circle-wall contacts within epsilon), plus soft one-sided
    non-overlap penalties and a smooth radius-growth term. The active
    set is re-derived from each fresh LP solution. Final radii always
    come from the exact LP on the final centers, guaranteeing validity.
    """
    from scipy.optimize import least_squares

    n = centers.shape[0]
    ii, jj = np.triu_indices(n, k=1)
    T, W_OV, W_GROW, W_TAN, EPS = 0.30, 20.0, 1.0, 1.0, 0.01
    best_c = centers.copy()
    best_r = radii.copy()
    best_s = float(sum_radii)
    cur_c, cur_r = centers.copy(), radii.copy()

    lb = np.concatenate([np.full(2 * n, 1e-3), np.zeros(n)])
    ub = np.concatenate([np.full(2 * n, 1.0 - 1e-3), np.full(n, 0.5)])
    ai = aj = aw = np.array([], dtype=int)

    def residuals(z):
        c = z[:2 * n].reshape(n, 2)
        r = z[2 * n:]
        d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
        wall = np.minimum.reduce(
            [c[:, 0], c[:, 1], 1.0 - c[:, 0], 1.0 - c[:, 1]])
        parts = [
            W_OV * np.maximum(0.0, r[ii] + r[jj] - d[ii, jj]),
            W_OV * np.maximum(0.0, r - wall),
            W_GROW * (T - r),
        ]
        if len(ai):
            parts.append(W_TAN * (d[ai, aj] - r[ai] - r[aj]))
        if len(aw):
            parts.append(W_TAN * (wall[aw] - r[aw]))
        return np.concatenate(parts)

    for _ in range(rounds):
        d = np.sqrt(((cur_c[:, None, :] - cur_c[None, :, :]) ** 2).sum(-1))
        wall = np.minimum.reduce(
            [cur_c[:, 0], cur_c[:, 1], 1.0 - cur_c[:, 0], 1.0 - cur_c[:, 1]])
        act = (d[ii, jj] - cur_r[ii] - cur_r[jj]) < EPS
        ai, aj = ii[act], jj[act]
        aw = np.where((wall - cur_r) < EPS)[0]
        z0 = np.concatenate([cur_c.ravel(), cur_r])
        try:
            sol = least_squares(residuals, z0, bounds=(lb, ub),
                                method='trf', x_scale='jac',
                                max_nfev=max_nfev)
        except Exception:
            break
        new_c = sol.x[:2 * n].reshape(n, 2)
        new_r = compute_max_radii(new_c)
        new_s = float(np.sum(new_r))
        if new_s > best_s + 1e-9:
            best_c, best_r, best_s = new_c.copy(), new_r.copy(), new_s
            cur_c, cur_r = new_c.copy(), new_r.copy()
        elif new_s > best_s - 5e-4:
            cur_c, cur_r = new_c.copy(), new_r.copy()
        else:
            break
    return best_c, best_r, best_s


def build_grid_layout(gap):
    """5x5 grid of circles at pitch 0.2 plus one extra circle inserted
    at the given gap position (interior gaps give ~0.041 extra radius)."""
    centers = [[0.1 + 0.2 * i, 0.1 + 0.2 * j]
               for j in range(5) for i in range(5)]
    centers.append(list(gap))
    return np.array(centers)


def refine(centers, radii, sum_radii, max_passes=30):
    """Deterministic hill-climb: perturb one circle at a time (random plus
    axis-aligned moves), re-solve the radius LP, keep improvements.
    Capped by a fixed number of passes (no wall-clock dependence), halving
    the step size whenever a full pass yields no gain."""
    rng = np.random.default_rng(0)
    centers = centers.copy()
    n = centers.shape[0]
    step = 0.02
    for _ in range(max_passes):
        improved = False
        for i in rng.permutation(n):
            base = centers[i].copy()
            moves = [rng.uniform(-step, step, 2),
                     rng.uniform(-step, step, 2),
                     (step, 0.0), (-step, 0.0), (0.0, step), (0.0, -step)]
            for dx, dy in moves:
                centers[i] = base + [dx, dy]
                if not (0.01 <= centers[i, 0] <= 0.99 and 0.01 <= centers[i, 1] <= 0.99):
                    continue
                r2 = compute_max_radii(centers)
                s2 = float(np.sum(r2))
                if s2 > sum_radii + 1e-9:
                    radii, sum_radii = r2, s2
                    improved = True
                    break
            else:
                centers[i] = base
        if not improved:
            step *= 0.5
            if step < 1e-4:
                break
    return centers, radii, sum_radii


def build_layout(counts, r0, stagger):
    """Staggered hex rows: horizontal pitch 2*r0, rows spread to fill square."""
    n = sum(counts)
    if n != 26:
        return None
    nrow = len(counts)
    dy = (1.0 - 2.0 * r0) / (nrow - 1)
    centers = np.zeros((n, 2))
    idx = 0
    for row, count in enumerate(counts):
        y = r0 + row * dy
        offset = stagger * r0 if row % 2 == 1 else 0.0
        start = 0.5 - (count - 1) * r0 + offset
        for k in range(count):
            centers[idx] = [start + 2.0 * r0 * k, y]
            idx += 1
    if (centers < 1e-9).any() or (centers > 1 - 1e-9).any():
        return None
    return centers


def compute_max_radii(centers):
    """
    Exact radius maximization via linear programming: for fixed centers,
    maximize sum(r_i) s.t. r_i + r_j <= d_ij and r_i <= wall distance.

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates

    Returns:
        np.array of shape (n) with radius of each circle
    """
    from scipy.optimize import linprog

    n = centers.shape[0]
    d = np.sqrt(((centers[:, None, :] - centers[None, :, :]) ** 2).sum(-1))
    wall = np.minimum.reduce([
        centers[:, 0], centers[:, 1], 1.0 - centers[:, 0], 1.0 - centers[:, 1]
    ])

    # Build sparse-style constraint matrix vectorized for speed.
    ii, jj = np.triu_indices(n, k=1)
    m = len(ii) + n
    A = np.zeros((m, n))
    A[np.arange(len(ii)), ii] = 1.0
    A[np.arange(len(ii)), jj] = 1.0
    A[len(ii) + np.arange(n), np.arange(n)] = 1.0
    b = np.concatenate([d[ii, jj], wall])

    res = linprog(-np.ones(n), A_ub=A, b_ub=b,
                  bounds=[(0.0, None)] * n, method="highs")
    if res.success:
        return res.x
    # fallback: iterative proportional shrinking
    radii = wall.copy()
    for _ in range(200):
        viol = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                s = radii[i] + radii[j]
                if s > d[i, j]:
                    sc = d[i, j] / s
                    radii[i] *= sc
                    radii[j] *= sc
                    viol = max(viol, s - d[i, j])
        if viol < 1e-12:
            break
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
