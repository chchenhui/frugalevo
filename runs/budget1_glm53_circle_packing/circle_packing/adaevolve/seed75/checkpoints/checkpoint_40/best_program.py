# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """
    Construct a specific arrangement of 26 circles in a unit square
    that attempts to maximize the sum of their radii.

    Returns:
        Tuple of (centers, radii, sum_of_radii)
        centers: np.array of shape (26, 2) with (x, y) coordinates
        radii: np.array of shape (26) with radius of each circle
        sum_of_radii: Sum of all radii
    """
    """Multiple layout families swept over lattice spacing and global
    shifts; top candidates by LP-optimal sum are polished by greedy
    coordinate descent, then perturbation restarts (mini basin-hopping)
    escape local optima. Best refined result is returned."""
    cands = []
    for r0 in np.arange(0.095, 0.135, 0.005):
        for dx in (-0.01, 0.0, 0.01):
            for dy in (-0.01, 0.0, 0.01):
                for layout in (layout_A, layout_B, layout_C, layout_D,
                               layout_E):
                    centers = np.clip(layout(r0, dx, dy), 0.001, 0.999)
                    radii = compute_max_radii(centers)
                    cands.append((float(np.sum(radii)), centers))
    cands.sort(key=lambda c: -c[0])
    best = None
    for _, centers in cands[:4]:
        centers, radii, s = refine(centers)
        if best is None or s > best[2]:
            best = (centers, radii, s)
    # Perturbation restarts: jitter the best arrangement and re-refine.
    rng = np.random.default_rng(12345)
    centers, radii, s = best
    for sigma in (0.006, 0.012, 0.020, 0.010, 0.008, 0.015):
        pert = np.clip(centers + rng.normal(0.0, sigma, centers.shape),
                       0.02, 0.98)
        c2, r2, s2 = refine(pert)
        if s2 > s:
            centers, radii, s = c2, r2, s2
    # Alternating SLSQP <-> greedy refine loop: each SLSQP local optimum
    # is re-polished by greedy moves, then re-optimized jointly. This
    # alternation escapes optima neither method can leave alone.
    for _ in range(3):
        prev_s = s
        centers, radii, s = slsqp_polish(centers, radii, s)
        centers, radii, s = refine(centers, rounds=2)
        if s <= prev_s + 1e-7:
            break
    return (centers, radii, s)


def hex_rows(counts, r0, dx, dy):
    """Staggered hex rows: horizontal spacing 2*r0, vertical sqrt(3)*r0."""
    t = 2.0 * r0
    s = np.sqrt(3.0) * r0
    pts = []
    for row, m in enumerate(counts):
        y = 0.5 + (row - (len(counts) - 1) / 2.0) * s + dy
        for i in range(m):
            x = 0.5 + (i - (m - 1) / 2.0) * t + dx
            pts.append([x, y])
    return pts


def layout_A(r0, dx, dy):
    """Rows 5,4,5,4,5 (23 circles) + 3 gap fillers near row ends/edges."""
    t = 2.0 * r0
    s = np.sqrt(3.0) * r0
    pts = hex_rows([5, 4, 5, 4, 5], r0, dx, dy)
    pts.append([0.5 - 2.0 * t - 0.02 + dx, 0.5 - 1.5 * s + dy])
    pts.append([0.5 + 2.0 * t + 0.02 + dx, 0.5 - 1.5 * s + dy])
    pts.append([0.5 + dx, 0.5 + 2.0 * s + 0.08 + dy])
    return np.array(pts)


def layout_B(r0, dx, dy):
    """Rows 4,5,4,5,4 (22 circles) + 4 corner circles (each bounded by
    two walls, so they can grow large) + 2 mid-edge fillers."""
    t = 2.0 * r0
    s = np.sqrt(3.0) * r0
    pts = hex_rows([4, 5, 4, 5, 4], r0, dx, dy)
    c = 0.5 * t
    pts.append([c, c])
    pts.append([1.0 - c, c])
    pts.append([c, 1.0 - c])
    pts.append([1.0 - c, 1.0 - c])
    pts.append([0.5 - 1.5 * t - 0.02 + dx, 0.5 + dy])
    pts.append([0.5 + 1.5 * t + 0.02 + dx, 0.5 + dy])
    return np.array(pts)


def layout_C(r0, dx, dy):
    """5 rows 4,5,4,5,4 + 2 corner circles (wall-bounded, can grow
    large) + 2 mid-edge fillers; asymmetric since the optimum packing
    in a square is not vertically symmetric."""
    t = 2.0 * r0
    s = np.sqrt(3.0) * r0
    pts = hex_rows([4, 5, 4, 5, 4], r0, dx, dy)
    pts.append([0.12 + dx, 0.12 + dy])
    pts.append([0.88 + dx, 0.12 + dy])
    pts.append([0.5 - 2.0 * t - 0.02 + dx, 0.5 - s + dy])
    pts.append([0.5 + 2.0 * t + 0.02 + dx, 0.5 - s + dy])
    return np.array(pts)


def layout_D(r0, dx, dy):
    """Boundary-ring layout: 4 corner + 4 mid-edge wall-bounded circles
    surrounding inner hex rows 4,5,4,5 (18 circles), for 26 total."""
    t = 2.0 * r0
    s = np.sqrt(3.0) * r0
    c = 0.11
    pts = []
    for cx in (c, 1.0 - c):
        for cy in (c, 1.0 - c):
            pts.append([cx, cy])
    pts.append([0.5 + dx, c])
    pts.append([0.5 + dx, 1.0 - c])
    pts.append([c, 0.5 + dy])
    pts.append([1.0 - c, 0.5 + dy])
    for row, m in enumerate([4, 5, 4, 5]):
        y = 0.5 + (row - 1.5) * s + dy
        for i in range(m):
            x = 0.5 + (i - (m - 1) / 2.0) * t + dx
            pts.append([x, y])
    return np.array(pts)


def slsqp_polish(centers, radii, s, margin=1e-7, iters=200):
    """Joint direct nonlinear optimization over all centers AND radii
    (78 variables) with SLSQP. Constraints are exact and smooth:
    pairwise (xi-xj)^2+(yi-yj)^2 >= (ri+rj+margin)^2 and wall
    constraints xi-ri>=margin etc. Warm start is the current best
    feasible arrangement, so SLSQP can only improve the sum.
    Perturbed feasible restarts escape local optima; best valid
    result is kept."""
    try:
        from scipy.optimize import minimize
    except Exception:
        return centers, radii, s
    n = centers.shape[0]
    ii, jj = np.triu_indices(n, 1)

    def unpack(z):
        return z[:2 * n].reshape(n, 2), z[2 * n:]

    def pack(c, r):
        return np.concatenate([c.ravel(), r])

    def objective(z):
        return -float(np.sum(z[2 * n:]))

    def objective_grad(z):
        g = np.zeros_like(z)
        g[2 * n:] = -1.0
        return g

    def constraints(z):
        c, r = unpack(z)
        x, y = c[:, 0], c[:, 1]
        dx = x[ii] - x[jj]
        dy = y[ii] - y[jj]
        rr = (r[ii] + r[jj] + 2.0 * margin) ** 2
        pair = dx * dx + dy * dy - rr
        walls = np.concatenate([x - r - margin, 1.0 - x - r - margin,
                                y - r - margin, 1.0 - y - r - margin])
        return np.concatenate([pair, walls])

    def constraints_jac(z):
        c, r = unpack(z)
        x, y = c[:, 0], c[:, 1]
        m = len(ii)
        J = np.zeros((m + 4 * n, 3 * n))
        rs = r[ii] + r[jj] + 2.0 * margin
        J[np.arange(m), 2 * ii] = 2.0 * (x[ii] - x[jj])
        J[np.arange(m), 2 * jj] = -2.0 * (x[ii] - x[jj])
        J[np.arange(m), 2 * ii + 1] = 2.0 * (y[ii] - y[jj])
        J[np.arange(m), 2 * jj + 1] = -2.0 * (y[ii] - y[jj])
        J[np.arange(m), 2 * n + ii] = -2.0 * rs
        J[np.arange(m), 2 * n + jj] = -2.0 * rs
        off = m
        for k in range(n):
            J[off + k, 2 * k] = 1.0
            J[off + k, 2 * n + k] = -1.0
            J[off + n + k, 2 * k] = -1.0
            J[off + n + k, 2 * n + k] = -1.0
            J[off + 2 * n + k, 2 * k + 1] = 1.0
            J[off + 2 * n + k, 2 * n + k] = -1.0
            J[off + 3 * n + k, 2 * k + 1] = -1.0
            J[off + 3 * n + k, 2 * n + k] = -1.0
        return J

    bounds = ([(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n)
    cons = [{"type": "ineq", "fun": constraints, "jac": constraints_jac}]

    def run_slsqp(z0):
        try:
            res = minimize(objective, z0, jac=objective_grad,
                           method="SLSQP", bounds=bounds,
                           constraints=cons,
                           options={"maxiter": iters, "ftol": 1e-12})
            return res.x
        except Exception:
            return z0

    def valid(c, r):
        if (c < -1e-9).any() or (c > 1 + 1e-9).any() or (r < 0).any():
            return False
        d = np.sqrt(((c[ii] - c[jj]) ** 2).sum(axis=1))
        if (r[ii] + r[jj] > d + 1e-9).any():
            return False
        b = np.minimum(np.minimum(c[:, 0], 1 - c[:, 0]),
                       np.minimum(c[:, 1], 1 - c[:, 1]))
        return bool((r <= b + 1e-9).all())

    best = (centers, radii, s)
    z = run_slsqp(pack(centers, radii))
    c, r = unpack(z)
    if valid(c, r):
        s2 = float(np.sum(r))
        if s2 > best[2]:
            best = (c.copy(), r.copy(), s2)
    # Perturbed feasible restarts: jitter centers, shrink radii to
    # restore feasibility, re-run SLSQP.
    rng = np.random.default_rng(777)
    cur_c, cur_r = best[0], best[1]
    for sigma in (0.002, 0.004, 0.006, 0.008, 0.012, 0.018,
                  0.003, 0.005, 0.010):
        c = np.clip(cur_c + rng.normal(0.0, sigma, cur_c.shape), 0.02, 0.98)
        b = np.minimum(np.minimum(c[:, 0], 1 - c[:, 0]),
                       np.minimum(c[:, 1], 1 - c[:, 1]))
        r = np.minimum(cur_r, b)
        d = np.sqrt(((c[ii] - c[jj]) ** 2).sum(axis=1))
        need = r[ii] + r[jj] > d
        for _ in range(200):
            if not need.any():
                break
            for a, bb, dist in zip(ii[need], jj[need], d[need]):
                tot = r[a] + r[bb]
                if tot > dist and tot > 0:
                    sc = dist / tot
                    r[a] *= sc
                    r[bb] *= sc
            need = r[ii] + r[jj] > d
        z = run_slsqp(pack(c, r))
        c2, r2 = unpack(z)
        if valid(c2, r2):
            s2 = float(np.sum(r2))
            if s2 > best[2]:
                best = (c2.copy(), r2.copy(), s2)
    return best


def layout_E(r0, dx, dy):
    """Corner-heavy layout: 4 large corner circles + 8 wall circles
    (2 per edge, all wall-bounded and able to grow) + 14 interior hex
    circles in rows 4,3,4,3. Wall circles benefit from one-wall
    constraints; corners from two walls each."""
    t = 2.0 * r0
    s = np.sqrt(3.0) * r0
    c = 0.13
    pts = []
    for cx in (c, 1.0 - c):
        for cy in (c, 1.0 - c):
            pts.append([cx, cy])
    # two wall circles per edge, symmetric about edge midpoints
    w = 0.5 * t
    for u in (0.5 - w, 0.5 + w):
        pts.append([u + dx, c])
        pts.append([u + dx, 1.0 - c])
        pts.append([c, u + dy])
        pts.append([1.0 - c, u + dy])
    # interior hex cluster of 14 circles
    for row, m in enumerate([4, 3, 4, 3]):
        y = 0.5 + (row - 1.5) * s + dy
        for i in range(m):
            x = 0.5 + (i - (m - 1) / 2.0) * t + dx
            pts.append([x, y])
    return np.array(pts)


def refine(centers, rounds=4):
    """Greedy per-circle displacement search: try 8-direction moves at
    decreasing step sizes down to 0.001; accept moves that increase the
    LP sum, then a pair-move pass shifting close pairs together."""
    centers = centers.copy()
    radii = compute_max_radii(centers)
    s = float(np.sum(radii))
    for step in (0.02, 0.008, 0.003, 0.001):
        for _ in range(rounds):
            improved = False
            for i in range(centers.shape[0]):
                best_move = None
                for dx, dy in ((step, 0), (-step, 0), (0, step), (0, -step),
                               (step, step), (step, -step),
                               (-step, step), (-step, -step)):
                    old = centers[i].copy()
                    centers[i] = old + [dx, dy]
                    if (centers[i] < 0.02).any() or (centers[i] > 0.98).any():
                        centers[i] = old
                        continue
                    r2 = compute_max_radii(centers)
                    s2 = float(np.sum(r2))
                    if s2 > s + 1e-9:
                        s, best_move = s2, (centers[i].copy(), r2)
                    centers[i] = old
                if best_move is not None:
                    centers[i], radii = best_move
                    improved = True
            if not improved:
                break
    # Pair-move pass: shift adjacent circle pairs together to escape
    # local optima where two circles need to move in the same direction.
    for step in (0.008, 0.003):
        for _ in range(2):
            improved = False
            n = centers.shape[0]
            for i in range(n):
                for j in range(i + 1, n):
                    if np.hypot(*(centers[i] - centers[j])) > 0.35:
                        continue
                    for dx, dy in ((step, 0), (-step, 0), (0, step),
                                   (0, -step)):
                        oi = centers[i].copy()
                        oj = centers[j].copy()
                        centers[i] = oi + [dx, dy]
                        centers[j] = oj + [dx, dy]
                        if ((centers[i] < 0.02).any() or
                                (centers[i] > 0.98).any() or
                                (centers[j] < 0.02).any() or
                                (centers[j] > 0.98).any()):
                            centers[i], centers[j] = oi, oj
                            continue
                        r2 = compute_max_radii(centers)
                        s2 = float(np.sum(r2))
                        if s2 > s + 1e-9:
                            s = s2
                            radii = r2
                            improved = True
                        else:
                            centers[i], centers[j] = oi, oj
            if not improved:
                break
    return centers, radii, s


def compute_max_radii(centers):
    """Maximize sum of radii subject to r_i + r_j <= dist(i,j) and
    r_i <= border distance, via scipy linprog with sparse constraints.
    Pairs with dist >= border_i + border_j can never bind (since
    r_i <= border_i always), so they are pruned; a sparse constraint
    matrix roughly triples LP throughput, enabling more refinement
    passes within the time budget. Converged greedy fallback."""
    n = centers.shape[0]
    x, y = centers[:, 0], centers[:, 1]
    border = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
    ii, jj = np.triu_indices(n, 1)
    d = np.sqrt(((centers[ii] - centers[jj]) ** 2).sum(axis=1))
    keep = d < border[ii] + border[jj] - 1e-12
    ik, jk, dk = ii[keep], jj[keep], d[keep]
    m = len(ik)
    try:
        from scipy.optimize import linprog
        from scipy.sparse import csr_matrix
        rows = np.concatenate([np.arange(m), np.arange(m)])
        cols = np.concatenate([ik, jk])
        A = csr_matrix((np.ones(2 * m), (rows, cols)), shape=(m, n))
        res = linprog(c=-np.ones(n), A_ub=A, b_ub=dk,
                      bounds=list(zip(np.zeros(n), border)), method="highs")
        if res.success:
            return np.maximum(res.x, 0.0)
    except Exception:
        pass
    radii = border.copy()
    for _ in range(500):
        over = radii[ik] + radii[jk] > dk
        if not over.any():
            break
        for a, b, dist in zip(ik[over], jk[over], dk[over]):
            tot = radii[a] + radii[b]
            if tot > dist and tot > 0:
                scale = dist / tot
                radii[a] *= scale
                radii[b] *= scale
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
