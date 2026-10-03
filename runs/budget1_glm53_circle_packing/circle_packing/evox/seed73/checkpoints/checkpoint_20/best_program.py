# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import time

import numpy as np


def _grid_layout(row_counts, jitter_rng=None, jitter=0.0):
    """
    Build an initial center arrangement from per-row circle counts.

    Rows are spaced evenly (with margin) and alternate rows are staggered
    by half a spacing for a hex-like pattern. Optionally add small uniform
    jitter (from a given RNG) to break symmetry and diversify the seed.

    Returns:
        np.array of shape (sum(row_counts), 2)
    """
    pts = []
    n_rows = len(row_counts)
    ys = np.linspace(0.1, 0.9, n_rows)
    for r, count in enumerate(row_counts):
        y = ys[r]
        xs = np.linspace(0.5 / count, 1 - 0.5 / count, count)
        if r % 2 == 1 and count > 1:
            xs = xs + 0.5 / count
            xs = np.clip(xs, 0.5 / count, 1 - 0.5 / count)
        for x in xs:
            pts.append([x, y])
    centers = np.array(pts)
    if jitter > 0.0 and jitter_rng is not None:
        centers = centers + jitter_rng.uniform(-jitter, jitter, centers.shape)
        centers = np.clip(centers, 0.02, 0.98)
    return centers


def _slsqp(centers0):
    """
    One SLSQP solve over z = [x1..xn, y1..yn, r1..rn] maximizing sum(r)
    subject to squared-distance non-overlap constraints
    (dx^2 + dy^2 - (ri + rj)^2 >= 0, smooth polynomial form with clean
    gradients) and wall constraints. Returns final centers or None.
    """
    from scipy.optimize import minimize
    n = centers0.shape[0]
    radii0 = compute_max_radii(centers0)
    z0 = np.concatenate([centers0.ravel(), radii0])

    def neg_sum(z):
        return -np.sum(z[2 * n:])

    def neg_sum_grad(z):
        g = np.zeros_like(z)
        g[2 * n:] = -1.0
        return g

    cons = []
    for i in range(n):
        for j in range(i + 1, n):
            def con_ij(z, i=i, j=j):
                dx = z[2 * i] - z[2 * j]
                dy = z[2 * i + 1] - z[2 * j + 1]
                s = z[2 * n + i] + z[2 * n + j]
                return dx * dx + dy * dy - s * s
            def con_ij_grad(z, i=i, j=j):
                g = np.zeros_like(z)
                dx = z[2 * i] - z[2 * j]
                dy = z[2 * i + 1] - z[2 * j + 1]
                s = z[2 * n + i] + z[2 * n + j]
                g[2 * i], g[2 * j] = 2 * dx, -2 * dx
                g[2 * i + 1], g[2 * j + 1] = 2 * dy, -2 * dy
                g[2 * n + i] = g[2 * n + j] = -2 * s
                return g
            cons.append({'type': 'ineq', 'fun': con_ij, 'jac': con_ij_grad})
    for i in range(n):
        def wall_i(z, i=i):
            x, y = z[2 * i], z[2 * i + 1]
            r = z[2 * n + i]
            return np.array([x - r, 1 - x - r, y - r, 1 - y - r])
        cons.append({'type': 'ineq', 'fun': wall_i})

    try:
        res = minimize(neg_sum, z0, jac=neg_sum_grad, constraints=cons,
                       method='SLSQP', options={'maxiter': 300, 'ftol': 1e-12})
        z = res.x
    except Exception:
        return None
    return np.clip(z[:2 * n].reshape(n, 2), 0.001, 0.999)


def construct_packing():
    """
    Multi-start SLSQP over all centers and radii (78 variables), the
    approach that achieved the best score historically. Each staggered
    hex-grid seed (row patterns plus deterministic jittered variants) is
    solved jointly for centers and radii, then radii are re-allocated
    exactly via the LP. Bounded basin hopping re-solves perturbed best
    layouts with SLSQP under a hard deadline. Returns
    (centers, radii, sum_of_radii).
    """
    patterns = [
        [6, 5, 6, 5, 4],
        [5, 6, 5, 6, 4],
        [4, 6, 6, 6, 4],
        [5, 5, 6, 5, 5],
        [6, 4, 6, 4, 6],
        [5, 4, 6, 6, 5],
        [7, 6, 7, 6],
        [6, 6, 7, 7],
        [5, 6, 6, 5, 4],
        [4, 5, 6, 5, 6],
    ]
    candidates = [_grid_layout(p) for p in patterns if sum(p) == 26]
    for pattern in ([6, 5, 6, 5, 4], [5, 6, 5, 6, 4], [7, 6, 7, 6]):
        for seed in (1, 2, 3):
            rng = np.random.default_rng(seed)
            candidates.append(_grid_layout(pattern, rng, 0.03))

    deadline = time.time() + 280.0
    best = None
    for c0 in candidates:
        if time.time() > deadline - 90.0:
            break
        c = _slsqp(c0)
        if c is None:
            continue
        radii = compute_max_radii(c)
        s = float(np.sum(radii))
        if best is None or s > best[0]:
            best = (s, c, radii)

    # Basin hopping: perturb the best layout and re-solve with SLSQP.
    for seed, amp in ((11, 0.02), (22, 0.02), (33, 0.01),
                      (44, 0.01), (55, 0.005), (66, 0.005)):
        if time.time() > deadline - 45.0:
            break
        rng = np.random.default_rng(seed)
        pert = np.clip(
            best[1] + rng.uniform(-amp, amp, best[1].shape), 0.02, 0.98
        )
        c = _slsqp(pert)
        if c is None:
            continue
        radii = compute_max_radii(c)
        s = float(np.sum(radii))
        if s > best[0]:
            best = (s, c, radii)

    sum_radii, centers, radii = best
    return centers, radii, sum_radii


def polish_centers_lp(centers, step=0.01, min_step=2e-4, max_sweeps=25,
                      deadline=None):
    """
    Coordinate descent on centers where every trial move is scored by the
    exact LP sum of radii for the whole layout. A move is accepted only
    if the total LP sum strictly increases, so we climb the true
    objective rather than a per-circle proxy. Step shrinks geometrically;
    sweeps stop early at the deadline.
    """
    n = centers.shape[0]
    best_sum = float(np.sum(compute_max_radii(centers)))
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (-1, -1), (1, -1), (-1, 1))
    sweeps = 0
    while step > min_step and sweeps < max_sweeps:
        if deadline is not None and time.time() > deadline:
            break
        sweeps += 1
        improved = False
        for i in range(n):
            base = centers[i].copy()
            best_pos = base
            for dx, dy in dirs:
                cand = base + np.array([dx, dy]) * step
                if not (0.001 < cand[0] < 0.999 and 0.001 < cand[1] < 0.999):
                    continue
                centers[i] = cand
                s = float(np.sum(compute_max_radii(centers)))
                if s > best_sum + 1e-9:
                    best_sum = s
                    best_pos = cand
            centers[i] = best_pos
            if best_pos is not base:
                improved = True
        if not improved:
            step *= 0.5
        else:
            step *= 0.9
    radii = compute_max_radii(centers)
    return centers, radii


def refine_centers(centers, radii):
    """
    Local coordinate-descent refinement of circle centers.

    For each circle, try moving its center by +/- step in x, y, and
    diagonals. Accept a move if it strictly increases that circle's
    feasible radius (limited by walls and non-overlap with all other
    circles at their current fixed radii). Step size shrinks geometrically
    so the process converges. This lets circles drift away from crowded
    neighbors and toward underused space (corners/edges), substantially
    raising the achievable sum of radii compared to a fixed grid.
    """
    n = centers.shape[0]
    dist = np.sqrt(((centers[:, None, :] - centers[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(dist, np.inf)

    def feasible_radius(idx, pos, d_row):
        x, y = pos
        wall = min(x, 1 - x, y, 1 - y)
        lim = np.min(d_row - radii) if n > 1 else np.inf
        return min(wall, lim)

    step = 0.05
    while step > 5e-5:
        improved = False
        for i in range(n):
            best_pos = centers[i]
            best_r = feasible_radius(i, centers[i], dist[i])
            for dx, dy in ((step, 0), (-step, 0), (0, step), (0, -step),
                           (step, step), (-step, -step), (step, -step), (-step, step)):
                cand = centers[i] + np.array([dx, dy])
                if not (0.001 < cand[0] < 0.999 and 0.001 < cand[1] < 0.999):
                    continue
                # distances from candidate to all others
                d_row = np.sqrt(((centers - cand) ** 2).sum(-1))
                d_row[i] = np.inf
                r = feasible_radius(i, cand, d_row)
                # Accept if own radius grows, or (tie) if slack to the
                # nearest neighbor grows: lets circles slide along
                # contact chains into underused regions.
                slack = np.min(d_row - radii) if n > 1 else np.inf
                cur_slack = np.min(dist[i] - radii) if n > 1 else np.inf
                if r > best_r + 1e-12 or (
                        abs(r - best_r) < 1e-13 and slack > cur_slack + 1e-12):
                    best_r = r
                    best_pos = cand
            if best_pos is not centers[i]:
                centers[i] = best_pos
                radii[i] = best_r
                # update distance matrix row/col
                diff = centers - centers[i]
                d = np.sqrt((diff ** 2).sum(-1))
                d[i] = np.inf
                dist[i] = d
                dist[:, i] = d
                improved = True
        if not improved:
            step *= 0.5
        else:
            step *= 0.98

    # Final full re-inflation to guarantee consistency and validity
    radii = compute_max_radii(centers)
    return centers, radii


def compute_max_radii(centers):
    """
    Exact optimal radii for fixed centers via a small linear program:
        maximize sum(r) s.t. r_i + r_j <= d_ij, 0 <= r_i <= wall_i.
    The LP removes the order dependence of greedy inflation and always
    allocates the best possible radii for the given centers. Radii are
    shrunk by a relative 1e-6 to guarantee strict feasibility. Falls back
    to convergent greedy inflation if SciPy is unavailable.
    """
    n = centers.shape[0]
    dist = np.sqrt(((centers[:, None, :] - centers[None, :, :]) ** 2).sum(-1))
    wall = np.minimum(
        np.minimum(centers[:, 0], 1 - centers[:, 0]),
        np.minimum(centers[:, 1], 1 - centers[:, 1]),
    )

    try:
        from scipy.optimize import linprog
        ii, jj = np.triu_indices(n, k=1)
        A_ub = np.zeros((ii.size, n))
        rows = np.arange(ii.size)
        A_ub[rows, ii] = 1.0
        A_ub[rows, jj] = 1.0
        res = linprog(
            -np.ones(n),
            A_ub=A_ub,
            b_ub=dist[ii, jj],
            bounds=[(0.0, w) for w in wall],
            method="highs",
        )
        if res.success:
            return np.maximum(res.x, 0.0) * 0.999999
    except Exception:
        pass

    # Greedy fallback: convergent inflation passes
    radii = np.zeros(n)
    np.fill_diagonal(dist, np.inf)
    for _ in range(200):
        improved = False
        for i in range(n):
            new_r = min(wall[i], np.min(dist[i] - radii))
            if new_r > radii[i] + 1e-15:
                radii[i] = new_r
                improved = True
        if not improved:
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
