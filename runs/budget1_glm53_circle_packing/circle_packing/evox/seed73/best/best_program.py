# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles.

Approach: multi-start SLSQP over all centers and radii jointly.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from itertools import combinations


def construct_packing():
    """
    Construct 26 circles in a unit square.

    Approach (fundamentally different from greedy coordinate descent):
    1. Seed layouts from several staggered row-count grid families
       (plus deterministic jittered variants) for multi-start diversity.
    2. For each seed, run a single SLSQP solve over ALL 78 variables
       (x_i, y_i, r_i) maximizing sum(r) subject to hard nonlinear
       constraints: pairwise distance >= r_i + r_j and wall clearance.
    3. Polish with a greedy maximal inflation of each radius.
    4. Return the best of all starts.

    Returns:
        Tuple of (centers, radii, sum_of_radii)
    """
    patterns = [
        [6, 5, 6, 5, 4],
        [5, 6, 5, 6, 4],
        [4, 6, 6, 6, 4],
        [5, 5, 6, 5, 5],
        [6, 4, 6, 4, 6],
        [5, 4, 6, 6, 5],
    ]

    candidates = []
    for pattern in patterns:
        if sum(pattern) == 26:
            candidates.append(_grid_layout(pattern))
    # Jittered variants (deterministic seeds) for extra diversity
    for pattern in patterns:
        if sum(pattern) != 26:
            continue
        for seed in (1, 2, 3):
            rng = np.random.default_rng(seed)
            candidates.append(_grid_layout(pattern, rng, 0.03))
    # Extra fine-jitter seeds around the two most promising families
    for pattern in (patterns[0], patterns[2]):
        for seed in (11, 12, 13, 14):
            rng = np.random.default_rng(seed)
            candidates.append(_grid_layout(pattern, rng, 0.015))

    best = None
    for centers in candidates:
        c, r = _slsqp_refine(centers)
        r = compute_max_radii(c)
        s = float(np.sum(r))
        if best is None or s > best[0]:
            best = (s, c, r)

    # Second stage: exploit the incumbent with perturbed re-convergences.
    # Each candidate is scored AFTER exact polishing so selection matches
    # the final output metric exactly.
    import time
    t0 = time.time()
    rng = np.random.default_rng(42)
    global_centers = best[1].copy()
    base_centers = global_centers.copy()
    for trial in range(400):
        if time.time() - t0 > 250.0:
            break  # time guard: leave budget for final polish stages
        # Every 20th trial, restart from the global best to escape basins
        if trial % 20 == 19:
            base_centers = global_centers.copy()
        scale = 0.02 * (0.93 ** trial) + 1e-4
        pert = base_centers + rng.normal(0.0, scale, base_centers.shape)
        pert = np.clip(pert, 0.03, 0.97)
        c, r = _slsqp_refine(pert)
        c = _position_polish(c)
        c, r = _exact_polish(c)
        s = float(np.sum(r))
        if s > best[0]:
            best = (s, c, r)
            base_centers = c.copy()
            global_centers = c.copy()

    # Symmetry exploitation: mirrored and 90-degree rotated variants of
    # the incumbent map the square onto itself but re-land in different
    # basins after re-optimization. Cheap near-free diversity.
    def _variants(c):
        yield c
        yield c * [1.0, -1.0] + [0.0, 1.0]      # vertical mirror
        yield c * [-1.0, 1.0] + [1.0, 0.0]      # horizontal mirror
        rot = np.column_stack([1.0 - c[:, 1], c[:, 0]])  # 90-degree rotation
        yield rot
        yield rot * [1.0, -1.0] + [0.0, 1.0]
    for vc in _variants(best[1]):
        if time.time() - t0 > 290.0:
            break
        c, r = _slsqp_refine(vc)
        c = _position_polish(c)
        c, r = _exact_polish(c)
        s = float(np.sum(r))
        if s > best[0]:
            best = (s, c, r)

    # Third stage: alternate position moves and exact radius inflation
    # until fixed point. The two operators target different non-smooth
    # structure of the landscape and each can unlock the other.
    sum_radii, centers, radii = best
    for _ in range(12):
        c = _position_polish(centers)
        c, r = _exact_polish(c)
        s = float(np.sum(r))
        if s <= sum_radii + 1e-9:
            centers, radii = c, r
            sum_radii = max(sum_radii, s)
            break
        centers, radii = c, r
        sum_radii = s
    return centers, radii, float(np.sum(radii))


def _position_polish(centers, steps=(0.01, 0.004, 0.0015, 0.0005, 0.00015)):
    """
    Discrete local search on center positions.

    Operators per iteration:
      1. Single-circle moves along 16 unit directions at several step
         sizes; accept a move only if the total greedy-inflated radius
         sum strictly increases.
      2. Pairwise separation: for each near-tangent pair (gap below a
         threshold), push both centers apart along their connecting line
         to create mutual slack. This unblocks configurations that
         single-circle moves cannot reach.
    Repeated until no move helps (fixed point).
    """
    n = centers.shape[0]
    dirs = np.array([[np.cos(t), np.sin(t)] for t in
                     np.linspace(0, 2 * np.pi, 16, endpoint=False)])
    cur = centers.copy()
    dist = cdist(cur, cur)
    np.fill_diagonal(dist, np.inf)
    cur_sum = float(np.sum(compute_max_radii(cur)))
    improved = True
    while improved:
        improved = False
        for i in range(n):
            best_move = None
            for st in steps:
                for d in dirs:
                    trial = cur.copy()
                    trial[i] = np.clip(cur[i] + st * d, 0.02, 0.98)
                    s = float(np.sum(compute_max_radii(trial)))
                    if s > cur_sum + 1e-12 and (
                        best_move is None or s > best_move[0]
                    ):
                        best_move = (s, trial[i])
            if best_move is not None:
                cur[i] = best_move[1]
                cur_sum = best_move[0]
                improved = True
        # Pairwise separation moves (limit work to close pairs)
        d = cdist(cur, cur)
        np.fill_diagonal(d, np.inf)
        close = [(i, j) for i in range(n) for j in range(i + 1, n)
                if d[i, j] < 0.25]
        for (i, j) in close:
            u = cur[j] - cur[i]
            norm = np.linalg.norm(u)
            if norm < 1e-12:
                continue
            u = u / norm
            for st in (0.008, 0.003, 0.001):
                trial = cur.copy()
                trial[i] = np.clip(cur[i] - st * u, 0.02, 0.98)
                trial[j] = np.clip(cur[j] + st * u, 0.02, 0.98)
                s = float(np.sum(compute_max_radii(trial)))
                if s > cur_sum + 1e-12:
                    cur[i], cur[j] = trial[i], trial[j]
                    cur_sum = s
                    improved = True
    return cur


def _exact_polish(centers, eps=1e-6):
    """Inflate radii by bisection to exact contact (walls/neighbors).

    Sweeps with deterministic shuffled orderings until a fixed point:
    inflating one circle changes the limits for others, so a single
    ordered pass is order-dependent. Uses the evaluator's full
    1e-6 tolerance (leave eps slack so validity stays 1.0).
    """
    n = centers.shape[0]
    d = cdist(centers, centers)
    np.fill_diagonal(d, np.inf)
    wall = np.minimum(
        np.minimum(centers[:, 0], 1 - centers[:, 0]),
        np.minimum(centers[:, 1], 1 - centers[:, 1]),
    )
    radii = compute_max_radii(centers)
    order_rng = np.random.default_rng(7)
    for _ in range(8):
        changed = False
        order = order_rng.permutation(n)
        for i in order:
            lo = radii[i]
            hi = min(wall[i] - eps, float(np.min(d[i] - radii)) - eps)
            if hi <= lo:
                continue
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                ok = (mid <= wall[i] - eps) and np.all(mid <= d[i] - radii - eps)
                if ok:
                    lo = mid
                else:
                    hi = mid
            if lo > radii[i] + 1e-12:
                radii[i] = lo
                changed = True
        if not changed:
            break
    return centers, radii


def _slsqp_refine(centers):
    """
    Joint SLSQP optimization of centers and radii.

    Variables z = [x_0, y_0, r_0, x_1, y_1, r_1, ...] (3n total).
    Objective: maximize sum(r) -> minimize -sum(r).
    Hard inequality constraints (all must be >= 0):
      - d_ij - r_i - r_j  for every pair (non-overlap)
      - x_i - r_i, 1 - x_i - r_i, y_i - r_i, 1 - y_i - r_i (walls)

    Initial radii are a slightly shrunken greedy inflation so the
    start is strictly feasible, which helps SLSQP converge.
    """
    n = centers.shape[0]
    pairs = np.array(list(combinations(range(n), 2)), dtype=int)

    r0 = compute_max_radii(centers) * 0.98
    z0 = np.empty(3 * n)
    z0[0::3] = centers[:, 0]
    z0[1::3] = centers[:, 1]
    z0[2::3] = r0

    def objective(z):
        return -np.sum(z[2::3])

    def objective_grad(z):
        g = np.zeros_like(z)
        g[2::3] = -1.0
        return g

    def constraints_fun(z):
        x = z[0::3]
        y = z[1::3]
        r = z[2::3]
        d = cdist(np.column_stack([x, y]), np.column_stack([x, y]))
        out = [d[pairs[:, 0], pairs[:, 1]] - r[pairs[:, 0]] - r[pairs[:, 1]],
               x - r, 1.0 - x - r, y - r, 1.0 - y - r]
        return np.concatenate(out)

    options = {"maxiter": 500, "ftol": 1e-14}
    cons = [{"type": "ineq", "fun": constraints_fun}]
    res = minimize(objective, z0, jac=objective_grad,
                   method="SLSQP", constraints=cons, options=options)
    # Restart from the first solution to escape premature termination.
    res = minimize(objective, res.x, jac=objective_grad,
                   method="SLSQP", constraints=cons, options=options)

    z = res.x
    centers = np.column_stack([z[0::3], z[1::3]])
    radii = np.maximum(z[2::3], 0.0)
    return centers, radii


def _grid_layout(row_counts, jitter_rng=None, jitter=0.0):
    """
    Build an initial center arrangement from per-row circle counts.
    Rows are evenly spaced with margin; alternate rows are staggered
    by half a spacing for a hex-like pattern. Optional uniform jitter
    breaks symmetry to diversify the multi-start seeds.
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


def compute_max_radii(centers):
    """
    Greedy inflation: start all radii at zero and repeatedly grow each
    circle to the maximum radius allowed by (a) the unit-square walls and
    (b) non-overlap with all other circles at their current radii.

    Iterating this growth pass many times converges to a valid packing
    with a locally maximal sum of radii (each pass never decreases any
    radius, and radii are bounded, so the process converges).

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates

    Returns:
        np.array of shape (n) with radius of each circle
    """
    n = centers.shape[0]
    radii = np.zeros(n)
    dist = np.sqrt(((centers[:, None, :] - centers[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(dist, np.inf)

    # Wall limit for each circle
    wall = np.minimum(
        np.minimum(centers[:, 0], 1 - centers[:, 0]),
        np.minimum(centers[:, 1], 1 - centers[:, 1]),
    )

    # Multiple growth passes: each circle takes the largest radius it can
    for _ in range(200):
        improved = False
        for i in range(n):
            # Max radius allowed by neighbors: dist[i,j] - radii[j]
            lim_by_others = np.min(dist[i] - radii)
            new_r = min(wall[i], lim_by_others)
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
