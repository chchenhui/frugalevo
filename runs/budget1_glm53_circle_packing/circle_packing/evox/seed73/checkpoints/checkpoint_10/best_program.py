# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
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


def construct_packing():
    """
    Construct 26 circles in a unit square via multi-start refinement.

    Several initial arrangements are generated: staggered hexagonal grids
    with different row-count patterns (6-5-6-5-4, 5-6-5-6-4, 7-6-7-6,
    4-6-6-6-4, 5-5-6-5-5) plus jittered variants (deterministic RNG seeds
    to break symmetry). Each seed is greedily inflated, then locally
    refined (twice, with re-inflation between rounds, to escape the
    single-pass local optimum). The arrangement with the largest sum of
    radii is returned.

    Returns:
        Tuple of (centers, radii, sum_of_radii)
    """
    patterns = [
        [6, 5, 6, 5, 4],
        [5, 6, 5, 6, 4],
        [7, 6, 7, 6],
        [4, 6, 6, 6, 4],
        [5, 5, 6, 5, 5],
        [6, 4, 6, 4, 6],
        [5, 4, 6, 6, 5],
        [4, 5, 6, 5, 6],
        [6, 6, 7, 7],
        [5, 6, 6, 5, 4],
    ]

    best = None  # (sum, centers, radii)

    candidates = []
    for pattern in patterns:
        if sum(pattern) != 26:
            continue
        candidates.append(_grid_layout(pattern))
    # Jittered variants of several patterns for extra diversity
    for pattern in ([6, 5, 6, 5, 4], [5, 6, 5, 6, 4], [4, 6, 6, 6, 4],
                    [5, 5, 6, 5, 5], [7, 6, 7, 6]):
        for seed in (1, 2, 3):
            rng = np.random.default_rng(seed)
            candidates.append(
                _grid_layout(pattern, jitter_rng=rng, jitter=0.03)
            )

    for centers in candidates:
        radii = compute_max_radii(centers)
        # Two rounds of refine + re-inflate to escape local optima
        for _ in range(2):
            centers, radii = refine_centers(centers, radii)
            radii = compute_max_radii(centers)
        s = np.sum(radii)
        if best is None or s > best[0]:
            best = (s, centers, radii)

    # Basin hopping: perturb the best solution and re-refine to escape
    # the shared local optimum; keep any improvement found.
    for seed in (11, 22, 33, 44):
        rng = np.random.default_rng(seed)
        pert = np.clip(
            best[1] + rng.uniform(-0.02, 0.02, best[1].shape), 0.02, 0.98
        )
        r2 = compute_max_radii(pert)
        for _ in range(2):
            pert, r2 = refine_centers(pert, r2)
            r2 = compute_max_radii(pert)
        s2 = np.sum(r2)
        if s2 > best[0]:
            best = (s2, pert, r2)

    sum_radii, centers, radii = best
    return centers, radii, sum_radii


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
    while step > 2e-5:
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
                if r > best_r + 1e-12:
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
