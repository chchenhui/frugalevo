# EVOLVE-BLOCK-START
"""Hexagonal-lattice constructor for packing 26 circles in a unit square."""
import numpy as np

N = 26
HEX_ASPECT = np.sqrt(3.0) / 2.0  # vertical spacing factor for hex lattice


def build_hex_rows():
    """
    Build 26 circle centers arranged in 6 hexagonal rows.

    Row layout (from bottom to top): 4, 5, 4, 5, 4, 4 circles.
    Rows with 5 circles are inset by half a spacing (hex offset),
    giving the densest local triangular arrangement.

    Returns:
        np.array of shape (26, 2)
    """
    s = 0.2                       # horizontal lattice spacing
    dy = s * HEX_ASPECT           # vertical lattice spacing
    row_counts = [4, 5, 4, 5, 4, 4]
    centers = []

    # vertical placement: center the 6 rows in the square
    total_h = dy * (len(row_counts) - 1)
    y0 = (1.0 - total_h) / 2.0

    for r, count in enumerate(row_counts):
        y = y0 + r * dy
        # 5-circle rows are offset inward by s/2 (hex staggering)
        x0 = (1.0 - s * (count - 1)) / 2.0
        for c in range(count):
            centers.append([x0 + c * s, y])

    return np.array(centers)


def solve_radii(centers, iterations=200):
    """
    Compute radii by proportional shrinking under two constraint types:
      1. Circle must stay inside unit square: r_i <= min(x, y, 1-x, 1-y)
      2. Circles must not overlap: r_i + r_j <= dist(i, j)

    Uses a bounded fixed-point loop: whenever a pair violates the
    distance constraint, both radii are scaled proportionally.

    Args:
        centers: (n, 2) array of centers
        iterations: max refinement passes

    Returns:
        np.array of shape (n,) of radii
    """
    n = centers.shape[0]

    # Wall constraints give an upper bound for each radius
    walls = np.minimum.reduce([
        centers[:, 0], centers[:, 1],
        1.0 - centers[:, 0], 1.0 - centers[:, 1]
    ])
    radii = walls.copy()

    # Precompute pairwise distances
    dist = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)

    for _ in range(iterations):
        # Pairwise violation: radii[i] + radii[j] > dist[i, j]
        excess = radii[:, None] + radii[None, :] - dist
        np.fill_diagonal(excess, -np.inf)
        worst = np.max(excess)
        if worst <= 1e-12:
            break
        # Fix all violating pairs proportionally, symmetric pass
        for i in range(n):
            for j in range(i + 1, n):
                d = dist[i, j]
                ssum = radii[i] + radii[j]
                if ssum > d + 1e-15 and ssum > 0:
                    scale = d / ssum
                    radii[i] *= scale
                    radii[j] *= scale

    # Never exceed wall bounds
    radii = np.minimum(radii, walls)
    return radii


def _pressures(centers, radii):
    """For each circle, compute 'pressure' = sum of violated room beyond
    its current radius, and the direction to relieve it (away from the
    tightest neighbor / toward square interior)."""
    n = centers.shape[0]
    diff = centers[:, None, :] - centers[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist, np.inf)

    # slack to each neighbor: dist - (r_i + r_j); negative => tight
    slack = dist - (radii[:, None] + radii[None, :])
    np.fill_diagonal(slack, -np.inf)
    # wall slack: distance from center to wall minus radius
    wl = centers[:, 0] - radii          # left wall slack
    wr = 1.0 - centers[:, 0] - radii
    wb = centers[:, 1] - radii
    wt = 1.0 - centers[:, 1] - radii

    pressure = np.zeros(n)
    move = np.zeros((n, 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        unit = np.where(dist[:, :, None] > 1e-12,
                        diff / np.maximum(dist[:, :, None], 1e-12), 0.0)
    # pairwise relief: push tight pairs apart proportionally
    tight = np.maximum(-slack, 0.0)     # amount by which pair is over-tight
    np.fill_diagonal(tight, 0.0)
    push = (unit * tight[:, :, None] * 0.5).sum(axis=1) \
         - (unit * tight[:, :, None] * 0.5).sum(axis=0)

    # wall relief: push inward when a wall is violated
    dx = np.maximum(-wl, 0.0) - np.maximum(-wr, 0.0)
    dy = np.maximum(-wb, 0.0) - np.maximum(-wt, 0.0)
    wall = np.stack([dx, dy], axis=1)

    pressure = tight.sum(axis=1) \
        + np.maximum(-wl, 0) + np.maximum(-wr, 0) \
        + np.maximum(-wb, 0) + np.maximum(-wt, 0)
    return pressure, push + wall


def _perturb_and_resolve(centers, rounds=60, step=0.15, seed=0):
    """Iteratively move centers away from tight contacts and re-solve
    radii, keeping the best feasible configuration seen."""
    rng = np.random.default_rng(seed)
    best_centers = centers.copy()
    radii = solve_radii(centers)
    best_sum = float(np.sum(radii))

    cur = centers.copy()
    cur_radii = radii
    for t in range(rounds):
        pressure, move = _pressures(cur, cur_radii)
        total = pressure.sum()
        # apply movement with decaying step
        alpha = step * max(0.2, 1.0 - t / rounds)
        cur = cur + alpha * move
        # small decaying jitter to escape symmetric stalls
        cur += rng.normal(0, 0.002 * max(0.2, 1.0 - t / rounds),
                          size=cur.shape)
        cur = np.clip(cur, 1e-4, 1.0 - 1e-4)

        cur_radii = solve_radii(cur)
        s = float(np.sum(cur_radii))
        if s > best_sum:
            best_sum = s
            best_centers = cur.copy()
        if total <= 1e-9:
            break
    return best_centers


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
    centers = build_hex_rows()
    centers = _perturb_and_resolve(centers)
    radii = solve_radii(centers)
    sum_radii = float(np.sum(radii))
    return centers, radii, sum_radii
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