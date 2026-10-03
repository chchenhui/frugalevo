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
    # Initialize arrays for 26 circles
    n = 26
    centers = np.zeros((n, 2))

    # Hexagonal lattice: rows of 5,4,5,4,5,3 circles.
    # Vertical spacing between rows is sqrt(3)*r (hex packing),
    # staggered rows are offset horizontally by r.
    r0 = 1.0 / (2.0 + 5.0 * np.sqrt(3.0))  # ~0.0939, fits 6 hex rows
    dy = np.sqrt(3.0) * r0

    row_counts = [5, 4, 5, 4, 5, 3]
    idx = 0
    for k, cnt in enumerate(row_counts):
        y = r0 + k * dy
        if cnt == 5:
            # Full row touching left/right walls
            xs = [r0 + 2.0 * r0 * j for j in range(5)]
        elif cnt == 4:
            # Staggered (offset) row
            xs = [2.0 * r0 + 2.0 * r0 * j for j in range(4)]
        else:
            # Top row of 3, centered horizontally
            xs = [0.5 - 2.0 * r0 + 2.0 * r0 * j for j in range(3)]
        for x in xs:
            centers[idx] = [x, y]
            idx += 1

    # Alternate vectorized radius growth with center relaxation:
    # 1) grow every radius to the largest value allowed by walls and the
    #    other circles' current radii (damped, then a final valid pass),
    # 2) nudge each center away from its most binding constraint
    #    (nearest wall or nearest crowded neighbor) with a decaying step,
    #    so circles drift into the corner/edge gaps the pure hex lattice
    #    leaves empty. Keep the best valid configuration found.
    rng = np.random.default_rng(0)

    def grow(c):
        """Damped growth to (near) feasibility limit, then a validity pass."""
        dd = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(dd, np.inf)
        wall = np.minimum(c.min(axis=1), (1.0 - c).min(axis=1))
        rad = np.full(len(c), 1e-6)
        for _ in range(120):
            nb = (dd - rad[None, :]).min(axis=1)
            lim = np.minimum(wall, nb)
            # damped growth so the fixed point is a balanced configuration
            rad = np.minimum(lim, rad * 1.4 + 1e-5)
        # one simultaneous pass back to strict feasibility
        nb = (dd - rad[None, :]).min(axis=1)
        rad = np.clip(np.minimum(rad, np.minimum(wall, nb)) - 1e-9, 1e-9, None)
        return rad, dd, wall

    rad, d, wall = grow(centers)
    best_sum = rad.sum()
    best_c, best_r = centers.copy(), rad.copy()

    for it in range(250):
        step = 0.012 * (1.0 - it / 250.0) + 0.0005
        # binding constraint for each circle: wall vs nearest neighbor slack
        nb_lim = (d - rad[None, :]).min(axis=1)
        j_min = (d - rad[None, :]).argmin(axis=1)
        for i in range(n):
            if wall[i] <= nb_lim[i]:
                # wall-limited: move away from the nearest wall(s)
                x, y = centers[i]
                dx = step if x < 0.5 else -step
                dy = step if y < 0.5 else -step
                if wall[i] == min(x, 1 - x):
                    dy = 0.0
                else:
                    dx = 0.0
                centers[i] += np.array([dx, dy])
            else:
                # neighbor-limited: slide directly away from that neighbor
                vec = centers[i] - centers[j_min[i]]
                nrm = np.linalg.norm(vec)
                if nrm > 1e-12:
                    centers[i] += (vec / nrm) * step
        # tiny deterministic jitter for exploration
        centers += rng.normal(0.0, 0.25 * step, centers.shape)
        centers = np.clip(centers, 0.005, 0.995)
        rad, d, wall = grow(centers)
        s = rad.sum()
        if s > best_sum:
            best_sum = s
            best_c, best_r = centers.copy(), rad.copy()
        else:
            centers = best_c.copy()
            rad, d, wall = grow(centers)

    return best_c, best_r, float(best_r.sum())


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