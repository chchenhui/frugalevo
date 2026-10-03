# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """
    Hexagonal-lattice packing of 26 circles in the unit square.

    Approach: generate several candidate triangular-lattice layouts
    (different row splits such as 5,4,5,4,5,3 and 4,5,4,5,4,4, and
    several lattice spacings), compute maximal valid radii for each
    via greedy grow-to-contact sweeps, and keep the layout with the
    largest radius sum. Short rows are offset by half a horizontal
    spacing to realize the hexagonal (triangular) lattice.

    Returns:
        Tuple of (centers, radii, sum_of_radii)
    """
    n = 26
    best = None

    row_sets = [
        [5, 4, 5, 4, 5, 3],
        [4, 5, 4, 5, 4, 4],
        [5, 4, 5, 4, 5, 3],
        [3, 5, 4, 5, 4, 5],
        [5, 5, 4, 5, 4, 3],
    ]
    base_r0 = 1.0 / (2.0 + 5.0 * np.sqrt(3.0))

    for row_counts in row_sets:
        for scale in (0.85, 0.95, 1.0, 1.05, 1.15):
            r0 = base_r0 * scale
            dy = np.sqrt(3.0) * r0
            dx = 2.0 * r0
            centers = np.zeros((n, 2))
            idx = 0
            for row, count in enumerate(row_counts):
                y = r0 + row * dy
                # short rows shifted by half a spacing (hex offset)
                x0 = r0 if count % 2 == 1 else 2.0 * r0
                for j in range(count):
                    centers[idx] = [x0 + j * dx, y]
                    idx += 1
            radii = compute_max_radii(centers)
            s = np.sum(radii)
            if best is None or s > best[0]:
                best = (s, centers, radii)

    sum_radii, centers, radii = best
    return centers, radii, sum_radii


def compute_max_radii(centers):
    """
    Compute valid radii for fixed centers.

    Method: greedy "grow-to-contact" sweeps. Starting from tiny radii,
    each circle in turn grows to the largest radius allowed by the walls
    and its neighbors' current radii (growth only, never shrink).
    Repeating sweeps converges toward a contact-saturated packing.
    The sweep order matters, so several orderings (forward, reverse,
    by wall distance) are tried and the best is kept. A final pairwise
    shrink pass guarantees the returned configuration is valid
    (overlap-free, inside the square).

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates

    Returns:
        np.array of shape (n) with radius of each circle
    """
    n = centers.shape[0]
    dist = np.sqrt(np.sum((centers[:, None, :] - centers[None, :, :]) ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)

    # distance to nearest wall caps every radius
    wall = np.minimum.reduce([centers[:, 0], centers[:, 1],
                             1.0 - centers[:, 0], 1.0 - centers[:, 1]])

    def grow_sweeps(order):
        radii = np.full(n, 1e-6)
        for _ in range(50):
            changed = False
            for i in order:
                limit = min(wall[i], np.min(dist[i] - radii))
                if limit > radii[i] + 1e-12:
                    radii[i] = limit
                    changed = True
            if not changed:
                break
        return radii

    def fix_pass(radii):
        for i in range(n):
            for j in range(i + 1, n):
                s = radii[i] + radii[j]
                d = dist[i, j]
                if s > d:
                    scale = d / s
                    radii[i] *= scale
                    radii[j] *= scale
        return radii

    orders = [
        list(range(n)),
        list(range(n - 1, -1, -1)),
        list(np.argsort(wall)),
        list(np.argsort(-wall)),
    ]

    best_radii = None
    best_sum = -1.0
    for order in orders:
        r = grow_sweeps(order)
        r = fix_pass(r)
        s = np.sum(r)
        if s > best_sum:
            best_sum = s
            best_radii = r
    return best_radii


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
