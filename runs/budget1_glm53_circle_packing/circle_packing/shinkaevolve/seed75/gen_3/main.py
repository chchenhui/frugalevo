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

    # Hexagonal lattice packing: locally the densest arrangement
    # Rows of 5,4,5,4,5,3 circles = 26 circles total.
    # Horizontal spacing s, vertical spacing s * sqrt(3)/2,
    # odd rows offset by s/2 to achieve the staggered hex pattern.
    s = 0.225
    dy = s * np.sqrt(3) / 2.0
    row_counts = [5, 4, 5, 4, 5, 3]
    total_height = 5 * dy
    y_margin = (1.0 - total_height) / 2.0

    idx = 0
    for row, cnt in enumerate(row_counts):
        y = y_margin + row * dy
        # Center each row horizontally
        x_start = (1.0 - (cnt - 1) * s) / 2.0
        offset = (s / 2.0) if (row % 2 == 1) else 0.0
        for k in range(cnt):
            centers[idx] = [x_start + k * s + offset, y]
            idx += 1

    # Compute maximum valid radii for this configuration
    radii = compute_max_radii(centers)

    # Calculate the sum of radii
    sum_radii = np.sum(radii)

    return centers, radii, sum_radii


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

    # Distance from each center to the nearest wall of the unit square
    wall_dist = np.minimum(
        np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
        np.minimum(centers[:, 1], 1.0 - centers[:, 1]),
    )

    # Precompute pairwise distances
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            dist[i, j] = d
            dist[j, i] = d

    # Start optimistic: each circle as large as its nearest wall allows
    radii = wall_dist.copy()

    # Iterative relaxation: repeatedly resolve overlaps by scaling the
    # offending pair proportionally, then re-clip to the walls.
    # This is a projection onto the feasible set and converges to a
    # valid packing that exploits the hexagonal lattice spacing
    # (interior radii approach s/2, the hex optimum).
    for _ in range(1000):
        max_violation = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                d = dist[i, j]
                rsum = radii[i] + radii[j]
                if rsum > d and rsum > 1e-15:
                    scale = d / rsum
                    radii[i] *= scale
                    radii[j] *= scale
                    violation = 1.0 - scale
                    if violation > max_violation:
                        max_violation = violation
        # Re-clip to walls after pair relaxation
        radii = np.minimum(radii, wall_dist)
        if max_violation < 1e-9:
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