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

    # Hexagonal (staggered-row) lattice packing of 26 equal circles.
    # Rows (top to bottom) have counts 5, 4, 5, 4, 5, 3 = 26 circles.
    # Horizontal spacing s, vertical row spacing h = s*sqrt(3)/2.
    # Width constraint: 5s <= 1 ; Height constraint: s + 5h <= 1.
    s = 1.0 / (1.0 + 5.0 * np.sqrt(3.0) / 2.0)
    h = s * np.sqrt(3.0) / 2.0
    r = s / 2.0

    row_counts = [5, 4, 5, 4, 5, 3]
    idx = 0
    for row, count in enumerate(row_counts):
        y = r + row * h
        if row % 2 == 0:
            # Unstaggered rows touch both side walls
            for j in range(count):
                centers[idx] = [r + j * s, y]
                idx += 1
        else:
            # Staggered rows are shifted by half a spacing
            for j in range(count):
                centers[idx] = [r + s / 2.0 + j * s, y]
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

    # Uniform radius: limited by border distance and half the minimum
    # pairwise center distance (valid for lattice-based layouts).
    r = np.inf
    for i in range(n):
        x, y = centers[i]
        r = min(r, x, y, 1.0 - x, 1.0 - y)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            r = min(r, dist / 2.0)

    return np.full(n, r)


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