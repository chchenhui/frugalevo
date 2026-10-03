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

    # Hexagonal lattice packing: alternating rows of 5 and 6 circles.
    # Rows: 5, 6, 5, 6, 5 = 27 positions; drop the last one to get 26.
    s = 0.17          # horizontal lattice spacing
    dy = s * np.sqrt(3) / 2.0  # vertical spacing between rows
    y0 = (1.0 - 4 * dy) / 2.0  # vertically center the 5 rows

    idx = 0
    for row in range(5):
        count = 5 if row % 2 == 0 else 6
        y = y0 + row * dy
        for j in range(count):
            x = 0.5 + (j - (count - 1) / 2.0) * s
            if idx < n:
                centers[idx] = [x, y]
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
    # Precompute pairwise distances
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[i, j] = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))

    # Start all radii large (limited only by walls) and iteratively
    # shrink each radius so it fits against walls and neighbors.
    radii = np.array([min(x, y, 1 - x, 1 - y) for x, y in centers])

    # Monotone shrink-to-constraint iterations: each pass only decreases
    # radii, so the result is always a valid (non-overlapping, in-square)
    # packing; it converges to a configuration where circles touch.
    for _ in range(100):
        max_change = 0.0
        for i in range(n):
            x, y = centers[i]
            new_r = min(x, y, 1 - x, 1 - y)
            for j in range(n):
                if j != i:
                    new_r = min(new_r, dist[i, j] - radii[j])
            new_r = max(new_r, 0.0)
            max_change = max(max_change, radii[i] - new_r)
            radii[i] = new_r
        if max_change < 1e-12:
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