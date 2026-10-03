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

    # Hexagonal-grid layout: rows of 5, 6, 5, 6, 4 circles.
    # Hexagonal arrangement is the densest local structure for
    # similar-sized circles, so it is the natural explicit layout.
    dx = 0.2   # horizontal spacing within a row
    dy = 0.19  # vertical spacing between rows (slightly compressed hex)
    row_counts = [5, 6, 5, 6, 4]
    idx = 0
    y = 0.11
    for r, count in enumerate(row_counts):
        # hexagonal offset: odd rows shifted by dx/2
        x0 = 0.5 - (count - 1) * dx / 2.0
        for c in range(count):
            centers[idx] = [x0 + c * dx + (r % 2) * dx * 0.5, y]
            idx += 1
        y += dy

    # Relax positions: push overlapping pairs apart and keep in square.
    radii = np.full(n, 0.09)
    for it in range(2000):
        moved = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dvec = centers[i] - centers[j]
                dist = np.sqrt(np.sum(dvec ** 2))
                min_dist = radii[i] + radii[j]
                if dist < min_dist and dist > 1e-12:
                    push = (min_dist - dist) * 0.5
                    u = dvec / dist
                    centers[i] += push * u
                    centers[j] -= push * u
                    moved += push
        # keep inside the square
        for i in range(n):
            r_cap = min(centers[i, 0], centers[i, 1],
                        1 - centers[i, 0], 1 - centers[i, 1])
            radii[i] = min(radii[i] + 1e-4, max(r_cap, 1e-6))
        centers = np.clip(centers, radii, 1 - radii)
        if moved < 1e-9:
            break

    # Compute maximum valid radii for the relaxed configuration
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
    # Cap every radius by its distance to the square boundary
    radii = np.minimum(
        np.minimum(centers[:, 0], centers[:, 1]),
        np.minimum(1 - centers[:, 0], 1 - centers[:, 1]),
    )

    # Iteratively enforce pairwise constraints until convergence.
    # A single pass is insufficient because shrinking one radius
    # changes the constraints for its neighbors.
    for _ in range(200):
        worst = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                if radii[i] + radii[j] > dist:
                    scale = dist / (radii[i] + radii[j])
                    radii[i] *= scale
                    radii[j] *= scale
                    worst = max(worst, 1.0 - scale)
        if worst < 1e-12:
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