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

    # Layout: 5x5 grid of circles with spacing 0.2 and radius 0.1.
    # Each grid circle is tangent to the square boundary on its row/column
    # extremes and tangent to its grid neighbors (spacing = 2r = 0.2),
    # so all 25 grid circles achieve the maximum border-limited radius 0.1.
    idx = 0
    for j in range(5):
        for i in range(5):
            centers[idx] = [0.1 + 0.2 * i, 0.1 + 0.2 * j]
            idx += 1

    # 26th circle: small circle nestled in the gap between the four
    # grid circles at (0.1,0.1), (0.3,0.1), (0.1,0.3), (0.3,0.3).
    # Distance from (0.2,0.2) to each neighbor is 0.2*sqrt(2)/2 ~ 0.1414,
    # so its radius is limited to 0.1414 - 0.1 ~ 0.0414 without
    # shrinking any of the large grid circles.
    centers[25] = [0.2, 0.2]

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
    radii = np.ones(n)

    # First, limit by distance to square borders
    for i in range(n):
        x, y = centers[i]
        # Distance to borders
        radii[i] = min(x, y, 1 - x, 1 - y)

    # Then, limit by distance to other circles.
    # Use iterative constraint propagation: repeatedly enforce
    # r_i + r_j <= dist for every pair by shrinking only the circle
    # that violates the constraint. Unlike proportional scaling of both
    # circles, this never needlessly shrinks circles that are already
    # valid, and it converges to the maximal radii for this layout.
    for _ in range(100):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                if radii[i] + radii[j] > dist + 1e-12:
                    # Shrink only the violating side(s), respecting
                    # the other circle's current radius.
                    new_ri = dist - radii[j]
                    new_rj = dist - radii[i]
                    if new_ri < radii[i]:
                        radii[i] = max(new_ri, 0.0)
                        changed = True
                    if new_rj < radii[j]:
                        radii[j] = max(new_rj, 0.0)
                        changed = True
        if not changed:
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