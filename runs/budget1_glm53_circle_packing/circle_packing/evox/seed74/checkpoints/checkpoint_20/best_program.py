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
    # Hexagonal lattice packing: 6 staggered rows with widths 5,4,5,4,5,3 (=26).
    # All circles share radius r. Vertical constraint is binding:
    #   2r + 5*(sqrt(3)*r) = 1  =>  r = 1/(2 + 5*sqrt(3)) ~ 0.09381
    # Adjacent rows are staggered horizontally by r so that neighbor
    # center-to-center distances are exactly 2r (hexagonal contacts).
    # Sum of radii = 26 * r ~ 2.439.
    n = 26
    centers = np.zeros((n, 2))

    r = 1.0 / (2.0 + 5.0 * np.sqrt(3.0))
    dy = np.sqrt(3.0) * r  # vertical spacing between adjacent rows

    row_widths = [5, 4, 5, 4, 5, 3]
    assert sum(row_widths) == n

    idx = 0
    for row, w in enumerate(row_widths):
        y = r + row * dy
        # Even-indexed rows sit on even multiples of r from center;
        # odd-indexed rows on odd multiples (stagger by r).
        if row % 2 == 0:
            c = w - 1  # even offset parity (w is odd here)
        else:
            c = w - 1 if (w - 1) % 2 == 1 else w  # force odd parity
        for k in range(w):
            x = 0.5 + (2 * k - c) * r
            centers[idx] = [x, y]
            idx += 1

    radii = np.full(n, r)
    sum_radii = float(n * r)
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
