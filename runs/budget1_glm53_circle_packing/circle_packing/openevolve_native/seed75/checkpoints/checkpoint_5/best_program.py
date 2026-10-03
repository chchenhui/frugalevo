# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles using a hexagonal lattice."""
import numpy as np


def construct_packing():
    """
    Construct a two-tier packing of 26 circles in a unit square.

    Approach: hybrid of large hexagonal block + smaller hex block on top.
    - Bottom tier: 4 staggered hex rows with counts 5, 4, 5, 4 (18 circles)
      at radius r = 0.1 (width-limited: a row of 5 spans 10r = 1).
      Row centers at y = r + k*sqrt(3)*r; row 4 center y = 0.61963.
    - Top tier: 2 staggered rows of 4 smaller circles (8 circles) at radius
      r2 ~ 0.0831. The lower top row is offset by 0.1 horizontally from
      row-4 centers, so clearance requires sqrt(0.01 + dy^2) >= 0.1 + r2,
      i.e. dy = sqrt(r2^2 + 0.2*r2). The height constraint
      0.61963 + sqrt(r2^2+0.2*r2) + (1+sqrt(3))*r2 <= 1 solves to
      r2 ~ 0.0831, making the top edge land at ~0.9999.
    - Sum of radii = 18*0.1 + 8*0.0831 ~ 2.465.

    Returns:
        Tuple of (centers, radii, sum_of_radii)
    """
    n = 26
    r = 0.1
    dy = np.sqrt(3.0) * r  # vertical spacing between staggered rows
    y4 = r + 3.0 * dy      # center y of the 4th (staggered) big row

    # Solve for top-tier radius r2:
    # y4 + sqrt(r2^2 + 0.2*r2) + (1+sqrt(3))*r2 = 1
    # => 6.46409*r2^2 - 2.27846*r2 + (1 - y4)^2 ... solved via quadratic.
    a = 1.0 + (1.0 + np.sqrt(3.0)) ** 2 - 1.0  # coefficient of r2^2 after squaring
    # Direct quadratic: r2^2 + 0.2*r2 = (1 - y4 - (1+sqrt(3))*r2)^2
    # Expand: (1 + (1+sqrt3)^2 - 1) r2^2 ... use numeric solve:
    from math import sqrt
    A = (1.0 + np.sqrt(3.0)) ** 2 - 1.0  # = (1+sqrt3)^2 - 1
    B = -2.0 * (1.0 - y4) * (1.0 + np.sqrt(3.0)) - 0.2
    C = (1.0 - y4) ** 2
    disc = B * B - 4.0 * A * C
    r2 = (-B - sqrt(disc)) / (2.0 * A)
    r2 = min(r2, 0.083)  # safety margin

    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    idx = 0

    # Bottom tier: rows 5, 4, 5, 4 of radius-r circles
    row_counts = [5, 4, 5, 4]
    for row, m in enumerate(row_counts):
        y = r + row * dy
        if m == 5:
            xs = r + 2.0 * r * np.arange(m)          # touches left/right walls
        else:
            xs = 2.0 * r + 2.0 * r * np.arange(m)    # staggered by r
        for x in xs:
            centers[idx] = [x, y]
            radii[idx] = r
            idx += 1

    # Top tier: lower row of 4 offset by 0.1 from row-4 centers,
    # upper row of 4 staggered by r2 above it.
    y1 = y4 + np.sqrt(r2 * r2 + 0.2 * r2)  # exact touching distance to row 4
    y2 = y1 + np.sqrt(3.0) * r2
    xs_low = [0.1, 0.3, 0.5, 0.7]
    xs_high = [0.2, 0.4, 0.6, 0.8]
    for x in xs_low:
        centers[idx] = [x, y1]
        radii[idx] = r2
        idx += 1
    for x in xs_high:
        centers[idx] = [x, y2]
        radii[idx] = r2
        idx += 1

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
