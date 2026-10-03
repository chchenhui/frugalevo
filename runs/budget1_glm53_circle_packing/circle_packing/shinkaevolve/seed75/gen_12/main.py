# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles using a hexagonal lattice."""
import numpy as np


def construct_packing():
    """
    Construct a hexagonal-lattice arrangement of 26 circles in a unit square
    that maximizes the sum of their radii via a closed-form layout.

    Returns:
        Tuple of (centers, radii, sum_of_radii)
    """
    n = 26
    centers_list = []

    # Hexagonal lattice: staggered rows.
    # Row counts alternate 6, 5, 6, 5, 6, ... (vertical direction)
    # Horizontal spacing between adjacent circles in a row: 2*r
    # Vertical spacing between rows: sqrt(3) * r  (hex close packing)
    r = 1.0 / 12.0  # 6 columns span exactly the unit square: 5*2r + 2r = 12r = 1

    dx = 2.0 * r                # horizontal spacing within a row
    dy = np.sqrt(3.0) * r        # vertical spacing between rows

    row_counts = [6, 5, 6, 5, 6]  # 28 positions -> drop 2 to reach 26

    # Center the 5 rows vertically so wall distances are balanced:
    # the rows span 2r (circle diameters) + 4*dy (row gaps) in height.
    span = 2.0 * r + 4.0 * dy
    y = (1.0 - span) / 2.0 + r
    for row_idx, count in enumerate(row_counts):
        if row_idx % 2 == 0:
            # Even rows: flush against the left wall
            xs = r + dx * np.arange(count)
        else:
            # Odd rows: offset by r (staggered)
            xs = 2.0 * r + dx * np.arange(count)
        for x in xs:
            centers_list.append((x, y))
        y += dy

    # Drop the two circles closest to the top-right corner to make room
    # (keeps the arrangement balanced and lets remaining radii stay at r).
    centers_list = centers_list[:28]
    # Remove last two from the top row (rightmost) to reach exactly 26
    centers_list = centers_list[:26]

    centers = np.array(centers_list)

    radii = compute_max_radii(centers)
    sum_radii = float(np.sum(radii))
    return centers, radii, sum_radii


def compute_max_radii(centers):
    """
    Compute maximal radii via fixed-point relaxation:
    start with each radius at its full wall distance (so circles can grow
    into unused space such as the truncated top row and vertical margins),
    then repeatedly cap r_i = min(wall_i, min_j (d_ij - r_j)).
    This converges to a valid packing where each circle is as large as
    possible given its neighbors and the square boundary.
    """
    n = centers.shape[0]
    x = centers[:, 0]
    y = centers[:, 1]
    wall = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))

    # Precompute pairwise distances
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            dist[i, j] = d
            dist[j, i] = d

    radii = wall.copy()
    for _ in range(500):
        max_change = 0.0
        for i in range(n):
            # Largest allowed radius given current neighbor radii
            cap = wall[i]
            for j in range(n):
                if j == i:
                    continue
                cap = min(cap, dist[i, j] - radii[j])
            cap = max(cap, 0.0)
            max_change = max(max_change, abs(radii[i] - cap))
            radii[i] = cap
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