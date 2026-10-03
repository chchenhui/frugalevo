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
    # Hexagonal-lattice layout: 5 staggered rows with counts 6,5,6,5,6.
    # Horizontal spacing 0.2, vertical spacing 0.1*sqrt(3) ~ 0.1732.
    # This lattice supports 28 circles of radius 0.1; we drop 2 adjacent
    # circles from the middle row (n=26) to leave a growth hole.
    n = 26
    centers_list = []
    y0 = 0.1
    dy = 0.1 * np.sqrt(3.0)
    counts = [6, 5, 6, 5, 6]
    skip = {(2, 2), (2, 3)}  # remove two adjacent circles in the middle row
    for row, cnt in enumerate(counts):
        y = y0 + row * dy
        for k in range(cnt):
            if (row, k) in skip:
                continue
            if cnt == 6:
                x = 0.2 * k  # touches walls at x=0 and x=1
            else:
                x = 0.1 + 0.2 * k  # staggered by half spacing
            centers_list.append([x, y])
    centers = np.array(centers_list)
    assert centers.shape == (n, 2)

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

    # Distance from each center to the nearest square border
    border = np.minimum(centers.min(axis=1), (1.0 - centers).min(axis=1))

    # For fixed centers, maximizing sum(r_i) subject to
    #   r_i <= border_i  and  r_i + r_j <= dist_ij
    # is a linear program -> solve it exactly when scipy is available.
    try:
        from scipy.optimize import linprog

        A_ub = []
        b_ub = []
        for i in range(n):
            row = np.zeros(n)
            row[i] = 1.0
            A_ub.append(row)
            b_ub.append(border[i])
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                row = np.zeros(n)
                row[i] = 1.0
                row[j] = 1.0
                A_ub.append(row)
                b_ub.append(d)
        res = linprog(
            c=-np.ones(n),
            A_ub=np.array(A_ub),
            b_ub=np.array(b_ub),
            bounds=[(0.0, None)] * n,
            method="highs",
        )
        if res.success:
            return res.x
    except Exception:
        pass

    # Fallback: multi-pass proportional scaling (converges to a valid,
    # reasonably tight set of radii if scipy is unavailable).
    radii = border.copy()
    for _ in range(200):
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                s = radii[i] + radii[j]
                if s > dist and s > 0:
                    scale = dist / s
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