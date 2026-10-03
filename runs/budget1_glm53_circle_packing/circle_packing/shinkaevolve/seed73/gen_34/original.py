# EVOLVE-BLOCK-START
"""Hexagonal staggered-row constructor for packing n=26 circles in a unit square."""
import numpy as np


def construct_packing():
    """
    Construct a staggered hexagonal-like arrangement of 26 circles in a
    unit square, maximizing the sum of their radii.

    Returns:
        centers, radii, sum_radii
    """
    n = 26
    centers_list = []

    # 5 staggered rows: counts 6, 5, 6, 5, 4 (total 26)
    row_counts = [6, 5, 6, 5, 4]
    # Vertical spacing tuned so rows nearly touch: hexagonal rows have
    # dy = r * sqrt(3) if circles in adjacent rows touch. With 5 rows
    # covering the unit height: r ~ 1/(2*4 + sqrt(3)) roughly.
    # Use dy slightly less than sqrt(3)*r to let radius solver find max.
    num_gaps_y = len(row_counts) - 1  # 4 gaps
    # For hex packing with r: first row at y=r, last at y=1-r
    # dy = sqrt(3)*r  =>  2r + 4*sqrt(3)*r = 1  =>  r = 1/(2+4*1.732)
    r_guess = 1.0 / (2 + 4 * np.sqrt(3))  # ~0.1177
    dy = np.sqrt(3) * r_guess * 0.98      # slight slack

    # Horizontal spacing per row: row with k circles spaced evenly,
    # margin r from walls: dx = (1 - 2r)/(k-1)
    y_cursor = r_guess * 0.98
    for row_idx, k in enumerate(row_counts):
        dx = (1.0 - 2 * r_guess) / (k - 1) if k > 1 else 0.0
        x_start = r_guess * 0.98
        for i in range(k):
            centers_list.append([x_start + i * dx, y_cursor])
        y_cursor += dy

    centers = np.array(centers_list)

    # Compute maximal valid radii: start from wall distances,
    # then iteratively resolve pairwise overlaps by proportional scaling.
    radii = compute_max_radii(centers)

    sum_radii = float(np.sum(radii))
    return centers, radii, sum_radii


def compute_max_radii(centers):
    """
    Compute radii limited by walls and pairwise distances,
    resolving overlaps by proportional pairwise scaling until valid.
    """
    n = centers.shape[0]
    x = centers[:, 0]
    y = centers[:, 1]
    radii = np.minimum.reduce([x, y, 1 - x, 1 - y])

    # Precompute pairwise distances
    dist = np.sqrt(((centers[:, None, :] - centers[None, :, :]) ** 2).sum(-1))

    # Iteratively fix overlaps
    for _ in range(50):
        worst = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                d = dist[i, j]
                s = radii[i] + radii[j]
                if s > d and s > 0:
                    scale = d / s
                    radii[i] *= scale
                    radii[j] *= scale
                    worst = max(worst, 1 - scale)
        if worst < 1e-12:
            break

    # Final safety: hard clip against walls and neighbors
    for i in range(n):
        for j in range(n):
            if i != j:
                allowed = dist[i, j] - radii[j]
                if radii[i] > allowed:
                    radii[i] = max(allowed, 1e-9)
    radii = np.minimum(radii, np.minimum.reduce([x, y, 1 - x, 1 - y]))
    radii = np.maximum(radii, 1e-9)

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