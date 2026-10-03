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
    n = 26
    centers = np.zeros((n, 2))

    # --- Seed layout: hexagonal-style rows with counts 6,5,6,5,4 ---
    row_counts = [6, 5, 6, 5, 4]
    n_rows = len(row_counts)
    idx = 0
    for r_i, k in enumerate(row_counts):
        y = (r_i + 0.5) / n_rows
        for j in range(k):
            x = (j + 0.5) / k
            centers[idx] = [x, y]
            idx += 1

    # --- Pressure-based refinement ---
    # Repeatedly inflate target radii and push circles apart /
    # away from walls to relieve the tightest constraints.
    lr = 0.5
    for step in range(400):
        radii = compute_max_radii(centers)
        target = radii * 1.03

        disp = np.zeros_like(centers)

        # Pairwise repulsion where inflated targets would overlap
        for i in range(n):
            for j in range(i + 1, n):
                diff = centers[i] - centers[j]
                dist = np.sqrt(np.sum(diff ** 2))
                if dist < 1e-12:
                    diff = np.array([1e-6, 0.0])
                    dist = 1e-6
                overlap = target[i] + target[j] - dist
                if overlap > 0:
                    push = (overlap * 0.5) * diff / dist
                    disp[i] += push
                    disp[j] -= push

        # Wall repulsion where inflated targets exceed walls
        for i in range(n):
            x, y = centers[i]
            r = target[i]
            if x - r < 0:
                disp[i, 0] += (r - x)
            if x + r > 1:
                disp[i, 0] -= (x + r - 1)
            if y - r < 0:
                disp[i, 1] += (r - y)
            if y + r > 1:
                disp[i, 1] -= (y + r - 1)

        centers = centers + lr * disp
        centers = np.clip(centers, 1e-6, 1 - 1e-6)

        # Decay step size for smooth convergence
        lr *= 0.995
        if lr < 0.02:
            lr = 0.02

    radii = compute_max_radii(centers)
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii


def compute_max_radii(centers):
    """
    Compute the maximum possible radii for each circle position
    such that they don't overlap and stay within the unit square.

    Uses iterative proportional scaling until convergence so that
    all pairwise and wall constraints are fully respected.

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates

    Returns:
        np.array of shape (n) with radius of each circle
    """
    n = centers.shape[0]
    radii = np.ones(n)

    # Limit by distance to square borders
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(x, y, 1 - x, 1 - y)

    # Precompute pairwise distances
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            dists.append((i, j, d))

    # Iteratively scale radii until all constraints hold
    for _ in range(200):
        max_change = 0.0
        for i, j, d in dists:
            s = radii[i] + radii[j]
            if s > d:
                scale = d / s
                new_i = radii[i] * scale
                new_j = radii[j] * scale
                max_change = max(max_change, abs(radii[i] - new_i))
                radii[i], radii[j] = new_i, new_j
        # Re-apply wall caps (scaling only shrinks, so walls stay valid)
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
