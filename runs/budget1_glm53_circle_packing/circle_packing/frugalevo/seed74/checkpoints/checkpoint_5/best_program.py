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
    """
    Hexagonal-row constructor with candidate selection and jitter refinement.

    Since compute_max_radii already yields feasible maximal radii, the old
    "relaxation" only shuffled circles and degraded the sum. Instead we:
      1. Generate several staggered-row hexagonal-like layouts (different row
         counts and spacings), compute the achievable sum for each, and keep
         the best.
      2. Hill-climb with small random jitters of individual circles,
         accepting only moves that increase the total sum of maximal radii.
    """
    n = 26
    rng = np.random.RandomState(0)

    def build(counts, spacing, stagger=True):
        c = np.zeros((n, 2))
        idx = 0
        y0 = spacing / 2
        for row, k in enumerate(counts):
            y = y0 + row * spacing
            total_w = (k - 1) * spacing
            x_start = (1.0 - total_w) / 2.0
            if stagger and row % 2 == 1:
                x_start += spacing / 2
            for j in range(k):
                c[idx] = [x_start + j * spacing, y]
                idx += 1
        return np.clip(c, 0.005, 0.995)

    # Candidate row patterns summing to 26 over 5 and 6 rows.
    # 6-row layouts better approximate hexagonal packing in the interior.
    patterns = [
        [4, 5, 6, 6, 5],
        [5, 5, 6, 5, 5],
        [4, 6, 6, 6, 4],
        [5, 6, 5, 6, 4],
        [6, 5, 5, 5, 5],
        [4, 5, 5, 6, 6],
        [4, 4, 6, 6, 6],
        [5, 4, 6, 6, 5],
        [3, 5, 6, 6, 6],
        [4, 5, 6, 5, 6],
        [4, 4, 5, 5, 4, 4],
        [4, 4, 5, 5, 4, 4][:5] + [4],
        [3, 4, 5, 5, 5, 4],
        [4, 4, 5, 4, 5, 4],
        [5, 4, 4, 4, 4, 5],
        [3, 5, 5, 5, 5, 3],
    ]
    best_centers, best_sum = None, -1.0
    for counts in patterns:
        if sum(counts) != n:
            continue
        for spacing in (0.16, 0.165, 0.17, 0.175, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23):
            for stagger in (True, False):
                c = build(counts, spacing, stagger)
                s = np.sum(compute_max_radii(c))
                if s > best_sum:
                    best_sum, best_centers = s, c.copy()

    # Hill-climbing jitter refinement with multiple restarts. Each restart
    # perturbs the best layout and re-optimizes; the global best is kept.
    def hill_climb(centers, iters, step0):
        radii = compute_max_radii(centers)
        best_sum = np.sum(radii)
        step = step0
        for it in range(iters):
            if it % (iters // 6) == (iters // 6) - 1:
                step *= 0.6
            if rng.rand() < 0.75:
                i = rng.randint(n)
                old = centers[i].copy()
                centers[i] = np.clip(
                    centers[i] + rng.uniform(-step, step, 2), 0.005, 0.995
                )
                new_radii = compute_max_radii(centers)
                new_sum = np.sum(new_radii)
                if new_sum > best_sum:
                    best_sum = new_sum
                    radii = new_radii
                else:
                    centers[i] = old
            else:
                i, j = rng.randint(n, size=2)
                if i == j:
                    continue
                old_i, old_j = centers[i].copy(), centers[j].copy()
                centers[i] = np.clip(
                    centers[i] + rng.uniform(-step, step, 2), 0.005, 0.995
                )
                centers[j] = np.clip(
                    centers[j] + rng.uniform(-step, step, 2), 0.005, 0.995
                )
                new_radii = compute_max_radii(centers)
                new_sum = np.sum(new_radii)
                if new_sum > best_sum:
                    best_sum = new_sum
                    radii = new_radii
                else:
                    centers[i], centers[j] = old_i, old_j
        return centers, radii, best_sum

    centers, radii, best_sum = hill_climb(best_centers, 200000, 0.012)

    # Restarts: perturb best and re-climb with fewer iterations, keep global best
    for _ in range(5):
        c2 = np.clip(best_centers + rng.uniform(-0.02, 0.02, best_centers.shape),
                     0.005, 0.995)
        c2, r2, s2 = hill_climb(c2, 80000, 0.008)
        if s2 > best_sum:
            best_sum, best_centers, radii = s2, c2.copy(), r2

    # Final fine-tuning with tiny steps
    centers, radii, best_sum = hill_climb(best_centers, 60000, 0.003)

    return centers, radii, best_sum


def compute_max_radii(centers):
    """
    Compute the maximum possible radii for each circle position
    such that they don't overlap and stay within the unit square.

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates

    Returns:
        np.array of shape (n) with radius of each circle
    """
    """
    Vectorized maximal-radii computation.

    Border distances give an upper bound; then pairwise proportional
    scaling is iterated to a fixed point so the radii are (approximately)
    maximal subject to non-overlap. Iterating strictly improves on a
    single scaling pass, increasing the achievable sum of radii.
    """
    n = centers.shape[0]
    # Distance to square borders
    radii = np.minimum(np.minimum(centers[:, 0], 1 - centers[:, 0]),
                       np.minimum(centers[:, 1], 1 - centers[:, 1]))

    # Pairwise distance matrix (excluding self-distance)
    diff = centers[:, None, :] - centers[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)

    # Iterate proportional scaling until convergence (fixed point)
    for _ in range(40):
        s = radii[:, None] + radii[None, :]
        scale = np.where(s > dist, dist / np.maximum(s, 1e-12), 1.0)
        new_radii = radii * np.min(scale, axis=1)
        if np.max(np.abs(new_radii - radii)) < 1e-11:
            radii = new_radii
            break
        radii = new_radii

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
