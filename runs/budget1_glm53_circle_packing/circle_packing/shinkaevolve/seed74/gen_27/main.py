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

    # Screen several explicit hex-style row layouts (and horizontal scales)
    # with the exact LP radius solver, keeping the best-scoring arrangement.
    candidates = [
        [5, 4, 5, 4, 5, 3],
        [6, 4, 5, 4, 6],
        [5, 5, 4, 5, 5],
        [6, 5, 4, 5, 6],
        [4, 4, 5, 4, 5, 4],
        [3, 4, 5, 4, 5, 5],
        [6, 5, 5, 5, 5],
        [7, 4, 4, 4, 7],
    ]
    scales = [1.0, 0.96, 1.02]

    best_centers = None
    best_radii = None
    best_sum = -1.0
    for rows in candidates:
        if sum(rows) != n:
            continue
        for sc in scales:
            centers_c = _build_layout(rows, sc)
            radii_c = _lp_radii(centers_c)
            s = float(np.sum(radii_c))
            if s > best_sum:
                best_sum = s
                best_centers = centers_c
                best_radii = radii_c

    centers = best_centers
    radii = best_radii

    # Calculate the sum of radii
    sum_radii = np.sum(radii)

    return centers, radii, sum_radii


def _build_layout(rows, scale=1.0, vpad=0.5):
    """Build staggered hex-like centers for a given row-count pattern."""
    n_rows = len(rows)
    dy = 1.0 / (n_rows + 1)
    centers = []
    for r, count in enumerate(rows):
        y = (r + 1) * dy
        dx = 1.0 / (count + 1)
        offset = 0.0
        if r % 2 == 1:
            offset = 0.5 * dx
        for c in range(count):
            # scale spreads rows slightly wider/narrower than the square
            x = 0.5 + ((c + 1) * dx + offset - 0.5) * scale
            x = min(max(x, 0.02), 0.98)
            centers.append([x, y])
    return np.array(centers)


def _lp_radii(centers):
    """Maximize sum(r) s.t. r_i + r_j <= d_ij, r_i <= border distance.

    Exact linear program; falls back to the iterative shrink solver.
    """
    n = centers.shape[0]
    radii = compute_max_radii(centers)
    try:
        from scipy.optimize import linprog

        A_ub = []
        b_ub = []
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                row = np.zeros(n)
                row[i] = 1.0
                row[j] = 1.0
                A_ub.append(row)
                b_ub.append(d)
        for i in range(n):
            x, y = centers[i]
            row = np.zeros(n)
            row[i] = 1.0
            A_ub.append(row)
            b_ub.append(min(x, y, 1 - x, 1 - y))
        res = linprog(c=-np.ones(n), A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                      bounds=[(0, None)] * n, method="highs")
        if res.success:
            radii = res.x
    except Exception:
        pass
    return radii


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

    # Then, limit by distance to other circles
    # Each pair of circles with centers at distance d can have
    # sum of radii at most d to avoid overlap
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))

            # If current radii would cause overlap
            if radii[i] + radii[j] > dist:
                # Scale both radii proportionally
                scale = dist / (radii[i] + radii[j])
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