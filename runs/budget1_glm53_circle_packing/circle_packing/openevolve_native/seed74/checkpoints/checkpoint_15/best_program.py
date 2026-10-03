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
    # Hexagonal lattice packing: rows of 5,4,5,4,5,3 circles.
    # Horizontal spacing 2r, vertical spacing sqrt(3)*r, with
    # r = 1/(2 + 5*sqrt(3)) so 6 rows fit exactly in the unit square.
    n = 26
    r0 = 1.0 / (2.0 + 5.0 * np.sqrt(3.0))
    dx = 2.0 * r0
    dy = np.sqrt(3.0) * r0

    def build_rows(row_counts, x_offset_factor=0.0):
        """Hexagonal lattice seed: rows with given counts, optionally
        shifting alternate rows horizontally by x_offset_factor*dx."""
        pts = []
        for row, count in enumerate(row_counts):
            y = r0 + row * dy
            shift = x_offset_factor * dx if row % 2 == 1 else 0.0
            start_x = r0 + (5 - count) * r0 + shift
            for k in range(count):
                pts.append([start_x + k * dx, y])
        return np.array(pts[:n])

    def finish_top_row(pts):
        """Re-seed final 3 circles as corners + center of top edge."""
        pts = pts.copy()
        pts[23] = [r0, 1.0 - r0]
        pts[24] = [0.5, 1.0 - r0]
        pts[25] = [1.0 - r0, 1.0 - r0]
        return pts

    # Several structurally different seeds; quick-refine each, keep the best.
    seeds = [
        finish_top_row(build_rows([5, 4, 5, 4, 5, 3])),
        build_rows([4, 5, 4, 5, 4, 4]),
        build_rows([5, 4, 5, 4, 5, 3], x_offset_factor=0.5),
        build_rows([4, 5, 4, 5, 4, 4], x_offset_factor=0.5),
    ]

    quick_steps = (0.01, 0.003)
    best_centers = None
    best_val = -1.0
    for seed in seeds:
        cand = refine_centers(seed, steps=quick_steps)
        val = np.sum(compute_max_radii(cand))
        if val > best_val:
            best_val = val
            best_centers = cand

    # Fully refine the most promising seed
    centers = refine_centers(best_centers)

    # Compute maximum valid radii for this configuration
    radii = compute_max_radii(centers)

    # Calculate the sum of radii
    sum_radii = np.sum(radii)

    return centers, radii, sum_radii


def refine_centers(centers, steps=(0.03, 0.01, 0.004, 0.0015, 0.0005, 0.0002)):
    """
    Deterministic coordinate descent on circle centers.

    For each center, tries +/- moves along x, y, and both diagonals at
    the current step size, keeping a move only if the LP-optimal sum of
    radii improves. Diagonal moves help slide circles along hexagonal
    (60-degree) constraint directions. Uses progressively smaller step
    sizes; centers stay strictly inside the square (margin 1e-6) so wall
    bounds remain valid.
    """
    n = centers.shape[0]
    best = centers.copy()
    best_val = np.sum(compute_max_radii(best))
    for step in steps:
        improved = True
        while improved:
            improved = False
            for i in range(n):
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1),
                               (1, 1), (1, -1), (-1, 1), (-1, -1)):
                    cand = best.copy()
                    nx = cand[i, 0] + dx * step
                    ny = cand[i, 1] + dy * step
                    if nx < 1e-6 or nx > 1.0 - 1e-6:
                        continue
                    if ny < 1e-6 or ny > 1.0 - 1e-6:
                        continue
                    cand[i, 0] = nx
                    cand[i, 1] = ny
                    val = np.sum(compute_max_radii(cand))
                    if val > best_val + 1e-9:
                        best = cand
                        best_val = val
                        improved = True
    return best


def compute_max_radii(centers):
    """
    Compute radii maximizing the sum of radii for fixed centers.
    Solves the LP: max sum(r) s.t. r_i + r_j <= dist(i,j) for all pairs,
    and 0 <= r_i <= min distance to the square walls.
    Uses scipy linprog (exact); falls back to iterative scaling.
    """
    n = centers.shape[0]
    # Wall distance upper bounds
    bounds_hi = np.minimum(
        np.minimum(centers[:, 0], 1 - centers[:, 0]),
        np.minimum(centers[:, 1], 1 - centers[:, 1]),
    )

    try:
        from scipy.optimize import linprog

        # Pairwise constraints: r_i + r_j <= d_ij
        A = []
        b = []
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                row = np.zeros(n)
                row[i] = 1.0
                row[j] = 1.0
                A.append(row)
                b.append(d)
        res = linprog(
            c=-np.ones(n),
            A_ub=np.array(A),
            b_ub=np.array(b),
            bounds=[(0, bounds_hi[i]) for i in range(n)],
            method="highs",
        )
        if res.success:
            return res.x
    except Exception:
        pass

    # Fallback: iterative proportional scaling (always valid)
    radii = bounds_hi.copy()
    for _ in range(200):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                if radii[i] + radii[j] > d:
                    scale = d / (radii[i] + radii[j])
                    radii[i] *= scale
                    radii[j] *= scale
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
