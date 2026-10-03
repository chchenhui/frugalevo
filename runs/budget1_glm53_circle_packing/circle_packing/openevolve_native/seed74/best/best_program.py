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
    # Try several hexagonal row layouts (summing to 26 circles),
    # refine each with coordinate descent, and keep the best result.
    n = 26
    layouts = [
        [5, 4, 5, 4, 5, 3],
        [4, 5, 4, 5, 4, 4],
        [5, 4, 5, 5, 4, 3],
        [5, 5, 4, 5, 4, 3],
        [4, 5, 5, 4, 5, 3],
        [6, 4, 5, 4, 5, 2],
        # 7-row layouts: more, smaller circles utilize edges better
        [4, 4, 4, 4, 4, 3, 3],
        [3, 4, 4, 4, 4, 4, 3],
        [4, 3, 4, 4, 4, 4, 3],
        [4, 4, 3, 5, 3, 4, 3],
        [5, 3, 4, 4, 4, 3, 3],
    ]
    best_centers = None
    best_val = -1.0
    for row_counts in layouts:
        centers = build_rows(row_counts)
        centers = refine_centers(centers, steps=(0.02, 0.005, 0.001))
        val = np.sum(compute_max_radii(centers))
        if val > best_val:
            best_val = val
            best_centers = centers
    centers = best_centers

    # Fine refinement plus shake perturbations to escape local optima
    centers = refine_centers(centers)
    centers = shake_and_refine(centers)
    centers = shake_and_refine(centers, rounds=10, amp=0.006)
    centers = refine_centers(centers, steps=(0.0008, 0.0003, 0.0001))

    # Compute maximum valid radii for this configuration
    radii = compute_max_radii(centers)

    # Calculate the sum of radii
    sum_radii = np.sum(radii)

    return centers, radii, sum_radii


def build_rows(row_counts):
    """Build hexagonal-lattice seed centers for the given per-row counts.
    Rows are spaced sqrt(3)*r vertically and 2r horizontally, with
    r chosen so all rows fit exactly in the unit square. Odd-count rows
    are centered; even-count rows are offset by one radius."""
    total = sum(row_counts)
    centers = np.zeros((total, 2))
    nrows = len(row_counts)
    r0 = 1.0 / (2.0 + (nrows - 1) * np.sqrt(3.0))
    dx = 2.0 * r0
    dy = np.sqrt(3.0) * r0
    maxc = max(row_counts)
    idx = 0
    for row, count in enumerate(row_counts):
        y = r0 + row * dy
        start_x = r0 + (maxc - count) * r0
        for k in range(count):
            centers[idx] = [start_x + k * dx, y]
            idx += 1
    return centers


def shake_and_refine(centers, rounds=12, amp=0.02):
    """Escape local optima: apply small deterministic collective jitters
    (sine-based, different per circle) to all centers, re-run coordinate
    descent, and keep the result only if the LP radii sum improves."""
    best = centers.copy()
    best_val = np.sum(compute_max_radii(best))
    n = best.shape[0]
    for t in range(rounds):
        cand = best.copy()
        phase = 0.7 * (t + 1)
        for i in range(n):
            cand[i, 0] += amp * np.sin(phase + 2.3 * i)
            cand[i, 1] += amp * np.cos(phase + 1.7 * i)
        np.clip(cand[:, 0], 1e-6, 1 - 1e-6, out=cand[:, 0])
        np.clip(cand[:, 1], 1e-6, 1 - 1e-6, out=cand[:, 1])
        cand = refine_centers(cand, steps=(0.01, 0.002, 0.0005, 0.0002))
        val = np.sum(compute_max_radii(cand))
        if val > best_val + 1e-9:
            best = cand
            best_val = val
    return best


def refine_centers(centers, steps=(0.03, 0.01, 0.004, 0.0015, 0.0005, 0.0002, 0.0001)):
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


_PAIRS = None  # cached upper-triangular pair indices


def compute_max_radii(centers):
    """
    Compute radii maximizing the sum of radii for fixed centers.
    Solves the LP: max sum(r) s.t. r_i + r_j <= dist(i,j) for all pairs,
    and 0 <= r_i <= min distance to the square walls.
    Constraints built vectorized with cached pair indices; falls back
    to iterative scaling if scipy is unavailable.
    """
    n = centers.shape[0]
    # Wall distance upper bounds
    bounds_hi = np.minimum(
        np.minimum(centers[:, 0], 1 - centers[:, 0]),
        np.minimum(centers[:, 1], 1 - centers[:, 1]),
    )

    try:
        from scipy.optimize import linprog

        # Pairwise constraints: r_i + r_j <= d_ij (vectorized, cached)
        global _PAIRS
        if _PAIRS is None or _PAIRS[0].shape[0] != n * (n - 1) // 2:
            _PAIRS = np.triu_indices(n, k=1)
        ii, jj = _PAIRS
        diffs = centers[ii] - centers[jj]
        b = np.sqrt((diffs ** 2).sum(axis=1))
        A = np.zeros((ii.shape[0], n))
        rows = np.arange(ii.shape[0])
        A[rows, ii] = 1.0
        A[rows, jj] = 1.0
        res = linprog(
            c=-np.ones(n),
            A_ub=A,
            b_ub=b,
            bounds=np.stack([np.zeros(n), bounds_hi], axis=1),
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
