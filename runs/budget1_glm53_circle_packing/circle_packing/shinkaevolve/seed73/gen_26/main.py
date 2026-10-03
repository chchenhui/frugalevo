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

    # Hexagonal-lattice layout: rows of [5, 4, 5, 4, 5, 3] circles.
    # Neighboring circles in a row touch (dx = 2r) and rows are offset
    # by r horizontally and dy = sqrt(3)*r vertically.
    sqrt3 = np.sqrt(3.0)
    # Horizontal constraint (5-circle row): 2r + 4*dx = 10r <= 1
    # Vertical constraint (6 rows): 2r + 5*dy = r*(2 + 5*sqrt(3)) <= 1
    r0 = min(1.0 / 10.0, 1.0 / (2.0 + 5.0 * sqrt3))
    dx = 2.0 * r0
    dy = sqrt3 * r0

    # Several candidate layouts: vary which rows get extra circles
    # (top row has leftover space in the 5,4,5,4,5,3 arrangement).
    candidates = []
    for counts in ([5, 4, 5, 4, 5, 3], [5, 4, 5, 4, 4, 4], [4, 5, 4, 5, 4, 4]):
        pts = []
        for row, c in enumerate(counts):
            y = r0 + row * dy
            span = 2.0 * r0 + (c - 1) * dx
            start = (1.0 - span) / 2.0 + r0
            for k in range(c):
                pts.append([start + k * dx, y])
        candidates.append(np.array(pts))

    best = None
    for cand in candidates:
        centers, radii, s = refine_layout(cand, r0)
        if best is None or s > best[2]:
            best = (centers, radii, s)

    centers, radii, sum_radii = best

    # Greedy void-filling pass: find the largest empty space, insert a
    # circle there, re-refine with all 27 circles, then drop the
    # smallest to restore n=26. Keep only if total radius improves.
    improved = True
    while improved:
        improved = False
        v, vr = find_largest_void(centers, radii)
        if vr > 0.04:
            cand = np.vstack([centers, v.reshape(1, 2)])
            c2, r2, s2 = refine_layout(cand, float(np.max(radii)), iters=40)
            # Drop smallest circle to restore n, then re-grow
            k = np.argmin(r2)
            keep = [i for i in range(c2.shape[0]) if i != k]
            c3 = c2[keep]
            c3, r3, s3 = refine_layout(c3, float(np.max(r2)), iters=40)
            if s3 > sum_radii + 1e-6 and c3.shape[0] == 26:
                centers, radii, sum_radii = c3, r3, s3
                improved = True
        else:
            break

    return centers, radii, sum_radii


def find_largest_void(centers, radii):
    """
    Grid-sample the unit square to find the point maximizing the
    distance to all circle boundaries and the walls.
    Returns (point, void_radius).
    """
    g = 40
    xs = (np.arange(g) + 0.5) / g
    pts = np.array([[x, y] for x in xs for y in xs])
    d = np.sqrt(((pts[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))
    clear_circ = np.min(d - radii[None, :], axis=1)
    clear_wall = np.minimum(np.minimum(pts[:, 0], 1.0 - pts[:, 0]),
                            np.minimum(pts[:, 1], 1.0 - pts[:, 1]))
    clear = np.minimum(clear_circ, clear_wall)
    i = int(np.argmax(clear))
    return pts[i], float(clear[i])


def refine_layout(centers, r0, iters=60):
    """
    Deterministically nudge centers toward free space and re-grow radii.
    Each circle is pushed away from its closest neighbor overlap and
    toward the nearest walls; only configurations with a higher total
    radius are kept. Returns the best (centers, radii, sum) found.
    """
    centers = centers.copy()
    radii = grow_radii(centers, r0)
    best_centers = centers.copy()
    best_radii = radii.copy()
    best_sum = float(np.sum(radii))

    for t in range(iters):
        step = 0.02 * (1.0 - t / iters)
        d = np.sqrt(((centers[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))
        np.fill_diagonal(d, np.inf)
        new_centers = centers.copy()
        for i in range(centers.shape[0]):
            # Push away from nearest neighbor (direction of max clearance)
            j = np.argmin(d[i])
            vec = centers[i] - centers[j]
            norm = np.linalg.norm(vec)
            if norm > 1e-9:
                new_centers[i] += step * vec / norm
            # Pull toward nearest wall to use boundary space
            walls = np.array([[0.0, centers[i, 1]], [1.0, centers[i, 1]],
                              [centers[i, 0], 0.0], [centers[i, 0], 1.0]])
            w = np.argmin(np.linalg.norm(walls - centers[i], axis=1))
            towall = walls[w] - centers[i]
            new_centers[i] += 0.5 * step * towall
        new_centers = np.clip(new_centers, 0.005, 0.995)
        # Keep minimum separation to avoid degenerate collapse
        new_radii = grow_radii(new_centers, r0)
        new_sum = float(np.sum(new_radii))
        if new_sum > best_sum:
            best_sum = new_sum
            best_centers = new_centers.copy()
            best_radii = new_radii.copy()
            centers = new_centers
        else:
            # Accept small moves anyway early on (exploration)
            if t < iters // 2:
                centers = new_centers
    return best_centers, best_radii, best_sum


def grow_radii(centers, r_init, iters=1000):
    """
    Grow radii from a small feasible starting point using a
    Gauss-Seidel style growth iteration. Each circle's radius can
    only grow up to the minimum of:
      - its distance to the square walls
      - (distance to any other circle - that circle's current radius)
    Since radii only grow when locally valid, the result is feasible.
    """
    n = centers.shape[0]
    # Wall distance caps
    b = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                   np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    # Pairwise distance matrix
    d = np.sqrt(((centers[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(d, np.inf)

    radii = np.full(n, 0.25 * r_init)
    for _ in range(iters):
        grew = False
        for i in range(n):
            # Max radius given current radii of all others
            cand = min(b[i], np.min(d[i, :] - radii))
            if cand > radii[i] + 1e-12:
                radii[i] = cand
                grew = True
        if not grew:
            break
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