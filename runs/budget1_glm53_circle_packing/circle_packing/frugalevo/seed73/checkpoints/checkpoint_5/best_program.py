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

    """Approach: staggered hexagonal pattern with row counts 6,5,6,5,4
    (total 26). Horizontal spacing 0.16, rows offset by 0.08, vertical
    pitch 0.2*sin(60°) ≈ 0.1732 so rows fit in [0.1, 0.9]. This starts
    near sum 2.6 (all radii ~0.1) — a much better basin than the 5x5
    grid's 2.5 — and the SLSQP refinement then pushes beyond it."""
    counts = [6, 5, 6, 5, 4]
    dx = 0.16
    dy = 0.2 * np.sqrt(3) / 2  # ≈ 0.1732
    k = 0
    for row, cnt in enumerate(counts):
        y = 0.1 + row * dy
        row_width = (cnt - 1) * dx
        x0 = 0.5 - row_width / 2  # center each row horizontally
        for col in range(cnt):
            centers[k] = [x0 + dx * col, y]
            k += 1

    # Compute maximum valid radii for this configuration
    radii = compute_max_radii(centers)

    # Locally refine (breaks the grid's symmetry so some circles can
    # exceed 0.1 while others shrink). Radii are always recomputed from
    # the refined centers via compute_max_radii, so validity holds.
    for _ in range(3):
        centers_new, radii_new = _refine(centers, radii)
        if np.sum(radii_new) <= np.sum(radii) + 1e-10:
            break
        centers, radii = centers_new, radii_new

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
    radii = np.ones(n)

    # Limit by distance to square borders
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(x, y, 1 - x, 1 - y)

    # Iteratively enforce pairwise non-overlap until convergence so that
    # shrinkage from one pair propagates correctly to all other pairs
    for _ in range(200):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                if radii[i] + radii[j] > dist:
                    # Shrink the larger radius first (keeps small circles
                    # small and preserves the intended size hierarchy)
                    excess = radii[i] + radii[j] - dist
                    if radii[i] >= radii[j]:
                        radii[i] = max(radii[i] - excess, 1e-12)
                    else:
                        radii[j] = max(radii[j] - excess, 1e-12)
                    changed = True
        if not changed:
            break

    return radii


def _refine(centers, radii):
    """Locally optimize centers and radii with SLSQP starting from a
    feasible configuration, maximizing the sum of radii subject to wall
    constraints (r <= min(x, y, 1-x, 1-y)) and pairwise non-overlap
    (r_i + r_j <= distance). The optimizer's radii are discarded; radii
    are recomputed from the refined centers via compute_max_radii so the
    returned configuration is always valid. Falls back to the input if
    scipy is unavailable or no strict improvement is found."""
    try:
        from scipy.optimize import minimize
    except ImportError:
        return centers, radii

    n = len(centers)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    def obj(z):
        return -np.sum(z[2 * n:])

    cons = []
    for i in range(n):
        cons.append({"type": "ineq",
                     "fun": lambda z, i=i: (np.array([z[2 * i], z[2 * i + 1],
                                                      1 - z[2 * i], 1 - z[2 * i + 1]])
                                            - z[2 * n + i])})
    for (i, j) in pairs:
        cons.append({"type": "ineq",
                     "fun": lambda z, i=i, j=j:
                     np.sqrt(np.sum((z[2 * i:2 * i + 2] - z[2 * j:2 * j + 2]) ** 2))
                     - z[2 * n + i] - z[2 * n + j]})

    z0 = np.concatenate([centers.ravel(), radii])
    try:
        res = minimize(obj, z0, method="SLSQP", constraints=cons,
                       options={"maxiter": 400, "ftol": 1e-10})
    except Exception:
        return centers, radii

    c2 = np.asarray(res.x[:2 * n]).reshape(n, 2)
    if np.all(c2 > 1e-9) and np.all(c2 < 1 - 1e-9):
        r2 = compute_max_radii(c2)
        if np.sum(r2) > np.sum(radii) + 1e-9:
            return c2, r2
    return centers, radii


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
