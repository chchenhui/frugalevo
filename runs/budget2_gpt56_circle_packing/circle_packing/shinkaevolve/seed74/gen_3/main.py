# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """
    Construct 26 disks by refining a staggered, nearly hexagonal seed.

    The final radius assignment is solved as a linear program for the
    resulting centers.  This avoids the order dependence of repeatedly
    shrinking pairs of disks.
    """
    from scipy.optimize import linprog, minimize

    n = 26

    # Five staggered layers give 26 well-separated feasible initial sites.
    # The small deterministic displacement breaks artificial lattice
    # symmetries, which lets boundary disks acquire different radii.
    counts = (5, 6, 5, 6, 4)
    seed = []
    for row, count in enumerate(counts):
        y = (row + 0.5) / len(counts)
        for col in range(count):
            x = (col + 0.5) / count
            k = len(seed)
            seed.append([
                x + 0.005 * np.sin(7.0 * k + 1.3 * row),
                y + 0.005 * np.cos(5.0 * k + 0.7 * row),
            ])
    seed = np.asarray(seed)
    z0 = np.r_[seed.ravel(), np.full(n, 0.04)]

    ii, jj = np.triu_indices(n, 1)

    def inequalities(z):
        c = z[:2 * n].reshape(n, 2)
        r = z[2 * n:]
        edge = np.minimum(np.minimum(c[:, 0], c[:, 1]),
                          np.minimum(1.0 - c[:, 0], 1.0 - c[:, 1]))
        d2 = np.sum((c[ii] - c[jj]) ** 2, axis=1)
        return np.r_[r, edge - r, d2 - (r[ii] + r[jj]) ** 2]

    # Squared-distance constraints are smooth at contacts.  The initial
    # point is strictly feasible, so SLSQP can preserve feasibility while
    # moving centers and allowing the radii to become non-uniform.
    result = minimize(
        lambda z: -np.sum(z[2 * n:]),
        z0,
        method="SLSQP",
        constraints={"type": "ineq", "fun": inequalities},
        options={"maxiter": 1400, "ftol": 1.0e-11, "disp": False},
    )
    z = result.x if result.success and np.min(inequalities(result.x)) >= -1e-7 else z0
    centers = z[:2 * n].reshape(n, 2)

    # For fixed centers, maximizing sum(r_i) is an exact linear program:
    # r_i + r_j <= distance(i,j), with the square edges as upper bounds.
    distances = np.sqrt(np.sum((centers[ii] - centers[jj]) ** 2, axis=1))
    pair_matrix = np.zeros((len(ii), n))
    pair_matrix[np.arange(len(ii)), ii] = 1.0
    pair_matrix[np.arange(len(ii)), jj] = 1.0
    edge = np.minimum(np.minimum(centers[:, 0], centers[:, 1]),
                      np.minimum(1.0 - centers[:, 0], 1.0 - centers[:, 1]))
    safety = 1.0 - 1.0e-10
    lp = linprog(
        -np.ones(n),
        A_ub=pair_matrix,
        b_ub=safety * distances,
        bounds=[(0.0, safety * value) for value in edge],
        method="highs",
    )

    if lp.success:
        radii = lp.x
    else:
        radii = compute_max_radii(centers)

    return centers, radii, float(np.sum(radii))


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