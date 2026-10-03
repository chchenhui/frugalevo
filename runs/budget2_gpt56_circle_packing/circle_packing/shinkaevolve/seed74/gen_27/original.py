# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """
    Jointly optimize centers and unequal radii from several staggered
    hexagonal seeds.  Unlike a greedy post-processing radius assignment,
    every radius is allowed to trade against every other radius.
    """
    from scipy.optimize import minimize

    n = 26
    pair_i, pair_j = np.triu_indices(n, 1)

    # Five staggered layers give 26 well-separated initial sites.  The
    # deliberate asymmetry (the six-circle middle row) is helpful near the
    # vertical boundaries, where a perfectly triangular lattice is cut off.
    row_counts = (5, 5, 6, 5, 5)
    base = []
    for row, count in enumerate(row_counts):
        y = 0.11 + 0.195 * row
        if count == 6:
            xs = np.linspace(0.10, 0.90, count)
        else:
            xs = np.linspace(0.13, 0.87, count)
        base.extend((x, y) for x in xs)
    base = np.asarray(base, dtype=float)

    def inequalities(z):
        c = z[:2 * n].reshape(n, 2)
        r = z[2 * n:]
        # Four wall inequalities for each circle.
        walls = np.column_stack((
            c[:, 0] - r, c[:, 1] - r,
            1.0 - c[:, 0] - r, 1.0 - c[:, 1] - r
        )).ravel()
        # Squared distances avoid square roots and are smooth away from
        # coincident centers.
        delta = c[pair_i] - c[pair_j]
        separation = np.einsum("ij,ij->i", delta, delta) - (r[pair_i] + r[pair_j]) ** 2
        return np.concatenate((walls, separation))

    bounds = [(0.0001, 0.9999)] * (2 * n) + [(0.000001, 0.5)] * n
    constraint = {"type": "ineq", "fun": inequalities}
    rng = np.random.default_rng(26031991)
    best = None

    # Small reproducible perturbations let the optimizer discover unequal,
    # boundary-adapted arrangements without an expensive global search.
    for trial in range(6):
        centers0 = base.copy()
        if trial:
            centers0 += rng.normal(0.0, 0.018, centers0.shape)
            centers0 = np.clip(centers0, 0.06, 0.94)
        z0 = np.concatenate((centers0.ravel(), np.full(n, 0.025)))
        result = minimize(
            lambda z: -np.sum(z[2 * n:]),
            z0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraint,
            options={"maxiter": 1400, "ftol": 1e-11, "disp": False},
        )
        if result.success and np.min(inequalities(result.x)) >= -2e-7:
            if best is None or result.x[2 * n:].sum() > best[2 * n:].sum():
                best = result.x

    # The unperturbed seed is feasible, so this fallback is only relevant if
    # an optimizer installation declines all numerical solves.
    if best is None:
        best = np.concatenate((base.ravel(), np.full(n, 0.025)))

    centers = best[:2 * n].reshape(n, 2)
    radii = best[2 * n:].copy()

    # SLSQP permits tiny feasibility errors.  One global scale factor is a
    # certificate: it simultaneously enforces every wall and pair constraint.
    wall_limit = np.minimum.reduce((
        centers[:, 0], centers[:, 1], 1.0 - centers[:, 0], 1.0 - centers[:, 1]
    ))
    scale = np.min(wall_limit / radii)
    distances = np.sqrt(np.sum((centers[pair_i] - centers[pair_j]) ** 2, axis=1))
    scale = min(scale, np.min(distances / (radii[pair_i] + radii[pair_j])))
    radii *= min(1.0, 0.999999 * scale)

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