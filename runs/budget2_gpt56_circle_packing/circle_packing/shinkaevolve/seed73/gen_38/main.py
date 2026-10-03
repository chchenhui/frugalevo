# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """
    Numerically refine several staggered, boundary-aware 26-circle layouts.

    The optimization variables are the 52 center coordinates and the 26
    individual radii.  Thus radii are not artificially tied together: circles
    adjacent to corners or narrow boundary gaps may become small while useful
    interior circles remain large.
    """
    from scipy.optimize import minimize

    n = 26
    ii, jj = np.triu_indices(n, 1)

    # Each tuple is a distinct five-row contact-graph hypothesis.  They all
    # contain 26 circles, but differ in where the denser six-circle layers
    # meet the square boundary.
    row_layouts = (
        (5, 6, 5, 6, 4),
        (4, 6, 6, 5, 5),
        (5, 5, 6, 6, 4),
        (6, 4, 6, 5, 5),
        (5, 5, 6, 4, 6),
        (4, 6, 5, 6, 5),
    )

    def row_start(row_sizes, transpose=False):
        points = []
        for row, count in enumerate(row_sizes):
            y = 0.10 + 0.20 * row
            xs = np.linspace(1.0 / (count + 1), count / (count + 1), count)
            points.extend((x, y) for x in xs)
        points = np.asarray(points, dtype=float)
        if transpose:
            points = points[:, ::-1]
        return points

    def constraints(z):
        c = z[:2 * n].reshape(n, 2)
        r = z[2 * n:]
        delta = c[ii] - c[jj]
        distances_squared = np.einsum("ij,ij->i", delta, delta)
        # Four wall inequalities per circle followed by all pair inequalities.
        return np.concatenate((
            c[:, 0] - r, 1.0 - c[:, 0] - r,
            c[:, 1] - r, 1.0 - c[:, 1] - r,
            distances_squared - (r[ii] + r[jj]) ** 2
        ))

    def constraints_jacobian(z):
        """Exact derivatives of wall and squared pair-contact constraints."""
        c = z[:2 * n].reshape(n, 2)
        r = z[2 * n:]
        jac = np.zeros((4 * n + len(ii), 3 * n))
        k = np.arange(n)

        jac[k, 2 * k] = 1.0
        jac[k, 2 * n + k] = -1.0
        jac[n + k, 2 * k] = -1.0
        jac[n + k, 2 * n + k] = -1.0
        jac[2 * n + k, 2 * k + 1] = 1.0
        jac[2 * n + k, 2 * n + k] = -1.0
        jac[3 * n + k, 2 * k + 1] = -1.0
        jac[3 * n + k, 2 * n + k] = -1.0

        rows = 4 * n + np.arange(len(ii))
        delta = c[ii] - c[jj]
        sums = r[ii] + r[jj]
        jac[rows, 2 * ii] = 2.0 * delta[:, 0]
        jac[rows, 2 * ii + 1] = 2.0 * delta[:, 1]
        jac[rows, 2 * jj] = -2.0 * delta[:, 0]
        jac[rows, 2 * jj + 1] = -2.0 * delta[:, 1]
        jac[rows, 2 * n + ii] = -2.0 * sums
        jac[rows, 2 * n + jj] = -2.0 * sums
        return jac

    objective_jacobian = np.zeros(3 * n)
    objective_jacobian[2 * n:] = -1.0

    starts = []
    # Transposition doubles the topology portfolio without increasing the
    # number of optimization runs beyond the intended twelve-start budget.
    for row_sizes in row_layouts:
        for transpose in (False, True):
            centers = row_start(row_sizes, transpose)
            # A boundary-aware radius profile is feasible initially and lets
            # the solver immediately explore unequal outer-layer circles.
            edge = np.minimum.reduce((
                centers[:, 0], centers[:, 1],
                1.0 - centers[:, 0], 1.0 - centers[:, 1]
            ))
            profile = 1.0 + 0.20 * (0.25 - edge) / 0.25
            radii = 0.035 * np.clip(profile, 0.78, 1.22)
            starts.append((centers, radii))

    bounds = [(0.0, 1.0)] * (2 * n) + [(1.0e-7, 0.5)] * n
    best_centers = starts[0][0]
    best_radii = starts[0][1]
    best_value = np.sum(best_radii)

    for initial_centers, initial_radii in starts:
        initial = np.concatenate((initial_centers.ravel(), initial_radii))
        result = minimize(
            lambda z: -np.sum(z[2 * n:]),
            initial,
            jac=lambda z: objective_jacobian,
            method="SLSQP",
            bounds=bounds,
            constraints={
                "type": "ineq", "fun": constraints, "jac": constraints_jacobian
            },
            options={"maxiter": 900, "ftol": 2.0e-11, "disp": False},
        )
        if result.x is None:
            continue
        candidate_centers = result.x[:2 * n].reshape(n, 2)
        candidate_radii = result.x[2 * n:]
        # SLSQP can report a useful point even if it ends on its iteration
        # limit, so assess candidates after the same conservative validation.
        candidate_radii = np.maximum(candidate_radii, 0.0)
        wall = np.minimum.reduce((
            candidate_centers[:, 0], candidate_centers[:, 1],
            1.0 - candidate_centers[:, 0], 1.0 - candidate_centers[:, 1]
        ))
        scale = np.min(wall / np.maximum(candidate_radii, 1.0e-15))
        delta = candidate_centers[ii] - candidate_centers[jj]
        distance = np.sqrt(np.einsum("ij,ij->i", delta, delta))
        pair_sum = candidate_radii[ii] + candidate_radii[jj]
        scale = min(scale, np.min(distance / np.maximum(pair_sum, 1.0e-15)))
        candidate_radii *= min(1.0, scale) * (1.0 - 2.0e-10)
        value = np.sum(candidate_radii)
        if value > best_value:
            best_centers, best_radii, best_value = candidate_centers, candidate_radii, value

    return best_centers, best_radii, float(best_value)


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