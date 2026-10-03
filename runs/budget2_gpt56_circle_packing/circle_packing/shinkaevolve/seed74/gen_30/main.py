# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """
    Construct 26 circles by opening up a square contact lattice.

    The initial configuration consists of 25 radius-.1 circles in a
    5 by 5 lattice and a small circle in a four-circle interstice.
    It is already feasible, and is a much better starting point than a
    radial pattern for a subsequent constrained improvement.
    """
    n = 26
    grid = np.linspace(0.1, 0.9, 5)
    centers = np.array([(x, y) for y in grid for x in grid], dtype=float)
    centers = np.vstack((centers, [0.2, 0.2]))
    radii = np.full(n, 0.1, dtype=float)
    radii[-1] = np.sqrt(0.02) - 0.1

    # Optimize the actual packing variables, rather than assigning radii
    # greedily in a pair-dependent order.  The lattice/interstice start
    # supplies a feasible fallback when scipy is not installed.
    initial_centers = centers.copy()
    initial_radii = radii.copy()
    ii, jj = np.triu_indices(n, 1)

    def clearance(z):
        c = z[:2 * n].reshape(n, 2)
        r = z[2 * n:]
        pair_distance = np.sqrt(np.sum((c[ii] - c[jj]) ** 2, axis=1))
        return np.concatenate((
            c[:, 0] - r, 1.0 - c[:, 0] - r,
            c[:, 1] - r, 1.0 - c[:, 1] - r,
            pair_distance - r[ii] - r[jj],
        ))

    def clearance_jacobian(z):
        """Analytic derivatives of walls and pairwise distance clearances."""
        c = z[:2 * n].reshape(n, 2)
        rows = 4 * n + len(ii)
        jac = np.zeros((rows, 3 * n), dtype=float)
        disk = np.arange(n)
        radius_column = 2 * n + disk

        jac[disk, 2 * disk] = 1.0
        jac[disk, radius_column] = -1.0
        jac[n + disk, 2 * disk] = -1.0
        jac[n + disk, radius_column] = -1.0
        jac[2 * n + disk, 2 * disk + 1] = 1.0
        jac[2 * n + disk, radius_column] = -1.0
        jac[3 * n + disk, 2 * disk + 1] = -1.0
        jac[3 * n + disk, radius_column] = -1.0

        delta = c[ii] - c[jj]
        distance = np.sqrt(np.einsum("ij,ij->i", delta, delta))
        direction = delta / np.maximum(distance[:, None], 1.e-12)
        pair_rows = 4 * n + np.arange(len(ii))
        jac[pair_rows, 2 * ii] = direction[:, 0]
        jac[pair_rows, 2 * ii + 1] = direction[:, 1]
        jac[pair_rows, 2 * jj] = -direction[:, 0]
        jac[pair_rows, 2 * jj + 1] = -direction[:, 1]
        jac[pair_rows, 2 * n + ii] = -1.0
        jac[pair_rows, 2 * n + jj] = -1.0
        return jac

    try:
        from scipy.optimize import minimize

        z0 = np.concatenate((centers.ravel(), radii))

        # Square packings have several distinct useful contact graphs.  The
        # regular grid is excellent for one of them, while staggered layers
        # more readily open the alternating boundary gaps.  All starts below
        # have deliberately small, strictly feasible radii.
        layered = []
        for row, count in enumerate((5, 6, 5, 6, 4)):
            y = (row + 0.5) / 5.0
            for col in range(count):
                layered.append(((col + 0.5) / count, y))
        layered = np.asarray(layered, dtype=float)

        rng = np.random.default_rng(26031991)
        starts = [z0, np.concatenate((layered.ravel(), np.full(n, 0.025)))]

        # Moving the interstitial disk to a different lattice cell changes
        # which four large disks compete for its local clearance.  The slight
        # deterministic displacement breaks the otherwise equivalent square
        # symmetries and exposes asymmetric boundary-contact continuations.
        interstices = (
            (0.2, 0.4), (0.4, 0.2), (0.4, 0.4),
            (0.8, 0.6), (0.6, 0.8),
        )
        local_indices = np.arange(n - 1)
        for k, point in enumerate(interstices):
            trial = initial_centers.copy()
            trial[-1] = point
            amplitude = 0.0035
            trial[:-1, 0] += amplitude * np.sin(1.37 * local_indices + 0.71 * k)
            trial[:-1, 1] += amplitude * np.cos(1.91 * local_indices + 0.43 * k)
            trial = np.clip(trial, 0.035, 0.965)
            starts.append(np.concatenate((trial.ravel(), np.full(n, 0.025))))

        for amplitude in (0.006, 0.012, 0.018, 0.024):
            trial = initial_centers + rng.uniform(-amplitude, amplitude,
                                                  initial_centers.shape)
            trial = np.clip(trial, 0.055, 0.945)
            starts.append(np.concatenate((trial.ravel(), np.full(n, 0.025))))

        best = z0
        best_value = np.sum(radii)
        optimize_options = {"maxiter": 900, "ftol": 3.e-12, "disp": False}
        bounds = [(0.0, 1.0)] * (2 * n) + [(1.e-6, 0.5)] * n
        for start in starts:
            result = minimize(
                lambda z: -np.sum(z[2 * n:]),
                start,
                method="SLSQP",
                bounds=bounds,
                constraints={
                    "type": "ineq",
                    "fun": clearance,
                    "jac": clearance_jacobian,
                },
                options=optimize_options,
            )
            if result.x is not None and np.min(clearance(result.x)) >= -2.e-6:
                value = np.sum(result.x[2 * n:])
                if value > best_value:
                    best, best_value = result.x, value

        # A final pass from the winning topology resolves contacts that were
        # only approximately active in its first optimization pass.
        if best_value > np.sum(radii):
            polished = minimize(
                lambda z: -np.sum(z[2 * n:]),
                best,
                method="SLSQP",
                bounds=bounds,
                constraints={
                    "type": "ineq",
                    "fun": clearance,
                    "jac": clearance_jacobian,
                },
                options={"maxiter": 1600, "ftol": 1.e-13, "disp": False},
            )
            if (polished.x is not None and
                    np.min(clearance(polished.x)) >= -2.e-6 and
                    np.sum(polished.x[2 * n:]) > best_value):
                best, best_value = polished.x, np.sum(polished.x[2 * n:])

            candidate_centers = best[:2 * n].reshape(n, 2)
            candidate_radii = best[2 * n:]
            # SLSQP can leave residuals at about its feasibility tolerance.
            # A common scaling preserves every contact inequality exactly.
            border = np.minimum.reduce((
                candidate_centers[:, 0], candidate_centers[:, 1],
                1.0 - candidate_centers[:, 0],
                1.0 - candidate_centers[:, 1],
            ))
            distances = np.sqrt(
                np.sum((candidate_centers[ii] - candidate_centers[jj]) ** 2,
                       axis=1)
            )
            scale = min(
                1.0,
                np.min(border / candidate_radii),
                np.min(distances / (candidate_radii[ii] + candidate_radii[jj])),
            )
            centers = candidate_centers
            radii = candidate_radii * (0.999999 * scale)
    except ImportError:
        centers, radii = initial_centers, initial_radii

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