# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """
    Construct an unequal-radius packing of 26 circles in the unit square.

    Several staggered lattice topologies are explored first.  The strongest
    candidates are then refined substantially further, since good unequal
    circle packings generally require resolving a delicate contact graph near
    the square boundary.
    """
    from scipy.optimize import minimize

    n = 26
    ii, jj = np.triu_indices(n, 1)
    pair_count = len(ii)
    variable_count = 3 * n

    def make_rows(row_sizes, alternate_shift=0.0, transpose=False,
                  jitter=None, vertical_phase=0.0):
        """Create a boundary-aware staggered triangular-lattice seed."""
        pts = []
        row_count = len(row_sizes)
        for row, count in enumerate(row_sizes):
            y = 0.095 + 0.81 * row / max(1, row_count - 1)
            y += vertical_phase * ((-1.0) ** row)

            # Centers of equally spaced row cells.  Alternate shifts make
            # nearby rows resemble a triangular lattice rather than a grid.
            spacing = 1.0 / count
            offset = alternate_shift * spacing * (1 if row % 2 else -1)
            xs = (np.arange(count, dtype=float) + 0.5) * spacing + offset

            # Keep deliberately shifted boundary centers inside the square.
            xs = np.clip(xs, 0.045, 0.955)
            pts.extend((x, y) for x in xs)

        pts = np.asarray(pts, dtype=float)
        if jitter is not None:
            pts = np.clip(pts + jitter, 0.025, 0.975)
        if transpose:
            pts = pts[:, ::-1]
        return pts

    def constraints(z):
        c = z[:2 * n].reshape(n, 2)
        r = z[2 * n:]
        d = c[ii] - c[jj]
        pair = np.einsum("ij,ij->i", d, d) - (r[ii] + r[jj]) ** 2
        return np.concatenate((
            c[:, 0] - r,
            1.0 - c[:, 0] - r,
            c[:, 1] - r,
            1.0 - c[:, 1] - r,
            pair
        ))

    def constraint_jacobian(z):
        c = z[:2 * n].reshape(n, 2)
        r = z[2 * n:]
        jac = np.zeros((4 * n + pair_count, variable_count), dtype=float)
        k = np.arange(n)

        jac[k, 2 * k] = 1.0
        jac[k, 2 * n + k] = -1.0

        jac[n + k, 2 * k] = -1.0
        jac[n + k, 2 * n + k] = -1.0

        jac[2 * n + k, 2 * k + 1] = 1.0
        jac[2 * n + k, 2 * n + k] = -1.0

        jac[3 * n + k, 2 * k + 1] = -1.0
        jac[3 * n + k, 2 * n + k] = -1.0

        rows = 4 * n + np.arange(pair_count)
        d = c[ii] - c[jj]
        rs = r[ii] + r[jj]

        jac[rows, 2 * ii] = 2.0 * d[:, 0]
        jac[rows, 2 * ii + 1] = 2.0 * d[:, 1]
        jac[rows, 2 * jj] = -2.0 * d[:, 0]
        jac[rows, 2 * jj + 1] = -2.0 * d[:, 1]
        jac[rows, 2 * n + ii] = -2.0 * rs
        jac[rows, 2 * n + jj] = -2.0 * rs
        return jac

    objective_gradient = np.zeros(variable_count)
    objective_gradient[2 * n:] = -1.0

    def objective(z):
        return -np.sum(z[2 * n:])

    def safely_scale(centers, radii):
        """Make an optimizer result strictly feasible with one global scale."""
        radii = np.maximum(np.asarray(radii, dtype=float), 0.0).copy()
        centers = np.asarray(centers, dtype=float).copy()

        wall = np.minimum.reduce((
            centers[:, 0], centers[:, 1],
            1.0 - centers[:, 0], 1.0 - centers[:, 1]
        ))
        scale = np.min(wall / np.maximum(radii, 1.0e-15))

        d = centers[ii] - centers[jj]
        dist = np.sqrt(np.einsum("ij,ij->i", d, d))
        sums = radii[ii] + radii[jj]
        scale = min(scale, np.min(dist / np.maximum(sums, 1.0e-15)))

        radii *= max(0.0, min(1.0, scale)) * (1.0 - 5.0e-11)
        return centers, radii

    bounds = [(0.0, 1.0)] * (2 * n) + [(1.0e-8, 0.5)] * n
    con = {
        "type": "ineq",
        "fun": constraints,
        "jac": constraint_jacobian,
    }

    # These row populations are close to a triangular lattice but have
    # different boundary defects.  Such defects are important for unequal
    # radii because they determine which outer circles can become larger.
    topologies = (
        (5, 5, 6, 5, 5),
        (5, 6, 5, 6, 4),
        (4, 6, 6, 6, 4),
        (5, 6, 5, 5, 5),
        (4, 5, 6, 6, 5),
        (5, 5, 6, 6, 4),
    )

    rng = np.random.default_rng(260319)
    starts = []
    for topology_index, rows in enumerate(topologies):
        for transpose in (False, True):
            for variant in range(2):
                if variant == 0:
                    jitter = None
                else:
                    jitter = rng.normal(0.0, 0.010, (n, 2))

                shift = 0.17 if (topology_index + variant) % 2 else 0.31
                phase = 0.003 if variant else 0.0
                centers = make_rows(
                    rows,
                    alternate_shift=shift,
                    transpose=transpose,
                    jitter=jitter,
                    vertical_phase=phase,
                )

                edge = np.minimum.reduce((
                    centers[:, 0], centers[:, 1],
                    1.0 - centers[:, 0], 1.0 - centers[:, 1]
                ))
                # Slightly varied radii break artificial row symmetry while
                # remaining safely below all initial nearest-neighbor gaps.
                radii = 0.033 + 0.006 * np.clip((0.18 - edge) / 0.18, -0.4, 1.0)
                if variant:
                    radii *= rng.uniform(0.94, 1.06, n)
                starts.append(np.concatenate((centers.ravel(), radii)))

    candidates = []
    for initial in starts:
        result = minimize(
            objective,
            initial,
            jac=lambda z: objective_gradient,
            method="SLSQP",
            bounds=bounds,
            constraints=con,
            options={
                "maxiter": 720,
                "ftol": 3.0e-11,
                "disp": False,
            },
        )
        if result.x is None or not np.all(np.isfinite(result.x)):
            continue

        centers, radii = safely_scale(
            result.x[:2 * n].reshape(n, 2),
            result.x[2 * n:],
        )
        candidates.append((float(np.sum(radii)), centers, radii))

    if not candidates:
        centers = make_rows((5, 5, 6, 5, 5), alternate_shift=0.25)
        radii = np.full(n, 0.03)
        return centers, radii, float(np.sum(radii))

    candidates.sort(key=lambda item: item[0], reverse=True)

    # Re-solving only the elite layouts permits much tighter contact-graph
    # convergence without paying this cost for every exploratory topology.
    refined = candidates[:min(3, len(candidates))]
    for _, initial_centers, initial_radii in refined:
        initial = np.concatenate((initial_centers.ravel(), initial_radii))
        result = minimize(
            objective,
            initial,
            jac=lambda z: objective_gradient,
            method="SLSQP",
            bounds=bounds,
            constraints=con,
            options={
                "maxiter": 4200,
                "ftol": 5.0e-13,
                "disp": False,
            },
        )
        if result.x is None or not np.all(np.isfinite(result.x)):
            continue

        centers, radii = safely_scale(
            result.x[:2 * n].reshape(n, 2),
            result.x[2 * n:],
        )
        candidates.append((float(np.sum(radii)), centers, radii))

    best_value, best_centers, best_radii = max(candidates, key=lambda item: item[0])
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