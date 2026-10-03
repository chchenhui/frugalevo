# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """
    Escape the original row contact graph using deleted-site and interstitial
    staggered lattices, briefly relax their centers by repulsion, LP-rank them,
    and jointly polish the eight best layouts with hard geometric constraints.
    """
    from scipy.optimize import linprog, minimize

    n = 26
    ii, jj = np.triu_indices(n, 1)
    pair_count = len(ii)
    rows = np.arange(pair_count)

    def feasible_radii(points):
        """Solve the fixed-center radius LP and apply a final numerical shrink."""
        delta = points[ii] - points[jj]
        distances = np.sqrt(np.einsum("ij,ij->i", delta, delta))
        border = np.minimum.reduce(
            (points[:, 0], points[:, 1], 1.0 - points[:, 0], 1.0 - points[:, 1])
        )
        matrix = np.zeros((pair_count, n), dtype=float)
        matrix[rows, ii] = 1.0
        matrix[rows, jj] = 1.0
        answer = linprog(
            -np.ones(n), A_ub=matrix, b_ub=distances,
            bounds=[(0.0, float(v)) for v in border], method="highs",
        )
        return answer.x * (1.0 - 2e-10) if answer.success else np.zeros(n)

    def constraints(z):
        """Return exact pair separation and four wall-clearance slacks."""
        points = z[:2 * n].reshape(n, 2)
        radii = z[2 * n:]
        delta = points[ii] - points[jj]
        pair = np.einsum("ij,ij->i", delta, delta) - (radii[ii] + radii[jj]) ** 2
        walls = np.column_stack((
            points[:, 0] - radii, points[:, 1] - radii,
            1.0 - points[:, 0] - radii, 1.0 - points[:, 1] - radii,
        )).ravel()
        return np.r_[pair, walls]

    def constraint_jacobian(z):
        """Return the analytic Jacobian of all hard packing constraints."""
        points = z[:2 * n].reshape(n, 2)
        radii = z[2 * n:]
        delta = points[ii] - points[jj]
        sums = radii[ii] + radii[jj]
        jac = np.zeros((pair_count + 4 * n, 3 * n), dtype=float)
        jac[rows, 2 * ii] = 2.0 * delta[:, 0]
        jac[rows, 2 * ii + 1] = 2.0 * delta[:, 1]
        jac[rows, 2 * jj] = -2.0 * delta[:, 0]
        jac[rows, 2 * jj + 1] = -2.0 * delta[:, 1]
        jac[rows, 2 * n + ii] = -2.0 * sums
        jac[rows, 2 * n + jj] = -2.0 * sums
        for k in range(n):
            q = pair_count + 4 * k
            jac[q, 2 * k], jac[q, 2 * n + k] = 1.0, -1.0
            jac[q + 1, 2 * k + 1], jac[q + 1, 2 * n + k] = 1.0, -1.0
            jac[q + 2, 2 * k], jac[q + 2, 2 * n + k] = -1.0, -1.0
            jac[q + 3, 2 * k + 1], jac[q + 3, 2 * n + k] = -1.0, -1.0
        return jac

    def relax(points):
        """Use 80 bounded L-BFGS iterations of pair and wall repulsion."""
        def energy(flat):
            p = flat.reshape(n, 2)
            d = p[ii] - p[jj]
            d2 = np.einsum("ij,ij->i", d, d)
            # The short-range term changes neighbours without imposing a grid.
            pair_energy = np.exp(-d2 / 0.020)
            wall_distance = np.minimum.reduce(
                (p[:, 0], p[:, 1], 1.0 - p[:, 0], 1.0 - p[:, 1])
            )
            return float(pair_energy.sum() + 0.24 * np.exp(-wall_distance / 0.045).sum())

        result = minimize(
            energy, np.asarray(points, dtype=float).ravel(),
            method="L-BFGS-B", bounds=[(0.012, 0.988)] * (2 * n),
            options={"maxiter": 80, "ftol": 1e-12},
        )
        return result.x.reshape(n, 2).copy()

    def staggered(count, phase):
        """Build a phase-shifted five-row triangular lattice of given width."""
        pts = []
        for row in range(5):
            y = 0.09 + 0.205 * row
            shift = (0.5 * (row & 1) + phase) / count
            for col in range(count):
                pts.append(((col + 0.5 + shift) / count, y))
        return np.asarray(pts, dtype=float)

    candidates = []
    golden = 0.6180339887498949
    for trial in range(12):
        phase = (trial * golden) % 1.0 - 0.5
        # A 5x5 lattice gains one interstitial point, changing its local graph.
        points = staggered(5, phase)
        insert = np.array([[0.5 + 0.12 * np.sin(6.283 * trial * golden),
                            0.5 + 0.10 * np.cos(6.283 * trial * golden)]])
        candidates.append(relax(np.vstack((points, insert))))

    for trial in range(12):
        phase = (trial * golden + 0.31) % 1.0 - 0.5
        # Four phase-dependent deleted sites from 6x5 leave 26 centers.
        points = staggered(6, phase)
        deleted = np.sort((np.arange(4) * 7 + 3 * trial) % 30)
        candidates.append(relax(np.delete(points, deleted, axis=0)))

    # Retain a simple valid layered layout as a fallback, not as a search path.
    fallback = staggered(5, 0.0)
    fallback = np.vstack((fallback, [[0.5, 0.5]]))
    best_points = fallback.copy()
    best_radii = feasible_radii(best_points)

    scored = [(feasible_radii(p).sum(), p) for p in candidates]
    scored.sort(key=lambda item: item[0], reverse=True)
    for _, points in scored[:8]:
        radii = feasible_radii(points) * 0.70
        result = minimize(
            lambda z: -np.sum(z[2 * n:]),
            np.r_[points.ravel(), radii],
            jac=lambda z: np.r_[np.zeros(2 * n), -np.ones(n)],
            method="SLSQP",
            bounds=[(0.001, 0.999)] * (2 * n) + [(0.0, 0.5)] * n,
            constraints={"type": "ineq", "fun": constraints, "jac": constraint_jacobian},
            options={"maxiter": 650, "ftol": 1e-10, "disp": False},
        )
        candidate = result.x[:2 * n].reshape(n, 2)
        candidate_radii = feasible_radii(candidate)
        if candidate_radii.sum() > best_radii.sum():
            best_points, best_radii = candidate.copy(), candidate_radii

    return best_points, best_radii, float(best_radii.sum())


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
