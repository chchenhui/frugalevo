# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """
    Jointly optimize centers and radii for 26 non-overlapping circles in the
    unit square.  Several layered and grid-like feasible seeds are used since
    boundary defects make different contact graphs locally competitive.
    """
    n = 26
    variable_count = 3 * n

    pair_i, pair_j = np.triu_indices(n, 1)
    pair_count = len(pair_i)
    pair_rows = np.arange(4 * n, 4 * n + pair_count)

    def pack(centers, radii):
        return np.concatenate((centers.ravel(), radii))

    def unpack(z):
        return z[:2 * n].reshape(n, 2), z[2 * n:]

    # Dense square seed: twenty-five near-equal circles and one small boundary
    # defect.  This is particularly reliable for obtaining a good baseline.
    grid = np.array(
        [(0.1 + 0.2 * col, 0.1 + 0.2 * row)
         for row in range(5) for col in range(5)],
        dtype=float,
    )
    grid_centers = np.vstack((grid, [[0.008, 0.008]]))
    grid_radii = np.r_[np.full(25, 0.099), 0.007]

    # Alternating rows give SLSQP a hexagonal/contact-network alternative.
    layered = []
    for row, count in enumerate((5, 6, 5, 6, 4)):
        y = 0.10 + 0.20 * row
        offset = 0.10 if count == 5 else 0.075
        spacing = 0.20 if count == 5 else 0.17
        layered.extend((offset + spacing * col, y) for col in range(count))
    layered_centers = np.asarray(layered, dtype=float)
    layered_radii = np.full(n, 0.034)

    # A second layered topology with the short row on the opposite side.
    asymmetric = []
    row_specs = (
        (5, 0.10, 0.20),
        (6, 0.075, 0.17),
        (5, 0.10, 0.20),
        (5, 0.10, 0.20),
        (5, 0.10, 0.20),
    )
    for row, (count, offset, spacing) in enumerate(row_specs):
        y = 0.10 + 0.20 * row
        asymmetric.extend((offset + spacing * col, y) for col in range(count))
    asymmetric_centers = np.asarray(asymmetric, dtype=float)
    asymmetric_radii = np.full(n, 0.033)

    # Static portion of the analytic constraint Jacobian.
    base_jacobian = np.zeros((4 * n + pair_count, variable_count))
    for i in range(n):
        x, y, r = 2 * i, 2 * i + 1, 2 * n + i
        row = 4 * i
        base_jacobian[row, x] = 1.0
        base_jacobian[row, r] = -1.0
        base_jacobian[row + 1, x] = -1.0
        base_jacobian[row + 1, r] = -1.0
        base_jacobian[row + 2, y] = 1.0
        base_jacobian[row + 2, r] = -1.0
        base_jacobian[row + 3, y] = -1.0
        base_jacobian[row + 3, r] = -1.0

    def constraint_values(z):
        centers, radii = unpack(z)
        border = np.empty(4 * n)
        border[0::4] = centers[:, 0] - radii
        border[1::4] = 1.0 - centers[:, 0] - radii
        border[2::4] = centers[:, 1] - radii
        border[3::4] = 1.0 - centers[:, 1] - radii

        delta = centers[pair_i] - centers[pair_j]
        distance_squared = np.einsum("ij,ij->i", delta, delta)
        pair_values = distance_squared - (radii[pair_i] + radii[pair_j]) ** 2
        return np.concatenate((border, pair_values))

    def constraint_jacobian(z):
        centers, radii = unpack(z)
        jacobian = base_jacobian.copy()
        delta = centers[pair_i] - centers[pair_j]
        radius_sum = radii[pair_i] + radii[pair_j]

        jacobian[pair_rows, 2 * pair_i] = 2.0 * delta[:, 0]
        jacobian[pair_rows, 2 * pair_i + 1] = 2.0 * delta[:, 1]
        jacobian[pair_rows, 2 * pair_j] = -2.0 * delta[:, 0]
        jacobian[pair_rows, 2 * pair_j + 1] = -2.0 * delta[:, 1]
        jacobian[pair_rows, 2 * n + pair_i] = -2.0 * radius_sum
        jacobian[pair_rows, 2 * n + pair_j] = -2.0 * radius_sum
        return jacobian

    starts = (
        pack(grid_centers, grid_radii),
        pack(layered_centers, layered_radii),
        pack(asymmetric_centers, asymmetric_radii),
    )
    best = starts[0].copy()
    best_value = float(np.sum(grid_radii))

    try:
        from scipy.optimize import minimize

        objective_gradient = np.zeros(variable_count)
        objective_gradient[2 * n:] = -1.0
        constraint = {
            "type": "ineq",
            "fun": constraint_values,
            "jac": constraint_jacobian,
        }
        bounds = [(0.0, 1.0)] * (2 * n) + [(1e-8, 0.5)] * n

        for start in starts:
            result = minimize(
                lambda z: -np.sum(z[2 * n:]),
                start,
                jac=lambda z: objective_gradient,
                method="SLSQP",
                bounds=bounds,
                constraints=(constraint,),
                options={"maxiter": 1200, "ftol": 2e-12, "disp": False},
            )
            if np.all(np.isfinite(result.x)):
                value = float(np.sum(result.x[2 * n:]))
                if value > best_value and np.min(constraint_values(result.x)) >= -2e-7:
                    best = result.x.copy()
                    best_value = value
    except ImportError:
        pass

    centers, radii = unpack(best)

    # Conservative final certification against containment and pair overlap.
    border_limit = np.min(
        np.column_stack((
            centers[:, 0] / radii,
            centers[:, 1] / radii,
            (1.0 - centers[:, 0]) / radii,
            (1.0 - centers[:, 1]) / radii,
        ))
    )
    delta = centers[pair_i] - centers[pair_j]
    distances = np.sqrt(np.einsum("ij,ij->i", delta, delta))
    pair_limit = np.min(distances / (radii[pair_i] + radii[pair_j]))
    scale = min(1.0, border_limit, pair_limit) * (1.0 - 1e-10)
    radii = radii * scale

    return centers, radii, float(np.sum(radii))


def compute_max_radii(centers):
    """
    Compute maximum valid radii for fixed centers using a conservative
    pairwise scaling pass.
    """
    n = centers.shape[0]
    radii = np.minimum.reduce((
        centers[:, 0], centers[:, 1],
        1.0 - centers[:, 0], 1.0 - centers[:, 1],
    ))

    for i in range(n):
        for j in range(i + 1, n):
            distance = np.linalg.norm(centers[i] - centers[j])
            total = radii[i] + radii[j]
            if total > distance and total > 0.0:
                scale = distance / total
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