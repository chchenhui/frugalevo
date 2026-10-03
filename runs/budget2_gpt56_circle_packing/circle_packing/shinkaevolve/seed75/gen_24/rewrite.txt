# EVOLVE-BLOCK-START
"""Layered nonlinear/LP constructor for 26 circles in a unit square."""
import numpy as np


def _safe_radii(centers, radii):
    """Return a strictly valid common-scale projection of a candidate packing."""
    centers = np.clip(np.asarray(centers, dtype=float).copy(), 0.0, 1.0)
    radii = np.maximum(np.asarray(radii, dtype=float).copy(), 0.0)

    border = np.minimum.reduce((
        centers[:, 0], centers[:, 1],
        1.0 - centers[:, 0], 1.0 - centers[:, 1],
    ))
    radii = np.minimum(radii, np.maximum(border, 0.0))

    n = len(radii)
    ii, jj = np.triu_indices(n, 1)
    delta = centers[ii] - centers[jj]
    distances = np.sqrt(np.einsum("ij,ij->i", delta, delta))
    sums = radii[ii] + radii[jj]

    scale = 1.0
    active = sums > 0.0
    if np.any(active):
        scale = min(scale, float(np.min(distances[active] / sums[active])))

    radii *= max(0.0, min(1.0, scale * (1.0 - 2.0e-10)))
    return centers, radii


def _row_seed(row_sizes, phase=0.0):
    """Create a slightly perturbed layered triangular-layout seed."""
    centers = []
    rows = len(row_sizes)

    for row, count in enumerate(row_sizes):
        y = 0.105 + 0.79 * row / max(1, rows - 1)
        inset = 0.072 + 0.010 * ((row + int(phase * 7.0)) % 2)
        xs = np.linspace(inset, 1.0 - inset, count)

        for col, x in enumerate(xs):
            dx = 0.0085 * np.sin(1.73 * (row + 1) * (col + 1) + phase)
            dy = 0.0065 * np.cos(1.21 * (row + 2) * (col + 1) + phase)
            centers.append((
                np.clip(x + dx, 0.045, 0.955),
                np.clip(y + dy, 0.045, 0.955),
            ))

    return np.asarray(centers, dtype=float)


def _fallback():
    """Dependency-free valid staggered packing."""
    centers = _row_seed((5, 6, 5, 5, 5), 0.0)
    n = len(centers)
    ii, jj = np.triu_indices(n, 1)

    border = np.minimum.reduce((
        centers[:, 0], centers[:, 1],
        1.0 - centers[:, 0], 1.0 - centers[:, 1],
    ))
    delta = centers[ii] - centers[jj]
    distances = np.sqrt(np.einsum("ij,ij->i", delta, delta))
    radius = min(float(np.min(border)), float(np.min(distances)) * 0.5)
    radii = np.full(n, radius * 0.999999, dtype=float)
    return centers, radii


def construct_packing():
    """
    Construct a high-sum valid packing of 26 circles.

    The nonlinear stage finds useful contact graphs; the LP stage then gives
    the exact best unequal-radius allocation for each resulting center set.
    """
    n = 26
    ii, jj = np.triu_indices(n, 1)
    pair_count = len(ii)
    index = np.arange(n)

    templates = [
        ((5, 6, 5, 5, 5), 0.00),
        ((5, 5, 6, 5, 5), 0.37),
        ((5, 5, 5, 6, 5), 0.74),
        ((6, 5, 5, 5, 5), 1.13),
        ((5, 5, 5, 5, 6), 1.49),
        ((4, 6, 5, 6, 5), 1.67),
        ((5, 4, 6, 5, 6), 2.03),
        ((6, 5, 4, 6, 5), 2.39),
        ((5, 6, 5, 4, 6), 2.76),
        ((6, 5, 6, 5, 4), 3.12),
        ((4, 6, 6, 5, 5), 3.48),
        ((5, 4, 6, 6, 5), 3.85),
        ((5, 5, 4, 6, 6), 4.21),
        ((6, 5, 5, 4, 6), 4.58),
        ((6, 6, 5, 5, 4), 4.94),
    ]

    best_centers, best_radii = _fallback()
    best_value = float(np.sum(best_radii))

    try:
        from scipy.optimize import linprog, minimize

        pair_matrix = np.zeros((pair_count, n), dtype=float)
        pair_matrix[np.arange(pair_count), ii] = 1.0
        pair_matrix[np.arange(pair_count), jj] = 1.0

        def optimal_radii(centers):
            delta = centers[ii] - centers[jj]
            distances = np.sqrt(np.einsum("ij,ij->i", delta, delta))
            walls = np.minimum.reduce((
                centers[:, 0], centers[:, 1],
                1.0 - centers[:, 0], 1.0 - centers[:, 1],
            ))
            result = linprog(
                -np.ones(n),
                A_ub=pair_matrix,
                b_ub=distances,
                bounds=[(0.0, max(0.0, float(w))) for w in walls],
                method="highs",
            )
            if result.success and result.x is not None:
                return result.x
            return np.full(n, 1.0e-8)

        def objective(z):
            return -float(np.sum(z[2 * n:]))

        def objective_jac(z):
            grad = np.zeros(3 * n)
            grad[2 * n:] = -1.0
            return grad

        def inequalities(z):
            x = z[:n]
            y = z[n:2 * n]
            r = z[2 * n:]

            walls = np.concatenate((
                x - r, 1.0 - x - r,
                y - r, 1.0 - y - r,
            ))
            dx = x[ii] - x[jj]
            dy = y[ii] - y[jj]
            pairs = dx * dx + dy * dy - (r[ii] + r[jj]) ** 2
            return np.concatenate((walls, pairs))

        def inequalities_jac(z):
            x = z[:n]
            y = z[n:2 * n]
            r = z[2 * n:]
            jac = np.zeros((4 * n + pair_count, 3 * n), dtype=float)

            jac[index, index] = 1.0
            jac[index, 2 * n + index] = -1.0

            jac[n + index, index] = -1.0
            jac[n + index, 2 * n + index] = -1.0

            jac[2 * n + index, n + index] = 1.0
            jac[2 * n + index, 2 * n + index] = -1.0

            jac[3 * n + index, n + index] = -1.0
            jac[3 * n + index, 2 * n + index] = -1.0

            rows = 4 * n + np.arange(pair_count)
            dx = x[ii] - x[jj]
            dy = y[ii] - y[jj]
            sr = r[ii] + r[jj]

            jac[rows, ii] = 2.0 * dx
            jac[rows, jj] = -2.0 * dx
            jac[rows, n + ii] = 2.0 * dy
            jac[rows, n + jj] = -2.0 * dy
            jac[rows, 2 * n + ii] = -2.0 * sr
            jac[rows, 2 * n + jj] = -2.0 * sr
            return jac

        constraints = {
            "type": "ineq",
            "fun": inequalities,
            "jac": inequalities_jac,
        }
        bounds = [(0.0, 1.0)] * (2 * n) + [(1.0e-8, 0.5)] * n

        for row_sizes, phase in templates:
            centers = _row_seed(row_sizes, phase)

            # LP provides a feasible, unequal-radius initialization and avoids
            # wasting nonlinear iterations rediscovering obvious wall limits.
            initial_radii = 0.985 * optimal_radii(centers)
            z0 = np.concatenate((
                centers[:, 0],
                centers[:, 1],
                initial_radii,
            ))

            result = minimize(
                objective,
                z0,
                jac=objective_jac,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={
                    "maxiter": 1200,
                    "ftol": 1.5e-11,
                    "disp": False,
                },
            )

            candidate = result.x if result.x is not None else z0
            candidate_centers = np.column_stack((
                candidate[:n],
                candidate[n:2 * n],
            ))

            # Reallocate radii exactly for the obtained geometry.  This is
            # never worse than retaining a feasible nonlinear radius vector.
            candidate_radii = optimal_radii(candidate_centers)
            candidate_centers, candidate_radii = _safe_radii(
                candidate_centers, candidate_radii
            )
            value = float(np.sum(candidate_radii))

            if value > best_value:
                best_centers = candidate_centers
                best_radii = candidate_radii
                best_value = value

    except Exception:
        pass

    return best_centers, best_radii, float(np.sum(best_radii))


def compute_max_radii(centers):
    """
    Compute conservative valid radii for supplied center positions.

    Args:
        centers: np.array of shape (n, 2)

    Returns:
        np.array of feasible radii.
    """
    centers = np.asarray(centers, dtype=float)
    n = centers.shape[0]
    radii = np.minimum.reduce((
        centers[:, 0], centers[:, 1],
        1.0 - centers[:, 0], 1.0 - centers[:, 1],
    )).astype(float)

    for _ in range(3):
        scale = 1.0
        for i in range(n):
            for j in range(i + 1, n):
                distance = np.linalg.norm(centers[i] - centers[j])
                total = radii[i] + radii[j]
                if total > distance and total > 0.0:
                    scale = min(scale, distance / total)
        radii *= scale

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