# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """Optimize a layered, near-hexagonal 26-circle packing."""
    from scipy.optimize import minimize

    n = 26
    rng = np.random.default_rng(260319)
    # Alternating rows provide an isotropic starting contact graph while
    # deliberately leaving room for the optimizer to break its symmetry.
    row_sizes = (5, 6, 5, 6, 4)
    base = []
    for row, count in enumerate(row_sizes):
        y = 0.10 + 0.20 * row
        if count == 5:
            xs = np.linspace(0.10, 0.90, count)
        elif count == 6:
            xs = np.linspace(1.0 / 12.0, 11.0 / 12.0, count)
        else:
            xs = np.linspace(0.16, 0.84, count)
        base.extend((x, y) for x in xs)
    base = np.asarray(base, dtype=float)

    ii, jj = np.triu_indices(n, 1)
    pair_count = len(ii)

    def constraints(z):
        c = z[:2 * n].reshape(n, 2)
        r = z[2 * n:]
        delta = c[ii] - c[jj]
        distances = np.sqrt(np.sum(delta * delta, axis=1))
        return np.concatenate((
            c[:, 0] - r, 1.0 - c[:, 0] - r,
            c[:, 1] - r, 1.0 - c[:, 1] - r,
            distances - r[ii] - r[jj],
        ))

    def constraint_jacobian(z):
        c = z[:2 * n].reshape(n, 2)
        jac = np.zeros((4 * n + pair_count, 3 * n))
        for k in range(n):
            xcol, ycol = 2 * k, 2 * k + 1
            rcol = 2 * n + k
            jac[4 * k, xcol], jac[4 * k, rcol] = 1.0, -1.0
            jac[4 * k + 1, xcol], jac[4 * k + 1, rcol] = -1.0, -1.0
            jac[4 * k + 2, ycol], jac[4 * k + 2, rcol] = 1.0, -1.0
            jac[4 * k + 3, ycol], jac[4 * k + 3, rcol] = -1.0, -1.0
        delta = c[ii] - c[jj]
        lengths = np.sqrt(np.sum(delta * delta, axis=1))
        lengths = np.maximum(lengths, 1e-12)
        for k, (a, b) in enumerate(zip(ii, jj)):
            row = 4 * n + k
            direction = delta[k] / lengths[k]
            jac[row, 2 * a:2 * a + 2] = direction
            jac[row, 2 * b:2 * b + 2] = -direction
            jac[row, 2 * n + a] = -1.0
            jac[row, 2 * n + b] = -1.0
        return jac

    best_centers = base.copy()
    best_score = -np.inf
    nonlinear_constraint = {"type": "ineq", "fun": constraints,
                            "jac": constraint_jacobian}
    bounds = [(0.0, 1.0)] * (2 * n) + [(1e-8, 0.5)] * n

    for trial in range(8):
        if trial == 0:
            centers0 = base.copy()
        else:
            centers0 = np.clip(base + rng.normal(0.0, 0.018, base.shape),
                               0.025, 0.975)
        # A fixed-center LP supplies unequal, feasible radii at a useful
        # scale instead of making the nonlinear solver discover them from zero.
        radii0 = 0.72 * compute_max_radii(centers0)
        start = np.concatenate((centers0.ravel(), radii0))
        result = minimize(
            lambda z: -np.sum(z[2 * n:]), start,
            jac=lambda z: np.r_[np.zeros(2 * n), -np.ones(n)],
            method="SLSQP", bounds=bounds, constraints=[nonlinear_constraint],
            options={"maxiter": 700, "ftol": 2e-10, "disp": False},
        )
        if np.all(np.isfinite(result.x)) and np.min(constraints(result.x)) >= -2e-7:
            candidate = result.x[:2 * n].reshape(n, 2)
            score = np.sum(compute_max_radii(candidate))
            if score > best_score:
                best_score = score
                best_centers = candidate

    radii = compute_max_radii(best_centers) * (1.0 - 2e-10)
    return best_centers, radii, float(np.sum(radii))


def compute_max_radii(centers):
    """
    Maximize the sum of radii for fixed centers.

    This is a linear program: each wall gives an individual upper bound and
    each pair contributes r_i + r_j <= distance(i, j).
    """
    from scipy.optimize import linprog

    n = centers.shape[0]
    ii, jj = np.triu_indices(n, 1)
    distances = np.sqrt(np.sum((centers[ii] - centers[jj]) ** 2, axis=1))
    pair_matrix = np.zeros((len(ii), n))
    pair_matrix[np.arange(len(ii)), ii] = 1.0
    pair_matrix[np.arange(len(ii)), jj] = 1.0
    wall_limits = np.minimum.reduce((
        centers[:, 0], centers[:, 1],
        1.0 - centers[:, 0], 1.0 - centers[:, 1],
    ))
    result = linprog(
        -np.ones(n), A_ub=pair_matrix, b_ub=distances,
        bounds=[(0.0, max(0.0, limit)) for limit in wall_limits],
        method="highs",
    )
    if result.success:
        return result.x

    # Conservative fallback, retained for environments without a successful LP.
    radii = 0.45 * np.maximum(wall_limits, 0.0)
    for k, (i, j) in enumerate(zip(ii, jj)):
        total = radii[i] + radii[j]
        if total > distances[k] and total > 0.0:
            scale = distances[k] / total
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