# EVOLVE-BLOCK-START
"""Nonlinear / LP hybrid constructor for 26 circles in a unit square."""

import numpy as np


def _radius_lp(centers):
    """
    For fixed centers, solve the linear program

        maximize sum(r_i)
        subject to r_i + r_j <= distance(i, j)
                   r_i <= distance to every square boundary.

    This gives the best possible radii for a given center arrangement.
    """
    n = len(centers)

    try:
        from scipy.optimize import linprog

        pair_i, pair_j = np.triu_indices(n, 1)
        distances = np.sqrt(np.sum(
            (centers[pair_i] - centers[pair_j]) ** 2, axis=1
        ))

        rows = []
        rhs = []

        for k, (i, j) in enumerate(zip(pair_i, pair_j)):
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            rows.append(row)
            rhs.append(distances[k])

        border = np.minimum(
            np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
            np.minimum(centers[:, 1], 1.0 - centers[:, 1]),
        )

        for i in range(n):
            row = np.zeros(n)
            row[i] = 1.0
            rows.append(row)
            rhs.append(border[i])

        result = linprog(
            c=-np.ones(n),
            A_ub=np.asarray(rows),
            b_ub=np.asarray(rhs),
            bounds=[(0.0, None)] * n,
            method="highs",
        )

        if result.success:
            return np.maximum(result.x, 0.0)

    except Exception:
        pass

    # Conservative dependency-free fallback.
    border = np.minimum(
        np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
        np.minimum(centers[:, 1], 1.0 - centers[:, 1]),
    )
    d = centers[:, None, :] - centers[None, :, :]
    distances = np.sqrt(np.sum(d * d, axis=2))
    distances += np.eye(n) * 10.0
    common = min(np.min(border), 0.5 * np.min(distances))
    return np.full(n, max(common, 1e-8))


def _safe_radii(centers, radii):
    """Repair tiny numerical violations and leave a strict safety margin."""
    radii = np.maximum(np.asarray(radii, dtype=float), 0.0).copy()
    n = len(radii)

    border = np.minimum(
        np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
        np.minimum(centers[:, 1], 1.0 - centers[:, 1]),
    )
    radii = np.minimum(radii, np.maximum(border, 0.0))

    for i in range(n):
        for j in range(i + 1, n):
            distance = float(np.linalg.norm(centers[i] - centers[j]))
            total = radii[i] + radii[j]
            if total > distance and total > 0.0:
                radii[i] *= distance / total
                radii[j] *= distance / total

    return radii * (1.0 - 2e-9)


def _initial_centers(seed, jitter):
    """
    A square-grid seed is substantially denser than the old ring layout.
    The 26th circle begins in one of the grid's interstices.  Small,
    asymmetric perturbations allow the nonlinear solve to escape the
    artificial five-by-five symmetry.
    """
    rng = np.random.default_rng(seed)

    grid = np.array(
        [[0.1 + 0.2 * col, 0.1 + 0.2 * row]
         for row in range(5) for col in range(5)],
        dtype=float,
    )

    gap_choices = np.array([
        [0.2, 0.2], [0.4, 0.2], [0.6, 0.2], [0.8, 0.2],
        [0.2, 0.4], [0.4, 0.4], [0.6, 0.4], [0.8, 0.4],
        [0.2, 0.6], [0.4, 0.6], [0.6, 0.6], [0.8, 0.6],
        [0.2, 0.8], [0.4, 0.8], [0.6, 0.8], [0.8, 0.8],
    ])

    extra = gap_choices[seed % len(gap_choices)].copy()
    centers = np.vstack((grid, extra))

    if jitter > 0.0:
        centers += rng.normal(scale=jitter, size=centers.shape)
        centers = np.clip(centers, 0.025, 0.975)

    return centers


def _nonlinear_improve(initial_centers, maxiter=450):
    """
    Jointly optimize centers and radii.  Squared-distance constraints avoid
    square roots inside the nonlinear optimizer:

        ||c_i-c_j||^2 >= (r_i+r_j)^2.

    The later LP pass is used as an exact radius polish and validity repair.
    """
    try:
        from scipy.optimize import minimize
    except Exception:
        return initial_centers

    n = len(initial_centers)
    pi, pj = np.triu_indices(n, 1)
    m = len(pi)

    initial_radii = 0.90 * _radius_lp(initial_centers)
    z0 = np.concatenate((initial_centers.ravel(), initial_radii))

    # Linear boundary constraints and their constant Jacobian.
    boundary_jac = np.zeros((4 * n, 3 * n))
    for i in range(n):
        xcol, ycol, rcol = 2 * i, 2 * i + 1, 2 * n + i
        boundary_jac[4 * i + 0, xcol] = 1.0
        boundary_jac[4 * i + 0, rcol] = -1.0
        boundary_jac[4 * i + 1, xcol] = -1.0
        boundary_jac[4 * i + 1, rcol] = -1.0
        boundary_jac[4 * i + 2, ycol] = 1.0
        boundary_jac[4 * i + 2, rcol] = -1.0
        boundary_jac[4 * i + 3, ycol] = -1.0
        boundary_jac[4 * i + 3, rcol] = -1.0

    def constraints(z):
        c = z[:2 * n].reshape(n, 2)
        r = z[2 * n:]

        boundary = np.empty(4 * n)
        boundary[0::4] = c[:, 0] - r
        boundary[1::4] = 1.0 - c[:, 0] - r
        boundary[2::4] = c[:, 1] - r
        boundary[3::4] = 1.0 - c[:, 1] - r

        delta = c[pi] - c[pj]
        pair = np.sum(delta * delta, axis=1) - (r[pi] + r[pj]) ** 2
        return np.concatenate((boundary, pair))

    def constraint_jacobian(z):
        c = z[:2 * n].reshape(n, 2)
        r = z[2 * n:]
        jac = np.zeros((4 * n + m, 3 * n))
        jac[:4 * n] = boundary_jac

        delta = c[pi] - c[pj]
        sums = r[pi] + r[pj]
        for k, (i, j) in enumerate(zip(pi, pj)):
            row = 4 * n + k
            jac[row, 2 * i:2 * i + 2] = 2.0 * delta[k]
            jac[row, 2 * j:2 * j + 2] = -2.0 * delta[k]
            jac[row, 2 * n + i] = -2.0 * sums[k]
            jac[row, 2 * n + j] = -2.0 * sums[k]

        return jac

    def objective(z):
        return -float(np.sum(z[2 * n:]))

    def objective_jacobian(z):
        grad = np.zeros_like(z)
        grad[2 * n:] = -1.0
        return grad

    bounds = [(1e-6, 1.0 - 1e-6)] * (2 * n) + [(0.0, 0.5)] * n

    try:
        result = minimize(
            objective,
            z0,
            method="SLSQP",
            jac=objective_jacobian,
            bounds=bounds,
            constraints={
                "type": "ineq",
                "fun": constraints,
                "jac": constraint_jacobian,
            },
            options={
                "maxiter": maxiter,
                "ftol": 1e-10,
                "disp": False,
            },
        )

        candidate = result.x[:2 * n].reshape(n, 2)
        if np.all(np.isfinite(candidate)):
            return np.clip(candidate, 1e-7, 1.0 - 1e-7)

    except Exception:
        pass

    return initial_centers


def construct_packing():
    """
    Construct a high-density valid packing of 26 circles in the unit square.

    Returns:
        centers: shape (26, 2)
        radii: shape (26,)
        sum_of_radii: float
    """
    best_centers = None
    best_radii = None
    best_score = -np.inf

    # A few deliberately different asymmetric starts provide substantially
    # better layouts than relying on one highly symmetric local optimum.
    starts = [
        (0, 0.000),
        (1, 0.010),
        (6, 0.014),
        (11, 0.018),
        (14, 0.022),
        (9, 0.028),
    ]

    for seed, jitter in starts:
        centers0 = _initial_centers(seed, jitter)
        centers = _nonlinear_improve(centers0)
        radii = _safe_radii(centers, _radius_lp(centers))
        score = float(np.sum(radii))

        if score > best_score:
            best_score = score
            best_centers = centers
            best_radii = radii

    return best_centers, best_radii, float(np.sum(best_radii))


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