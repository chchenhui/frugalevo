# EVOLVE-BLOCK-START
"""Multistart nonlinear constructor for 26 unequal circles in a unit square."""
import numpy as np


def construct_packing():
    """
    Construct a high-quality packing of 26 non-overlapping circles in [0,1]^2.

    Returns:
        centers: array of shape (26, 2)
        radii: array of shape (26,)
        sum_of_radii: float
    """
    n = 26
    ii, jj = np.triu_indices(n, 1)
    pair_count = len(ii)
    variable_count = 3 * n

    # Squared distance constraints avoid square roots and have inexpensive,
    # exact derivatives.  They are equivalent to ordinary separation because
    # all radii are constrained to be nonnegative.
    def constraints(z):
        p = z[:2 * n].reshape(n, 2)
        r = z[2 * n:]

        dx = p[ii, 0] - p[jj, 0]
        dy = p[ii, 1] - p[jj, 1]
        pair = dx * dx + dy * dy - (r[ii] + r[jj]) ** 2

        return np.concatenate((
            p[:, 0] - r,
            p[:, 1] - r,
            1.0 - p[:, 0] - r,
            1.0 - p[:, 1] - r,
            pair,
        ))

    def constraint_jacobian(z):
        p = z[:2 * n].reshape(n, 2)
        r = z[2 * n:]
        jac = np.zeros((4 * n + pair_count, variable_count), dtype=float)

        ind = np.arange(n)

        # Wall contacts.
        jac[ind, 2 * ind] = 1.0
        jac[ind, 2 * n + ind] = -1.0

        jac[n + ind, 2 * ind + 1] = 1.0
        jac[n + ind, 2 * n + ind] = -1.0

        jac[2 * n + ind, 2 * ind] = -1.0
        jac[2 * n + ind, 2 * n + ind] = -1.0

        jac[3 * n + ind, 2 * ind + 1] = -1.0
        jac[3 * n + ind, 2 * n + ind] = -1.0

        # Circle-circle contacts.
        row = 4 * n + np.arange(pair_count)
        dx = p[ii, 0] - p[jj, 0]
        dy = p[ii, 1] - p[jj, 1]
        sr = r[ii] + r[jj]

        jac[row, 2 * ii] = 2.0 * dx
        jac[row, 2 * ii + 1] = 2.0 * dy
        jac[row, 2 * jj] = -2.0 * dx
        jac[row, 2 * jj + 1] = -2.0 * dy
        jac[row, 2 * n + ii] = -2.0 * sr
        jac[row, 2 * n + jj] = -2.0 * sr
        return jac

    def row_seed(row_sizes, phase, rng, noise):
        """A finite hexagonal lattice seed with deliberately broken symmetry."""
        rows = len(row_sizes)
        points = []
        for row, count in enumerate(row_sizes):
            y = (row + 0.5) / rows
            # Alternating lateral phases imitate triangular lattice contacts.
            shift = 0.14 * np.sin((row + phase) * np.pi * 0.93)
            for col in range(count):
                x = (col + 0.5 + shift) / count
                points.append((x, y))

        points = np.asarray(points, dtype=float)
        if noise:
            points += rng.normal(0.0, noise, points.shape)
            points = np.clip(points, 0.055, 0.945)
        return points

    # Several row populations have nearly hexagonal interiors but substantially
    # different boundary contact patterns.  This is important for unequal
    # circles, where the best square boundary pattern is not fully symmetric.
    layouts = (
        (5, 6, 5, 5, 5),
        (5, 5, 6, 5, 5),
        (5, 5, 5, 6, 5),
        (5, 6, 5, 6, 4),
        (4, 6, 5, 6, 5),
        (6, 5, 5, 5, 5),
        (4, 5, 6, 6, 5),
        (5, 6, 6, 5, 4),
    )

    # A small feasible incumbent is always available, including on platforms
    # that do not provide scipy.
    fallback_centers = row_seed(layouts[0], 0.0, np.random.RandomState(1), 0.0)
    best_centers = fallback_centers.copy()
    best_radii = np.full(n, 0.018, dtype=float)
    best_value = float(np.sum(best_radii))

    try:
        from scipy.optimize import minimize

        bounds = [(0.0, 1.0)] * (2 * n) + [(1e-7, 0.5)] * n
        nonlinear_constraint = {
            "type": "ineq",
            "fun": constraints,
            "jac": constraint_jacobian,
        }

        # The schedule is deterministic, making the returned constructor
        # reproducible while retaining the escape power of random restarts.
        rng = np.random.RandomState(26419)
        starts = []
        for layout_number, layout in enumerate(layouts):
            starts.append(row_seed(layout, 0.37 * layout_number, rng, 0.0))
            starts.append(row_seed(layout, 0.37 * layout_number + 0.21, rng, 0.020))
            starts.append(row_seed(layout, 0.37 * layout_number + 0.44, rng, 0.038))

        for attempt, seed in enumerate(starts):
            # Mildly varied radii make early contact selection less symmetric.
            initial_radii = 0.028 + rng.uniform(-0.006, 0.006, n)
            z0 = np.concatenate((seed.ravel(), initial_radii))

            # A randomly tilted objective first discovers asymmetric contact
            # networks; the final uniform objective restores the true target.
            if attempt % 3 == 2:
                weights = 1.0 + rng.uniform(-0.10, 0.10, n)
            else:
                weights = np.ones(n)

            result = minimize(
                lambda z, w=weights: -np.dot(w, z[2 * n:]),
                z0,
                jac=lambda z, w=weights: np.concatenate((
                    np.zeros(2 * n), -w
                )),
                method="SLSQP",
                bounds=bounds,
                constraints=nonlinear_constraint,
                options={
                    "maxiter": 900,
                    "ftol": 2e-11,
                    "disp": False,
                },
            )

            if not np.all(np.isfinite(result.x)):
                continue

            # Always polish candidates under the actual unweighted objective.
            polish = minimize(
                lambda z: -np.sum(z[2 * n:]),
                result.x,
                jac=lambda z: np.concatenate((np.zeros(2 * n), -np.ones(n))),
                method="SLSQP",
                bounds=bounds,
                constraints=nonlinear_constraint,
                options={
                    "maxiter": 700,
                    "ftol": 5e-12,
                    "disp": False,
                },
            )
            z = polish.x if np.all(np.isfinite(polish.x)) else result.x
            c = z[:2 * n].reshape(n, 2)
            r = z[2 * n:]

            # Permit only candidates which are feasible up to ordinary solver
            # rounding; an exact safety projection is done below.
            if np.min(constraints(z)) >= -2e-7:
                value = float(np.sum(r))
                if value > best_value:
                    best_value = value
                    best_centers = c.copy()
                    best_radii = r.copy()
    except Exception:
        pass

    centers = np.clip(best_centers, 0.0, 1.0)
    radii = np.maximum(best_radii, 0.0)

    # Uniform shrinking preserves every packing inequality, and makes the
    # answer strictly valid even when a solver stopped at a contact boundary.
    border = np.minimum.reduce((
        centers[:, 0],
        centers[:, 1],
        1.0 - centers[:, 0],
        1.0 - centers[:, 1],
    ))
    scale = np.min(border / np.maximum(radii, 1e-15))

    dx = centers[ii, 0] - centers[jj, 0]
    dy = centers[ii, 1] - centers[jj, 1]
    distance = np.sqrt(dx * dx + dy * dy)
    pair_scale = distance / np.maximum(radii[ii] + radii[jj], 1e-15)
    scale = min(1.0, scale, float(np.min(pair_scale)))
    radii *= max(0.0, scale * (1.0 - 2e-10))

    return centers, radii, float(np.sum(radii))


def compute_max_radii(centers):
    """
    Compute conservative feasible radii for fixed centers.

    This helper remains part of the public interface.  It uses repeated pair
    projection, which is suitable for arbitrary supplied centers.
    """
    centers = np.asarray(centers, dtype=float)
    n = len(centers)
    radii = np.minimum.reduce((
        centers[:, 0],
        centers[:, 1],
        1.0 - centers[:, 0],
        1.0 - centers[:, 1],
    ))
    radii = np.maximum(radii, 0.0)

    for _ in range(5):
        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.linalg.norm(centers[i] - centers[j]))
                s = radii[i] + radii[j]
                if s > d and s > 0.0:
                    factor = d / s
                    radii[i] *= factor
                    radii[j] *= factor
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