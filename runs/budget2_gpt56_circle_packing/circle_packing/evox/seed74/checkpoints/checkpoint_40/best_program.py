# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """Build layered center layouts, then recover optimal feasible radii by LP."""
    from scipy.optimize import minimize, linprog

    n = 26
    ii, jj = np.triu_indices(n, 1)
    bounds = [(1e-5, 0.99999)] * (2 * n) + [(1e-5, 0.5)] * n

    # For fixed centers, every pair constraint is r_i + r_j <= distance.
    pair_matrix = np.zeros((len(ii), n))
    pair_matrix[np.arange(len(ii)), ii] = 1.0
    pair_matrix[np.arange(len(ii)), jj] = 1.0
    patterns = (
        (5, 6, 5, 6, 4), (4, 6, 6, 6, 4), (5, 5, 6, 5, 5),
        (5, 6, 6, 5, 4), (4, 5, 6, 6, 5), (5, 5, 5, 6, 5),
    )

    def unpack(z):
        return z[:2 * n].reshape(n, 2), z[2 * n:]

    def feasibility(z):
        c, r = unpack(z)
        boundary = np.concatenate((
            c[:, 0] - r, 1.0 - c[:, 0] - r,
            c[:, 1] - r, 1.0 - c[:, 1] - r,
        ))
        d = c[ii] - c[jj]
        separation = np.sum(d * d, axis=1) - (r[ii] + r[jj]) ** 2
        return np.concatenate((boundary, separation))

    def feasibility_jacobian(z):
        """Return exact derivatives of wall and squared pair-separation slacks."""
        c, r = unpack(z)
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

        d = c[ii] - c[jj]
        rs = r[ii] + r[jj]
        row = 4 * n + np.arange(len(ii))
        jac[row, 2 * ii] = 2.0 * d[:, 0]
        jac[row, 2 * ii + 1] = 2.0 * d[:, 1]
        jac[row, 2 * jj] = -2.0 * d[:, 0]
        jac[row, 2 * jj + 1] = -2.0 * d[:, 1]
        jac[row, 2 * n + ii] = -2.0 * rs
        jac[row, 2 * n + jj] = -2.0 * rs
        return jac

    def make_safe(z):
        """Fix centers, maximize radii by LP, then apply numerical clearance."""
        c, fallback = unpack(z.copy())
        c = np.clip(c, 1e-9, 1.0 - 1e-9)
        distance = np.sqrt(np.sum((c[ii] - c[jj]) ** 2, axis=1))

        # r_i <= each distance from center i to a wall.
        wall = np.minimum.reduce((
            c[:, 0], c[:, 1], 1.0 - c[:, 0], 1.0 - c[:, 1]
        ))

        lp = linprog(
            -np.ones(n), A_ub=pair_matrix, b_ub=distance,
            bounds=[(1e-10, float(w)) for w in wall],
            method="highs",
        )
        if lp.success and lp.x is not None:
            # Keep a numerical clearance while retaining essentially all LP value.
            return c, np.maximum(lp.x * (1.0 - 1e-9), 1e-10)

        # Rare solver-failure fallback: retain the prior conservative repair.
        r = np.maximum(fallback, 1e-8)
        border = np.minimum.reduce(c, axis=1) / r
        border = np.minimum(border, np.minimum.reduce(1.0 - c, axis=1) / r)
        scale = min(1.0, float(np.min(border)),
                    float(np.min(distance / (r[ii] + r[jj]))))
        return c, r * max(0.999999 * scale, 1e-10)

    best_sum = -np.inf
    best_centers = best_radii = None
    # Multiple row counts and phase offsets expose distinct boundary contact
    # graphs.  Unequal-radius optima can be slightly asymmetric.
    rich_patterns = patterns + (
        (6, 5, 5, 5, 5), (5, 5, 5, 5, 6), (4, 6, 5, 6, 5),
        (5, 4, 6, 6, 5), (6, 5, 4, 6, 5), (5, 6, 4, 5, 6),
    )
    objective_jacobian = np.concatenate((np.zeros(2 * n), -np.ones(n)))

    for trial in range(120):
        rng = np.random.default_rng(2609 + trial)
        phase = trial // len(rich_patterns)
        row_gap = 0.184 + 0.0025 * (phase % 9)
        horizontal_shift = 0.05 + 0.075 * (phase % 8)
        seed = []
        for row, count in enumerate(rich_patterns[trial % len(rich_patterns)]):
            y = 0.5 + (row - 2) * row_gap
            stagger = 0.5 if row % 2 else 0.0
            for col in range(count):
                seed.append(((col + 0.5 + horizontal_shift * stagger) / count, y))
        seed = np.asarray(seed, dtype=float)
        seed += rng.normal(0.0, 0.006 + 0.0012 * (trial % 10), seed.shape)
        if trial & 1:
            seed[:, 0] = 1.0 - seed[:, 0]
        seed = np.clip(seed, 0.025, 0.975)

        z0 = np.concatenate((seed.ravel(), np.full(n, 0.052)))
        result = minimize(
            lambda z: -np.sum(z[2 * n:]), z0,
            jac=lambda z: objective_jacobian,
            method="SLSQP", bounds=bounds,
            constraints={
                "type": "ineq",
                "fun": feasibility,
                "jac": feasibility_jacobian,
            },
            options={"maxiter": 2200, "ftol": 1e-11, "disp": False},
        )
        centers, radii = make_safe(result.x if result.x is not None else z0)
        total = float(np.sum(radii))
        if total > best_sum:
            best_centers, best_radii, best_sum = centers, radii, total

    # Polish the best discovered contact graph, including small deterministic
    # asymmetric perturbations, while retaining the incumbent monotonically.
    for polish in range(7):
        centers0 = best_centers.copy()
        if polish:
            rng = np.random.default_rng(90260 + polish)
            centers0 += rng.normal(0.0, 0.0015 * polish, centers0.shape)
            centers0 = np.clip(centers0, 1e-5, 1.0 - 1e-5)

        z0 = np.concatenate((centers0.ravel(), best_radii))
        result = minimize(
            lambda z: -np.sum(z[2 * n:]), z0,
            jac=lambda z: objective_jacobian,
            method="SLSQP", bounds=bounds,
            constraints={
                "type": "ineq",
                "fun": feasibility,
                "jac": feasibility_jacobian,
            },
            options={"maxiter": 2600, "ftol": 3e-12, "disp": False},
        )
        centers, radii = make_safe(result.x if result.x is not None else z0)
        total = float(np.sum(radii))
        if total > best_sum:
            best_centers, best_radii, best_sum = centers, radii, total

    return best_centers, best_radii, best_sum


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
