# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """
    Optimize centers and unequal radii for 26 circles in the unit square.

    A layered near-hexagonal arrangement supplies good initial geometry.
    SLSQP then jointly moves every center and radius.  Explicit derivatives
    make repeated local searches substantially cheaper than finite-difference
    constrained optimization.
    """
    from scipy.optimize import minimize, linprog

    n = 26
    nc = 2 * n
    pair_i, pair_j = np.triu_indices(n, 1)
    m_pairs = len(pair_i)
    m_constraints = 4 * n + m_pairs

    # The fixed-center radius problem is linear: each pair contributes
    # r_i + r_j <= distance(i, j).  Build this matrix once for all starts.
    pair_matrix = np.zeros((m_pairs, n), dtype=float)
    pair_rows = np.arange(m_pairs)
    pair_matrix[pair_rows, pair_i] = 1.0
    pair_matrix[pair_rows, pair_j] = 1.0

    # Five horizontal layers are appropriate for 26 disks: the central
    # six-circle layer absorbs the otherwise unavoidable boundary mismatch.
    row_counts = (5, 5, 6, 5, 5)
    rows = []
    for row, count in enumerate(row_counts):
        y = 0.11 + 0.195 * row
        if count == 6:
            xs = np.linspace(0.10, 0.90, count)
        else:
            xs = np.linspace(0.13, 0.87, count)
        rows.extend((x, y) for x in xs)
    base = np.asarray(rows, dtype=float)
    row_index = np.repeat(np.arange(5), row_counts)

    # These retain the useful five-layer population while presenting SLSQP
    # with distinct initial contact graphs near the left and right boundaries.
    staggered = base.copy()
    staggered[:, 0] += np.where(row_index % 2, 0.040, -0.040)

    asymmetric = base.copy()
    asymmetric[:, 0] += np.array(
        [-0.028, 0.045, -0.010, 0.038, -0.022], dtype=float
    )[row_index]

    seed_patterns = (base, staggered, asymmetric)

    def inequalities(z):
        c = z[:nc].reshape(n, 2)
        r = z[nc:]
        walls = np.column_stack((
            c[:, 0] - r,
            c[:, 1] - r,
            1.0 - c[:, 0] - r,
            1.0 - c[:, 1] - r
        )).ravel()

        delta = c[pair_i] - c[pair_j]
        sep = np.einsum("ij,ij->i", delta, delta)
        sep -= (r[pair_i] + r[pair_j]) ** 2
        return np.concatenate((walls, sep))

    def inequality_jacobian(z):
        """Analytic Jacobian of walls and squared-distance constraints."""
        c = z[:nc].reshape(n, 2)
        r = z[nc:]
        jac = np.zeros((m_constraints, 3 * n), dtype=float)

        # Per-circle wall rows, in exactly the order used above.
        for k in range(n):
            row = 4 * k
            xcol, ycol, rcol = 2 * k, 2 * k + 1, nc + k

            jac[row, xcol] = 1.0
            jac[row, rcol] = -1.0

            jac[row + 1, ycol] = 1.0
            jac[row + 1, rcol] = -1.0

            jac[row + 2, xcol] = -1.0
            jac[row + 2, rcol] = -1.0

            jac[row + 3, ycol] = -1.0
            jac[row + 3, rcol] = -1.0

        delta = c[pair_i] - c[pair_j]
        rs = r[pair_i] + r[pair_j]
        q = np.arange(m_pairs) + 4 * n

        jac[q, 2 * pair_i] = 2.0 * delta[:, 0]
        jac[q, 2 * pair_i + 1] = 2.0 * delta[:, 1]
        jac[q, 2 * pair_j] = -2.0 * delta[:, 0]
        jac[q, 2 * pair_j + 1] = -2.0 * delta[:, 1]
        jac[q, nc + pair_i] = -2.0 * rs
        jac[q, nc + pair_j] = -2.0 * rs
        return jac

    def objective(z):
        return -float(np.sum(z[nc:]))

    def objective_jacobian(z):
        grad = np.zeros(3 * n, dtype=float)
        grad[nc:] = -1.0
        return grad

    bounds = [(0.00001, 0.99999)] * nc + [(0.000001, 0.5)] * n
    constraint = {
        "type": "ineq",
        "fun": inequalities,
        "jac": inequality_jacobian,
    }

    def lp_polish(centers, fallback):
        """Maximize total radius exactly after center coordinates are fixed."""
        wall_limit = np.minimum.reduce((
            centers[:, 0],
            centers[:, 1],
            1.0 - centers[:, 0],
            1.0 - centers[:, 1],
        ))
        distances = np.sqrt(np.sum(
            (centers[pair_i] - centers[pair_j]) ** 2, axis=1
        ))
        polished = linprog(
            c=-np.ones(n),
            A_ub=pair_matrix,
            b_ub=distances,
            bounds=[(0.0, max(0.0, float(w))) for w in wall_limit],
            method="highs",
        )
        if polished.success and polished.x is not None:
            return np.maximum(polished.x, 0.0)
        return fallback.copy()

    rng = np.random.default_rng(26031991)
    best = None
    best_radii = None
    best_value = -np.inf

    # The three clean starts have intentionally different boundary contact
    # graphs.  Remaining starts perturb them asymmetrically to release the
    # artificial row symmetry of the initial construction.
    for trial in range(24):
        centers0 = seed_patterns[trial % len(seed_patterns)].copy()

        if trial >= 3:
            stagger = (row_index % 2) * (0.009 if trial % 3 else -0.009)
            centers0[:, 0] += stagger

            noise = 0.010 + 0.0025 * (trial % 4)
            centers0 += rng.normal(0.0, noise, centers0.shape)

            # A few starts deliberately bias the middle and outer layers in
            # opposite directions, preserving a near-hexagonal topology.
            if trial % 4 == 0:
                centers0[row_index == 2, 0] += 0.012
            elif trial % 4 == 1:
                centers0[row_index != 2, 0] -= 0.007

            centers0 = np.clip(centers0, 0.045, 0.955)

        z0 = np.concatenate((centers0.ravel(), np.full(n, 0.024)))
        result = minimize(
            objective,
            z0,
            jac=objective_jacobian,
            method="SLSQP",
            bounds=bounds,
            constraints=constraint,
            options={"maxiter": 1500, "ftol": 2e-12, "disp": False},
        )

        if np.all(np.isfinite(result.x)):
            violation = np.min(inequalities(result.x))
            candidate_centers = result.x[:nc].reshape(n, 2)
            candidate_radii = lp_polish(candidate_centers, result.x[nc:])

            # LP polishing can recover radius allocation that was lost when
            # SLSQP terminated at a center/radius compromise.
            value = float(np.sum(candidate_radii))
            if violation >= -3e-7 and value > best_value:
                best = result.x.copy()
                best_radii = candidate_radii.copy()
                best_value = value

    # A feasible layered fallback is retained for environments where an
    # optimizer is unavailable or numerical optimization unexpectedly fails.
    if best is None:
        best = np.concatenate((base.ravel(), np.full(n, 0.024)))
        best_radii = lp_polish(base, best[nc:])

    centers = best[:nc].reshape(n, 2).copy()
    radii = best_radii.copy()

    # One global scale factor is a simple final feasibility certificate:
    # it satisfies all wall and pair constraints simultaneously.
    wall_limit = np.minimum.reduce((
        centers[:, 0],
        centers[:, 1],
        1.0 - centers[:, 0],
        1.0 - centers[:, 1],
    ))
    scale = float(np.min(wall_limit / radii))

    distances = np.sqrt(np.sum(
        (centers[pair_i] - centers[pair_j]) ** 2, axis=1
    ))
    pair_scale = np.min(distances / (radii[pair_i] + radii[pair_j]))
    scale = min(scale, float(pair_scale))
    radii *= min(1.0, 0.999999 * scale)

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