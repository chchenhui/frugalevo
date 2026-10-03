# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """
    Construct 26 unequal circles by directly optimizing centers and radii.

    A staggered lattice is a useful feasible starting point, but the square
    boundary strongly favors unequal boundary circles.  Consequently the
    radii are optimization variables rather than being computed greedily
    after the centers have been fixed.
    """
    from scipy.optimize import minimize

    n = 26
    rows = (5, 5, 6, 5, 5)
    base_centers = []
    for row, count in enumerate(rows):
        # The six-circle middle row is shifted half a lattice step relative
        # to the five-circle rows, giving a triangular-lattice seed.
        xs = np.linspace(1.0 / (count + 1), count / (count + 1), count)
        y = 0.10 + 0.20 * row
        base_centers.extend((x, y) for x in xs)
    base_centers = np.asarray(base_centers, dtype=float)

    pi, pj = np.triu_indices(n, 1)

    def constraints(z):
        c = z[:2 * n].reshape(n, 2)
        r = z[2 * n:]
        dx = c[pi, 0] - c[pj, 0]
        dy = c[pi, 1] - c[pj, 1]
        pair_clearance = dx * dx + dy * dy - (r[pi] + r[pj]) ** 2
        wall_clearance = np.concatenate((
            c[:, 0] - r, c[:, 1] - r,
            1.0 - c[:, 0] - r, 1.0 - c[:, 1] - r
        ))
        return np.concatenate((wall_clearance, pair_clearance))

    def constraint_jacobian(z):
        c = z[:2 * n].reshape(n, 2)
        r = z[2 * n:]
        jac = np.zeros((4 * n + len(pi), 3 * n))
        for k in range(n):
            jac[k, 2 * k] = 1.0
            jac[k, 2 * n + k] = -1.0
            jac[n + k, 2 * k + 1] = 1.0
            jac[n + k, 2 * n + k] = -1.0
            jac[2 * n + k, 2 * k] = -1.0
            jac[2 * n + k, 2 * n + k] = -1.0
            jac[3 * n + k, 2 * k + 1] = -1.0
            jac[3 * n + k, 2 * n + k] = -1.0

        start = 4 * n
        dx = c[pi, 0] - c[pj, 0]
        dy = c[pi, 1] - c[pj, 1]
        rs = r[pi] + r[pj]
        q = np.arange(len(pi)) + start
        jac[q, 2 * pi] = 2.0 * dx
        jac[q, 2 * pi + 1] = 2.0 * dy
        jac[q, 2 * pj] = -2.0 * dx
        jac[q, 2 * pj + 1] = -2.0 * dy
        jac[q, 2 * n + pi] = -2.0 * rs
        jac[q, 2 * n + pj] = -2.0 * rs
        return jac

    bounds = [(0.0, 1.0)] * (2 * n) + [(1e-7, 0.5)] * n
    con = {"type": "ineq", "fun": constraints, "jac": constraint_jacobian}
    rng = np.random.default_rng(271828)

    # The contact graph at the best variable-radius packing is not usually
    # reflection symmetric.  Search several deterministic symmetry-breaking
    # perturbations, from gentle lattice changes to larger graph-changing
    # displacements.  Small unequal initial radii also avoid a tendency for
    # SLSQP to preserve the equal-radius contact graph of the seed.
    starts = [(base_centers, np.full(n, 0.055))]
    for amplitude in (0.004, 0.009, 0.015, 0.022, 0.030):
        for _ in range(2):
            seed_centers = base_centers + rng.normal(0.0, amplitude,
                                                     base_centers.shape)
            seed_centers = np.clip(seed_centers, 0.045, 0.955)
            seed_radii = 0.040 + rng.uniform(-0.012, 0.012, n)
            starts.append((seed_centers, seed_radii))

    def feasible_scale(z):
        """Largest common radius scale that makes an iterate valid."""
        c = z[:2 * n].reshape(n, 2)
        r = np.maximum(z[2 * n:], 0.0)
        scale = 1.0
        positive = r > 0.0
        if np.any(positive):
            wall = np.minimum.reduce((
                c[:, 0], c[:, 1], 1.0 - c[:, 0], 1.0 - c[:, 1]))
            scale = min(scale, float(np.min(wall[positive] / r[positive])))
        sums = r[pi] + r[pj]
        active = sums > 0.0
        if np.any(active):
            distances = np.hypot(c[pi, 0] - c[pj, 0],
                                 c[pi, 1] - c[pj, 1])
            scale = min(scale, float(np.min(distances[active] / sums[active])))
        return max(0.0, min(1.0, scale))

    best = None
    best_valid_sum = -np.inf
    for seed_centers, seed_radii in starts:
        z0 = np.concatenate((seed_centers.ravel(), seed_radii))
        result = minimize(
            lambda z: -np.sum(z[2 * n:]), z0,
            jac=lambda z: np.r_[np.zeros(2 * n), -np.ones(n)],
            method="SLSQP", bounds=bounds, constraints=con,
            options={"maxiter": 1400, "ftol": 1e-11, "disp": False}
        )
        candidate_sum = feasible_scale(result.x) * np.sum(
            np.maximum(result.x[2 * n:], 0.0))
        if candidate_sum > best_valid_sum:
            best = result.x
            best_valid_sum = candidate_sum

    centers = best[:2 * n].reshape(n, 2)
    radii = np.maximum(best[2 * n:], 0.0)

    # Optimizers legitimately stop within a tiny constraint tolerance.  This
    # common scale factor converts such a result into a strictly valid one.
    factors = [1.0]
    for i in range(n):
        if radii[i] > 0.0:
            factors.append(min(centers[i, 0], centers[i, 1],
                               1.0 - centers[i, 0], 1.0 - centers[i, 1])
                           / radii[i])
    for i, j in zip(pi, pj):
        denom = radii[i] + radii[j]
        if denom > 0.0:
            factors.append(np.hypot(*(centers[i] - centers[j])) / denom)
    radii *= max(0.0, min(1.0, min(factors) * (1.0 - 1e-10)))

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