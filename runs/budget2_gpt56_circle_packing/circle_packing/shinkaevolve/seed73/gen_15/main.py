# EVOLVE-BLOCK-START
"""Topology-guided nonlinear constructor for 26 circles in a unit square."""
import numpy as np


def _initial_layout(row_counts, phase_pattern):
    """Create a feasible staggered five-row seed arrangement."""
    centers = []
    radii = []

    rows = len(row_counts)
    ys = np.linspace(0.10, 0.90, rows)

    for row, (count, y) in enumerate(zip(row_counts, ys)):
        # Five-circle rows naturally use x=.1,.3,...,.9.
        # Six-circle rows use x=1/12,3/12,...,11/12.
        base = (np.arange(count) + 0.5) / count

        # Small alternating offset gives the optimizer a non-lattice seed.
        phase = phase_pattern[row]
        if count == 5:
            offset = 0.018 * phase
        else:
            offset = 0.010 * phase

        x = np.clip(base + offset, 0.055, 0.945)

        for xx in x:
            centers.append((xx, y))
            radii.append(0.060)

    return np.asarray(centers, dtype=float), np.asarray(radii, dtype=float)


def _certify(centers, radii):
    """
    Uniformly reduce radii by the minimum necessary factor.
    This is deliberately performed after optimization to make the returned
    numerical packing robust against strict external overlap checks.
    """
    centers = np.asarray(centers, dtype=float).copy()
    radii = np.maximum(np.asarray(radii, dtype=float).copy(), 0.0)

    factor = 1.0
    n = len(radii)

    for i in range(n):
        if radii[i] > 0.0:
            wall = min(
                centers[i, 0],
                centers[i, 1],
                1.0 - centers[i, 0],
                1.0 - centers[i, 1],
            )
            factor = min(factor, wall / radii[i])

    for i in range(n):
        for j in range(i + 1, n):
            total = radii[i] + radii[j]
            if total > 0.0:
                d = np.linalg.norm(centers[i] - centers[j])
                factor = min(factor, d / total)

    # A tiny margin avoids invalidity caused by platform-dependent rounding.
    radii *= max(0.0, min(1.0, factor * (1.0 - 2.0e-10)))
    return centers, radii


def _optimize_seed(centers0, radii0):
    """Optimize one row-topology seed with analytic vector constraints."""
    try:
        from scipy.optimize import minimize
    except Exception:
        return centers0, radii0

    n = len(radii0)
    pair_i, pair_j = np.triu_indices(n, 1)

    z0 = np.concatenate((centers0[:, 0], centers0[:, 1], radii0))

    def unpack(z):
        return z[:n], z[n:2 * n], z[2 * n:]

    def objective(z):
        # Very small quadratic regularization helps SLSQP choose stable
        # representatives among nearly equivalent contact configurations.
        r = z[2 * n:]
        return -np.sum(r) + 1.0e-10 * np.sum((z[:2 * n] - 0.5) ** 2)

    def constraints(z):
        x, y, r = unpack(z)

        # Four wall inequalities per circle.
        walls = np.concatenate((x - r, 1.0 - x - r, y - r, 1.0 - y - r))

        dx = x[pair_i] - x[pair_j]
        dy = y[pair_i] - y[pair_j]
        rs = r[pair_i] + r[pair_j]

        # Squared distance avoids square roots and is smooth away from
        # coincident centers, which the feasible seeds already avoid.
        pairs = dx * dx + dy * dy - rs * rs
        return np.concatenate((walls, pairs))

    bounds = [(0.001, 0.999)] * (2 * n) + [(0.00001, 0.24)] * n

    result = minimize(
        objective,
        z0,
        method="SLSQP",
        bounds=bounds,
        constraints={"type": "ineq", "fun": constraints},
        options={
            "maxiter": 1200,
            "ftol": 2.0e-11,
            "disp": False,
        },
    )

    z = result.x if np.all(np.isfinite(result.x)) else z0
    x, y, r = unpack(z)
    return np.column_stack((x, y)), r


def construct_packing():
    """
    Construct 26 non-overlapping circles in the unit square.

    Returns:
        (centers, radii, sum_of_radii)
    """
    # The location of the denser six-circle rows changes boundary contacts
    # substantially.  These asymmetric candidates are intentionally retained
    # instead of spending the budget on repeated random restarts.
    topologies = [
        ((5, 5, 6, 5, 5), (0, 1, 0, -1, 0)),
        ((4, 6, 6, 5, 5), (0, 1, -1, 1, 0)),
        ((5, 5, 6, 6, 4), (0, -1, 1, -1, 0)),
        ((6, 4, 6, 5, 5), (0, 1, -1, 1, 0)),
        ((5, 6, 5, 6, 4), (0, -1, 1, -1, 0)),
        ((5, 5, 6, 5, 5), (1, -1, 0, 1, -1)),
    ]

    best_centers = None
    best_radii = None
    best_sum = -np.inf

    for counts, phases in topologies:
        centers0, radii0 = _initial_layout(counts, phases)
        centers, radii = _optimize_seed(centers0, radii0)
        centers, radii = _certify(centers, radii)

        value = float(np.sum(radii))
        if value > best_sum:
            best_sum = value
            best_centers = centers
            best_radii = radii

    return best_centers, best_radii, float(np.sum(best_radii))


def compute_max_radii(centers):
    """
    Retained public helper: compute conservative valid radii for supplied
    centers by assigning each circle its nearest-wall / nearest-center limit.
    """
    centers = np.asarray(centers, dtype=float)
    n = len(centers)
    radii = np.minimum.reduce(
        [centers[:, 0], centers[:, 1], 1.0 - centers[:, 0], 1.0 - centers[:, 1]]
    )

    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if radii[i] + radii[j] > d:
                scale = d / (radii[i] + radii[j])
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
