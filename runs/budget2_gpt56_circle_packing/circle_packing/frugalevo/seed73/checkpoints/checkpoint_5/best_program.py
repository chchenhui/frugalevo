# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """Optimize several staggered five-layer seeds and return the best feasible packing."""
    # Multiple deterministic seeds expose SLSQP to different contact graphs.
    # The six-circle row is tried in both an outer and a central layer, while
    # alternating offsets provide hexagonal diagonal contacts.
    n = 26
    seeds = []

    # Use several different layer spacings and six-circle-row locations.
    # These seeds deliberately cover both rectangular and hexagonal-like
    # contact graphs, which are substantially different local optima for
    # the nonlinear optimizer.
    y_patterns = (
        (0.102, 0.299, 0.496, 0.693, 0.890),
        (0.105, 0.300, 0.495, 0.690, 0.895),
        (0.100, 0.296, 0.492, 0.688, 0.884),
        (0.108, 0.303, 0.498, 0.693, 0.888),
    )
    for y_values in y_patterns:
        for six_row in (1, 2, 3):
            for phase in (0, 1):
                centers = []
                radii = []
                for row, y in enumerate(y_values):
                    if row == six_row:
                        xs = np.linspace(1.0 / 12.0, 11.0 / 12.0, 6)
                        rr = [0.080] * 6
                    else:
                        # Slightly inset rows leave room for diagonal
                        # contacts with the narrower six-circle layer.
                        if phase:
                            left, right = (0.097, 0.903) if row % 2 else (0.103, 0.897)
                        else:
                            left, right = (0.103, 0.897) if row % 2 else (0.097, 0.903)
                        xs = np.linspace(left, right, 5)
                        rr = [0.097] * 5
                    centers.extend((float(x), float(y)) for x in xs)
                    radii.extend(rr)
                seeds.append((np.asarray(centers, dtype=float),
                              np.asarray(radii, dtype=float)))

    best_centers, best_radii = seeds[0]
    best_sum = -np.inf

    try:
        from scipy.optimize import minimize

        def objective(z):
            return -np.sum(z[2 * n:])

        def inequalities(z):
            c = z[:2 * n].reshape(n, 2)
            r = z[2 * n:]
            values = [
                c[:, 0] - r, c[:, 1] - r,
                1.0 - c[:, 0] - r, 1.0 - c[:, 1] - r
            ]
            for i in range(n):
                d = c[i + 1:] - c[i]
                values.append(np.sum(d * d, axis=1) -
                              (r[i] + r[i + 1:]) ** 2)
            return np.concatenate([np.asarray(v).ravel() for v in values])

        bounds = [(0.0, 1.0)] * (2 * n) + [(0.005, 0.2)] * n
        for seed_centers, seed_radii in seeds:
            z0 = np.concatenate((seed_centers.ravel(), seed_radii))
            result = minimize(
                objective, z0, method="SLSQP", bounds=bounds,
                constraints={"type": "ineq", "fun": inequalities},
                options={"maxiter": 600, "ftol": 2e-10, "disp": False}
            )
            if np.all(np.isfinite(result.x)):
                c = result.x[:2 * n].reshape(n, 2)
                r = result.x[2 * n:]
                if np.min(inequalities(result.x)) >= -2e-7:
                    total = float(np.sum(r))
                    if total > best_sum:
                        best_centers, best_radii, best_sum = c, r, total
    except Exception:
        pass

    # Conservative scaling removes tiny floating-point contact violations.
    best_radii *= 0.999995
    return best_centers, best_radii, float(np.sum(best_radii))


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
