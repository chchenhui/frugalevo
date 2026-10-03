# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """Construct a multi-start optimized staggered packing of 26 circles."""
    n = 26

    # Nearby hexagonal layer patterns.  The differing row orders alter the
    # boundary contacts and give SLSQP several distinct local basins.
    patterns = (
        (4, 5, 4, 5, 4, 4),
        (4, 4, 5, 4, 5, 4),
        (5, 4, 5, 4, 4, 4),
        (4, 5, 4, 4, 5, 4),
    )

    def make_seed(counts, jitter=None):
        pts = []
        for row, count in enumerate(counts):
            y = 0.125 + 0.15 * row
            # Half-column staggering gives triangular contacts between rows.
            x0 = 0.20 if count == 5 else 0.275
            for col in range(count):
                pts.append((x0 + 0.15 * col, y))
        c = np.asarray(pts, dtype=float)
        if jitter is not None:
            c += jitter
            c = np.clip(c, 0.075, 0.925)
        return c

    centers0 = make_seed(patterns[0])
    radii0 = np.full(n, 0.068, dtype=float)

    try:
        from scipy.optimize import minimize

        ii, jj = np.triu_indices(n, 1)
        m = len(ii)
        rows = np.arange(n)

        def constraints(z):
            c = z[:2 * n].reshape(n, 2)
            r = z[2 * n:]
            border = np.column_stack((
                c[:, 0] - r,
                c[:, 1] - r,
                1.0 - c[:, 0] - r,
                1.0 - c[:, 1] - r,
            )).ravel()
            d = c[ii] - c[jj]
            sep = np.sqrt(np.sum(d * d, axis=1)) - r[ii] - r[jj]
            return np.concatenate((border, sep))

        def constraint_jacobian(z):
            c = z[:2 * n].reshape(n, 2)
            jac = np.zeros((4 * n + m, 3 * n), dtype=float)

            base = 4 * rows
            jac[base, 2 * rows] = 1.0
            jac[base, 2 * n + rows] = -1.0
            jac[base + 1, 2 * rows + 1] = 1.0
            jac[base + 1, 2 * n + rows] = -1.0
            jac[base + 2, 2 * rows] = -1.0
            jac[base + 2, 2 * n + rows] = -1.0
            jac[base + 3, 2 * rows + 1] = -1.0
            jac[base + 3, 2 * n + rows] = -1.0

            d = c[ii] - c[jj]
            length = np.sqrt(np.sum(d * d, axis=1))
            length = np.maximum(length, 1.0e-12)
            u = d / length[:, None]
            q = 4 * n + np.arange(m)
            jac[q, 2 * ii] = u[:, 0]
            jac[q, 2 * ii + 1] = u[:, 1]
            jac[q, 2 * jj] = -u[:, 0]
            jac[q, 2 * jj + 1] = -u[:, 1]
            jac[q, 2 * n + ii] = -1.0
            jac[q, 2 * n + jj] = -1.0
            return jac

        rng = np.random.default_rng(26031991)
        starts = []
        for k, pattern in enumerate(patterns):
            seed = make_seed(pattern)
            starts.append(seed)
            if k < 2:
                # Controlled asymmetric displacement can improve corner and
                # side fitting while retaining the useful hexagonal topology.
                starts.append(make_seed(
                    pattern, rng.normal(0.0, 0.0065, size=(n, 2))
                ))

        best_z = np.concatenate((centers0.ravel(), radii0))
        best_value = -np.inf
        bounds = [(0.0, 1.0)] * (2 * n) + [(1.0e-6, 0.5)] * n
        objective_jac = np.r_[np.zeros(2 * n), -np.ones(n)]

        for seed in starts:
            z0 = np.concatenate((seed.ravel(), radii0))
            result = minimize(
                lambda z: -np.sum(z[2 * n:]),
                z0,
                method="SLSQP",
                jac=lambda z: objective_jac,
                bounds=bounds,
                constraints={
                    "type": "ineq",
                    "fun": constraints,
                    "jac": constraint_jacobian,
                },
                options={"maxiter": 1100, "ftol": 2.0e-11, "disp": False},
            )
            candidate = result.x
            feasible = np.min(constraints(candidate)) >= -2.0e-7
            value = float(np.sum(candidate[2 * n:]))
            if feasible and value > best_value:
                best_z = candidate
                best_value = value

        centers = best_z[:2 * n].reshape(n, 2)
        # A tiny inward reduction makes tangencies robust under independent
        # evaluator floating-point checks.
        radii = best_z[2 * n:] * (1.0 - 2.0e-9)
    except ImportError:
        centers = centers0
        radii = radii0

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

    for i in range(n):
        x, y = centers[i]
        radii[i] = min(x, y, 1 - x, 1 - y)

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if radii[i] + radii[j] > dist:
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