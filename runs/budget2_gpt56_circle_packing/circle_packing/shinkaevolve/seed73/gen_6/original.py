# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """Construct a locally optimized, staggered hexagonal 26-circle packing."""
    n = 26

    # Six close-packed layers, with four circles removed from alternating
    # layers.  Unlike a rectangular grid this gives every interior circle
    # diagonal contacts, while still leaving the optimizer freedom to adapt
    # the outer layers to the square boundary.
    counts = (4, 5, 4, 5, 4, 4)
    seed = []
    for row, count in enumerate(counts):
        y = 0.125 + 0.15 * row
        x0 = 0.20 if count == 5 else 0.275
        for col in range(count):
            seed.append((x0 + 0.15 * col, y))
    centers0 = np.asarray(seed, dtype=float)
    radii0 = np.full(n, 0.070, dtype=float)

    try:
        from scipy.optimize import minimize

        # Variables are x_0,y_0,...,x_25,y_25,r_0,...,r_25.
        z0 = np.concatenate((centers0.ravel(), radii0))
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

        def constraints(z):
            c = z[:2 * n].reshape(n, 2)
            r = z[2 * n:]
            border = np.column_stack((
                c[:, 0] - r, c[:, 1] - r,
                1.0 - c[:, 0] - r, 1.0 - c[:, 1] - r
            )).ravel()
            sep = np.empty(len(pairs))
            for k, (i, j) in enumerate(pairs):
                d = c[i] - c[j]
                sep[k] = np.hypot(d[0], d[1]) - r[i] - r[j]
            return np.concatenate((border, sep))

        def constraint_jacobian(z):
            c = z[:2 * n].reshape(n, 2)
            jac = np.zeros((4 * n + len(pairs), 3 * n))
            for i in range(n):
                q = 4 * i
                jac[q, 2 * i] = 1.0
                jac[q, 2 * n + i] = -1.0
                jac[q + 1, 2 * i + 1] = 1.0
                jac[q + 1, 2 * n + i] = -1.0
                jac[q + 2, 2 * i] = -1.0
                jac[q + 2, 2 * n + i] = -1.0
                jac[q + 3, 2 * i + 1] = -1.0
                jac[q + 3, 2 * n + i] = -1.0
            for k, (i, j) in enumerate(pairs):
                d = c[i] - c[j]
                length = np.hypot(d[0], d[1])
                q = 4 * n + k
                if length > 1.e-12:
                    u = d / length
                    jac[q, 2 * i:2 * i + 2] = u
                    jac[q, 2 * j:2 * j + 2] = -u
                jac[q, 2 * n + i] = -1.0
                jac[q, 2 * n + j] = -1.0
            return jac

        result = minimize(
            lambda z: -np.sum(z[2 * n:]),
            z0,
            method="SLSQP",
            jac=lambda z: np.r_[np.zeros(2 * n), -np.ones(n)],
            bounds=[(0.0, 1.0)] * (2 * n) + [(1.e-6, 0.5)] * n,
            constraints={"type": "ineq", "fun": constraints, "jac": constraint_jacobian},
            options={"maxiter": 1800, "ftol": 1.e-11, "disp": False},
        )
        z = result.x if result.success and np.min(constraints(result.x)) >= -1.e-7 else z0
        centers = z[:2 * n].reshape(n, 2)
        radii = z[2 * n:] * (1.0 - 1.e-9)
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