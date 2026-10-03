# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """
    Construct 26 circles by relaxing a staggered, hexagonal-like layout.
    The variables are the 52 center coordinates and the 26 radii.
    """
    n = 26
    pairs = np.array([(i, j) for i in range(n) for j in range(i + 1, n)],
                     dtype=int)

    # Five staggered rows are a good interior seed, while the unequal row
    # lengths give the optimizer freedom to exploit the square boundary.
    row_sizes = (5, 6, 5, 6, 4)
    seed = []
    for row, count in enumerate(row_sizes):
        y = 0.12 + 0.19 * row
        if count == 6:
            xs = np.linspace(0.075, 0.925, count)
        elif count == 5:
            xs = np.linspace(0.14, 0.86, count)
        else:
            xs = np.linspace(0.16, 0.84, count)
        seed.extend((x, y) for x in xs)
    seed = np.asarray(seed, dtype=float)

    def constraints(z):
        c = z[:2 * n].reshape(n, 2)
        r = z[2 * n:]
        walls = np.column_stack((c[:, 0] - r, c[:, 1] - r,
                                 1.0 - c[:, 0] - r, 1.0 - c[:, 1] - r))
        d = c[pairs[:, 0]] - c[pairs[:, 1]]
        pair_gaps = np.einsum("ij,ij->i", d, d) - (r[pairs[:, 0]] +
                                                   r[pairs[:, 1]]) ** 2
        return np.concatenate((walls.ravel(), pair_gaps))

    def constraint_jacobian(z):
        c = z[:2 * n].reshape(n, 2)
        r = z[2 * n:]
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
        offset = 4 * n
        for k, (i, j) in enumerate(pairs):
            dx, dy = c[i] - c[j]
            s = r[i] + r[j]
            jac[offset + k, 2 * i:2 * i + 2] = (2.0 * dx, 2.0 * dy)
            jac[offset + k, 2 * j:2 * j + 2] = (-2.0 * dx, -2.0 * dy)
            jac[offset + k, 2 * n + i] = -2.0 * s
            jac[offset + k, 2 * n + j] = -2.0 * s
        return jac

    # If scipy is unavailable, the seed is still a valid conservative packing.
    try:
        from scipy.optimize import minimize
    except ImportError:
        radii = np.full(n, 0.06)
        return seed, radii, float(np.sum(radii))

    best = None
    # Deterministic perturbations avoid imposing unnecessary reflection symmetry.
    for phase in (0.0, 1.7, 3.4, 5.1):
        c0 = seed.copy()
        k = np.arange(n)
        c0[:, 0] += 0.012 * np.sin(1.91 * k + phase)
        c0[:, 1] += 0.009 * np.cos(1.37 * k + phase)
        z0 = np.concatenate((c0.ravel(), np.full(n, 0.045)))
        result = minimize(
            lambda z: -np.sum(z[2 * n:]), z0,
            jac=lambda z: np.r_[np.zeros(2 * n), -np.ones(n)],
            method="SLSQP",
            bounds=[(1.e-7, 1. - 1.e-7)] * (2 * n) +
                   [(1.e-8, 0.5)] * n,
            constraints={"type": "ineq", "fun": constraints,
                         "jac": constraint_jacobian},
            options={"maxiter": 700, "ftol": 1.e-11, "disp": False},
        )
        if np.all(np.isfinite(result.x)) and np.min(constraints(result.x)) >= -1.e-7:
            if best is None or np.sum(result.x[2 * n:]) > np.sum(best[2 * n:]):
                best = result.x

    if best is None:
        best = np.concatenate((seed.ravel(), np.full(n, 0.06)))

    centers = best[:2 * n].reshape(n, 2)
    radii = np.maximum(best[2 * n:], 0.0)

    # SLSQP's stopping tolerance can leave a tiny overlap.  Scaling every
    # radius by one common factor is guaranteed not to alter center validity.
    scale = 1.0
    for i in range(n):
        if radii[i] > 0.0:
            scale = min(scale, centers[i, 0] / radii[i],
                        centers[i, 1] / radii[i],
                        (1.0 - centers[i, 0]) / radii[i],
                        (1.0 - centers[i, 1]) / radii[i])
    for i, j in pairs:
        total = radii[i] + radii[j]
        if total > 0.0:
            scale = min(scale, np.linalg.norm(centers[i] - centers[j]) / total)
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