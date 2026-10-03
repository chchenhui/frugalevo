# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """Polish several deterministic staggered-lattice seeds and return the best certified packing."""
    n = 26
    ii, jj = np.triu_indices(n, 1)

    def lattice(rows, radius, stretch=1.0):
        """Create a centered alternating triangular lattice with prescribed row populations."""
        points = []
        height = np.sqrt(3.0) * radius * (len(rows) - 1)
        y0 = 0.5 - 0.5 * height * stretch
        for row, count in enumerate(rows):
            y = y0 + row * np.sqrt(3.0) * radius * stretch
            start = 0.5 - (count - 1) * radius - (radius if row & 1 else 0.0)
            points.extend((start + 2.0 * radius * k, y) for k in range(count))
        return np.asarray(points, dtype=float)

    def make_feasible(z):
        """Clip centers and uniformly contract all radii until every packing inequality is strict."""
        p = np.clip(np.asarray(z[:2 * n], dtype=float).reshape(n, 2), 0.0, 1.0)
        r = np.maximum(np.asarray(z[2 * n:], dtype=float), 0.0)
        cap = np.min(np.column_stack((p[:, 0], p[:, 1], 1.0 - p[:, 0], 1.0 - p[:, 1])) /
                     np.maximum(r[:, None], 1e-300))
        d = np.sqrt(np.sum((p[ii] - p[jj]) ** 2, axis=1))
        cap = min(cap, np.min(d / np.maximum(r[ii] + r[jj], 1e-300)))
        r *= max(0.0, min(1.0, cap * (1.0 - 1e-9)))
        return p, r

    # Every choice of the two long rows has a different set of boundary
    # contacts.  These 15 arrangements are not equivalent during local
    # optimization because the square breaks the lattice's vertical symmetry.
    specifications = [
        ((5, 5, 4, 4, 4, 4), 0.0890, 1.00),
        ((5, 4, 5, 4, 4, 4), 0.0890, 1.00),
        ((5, 4, 4, 5, 4, 4), 0.0890, 1.00),
        ((5, 4, 4, 4, 5, 4), 0.0890, 1.00),
        ((5, 4, 4, 4, 4, 5), 0.0890, 1.00),
        ((4, 5, 5, 4, 4, 4), 0.0890, 1.00),
        ((4, 5, 4, 5, 4, 4), 0.0890, 0.96),
        ((4, 5, 4, 5, 4, 4), 0.0890, 1.00),
        ((4, 5, 4, 5, 4, 4), 0.0890, 1.04),
        ((4, 5, 4, 4, 5, 4), 0.0890, 1.00),
        ((4, 5, 4, 4, 4, 5), 0.0890, 1.00),
        ((4, 4, 5, 5, 4, 4), 0.0890, 1.00),
        ((4, 4, 5, 4, 5, 4), 0.0890, 1.00),
        ((4, 4, 5, 4, 4, 5), 0.0890, 1.00),
        ((4, 4, 4, 5, 5, 4), 0.0890, 1.00),
        ((4, 4, 4, 5, 4, 5), 0.0890, 1.00),
        ((4, 4, 4, 4, 5, 5), 0.0890, 1.00),
    ]
    rng = np.random.default_rng(260319)
    starts = []
    for number, (rows, radius, stretch) in enumerate(specifications):
        p = lattice(rows, radius, stretch)
        starts.append(np.r_[p.ravel(), np.full(n, radius)])

        # A moderately displaced start can leave the symmetric lattice basin;
        # feasibility projection retains a well-scaled initial radius vector.
        scale = 0.018 if number % 2 else 0.011
        q = np.clip(p + rng.normal(0.0, scale, size=p.shape), 0.012, 0.988)
        q, rr = make_feasible(np.r_[q.ravel(), np.full(n, radius)])
        starts.append(np.r_[q.ravel(), rr])

    best_p, best_r = make_feasible(starts[0])
    best_value = float(np.sum(best_r))

    try:
        from scipy.optimize import minimize

        def slack(z):
            """Return border and pairwise non-overlap slacks for SLSQP."""
            p = z[:2 * n].reshape(n, 2)
            r = z[2 * n:]
            border = np.column_stack((
                p[:, 0] - r, p[:, 1] - r,
                1.0 - p[:, 0] - r, 1.0 - p[:, 1] - r
            )).ravel()
            d = np.sqrt(np.sum((p[ii] - p[jj]) ** 2, axis=1))
            return np.r_[border, d - r[ii] - r[jj]]

        def slack_jacobian(z):
            """Return the exact Jacobian of border and pairwise distance slacks."""
            p = z[:2 * n].reshape(n, 2)
            m = len(ii)
            jac = np.zeros((4 * n + m, 3 * n), dtype=float)

            for k in range(n):
                row = 4 * k
                jac[row, 2 * k] = 1.0
                jac[row, 2 * n + k] = -1.0
                jac[row + 1, 2 * k + 1] = 1.0
                jac[row + 1, 2 * n + k] = -1.0
                jac[row + 2, 2 * k] = -1.0
                jac[row + 2, 2 * n + k] = -1.0
                jac[row + 3, 2 * k + 1] = -1.0
                jac[row + 3, 2 * n + k] = -1.0

            delta = p[ii] - p[jj]
            distance = np.sqrt(np.sum(delta * delta, axis=1))
            unit = delta / np.maximum(distance[:, None], 1e-14)
            rows = 4 * n + np.arange(m)
            jac[rows, 2 * ii] = unit[:, 0]
            jac[rows, 2 * ii + 1] = unit[:, 1]
            jac[rows, 2 * jj] = -unit[:, 0]
            jac[rows, 2 * jj + 1] = -unit[:, 1]
            jac[rows, 2 * n + ii] = -1.0
            jac[rows, 2 * n + jj] = -1.0
            return jac

        objective_jacobian = np.r_[np.zeros(2 * n), -np.ones(n)]
        bounds = [(0.0, 1.0)] * (2 * n) + [(1e-7, 0.25)] * n
        constraints = {"type": "ineq", "fun": slack, "jac": slack_jacobian}
        for z0 in starts:
            result = minimize(
                lambda z: -np.sum(z[2 * n:]), z0,
                jac=lambda z: objective_jacobian,
                method="SLSQP", bounds=bounds, constraints=constraints,
                options={"maxiter": 700, "ftol": 3e-12, "disp": False}
            )
            if np.all(np.isfinite(result.x)):
                p, r = make_feasible(result.x)
                value = float(np.sum(r))
                if value > best_value:
                    best_p, best_r, best_value = p, r, value
    except Exception:
        pass

    return best_p, best_r, best_value


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
