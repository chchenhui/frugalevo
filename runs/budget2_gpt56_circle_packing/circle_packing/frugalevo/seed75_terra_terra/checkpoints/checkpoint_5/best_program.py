# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """
    Optimize a 4-by-4/interstitial packing with exact analytic clearance
    derivatives, using wide asymmetric restarts and incumbent polishing.
    """
    from scipy.optimize import minimize

    n = 26
    nv = 3 * n
    pairs = np.array([(i, j) for i in range(n) for j in range(i + 1, n)])
    ii, jj = pairs[:, 0], pairs[:, 1]

    # This seed has large circles on a square core, smaller circles in its
    # diagonal voids, and one boundary-layer circle.  It is a useful starting
    # topology, but the search is deliberately allowed to break all symmetry.
    large = np.array(
        [[.125 + .25 * x, .125 + .25 * y]
         for y in range(4) for x in range(4)], dtype=float)
    inner = np.array(
        [[.25 * x, .25 * y] for y in range(1, 4) for x in range(1, 4)],
        dtype=float)
    centers0 = np.vstack((large, inner, [[.25, .03125]]))
    radii0 = np.r_[np.full(16, .125),
                   np.full(9, np.sqrt(2.0) / 8.0 - .125), .03125]
    base = np.r_[centers0.ravel(), radii0]

    def clearance(z):
        """Return all border and circle-pair nonoverlap clearances."""
        c = z[:2 * n].reshape(n, 2)
        r = z[2 * n:]
        d = np.sqrt(np.sum((c[ii] - c[jj]) ** 2, axis=1))
        return np.r_[c[:, 0] - r, 1.0 - c[:, 0] - r,
                     c[:, 1] - r, 1.0 - c[:, 1] - r,
                     d - r[ii] - r[jj]]

    def clearance_jacobian(z):
        """Return the exact Jacobian of border and pairwise clearances."""
        c = z[:2 * n].reshape(n, 2)
        jac = np.zeros((4 * n + len(ii), nv))
        k = np.arange(n)
        jac[k, 2 * k] = 1.0
        jac[k, 2 * n + k] = -1.0
        jac[n + k, 2 * k] = -1.0
        jac[n + k, 2 * n + k] = -1.0
        jac[2 * n + k, 2 * k + 1] = 1.0
        jac[2 * n + k, 2 * n + k] = -1.0
        jac[3 * n + k, 2 * k + 1] = -1.0
        jac[3 * n + k, 2 * n + k] = -1.0

        delta = c[ii] - c[jj]
        d = np.sqrt(np.sum(delta * delta, axis=1))
        unit = delta / np.maximum(d[:, None], 1e-14)
        row = 4 * n + np.arange(len(ii))
        jac[row, 2 * ii] = unit[:, 0]
        jac[row, 2 * ii + 1] = unit[:, 1]
        jac[row, 2 * jj] = -unit[:, 0]
        jac[row, 2 * jj + 1] = -unit[:, 1]
        jac[row, 2 * n + ii] = -1.0
        jac[row, 2 * n + jj] = -1.0
        return jac

    def make_valid(z):
        """Clip centers and uniformly shrink radii to obtain a safe packing."""
        c = np.clip(z[:2 * n].reshape(n, 2), 0.0, 1.0).copy()
        r = np.maximum(z[2 * n:].copy(), 1e-9)
        edge = np.minimum.reduce(
            (c[:, 0], c[:, 1], 1.0 - c[:, 0], 1.0 - c[:, 1]))
        scale = min(1.0, np.min(edge / r))
        d = np.sqrt(np.sum((c[ii] - c[jj]) ** 2, axis=1))
        scale = min(scale, np.min(d / (r[ii] + r[jj])))
        return c, r * max(0.0, scale * (1.0 - 2e-10))

    best_c, best_r = make_valid(base)
    best_value = float(best_r.sum())
    rng = np.random.default_rng(26031991)
    bounds = [(0.0, 1.0)] * (2 * n) + [(1e-9, 0.25)] * n
    objective_jac = np.r_[np.zeros(2 * n), -np.ones(n)]
    constraint = {"type": "ineq", "fun": clearance, "jac": clearance_jacobian}

    # Supplying derivatives removes SLSQP's expensive 79-point finite
    # differences.  The saved budget is used for substantially wider starts,
    # which are important because the best arrangement is slightly asymmetric.
    # Analytic derivatives make restarts cheap.  Spend most of the budget on
    # the productive asymmetric square-core basin, then repeatedly perturb the
    # best incumbent so that different contact graphs can emerge.
    for trial in range(320):
        if trial == 0:
            z0 = base.copy()
        elif trial < 120:
            z0 = base.copy()
            sigma = (.008, .018, .035, .060, .085)[trial % 5]
            z0[:2 * n] += rng.normal(0.0, sigma, 2 * n)
            z0[:2 * n] = np.clip(z0[:2 * n], .0005, .9995)
            z0[2 * n:] *= .55 + .08 * (trial % 4)
        elif trial < 270:
            z0 = np.r_[best_c.ravel(), best_r * (.70 + .06 * (trial % 4))]
            sigma = (.003, .008, .018, .035)[trial % 4]
            z0[:2 * n] += rng.normal(0.0, sigma, 2 * n)
            z0[:2 * n] = np.clip(z0[:2 * n], .0005, .9995)
        else:
            # A triangular-lattice phase supplies a genuinely different basin.
            dy = np.sqrt(3.0) / 10.0
            rows = [5, 4, 5, 4, 4, 4]
            pts = []
            for row, count in enumerate(rows):
                offset = .1 if count == 5 else .2
                pts.extend((offset + .2 * col, .07 + row * dy)
                           for col in range(count))
            z0 = np.r_[np.asarray(pts[:n]).ravel(), np.full(n, .045)]
            z0[:2 * n] += rng.normal(0.0, .018, 2 * n)
            z0[:2 * n] = np.clip(z0[:2 * n], .0005, .9995)

        c0, r0 = make_valid(z0)
        result = minimize(
            lambda z: -np.sum(z[2 * n:]), np.r_[c0.ravel(), r0],
            jac=lambda z: objective_jac, method="SLSQP", bounds=bounds,
            constraints=constraint,
            options={"maxiter": 900, "ftol": 2e-12, "disp": False},
        )
        c, r = make_valid(result.x)
        value = float(r.sum())
        if value > best_value:
            best_c, best_r, best_value = c, r, value

    return best_c, best_r, best_value


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
