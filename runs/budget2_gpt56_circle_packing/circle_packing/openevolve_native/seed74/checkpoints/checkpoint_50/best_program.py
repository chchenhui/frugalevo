# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """Optimize layered 5/6/5/5/5 seeds and select the largest uniformly certified result."""
    from scipy.optimize import minimize

    n = 26
    ii, jj = np.triu_indices(n, 1)
    m = len(ii)
    rng = np.random.default_rng(91726)

    def constraints(z):
        """Return boundary and pairwise non-overlap slack values."""
        c = z[:2 * n].reshape(n, 2)
        r = z[2 * n:]
        d = c[ii] - c[jj]
        pair = np.sum(d * d, axis=1) - (r[ii] + r[jj]) ** 2
        return np.concatenate((
            c[:, 0] - r, 1.0 - c[:, 0] - r,
            c[:, 1] - r, 1.0 - c[:, 1] - r, pair
        ))

    def constraint_jacobian(z):
        """Return analytic derivatives of boundary and pairwise slacks."""
        c = z[:2 * n].reshape(n, 2)
        r = z[2 * n:]
        jac = np.zeros((4 * n + m, 3 * n))
        for k in range(n):
            xcol, ycol, rcol = 2 * k, 2 * k + 1, 2 * n + k
            jac[k, xcol], jac[k, rcol] = 1.0, -1.0
            jac[n + k, xcol], jac[n + k, rcol] = -1.0, -1.0
            jac[2 * n + k, ycol], jac[2 * n + k, rcol] = 1.0, -1.0
            jac[3 * n + k, ycol], jac[3 * n + k, rcol] = -1.0, -1.0

        row = 4 * n
        for k, (a, b) in enumerate(zip(ii, jj)):
            dx, dy = c[a] - c[b]
            s = r[a] + r[b]
            jac[row + k, 2 * a] = 2.0 * dx
            jac[row + k, 2 * a + 1] = 2.0 * dy
            jac[row + k, 2 * b] = -2.0 * dx
            jac[row + k, 2 * b + 1] = -2.0 * dy
            jac[row + k, 2 * n + a] = -2.0 * s
            jac[row + k, 2 * n + b] = -2.0 * s
        return jac

    def certified_scale(z):
        """Return the largest common radius multiplier feasible at the given centers."""
        c = z[:2 * n].reshape(n, 2)
        r = np.maximum(z[2 * n:], 0.0)
        scale = 1.0
        for k in range(n):
            if r[k] > 0:
                scale = min(scale, c[k, 0] / r[k], c[k, 1] / r[k],
                            (1.0 - c[k, 0]) / r[k],
                            (1.0 - c[k, 1]) / r[k])
        d = c[ii] - c[jj]
        distances = np.sqrt(np.sum(d * d, axis=1))
        sums = r[ii] + r[jj]
        active = sums > 0
        if np.any(active):
            scale = min(scale, np.min(distances[active] / sums[active]))
        return min(1.0, max(0.0, scale))

    best = None
    best_value = -np.inf
    # Cover each possible six-circle row, then perturb the best incumbent.
    # This compact restart schedule reaches the same contact graph as many
    # more local restarts while avoiding redundant expensive SLSQP solves.
    for trial in range(25):
        if trial < 5 or best is None:
            wide_row = trial % 5
            seed_centers = []
            for row in range(5):
                y = 0.1 + 0.2 * row
                count = 6 if row == wide_row else 5
                xs = ((np.arange(6) + 0.5) / 6.0 if count == 6
                      else 0.1 + 0.2 * np.arange(5))
                seed_centers.extend((x, y) for x in xs)
            centers = np.asarray(seed_centers, dtype=float)
            radii = np.full(n, 0.068)
            if trial:
                centers += rng.uniform(-0.012, 0.012, centers.shape)
        else:
            incumbent_centers = best[:2 * n].reshape(n, 2)
            incumbent_radii = best[2 * n:]
            # Broad shakes can change the contact graph; later shakes refine
            # the most promising asymmetric realization.
            if trial < 15:
                amplitude, radius_noise = 0.030, 0.006
            else:
                amplitude, radius_noise = 0.016, 0.003
            centers = incumbent_centers + rng.uniform(
                -amplitude, amplitude, incumbent_centers.shape
            )
            centers = np.clip(centers, 1e-5, 1.0 - 1e-5)
            radii = np.clip(
                incumbent_radii + rng.normal(0.0, radius_noise, n),
                1e-5, 0.15
            )

        z0 = np.concatenate((centers.ravel(), radii))
        result = minimize(
            lambda z: -np.sum(z[2 * n:]),
            z0,
            jac=lambda z: np.concatenate((np.zeros(2 * n), -np.ones(n))),
            method="SLSQP",
            bounds=[(1e-6, 1.0 - 1e-6)] * (2 * n) + [(1e-7, 0.5)] * n,
            constraints={"type": "ineq", "fun": constraints,
                         "jac": constraint_jacobian},
            options={"maxiter": 1400, "ftol": 1e-11, "disp": False},
        )
        # SLSQP often stops with contacts displaced by roundoff-scale residuals.
        # Rank candidates after exactly the common shrink used for final output,
        # rather than comparing optimistic, potentially infeasible objectives.
        value = np.sum(np.maximum(result.x[2 * n:], 0.0)) * certified_scale(result.x)
        if value > best_value:
            best = result.x
            best_value = value

    if best is None:
        best = z0

    centers = best[:2 * n].reshape(n, 2)
    radii = np.maximum(best[2 * n:], 0.0)

    # A common shrink factor certifies every contact after finite-precision
    # optimization and is much less destructive than independently clipping.
    scale = 1.0
    for k in range(n):
        if radii[k] > 0:
            scale = min(scale, centers[k, 0] / radii[k],
                        centers[k, 1] / radii[k],
                        (1.0 - centers[k, 0]) / radii[k],
                        (1.0 - centers[k, 1]) / radii[k])
    for a, b in zip(ii, jj):
        if radii[a] + radii[b] > 0:
            scale = min(scale, np.linalg.norm(centers[a] - centers[b]) /
                        (radii[a] + radii[b]))
    radii *= min(1.0, scale) * (1.0 - 1e-10)

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
