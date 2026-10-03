# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """
    Numerically refine several dense, slightly asymmetric 26-circle layouts.

    The radii are optimization variables rather than being assigned by a
    greedy postprocessing pass.  This is important because a small circle
    near a boundary can permit several surrounding circles to become larger.
    """
    from scipy.optimize import minimize, linprog

    n = 26
    ii, jj = np.triu_indices(n, 1)

    # A 5 by 5 lattice is a good dense starting scaffold.  The twenty-sixth
    # disk is inserted in an interstice; the optimizer is then free to break
    # all lattice symmetry.
    grid = np.array([[x, y] for y in np.linspace(.1, .9, 5)
                     for x in np.linspace(.1, .9, 5)])
    base_centers = np.vstack((grid, [0.4, 0.4]))

    def constraints(z):
        c = z[:2 * n].reshape(n, 2)
        r = z[2 * n:]
        delta = c[ii] - c[jj]
        # Squared distances avoid square roots in the large pair constraint.
        pair_clearance = np.sum(delta * delta, axis=1) - (r[ii] + r[jj]) ** 2
        boundary_clearance = np.concatenate((
            c[:, 0] - r, 1.0 - c[:, 0] - r,
            c[:, 1] - r, 1.0 - c[:, 1] - r
        ))
        return np.concatenate((pair_clearance, boundary_clearance))

    def constraint_jacobian(z):
        """Exact Jacobian of all wall and squared-distance constraints."""
        c = z[:2 * n].reshape(n, 2)
        r = z[2 * n:]
        pair_count = len(ii)
        jac = np.zeros((pair_count + 4 * n, 3 * n))
        rows = np.arange(pair_count)
        delta = c[ii] - c[jj]
        summed = r[ii] + r[jj]

        jac[rows, 2 * ii] = 2.0 * delta[:, 0]
        jac[rows, 2 * ii + 1] = 2.0 * delta[:, 1]
        jac[rows, 2 * jj] = -2.0 * delta[:, 0]
        jac[rows, 2 * jj + 1] = -2.0 * delta[:, 1]
        jac[rows, 2 * n + ii] = -2.0 * summed
        jac[rows, 2 * n + jj] = -2.0 * summed

        k = np.arange(n)
        offset = pair_count
        jac[offset + k, 2 * k] = 1.0
        jac[offset + k, 2 * n + k] = -1.0
        jac[offset + n + k, 2 * k] = -1.0
        jac[offset + n + k, 2 * n + k] = -1.0
        jac[offset + 2 * n + k, 2 * k + 1] = 1.0
        jac[offset + 2 * n + k, 2 * n + k] = -1.0
        jac[offset + 3 * n + k, 2 * k + 1] = -1.0
        jac[offset + 3 * n + k, 2 * n + k] = -1.0
        return jac

    pair_matrix = np.zeros((len(ii), n))
    pair_matrix[np.arange(len(ii)), ii] = 1.0
    pair_matrix[np.arange(len(ii)), jj] = 1.0

    def optimal_radii(c, fallback):
        """Maximize radius sum exactly after the centers have been chosen."""
        wall = np.minimum.reduce((
            c[:, 0], c[:, 1], 1.0 - c[:, 0], 1.0 - c[:, 1],
        ))
        distances = np.linalg.norm(c[ii] - c[jj], axis=1)
        try:
            result = linprog(
                -np.ones(n),
                A_ub=pair_matrix,
                b_ub=distances,
                bounds=[(0.0, max(0.0, float(bound))) for bound in wall],
                method="highs",
            )
            if result.success:
                return result.x
        except Exception:
            pass
        return fallback

    def polish(c, r):
        # A common scale factor preserves every pairwise relation and removes
        # tiny negative slacks caused by finite optimizer termination tolerances.
        factors = [1.0]
        for k in range(n):
            if r[k] > 0:
                factors.append(c[k, 0] / r[k])
                factors.append((1.0 - c[k, 0]) / r[k])
                factors.append(c[k, 1] / r[k])
                factors.append((1.0 - c[k, 1]) / r[k])
        d = np.sqrt(np.sum((c[ii] - c[jj]) ** 2, axis=1))
        sums = r[ii] + r[jj]
        factors.extend((d[sums > 0] / sums[sums > 0]).tolist())
        return r * min(1.0, 0.999999 * min(factors))

    rng = np.random.default_rng(260319)
    best_centers = base_centers.copy()
    best_radii = np.full(n, 0.003)
    best_value = float(np.sum(best_radii))

    # Different perturbations lead to different contact graphs, while the
    # small initial radii make every start strictly feasible.
    for start in range(10):
        c0 = base_centers.copy()
        if start:
            c0 += rng.normal(0.0, 0.028, c0.shape)
            c0 = np.clip(c0, 0.055, 0.945)
        r0 = np.full(n, 0.003)
        z0 = np.concatenate((c0.ravel(), r0))
        result = minimize(
            lambda z: -np.sum(z[2 * n:]),
            z0,
            method="SLSQP",
            jac=lambda z: np.r_[np.zeros(2 * n), -np.ones(n)],
            bounds=[(0.0001, 0.9999)] * (2 * n) + [(0.000001, 0.5)] * n,
            constraints={"type": "ineq", "fun": constraints,
                         "jac": constraint_jacobian},
            options={"maxiter": 900, "ftol": 1e-10, "disp": False},
        )
        z = result.x
        c = z[:2 * n].reshape(n, 2)
        lp_radii = optimal_radii(c, z[2 * n:])

        # Re-enter the nonlinear problem from the LP allocation.  This often
        # permits a small center motion that improves an asymmetric local
        # contact graph exposed by the linear-program solution.
        refined = minimize(
            lambda q: -np.sum(q[2 * n:]),
            np.concatenate((c.ravel(), lp_radii)),
            method="SLSQP",
            jac=lambda q: np.r_[np.zeros(2 * n), -np.ones(n)],
            bounds=[(0.0001, 0.9999)] * (2 * n) + [(0.000001, 0.5)] * n,
            constraints={"type": "ineq", "fun": constraints,
                         "jac": constraint_jacobian},
            options={"maxiter": 260, "ftol": 1e-11, "disp": False},
        )
        if refined.x is not None:
            refined_c = refined.x[:2 * n].reshape(n, 2)
            refined_r = optimal_radii(refined_c, refined.x[2 * n:])
            if np.sum(refined_r) > np.sum(lp_radii):
                c, lp_radii = refined_c, refined_r
        r = polish(c, lp_radii)
        value = float(np.sum(r))
        if value > best_value:
            best_centers, best_radii, best_value = c, r, value

    return best_centers, best_radii, best_value


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