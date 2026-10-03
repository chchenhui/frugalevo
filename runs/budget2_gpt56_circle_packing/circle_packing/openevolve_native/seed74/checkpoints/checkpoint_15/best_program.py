# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """Optimize all five 5/6-row seeds, then polish the best with 12 deterministic asymmetric restarts."""
    from scipy.optimize import minimize

    n = 26
    ii, jj = np.triu_indices(n, 1)
    pairs = len(ii)
    rng = np.random.default_rng(91726)

    def slacks(z):
        c = z[:2 * n].reshape(n, 2)
        r = z[2 * n:]
        delta = c[ii] - c[jj]
        separation = np.sum(delta * delta, axis=1) - (r[ii] + r[jj]) ** 2
        return np.concatenate((
            c[:, 0] - r, 1.0 - c[:, 0] - r,
            c[:, 1] - r, 1.0 - c[:, 1] - r, separation
        ))

    def slack_jacobian(z):
        c = z[:2 * n].reshape(n, 2)
        r = z[2 * n:]
        jac = np.zeros((4 * n + pairs, 3 * n))

        for k in range(n):
            x, y, q = 2 * k, 2 * k + 1, 2 * n + k
            jac[k, x], jac[k, q] = 1.0, -1.0
            jac[n + k, x], jac[n + k, q] = -1.0, -1.0
            jac[2 * n + k, y], jac[2 * n + k, q] = 1.0, -1.0
            jac[3 * n + k, y], jac[3 * n + k, q] = -1.0, -1.0

        row = 4 * n
        for k, (a, b) in enumerate(zip(ii, jj)):
            dx, dy = c[a] - c[b]
            total = r[a] + r[b]
            jac[row + k, 2 * a] = 2.0 * dx
            jac[row + k, 2 * a + 1] = 2.0 * dy
            jac[row + k, 2 * b] = -2.0 * dx
            jac[row + k, 2 * b + 1] = -2.0 * dy
            jac[row + k, 2 * n + a] = -2.0 * total
            jac[row + k, 2 * n + b] = -2.0 * total
        return jac

    best = None
    # One row has six circles and the others have five.  Testing every
    # possible wide row permits useful vertical asymmetry near the boundary.
    for wide_row in range(5):
        points = []
        for row in range(5):
            y = 0.1 + 0.2 * row
            if row == wide_row:
                points.extend(((k + 0.5) / 6.0, y) for k in range(6))
            else:
                points.extend((0.1 + 0.2 * k, y) for k in range(5))

        centers = np.asarray(points, dtype=float)
        if wide_row:
            centers += rng.uniform(-0.008, 0.008, centers.shape)
        radii = np.full(n, 0.06)
        z0 = np.concatenate((centers.ravel(), radii))

        result = minimize(
            lambda z: -np.sum(z[2 * n:]),
            z0,
            jac=lambda z: np.concatenate((np.zeros(2 * n), -np.ones(n))),
            method="SLSQP",
            bounds=[(1e-6, 1.0 - 1e-6)] * (2 * n) + [(1e-7, 0.5)] * n,
            constraints={"type": "ineq", "fun": slacks, "jac": slack_jacobian},
            options={"maxiter": 1400, "ftol": 1e-11, "disp": False},
        )
        if np.min(slacks(result.x)) >= -1e-7:
            if best is None or np.sum(result.x[2 * n:]) > np.sum(best[2 * n:]):
                best = result.x

    if best is None:
        best = z0
    else:
        # SLSQP often reaches a good contact graph from a row seed but does
        # not change that graph.  Starting again from slightly loosened,
        # deterministic perturbations of the incumbent can discover nearby
        # asymmetric contact patterns with a larger total radius.
        incumbent_rng = np.random.default_rng(260917)
        incumbent = best.copy()
        # A compact restart portfolio is sufficient to test local contact
        # exchanges near row transitions without spending most runtime on
        # repeatedly rediscovering the same layered optimum.
        for trial in range(12):
            candidate = incumbent.copy()
            c = candidate[:2 * n].reshape(n, 2)
            r = candidate[2 * n:]

            # Include very small moves for contact-graph polishing and wider
            # moves capable of escaping the original five-row structure.
            magnitude = (0.002, 0.004, 0.007, 0.012, 0.019, 0.027)[trial % 6]
            c += incumbent_rng.normal(0.0, magnitude, c.shape)
            c[:] = np.clip(c, 1e-6, 1.0 - 1e-6)

            # Broad moves need additional radial slack; otherwise SLSQP can
            # remain trapped repairing the old set of contacts.
            r *= 0.87 if magnitude <= 0.007 else 0.76

            result = minimize(
                lambda z: -np.sum(z[2 * n:]),
                candidate,
                jac=lambda z: np.concatenate((np.zeros(2 * n), -np.ones(n))),
                method="SLSQP",
                bounds=[(1e-6, 1.0 - 1e-6)] * (2 * n) + [(1e-7, 0.5)] * n,
                constraints={"type": "ineq", "fun": slacks,
                             "jac": slack_jacobian},
                options={"maxiter": 1400, "ftol": 1e-11, "disp": False},
            )
            if np.min(slacks(result.x)) >= -1e-7:
                if np.sum(result.x[2 * n:]) > np.sum(incumbent[2 * n:]):
                    incumbent = result.x
        best = incumbent

    centers = best[:2 * n].reshape(n, 2)
    radii = np.maximum(best[2 * n:], 0.0)

    # Uniformly shrink only by the tiny amount needed to make numerical
    # contact constraints strictly valid under the evaluator's checks.
    scale = 1.0
    for k in range(n):
        if radii[k] > 0.0:
            scale = min(
                scale, centers[k, 0] / radii[k], centers[k, 1] / radii[k],
                (1.0 - centers[k, 0]) / radii[k],
                (1.0 - centers[k, 1]) / radii[k]
            )
    for a, b in zip(ii, jj):
        total = radii[a] + radii[b]
        if total > 0.0:
            scale = min(scale, np.linalg.norm(centers[a] - centers[b]) / total)

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
