# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """Optimize 26 independently sized circles from several hexagonal seeds.

    The variables are all center coordinates and radii.  SLSQP maximizes the
    radius sum subject to square-boundary and every pairwise distance
    constraint.  Several slightly asymmetric triangular-lattice seeds are
    useful because the best finite-square packing is not perfectly symmetric.
    """
    n = 26
    ii, jj = np.triu_indices(n, 1)

    # Each possible vertical location of the six-circle layer defines a
    # distinct boundary-contact graph.  They are equivalent in the infinite
    # lattice but not in a finite square.
    bases = []
    for six_row in range(5):
        points = []
        for row in range(5):
            count = 6 if row == six_row else 5
            xs = np.linspace(0.10, 0.90, 6) if count == 6 else np.linspace(0.18, 0.82, 5)
            points.extend((x, 0.14 + 0.18 * row) for x in xs)
        bases.append(np.asarray(points, dtype=float))
    base = bases[0]

    def constraints(z):
        """Return boundary and squared pair-separation slacks."""
        c = z[:2 * n].reshape(n, 2)
        r = z[2 * n:]
        boundary = np.concatenate((c[:, 0] - r, 1.0 - c[:, 0] - r,
                                   c[:, 1] - r, 1.0 - c[:, 1] - r))
        delta = c[ii] - c[jj]
        separation = np.einsum("ij,ij->i", delta, delta) - (r[ii] + r[jj]) ** 2
        return np.concatenate((boundary, separation))

    def constraint_jacobian(z):
        """Return the exact dense Jacobian of all packing slacks."""
        c = z[:2 * n].reshape(n, 2)
        r = z[2 * n:]
        m = len(ii)
        J = np.zeros((4 * n + m, 3 * n), dtype=float)
        k = np.arange(n)

        J[k, 2 * k] = 1.0
        J[k, 2 * n + k] = -1.0
        J[n + k, 2 * k] = -1.0
        J[n + k, 2 * n + k] = -1.0
        J[2 * n + k, 2 * k + 1] = 1.0
        J[2 * n + k, 2 * n + k] = -1.0
        J[3 * n + k, 2 * k + 1] = -1.0
        J[3 * n + k, 2 * n + k] = -1.0

        q = 4 * n + np.arange(m)
        d = c[ii] - c[jj]
        rs = r[ii] + r[jj]
        J[q, 2 * ii] = 2.0 * d[:, 0]
        J[q, 2 * ii + 1] = 2.0 * d[:, 1]
        J[q, 2 * jj] = -2.0 * d[:, 0]
        J[q, 2 * jj + 1] = -2.0 * d[:, 1]
        J[q, 2 * n + ii] = -2.0 * rs
        J[q, 2 * n + jj] = -2.0 * rs
        return J

    def scaled_objective(z):
        """Return the radius sum after uniform scaling to strict feasibility."""
        c = z[:2 * n].reshape(n, 2)
        r = z[2 * n:]
        border = np.minimum.reduce((c[:, 0], c[:, 1], 1.0 - c[:, 0], 1.0 - c[:, 1]))
        d = c[ii] - c[jj]
        distance = np.sqrt(np.einsum("ij,ij->i", d, d))
        factor = min(1.0, np.min(border / r), np.min(distance / (r[ii] + r[jj])))
        return np.sum(r) * max(0.0, factor)

    best = None
    best_score = -np.inf
    try:
        from scipy.optimize import minimize

        objective_gradient = np.r_[np.zeros(2 * n), -np.ones(n)]
        bounds = [(0.0, 1.0)] * (2 * n) + [(1.0e-5, 0.5)] * n

        def penalized_value_and_gradient(z, weight):
            """Return augmented-penalty objective and analytic gradient.

            Negative geometric slacks are penalized quadratically.  A
            continuation in ``weight`` permits substantial rearrangements of
            the contact graph before the final exact constrained solve.
            """
            slack = constraints(z)
            violated = np.minimum(slack, 0.0)
            value = -np.sum(z[2 * n:]) + weight * np.dot(violated, violated)
            gradient = objective_gradient + 2.0 * weight * (
                constraint_jacobian(z).T @ violated
            )
            return value, gradient

        # L-BFGS-B penalty continuation is deliberately used before SLSQP:
        # it can cross temporarily infeasible configurations and consequently
        # explores layer/contact patterns unavailable to a feasible-only run.
        for seed in range(80):
            rng = np.random.default_rng(731 + seed)
            centers0 = bases[seed % len(bases)].copy()
            if seed:
                centers0 += rng.normal(0.0, 0.013, centers0.shape)
            z = np.concatenate((centers0.ravel(), np.full(n, 0.038)))

            for weight in (2.0e2, 5.0e3, 1.0e5):
                penalty_result = minimize(
                    lambda q, w=weight: penalized_value_and_gradient(q, w),
                    z,
                    jac=True,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": 180, "ftol": 1.0e-13, "gtol": 1.0e-8},
                )
                z = penalty_result.x

            # The penalty point is normally very close to feasible.  SLSQP
            # now only needs to enforce exact contacts and maximize locally.
            result = minimize(
                lambda q: -np.sum(q[2 * n:]),
                z,
                jac=lambda q: objective_gradient,
                method="SLSQP",
                bounds=bounds,
                constraints={"type": "ineq", "fun": constraints,
                             "jac": constraint_jacobian},
                options={"maxiter": 1000, "ftol": 1.0e-11, "disp": False},
            )

            for candidate in (z, result.x):
                score = scaled_objective(candidate)
                if np.isfinite(score) and score > best_score:
                    best = candidate.copy()
                    best_score = score
    except Exception:
        best = None

    if best is None:
        centers = base
        radii = np.full(n, 0.038)
    else:
        centers = best[:2 * n].reshape(n, 2)
        radii = best[2 * n:].copy()

    # Optimizers may return constraints violated at roundoff level.  A common
    # scale factor retains the optimized radius proportions while making the
    # reported packing strictly feasible.
    border_limit = np.min(np.concatenate(
        (centers[:, 0], centers[:, 1], 1.0 - centers[:, 0], 1.0 - centers[:, 1])
    ).reshape(4, n), axis=0)
    delta = centers[ii] - centers[jj]
    distances = np.sqrt(np.einsum("ij,ij->i", delta, delta))
    pair_limit = np.min(distances / (radii[ii] + radii[jj]))
    # The limit above is computed from every boundary and pair constraint.
    # Keep only a double-precision-sized inward margin rather than discarding
    # one part per million of the optimized objective.
    scale = min(1.0, np.min(border_limit / radii), pair_limit) * (1.0 - 1.0e-10)
    radii *= scale

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
