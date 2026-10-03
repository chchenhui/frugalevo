# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """Refine perturbed 5,6,5,6,4 staggered hexagonal packings by SLSQP."""
    n = 26
    ii, jj = np.triu_indices(n, 1)
    r0 = 0.1

    # Both layouts are exact r=.1 staggered realizations.  Moving the
    # four-circle row from the boundary into the middle changes the contact
    # graph and gives SLSQP access to a materially different local basin.
    seed_sets = []
    for counts in ((5, 6, 5, 6, 4), (5, 6, 4, 6, 5)):
        seed_centers = []
        for row, count in enumerate(counts):
            x0 = r0 if count == 6 else 2.0 * r0
            xs = x0 + 2.0 * r0 * np.arange(count)
            y = r0 + row * np.sqrt(3.0) * r0
            seed_centers.extend(zip(xs, np.full(count, y)))
        seed_sets.append(np.asarray(seed_centers, dtype=float))

    # Keep the original two starts first, then add independent reflected
    # realizations.  Although reflection preserves the mathematical problem,
    # its independently generated perturbations sample additional asymmetric
    # contact graphs.  Thus this cannot discard the prior best candidate.
    seed_sets.extend([
        np.column_stack((1.0 - c[:, 0], c[:, 1])) for c in seed_sets.copy()
    ])

    def slacks(z):
        """Return boundary and pairwise non-overlap margins."""
        x, y, r = z[:n], z[n:2 * n], z[2 * n:]
        return np.concatenate((
            x - r, 1.0 - x - r, y - r, 1.0 - y - r,
            np.hypot(x[ii] - x[jj], y[ii] - y[jj]) - r[ii] - r[jj],
        ))

    def jacobian(z):
        """Return exact derivatives of all geometric feasibility margins."""
        x, y = z[:n], z[n:2 * n]
        jac = np.zeros((4 * n + len(ii), 3 * n))
        k = np.arange(n)
        jac[k, k], jac[k, 2 * n + k] = 1.0, -1.0
        jac[n + k, k], jac[n + k, 2 * n + k] = -1.0, -1.0
        jac[2 * n + k, n + k], jac[2 * n + k, 2 * n + k] = 1.0, -1.0
        jac[3 * n + k, n + k], jac[3 * n + k, 2 * n + k] = -1.0, -1.0
        dx, dy = x[ii] - x[jj], y[ii] - y[jj]
        d = np.maximum(np.hypot(dx, dy), 1e-14)
        row = 4 * n + np.arange(len(ii))
        jac[row, ii], jac[row, jj] = dx / d, -dx / d
        jac[row, n + ii], jac[row, n + jj] = dy / d, -dy / d
        jac[row, 2 * n + ii], jac[row, 2 * n + jj] = -1.0, -1.0
        return jac

    def certify(z):
        """Uniformly reduce radii by the worst computed feasibility ratio."""
        z = np.asarray(z, dtype=float).copy()
        c = np.column_stack((z[:n], z[n:2 * n]))
        r = z[2 * n:]
        ratios = [
            np.min(c[:, 0] / r), np.min(c[:, 1] / r),
            np.min((1.0 - c[:, 0]) / r), np.min((1.0 - c[:, 1]) / r),
        ]
        d = np.hypot(c[ii, 0] - c[jj, 0], c[ii, 1] - c[jj, 1])
        ratios.append(np.min(d / (r[ii] + r[jj])))
        z[2 * n:] *= min(1.0, max(0.0, min(ratios))) * (1.0 - 1e-10)
        return z

    # Small noise refines the original contact graph; larger deterministic
    # perturbations permit contact exchanges and asymmetric boundary effects.
    rng = np.random.default_rng(26091)
    starts = []
    for layout, seed_centers in enumerate(seed_sets):
        for trial in range(24):
            c = seed_centers.copy()
            if trial:
                amplitude = 0.0015 + 0.00125 * trial
                c += rng.uniform(-amplitude, amplitude, c.shape)
            r = np.full(n, 0.095)
            r *= 1.0 + rng.uniform(-0.06, 0.06, n)
            starts.append(np.concatenate((c[:, 0], c[:, 1], r)))

    best = certify(starts[0])
    try:
        from scipy.optimize import minimize
        objective_gradient = np.r_[np.zeros(2 * n), -np.ones(n)]
        constraint = {"type": "ineq", "fun": slacks, "jac": jacobian}
        for start in starts:
            result = minimize(
                lambda z: -np.sum(z[2 * n:]), start,
                jac=lambda z: objective_gradient, method="SLSQP",
                bounds=[(0.0, 1.0)] * (2 * n) + [(1e-6, 0.5)] * n,
                constraints=constraint,
                options={"maxiter": 1600, "ftol": 1e-12, "disp": False},
            )
            if np.all(np.isfinite(result.x)):
                candidate = certify(result.x)
                if np.sum(candidate[2 * n:]) > np.sum(best[2 * n:]):
                    best = candidate
    except Exception:
        pass

    centers = np.column_stack((best[:n], best[n:2 * n]))
    radii = best[2 * n:].copy()
    return centers, radii, float(np.sum(radii))


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
