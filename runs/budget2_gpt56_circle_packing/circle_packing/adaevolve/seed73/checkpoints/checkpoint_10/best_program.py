# EVOLVE-BLOCK-START
"""Numerically refined heterogeneous packing of 26 circles in a unit square."""
import numpy as np


def _make_feasible(centers, radii):
    """Uniformly scale all radii using the exact tightest wall or pair bound."""
    centers = np.asarray(centers, dtype=float)
    radii = np.maximum(np.asarray(radii, dtype=float), 1.0e-12)
    n = len(radii)
    ii, jj = np.triu_indices(n, 1)

    wall_scale = np.min(np.concatenate((
        centers[:, 0] / radii,
        centers[:, 1] / radii,
        (1.0 - centers[:, 0]) / radii,
        (1.0 - centers[:, 1]) / radii,
    )))
    distances = np.hypot(centers[ii, 0] - centers[jj, 0],
                         centers[ii, 1] - centers[jj, 1])
    pair_scale = np.min(distances / (radii[ii] + radii[jj]))
    return radii * min(1.0, wall_scale, pair_scale) * (1.0 - 2.0e-9)


def construct_packing():
    """Use exact-Jacobian SLSQP on many asymmetric five/six-row hexagonal seeds."""
    n = 26
    ii, jj = np.triu_indices(n, 1)

    def layered_seed(profile, phase=0.0):
        """Create a near-hexagonal layered seed with a controlled horizontal phase."""
        seed = []
        for row, count in enumerate(profile):
            y = 0.10 + 0.20 * row
            if count == 5:
                xs = np.linspace(0.10, 0.90, 5) + phase * (-1 if row % 2 else 1)
            else:
                xs = np.linspace(0.08, 0.92, 6) + phase * (-1 if row % 2 else 1)
            seed.extend((x, y) for x in xs)
        return np.asarray(seed, dtype=float)

    centers0 = layered_seed((5, 6, 5, 5, 5))
    radii0 = np.full(n, 0.070)

    def unpack(z):
        """Split the optimization vector into centers and individual radii."""
        return z[:2 * n].reshape(n, 2), z[2 * n:]

    def constraints(z):
        """Return wall and Euclidean pair-clearance inequalities in vectorized form."""
        c, r = unpack(z)
        d = c[ii] - c[jj]
        return np.concatenate((
            c[:, 0] - r, c[:, 1] - r,
            1.0 - c[:, 0] - r, 1.0 - c[:, 1] - r,
            np.hypot(d[:, 0], d[:, 1]) - r[ii] - r[jj],
        ))

    def constraint_jacobian(z):
        """Return the exact dense Jacobian of walls and pair-clearance constraints."""
        c, _ = unpack(z)
        m = len(ii)
        jac = np.zeros((4 * n + m, 3 * n))
        k = np.arange(n)
        jac[k, 2 * k] = 1.0
        jac[k, 2 * n + k] = -1.0
        jac[n + k, 2 * k + 1] = 1.0
        jac[n + k, 2 * n + k] = -1.0
        jac[2 * n + k, 2 * k] = -1.0
        jac[2 * n + k, 2 * n + k] = -1.0
        jac[3 * n + k, 2 * k + 1] = -1.0
        jac[3 * n + k, 2 * n + k] = -1.0

        d = c[ii] - c[jj]
        length = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1.0e-12)
        rows = 4 * n + np.arange(m)
        for axis in range(2):
            deriv = d[:, axis] / length
            jac[rows, 2 * ii + axis] = deriv
            jac[rows, 2 * jj + axis] = -deriv
        jac[rows, 2 * n + ii] = -1.0
        jac[rows, 2 * n + jj] = -1.0
        return jac

    centers, radii, best = centers0, radii0, float(np.sum(radii0))
    try:
        from scipy.optimize import minimize

        rng = np.random.default_rng(260917)
        profiles = (
            (6, 5, 5, 5, 5), (5, 6, 5, 5, 5), (5, 5, 6, 5, 5),
            (5, 5, 5, 6, 5), (5, 5, 5, 5, 6),
        )
        starts = []
        for profile in profiles:
            for phase in (0.0, 0.018, -0.018):
                seed = layered_seed(profile, phase)
                starts.extend((seed, seed[:, ::-1]))

        # Exact derivatives make these wider asymmetric perturbations inexpensive.
        for _ in range(16):
            anchor = starts[rng.integers(len(starts))]
            starts.append(np.clip(anchor + rng.uniform(-0.045, 0.045, (n, 2)),
                                  0.052, 0.948))

        bounds = [(0.0, 1.0)] * (2 * n) + [(1.0e-5, 0.5)] * n
        objective_jac = np.concatenate((np.zeros(2 * n), -np.ones(n)))
        for start in starts:
            result = minimize(
                lambda z: -np.sum(z[2 * n:]),
                np.concatenate((start.ravel(), np.full(n, 0.064))),
                jac=lambda z: objective_jac,
                method="SLSQP",
                bounds=bounds,
                constraints={"type": "ineq", "fun": constraints,
                             "jac": constraint_jacobian},
                options={"maxiter": 1800, "ftol": 2.0e-12, "disp": False},
            )
            candidate_centers, candidate_radii = unpack(result.x)
            candidate_radii = _make_feasible(candidate_centers, candidate_radii)
            value = float(np.sum(candidate_radii))
            if value > best:
                centers, radii, best = candidate_centers, candidate_radii, value
    except Exception:
        pass

    radii = _make_feasible(centers, radii)
    return centers, radii, float(np.sum(radii))


def compute_max_radii(centers):
    """Return equal safe radii for compatibility with the former helper API."""
    centers = np.asarray(centers, dtype=float)
    radii = np.full(len(centers), 0.1)
    return _make_feasible(centers, radii)


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
