# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """Refine explicit square-grid and hexagonal seeds by nonlinear packing.

    Centers and unequal radii are optimized simultaneously under exact wall and
    pairwise separation constraints.  The grid-with-interstice construction is
    retained as a feasible fallback, and every numerical result is conservatively
    repaired before it is returned.
    """
    n = 26
    safety = 1.0 - 2e-9
    grid = np.array(
        [[(2 * i + 1) / 10.0, (2 * j + 1) / 10.0]
         for j in range(5) for i in range(5)],
        dtype=float,
    )
    base_centers = np.vstack((grid, [[0.4, 0.4]]))
    base_radii = np.r_[np.full(25, 0.1), np.sqrt(0.02) - 0.1]

    pairs = [(i, j) for i in range(n) for j in range(i)]
    pair_i = np.array([p[0] for p in pairs])
    pair_j = np.array([p[1] for p in pairs])

    def repair(c, r):
        """Clip wall radii, then uniformly shrink only if a pair is tight."""
        c = np.clip(np.asarray(c, dtype=float), 0.0, 1.0)
        r = np.maximum(np.asarray(r, dtype=float), 0.0)
        r = np.minimum(r, np.minimum.reduce(
            [c[:, 0], c[:, 1], 1.0 - c[:, 0], 1.0 - c[:, 1]]))
        d = np.linalg.norm(c[pair_i] - c[pair_j], axis=1)
        totals = r[pair_i] + r[pair_j]
        active = totals > 0.0
        if np.any(active):
            r *= min(1.0, safety * np.min(d[active] / totals[active]))
        return c, r

    best_c, best_r = repair(base_centers, base_radii)
    best_value = float(best_r.sum())

    # A staggered six-row seed has a different contact graph from the grid.
    rr = 0.085
    hex_centers = []
    for row in range(6):
        count = 5 if row % 2 == 0 else 4
        offset = rr if row % 2 == 0 else 2.0 * rr
        for col in range(count):
            hex_centers.append((offset + 2.0 * rr * col,
                                rr + np.sqrt(3.0) * rr * row))
    hex_centers = np.asarray(hex_centers[:n], dtype=float)

    try:
        from scipy.optimize import minimize

        def constraints(z):
            """Return wall and squared-distance slacks for SLSQP."""
            c = z[:2 * n].reshape(n, 2)
            r = z[2 * n:]
            delta = c[pair_i] - c[pair_j]
            return np.r_[c[:, 0] - r, c[:, 1] - r,
                         1.0 - c[:, 0] - r, 1.0 - c[:, 1] - r,
                         np.einsum("ij,ij->i", delta, delta)
                         - (r[pair_i] + r[pair_j]) ** 2]

        def constraint_jacobian(z):
            """Provide analytic derivatives of walls and pair constraints."""
            c = z[:2 * n].reshape(n, 2)
            r = z[2 * n:]
            jac = np.zeros((4 * n + len(pairs), 3 * n))
            k = np.arange(n)
            jac[k, 2 * k] = 1.0
            jac[k, 2 * n + k] = -1.0
            jac[n + k, 2 * k + 1] = 1.0
            jac[n + k, 2 * n + k] = -1.0
            jac[2 * n + k, 2 * k] = -1.0
            jac[2 * n + k, 2 * n + k] = -1.0
            jac[3 * n + k, 2 * k + 1] = -1.0
            jac[3 * n + k, 2 * n + k] = -1.0

            row = 4 * n + np.arange(len(pairs))
            delta = c[pair_i] - c[pair_j]
            jac[row, 2 * pair_i] = 2.0 * delta[:, 0]
            jac[row, 2 * pair_i + 1] = 2.0 * delta[:, 1]
            jac[row, 2 * pair_j] = -2.0 * delta[:, 0]
            jac[row, 2 * pair_j + 1] = -2.0 * delta[:, 1]
            jac[row, 2 * n + pair_i] = -2.0 * (r[pair_i] + r[pair_j])
            jac[row, 2 * n + pair_j] = -2.0 * (r[pair_i] + r[pair_j])
            return jac

        # Use deterministic asymmetric multistarts.  Exact grid symmetry tends
        # to preserve an inferior square contact graph, whereas small distinct
        # displacements permit SLSQP to discover unequal boundary circles.
        rng = np.random.default_rng(26026)
        starts = [
            np.r_[base_centers.ravel(), base_radii],
            np.r_[hex_centers.ravel(), np.full(n, rr)],
        ]

        # Perturb the grid at several scales.  The extra circle is moved with
        # the others rather than held at its symmetric interstice.
        for scale in (0.008, 0.016, 0.028, 0.045, 0.065):
            c = base_centers + rng.normal(0.0, scale, base_centers.shape)
            c = np.clip(c, 0.015, 0.985)
            starts.append(np.r_[c.ravel(), np.full(n, 0.065)])

        # Staggered rows supply a qualitatively different near-hexagonal
        # contact graph.  Reflection and noise prevent repeated symmetric
        # stationary points.
        for flip in (False, True):
            c = hex_centers.copy()
            c[:, 0] = 0.5 + 0.92 * (c[:, 0] - c[:, 0].mean())
            c[:, 1] = 0.5 + 0.92 * (c[:, 1] - c[:, 1].mean())
            if flip:
                c[:, 0] = 1.0 - c[:, 0]
            c += rng.normal(0.0, 0.018, c.shape)
            starts.append(np.r_[np.clip(c, 0.015, 0.985).ravel(),
                                np.full(n, 0.065)])

        # Randomized 5-by-5 cells retain broad coverage of the square while
        # placing the 26th center in a randomly selected interstice.
        for _ in range(3):
            c = base_centers.copy()
            c[:25] += rng.uniform(-0.055, 0.055, (25, 2))
            c[-1] = rng.uniform(0.18, 0.82, 2)
            starts.append(np.r_[np.clip(c, 0.015, 0.985).ravel(),
                                np.full(n, 0.052)])

        bounds = [(0.0, 1.0)] * (2 * n) + [(1e-7, 0.5)] * n
        for start in starts:
            result = minimize(
                lambda z: -np.sum(z[2 * n:]), start,
                jac=lambda z: np.r_[np.zeros(2 * n), -np.ones(n)],
                method="SLSQP", bounds=bounds,
                constraints={"type": "ineq", "fun": constraints,
                             "jac": constraint_jacobian},
                options={"maxiter": 1200, "ftol": 3e-12, "disp": False},
            )
            if result.success and np.all(np.isfinite(result.x)):
                c, r = repair(result.x[:2 * n].reshape(n, 2), result.x[2 * n:])
                if r.sum() > best_value:
                    best_c, best_r, best_value = c, r, float(r.sum())
    except Exception:
        pass

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
