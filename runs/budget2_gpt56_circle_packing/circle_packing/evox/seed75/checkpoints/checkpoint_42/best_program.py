# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """Polish symmetric and unequal-radius perturbed layered hexagonal seeds, retaining the best certified packing."""
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

    # The locations of the two five-circle rows determine different boundary
    # contact graphs.  Search all such layered hexagonal arrangements, with one
    # asymmetric realization of each to permit unequal boundary radii.
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
        # Five-row families permit wider boundary circles and can converge to
        # contact graphs unavailable from the six-row 5/4 seeds.
        ((6, 5, 5, 5, 5), 0.0940, 0.94),
        ((5, 6, 5, 5, 5), 0.0940, 1.00),
        ((5, 5, 6, 5, 5), 0.0940, 1.06),
        ((5, 5, 5, 6, 5), 0.0940, 1.00),
        ((5, 5, 5, 5, 6), 0.0940, 0.94),
        # Seven-row seeds deliberately introduce narrow layers.  They are not
        # competitive as equal-radius packings, but are useful entry points
        # for asymmetric, boundary-dominated unequal-radius contact graphs.
        ((4, 4, 4, 4, 4, 3, 3), 0.0760, 1.00),
        ((4, 4, 4, 3, 4, 4, 3), 0.0760, 0.96),
        ((4, 4, 3, 4, 4, 3, 4), 0.0760, 1.00),
        ((4, 3, 4, 4, 3, 4, 4), 0.0760, 1.04),
        ((3, 4, 4, 3, 4, 4, 4), 0.0760, 1.00),
    ]
    rng = np.random.default_rng(260319)
    starts = []
    for number, (rows, radius, stretch) in enumerate(specifications):
        p = lattice(rows, radius, stretch)
        # Preserve the original deterministic starts exactly: these include
        # the previously strongest symmetric and mildly asymmetric basins.
        starts.append(np.r_[p.ravel(), np.full(n, radius)])
        for trial, scale in enumerate((0.008, 0.016, 0.025)):
            q = np.clip(p + rng.normal(0.0, scale, size=p.shape), 0.012, 0.988)
            spread = (0.06, 0.13, 0.20)[trial]
            initial_radii = radius * np.exp(rng.normal(0.0, spread, size=n))
            q, rr = make_feasible(np.r_[q.ravel(), initial_radii])
            starts.append(np.r_[q.ravel(), rr])

    # Wider perturbations are generated independently so that adding these
    # exploratory starts cannot alter the established seed sequence above.
    # They help SLSQP reach less symmetric boundary-contact graphs.
    exploratory_rng = np.random.default_rng(914726)
    for rows, radius, stretch in specifications:
        p = lattice(rows, radius, stretch)
        for scale, spread in ((0.032, 0.24), (0.045, 0.34), (0.060, 0.46)):
            q = np.clip(
                p + exploratory_rng.normal(0.0, scale, size=p.shape),
                0.008, 0.992
            )
            initial_radii = radius * np.exp(
                exploratory_rng.normal(0.0, spread, size=n)
            )
            q, rr = make_feasible(np.r_[q.ravel(), initial_radii])
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
            """Return the exact Jacobian of all border and pairwise packing slacks."""
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
        def polish(z0):
            """Run one constrained local maximization and certify its output."""
            result = minimize(
                lambda z: -np.sum(z[2 * n:]), z0,
                jac=lambda z: objective_jacobian, method="SLSQP",
                bounds=bounds, constraints=constraints,
                options={"maxiter": 950, "ftol": 2e-12, "disp": False}
            )
            if np.all(np.isfinite(result.x)):
                return make_feasible(result.x)
            return None

        # First search all independently constructed layered contact graphs.
        for z0 in starts:
            candidate = polish(z0)
            if candidate is not None:
                p, r = candidate
                value = float(np.sum(r))
                if value > best_value:
                    best_p, best_r, best_value = p, r, value

        # Perform deterministic incumbent basin-hopping.  In addition to
        # independent displacements, the shears alter the relative horizontal
        # phase of layers; this is useful because a good 26-circle graph is
        # close to, but generally not exactly, a triangular lattice.
        hop_rng = np.random.default_rng(723941)
        for wave, (scale, spread, count) in enumerate((
            # Fine perturbations preserve promising contact graphs while
            # allowing SLSQP to exchange active boundary contacts.
            (0.0025, 0.018, 10),
            (0.0060, 0.045, 12),
            (0.0120, 0.080, 14),
            # Larger restarts are especially useful after a strong incumbent
            # has been found from one of the new row-population families.
            (0.0200, 0.120, 14),
            (0.0320, 0.180, 10),
        )):
            incumbent_p, incumbent_r = best_p.copy(), best_r.copy()
            elite_starts = []
            for trial in range(count):
                q = incumbent_p + hop_rng.normal(0.0, scale, size=incumbent_p.shape)
                # A bounded shear moves alternating layers coherently instead
                # of destroying all contacts as independent noise can do.
                shear = hop_rng.normal(0.0, 0.55 * scale)
                q[:, 0] += shear * (incumbent_p[:, 1] - 0.5)
                if (trial + wave) & 1:
                    q[:, 1] += shear * (incumbent_p[:, 0] - 0.5)
                q = np.clip(q, 0.004, 0.996)
                rr = incumbent_r * np.exp(hop_rng.normal(0.0, spread, size=n))
                q, rr = make_feasible(np.r_[q.ravel(), rr])
                elite_starts.append(np.r_[q.ravel(), rr])

            for z0 in elite_starts:
                candidate = polish(z0)
                if candidate is not None:
                    p, r = candidate
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
