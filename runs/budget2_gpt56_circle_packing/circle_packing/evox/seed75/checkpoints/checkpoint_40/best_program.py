# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """Polish layered hexagonal seeds and deterministic asymmetric perturbations."""
    n = 26
    ii, jj = np.triu_indices(n, 1)
    pair_count = len(ii)

    def lattice(rows, radius, stretch=1.0):
        """Create an alternating, centered triangular-lattice row arrangement."""
        points = []
        height = np.sqrt(3.0) * radius * (len(rows) - 1)
        y0 = 0.5 - 0.5 * height * stretch
        for row, count in enumerate(rows):
            y = y0 + row * np.sqrt(3.0) * radius * stretch
            start = 0.5 - (count - 1) * radius - (radius if row & 1 else 0.0)
            points.extend((start + 2.0 * radius * k, y) for k in range(count))
        return np.asarray(points, dtype=float)

    def make_feasible(z):
        """Uniformly shrink candidate radii to certify square and pair constraints."""
        p = np.clip(np.asarray(z[:2 * n], dtype=float).reshape(n, 2), 0.0, 1.0)
        r = np.maximum(np.asarray(z[2 * n:], dtype=float), 0.0)
        edge = np.column_stack((p[:, 0], p[:, 1], 1.0 - p[:, 0], 1.0 - p[:, 1]))
        cap = np.min(edge / np.maximum(r[:, None], 1e-300))
        d = np.sqrt(np.sum((p[ii] - p[jj]) ** 2, axis=1))
        cap = min(cap, np.min(d / np.maximum(r[ii] + r[jj], 1e-300)))
        r *= max(0.0, min(1.0, cap * (1.0 - 1e-9)))
        return p, r

    # Enumerate every placement of the two long rows.  Square boundaries make
    # these layered triangular seeds genuinely different optimization basins.
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
    # Use both nearly symmetric and deliberately unequal-radius starts.  The
    # latter are particularly useful near the four square boundaries, where
    # sacrificing a small circle can release several larger neighbors.
    rng = np.random.default_rng(260319)
    starts = []
    for rows, radius, stretch in specifications:
        p = lattice(rows, radius, stretch)
        starts.append(np.r_[p.ravel(), np.full(n, radius)])
        for scale, spread in ((0.008, 0.06), (0.016, 0.13), (0.025, 0.20)):
            q = np.clip(p + rng.normal(0.0, scale, size=p.shape), 0.012, 0.988)
            rr = radius * np.exp(rng.normal(0.0, spread, size=n))
            q, rr = make_feasible(np.r_[q.ravel(), rr])
            starts.append(np.r_[q.ravel(), rr])

    # Independent broad perturbations avoid changing the reproducible sequence
    # above while supplying starts from less layered contact-graph basins.
    exploratory_rng = np.random.default_rng(914726)
    for rows, radius, stretch in specifications:
        p = lattice(rows, radius, stretch)
        for scale, spread in ((0.032, 0.24), (0.045, 0.34), (0.060, 0.46)):
            q = np.clip(
                p + exploratory_rng.normal(0.0, scale, size=p.shape), 0.008, 0.992
            )
            rr = radius * np.exp(exploratory_rng.normal(0.0, spread, size=n))
            q, rr = make_feasible(np.r_[q.ravel(), rr])
            starts.append(np.r_[q.ravel(), rr])

    best_p, best_r = make_feasible(starts[0])
    best_value = float(np.sum(best_r))

    try:
        from scipy.optimize import minimize

        def slack(z):
            """Return all border and non-overlap inequalities."""
            p = z[:2 * n].reshape(n, 2)
            r = z[2 * n:]
            border = np.column_stack((
                p[:, 0] - r, p[:, 1] - r,
                1.0 - p[:, 0] - r, 1.0 - p[:, 1] - r
            )).ravel()
            delta = p[ii] - p[jj]
            d = np.sqrt(np.sum(delta * delta, axis=1))
            return np.r_[border, d - r[ii] - r[jj]]

        def slack_jacobian(z):
            """Return the exact dense Jacobian of all packing inequalities."""
            p = z[:2 * n].reshape(n, 2)
            jac = np.zeros((4 * n + pair_count, 3 * n), dtype=float)

            for k in range(n):
                base = 4 * k
                jac[base, 2 * k] = 1.0
                jac[base, 2 * n + k] = -1.0
                jac[base + 1, 2 * k + 1] = 1.0
                jac[base + 1, 2 * n + k] = -1.0
                jac[base + 2, 2 * k] = -1.0
                jac[base + 2, 2 * n + k] = -1.0
                jac[base + 3, 2 * k + 1] = -1.0
                jac[base + 3, 2 * n + k] = -1.0

            delta = p[ii] - p[jj]
            distance = np.sqrt(np.sum(delta * delta, axis=1))
            unit = delta / np.maximum(distance[:, None], 1e-14)
            rows = 4 * n + np.arange(pair_count)
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
            """Run one constrained local maximization and certify its result."""
            result = minimize(
                lambda z: -np.sum(z[2 * n:]), z0, jac=lambda z: objective_jacobian,
                method="SLSQP", bounds=bounds, constraints=constraints,
                options={"maxiter": 950, "ftol": 2e-12, "disp": False}
            )
            if np.all(np.isfinite(result.x)):
                return make_feasible(result.x)
            return None

        for z0 in starts:
            candidate = polish(z0)
            if candidate is not None:
                p, r = candidate
                value = float(np.sum(r))
                if value > best_value:
                    best_p, best_r, best_value = p, r, value

        # Locally perturb the best discovered contact graph.  These restarts
        # preserve its useful structure but can break a nonessential contact
        # and allow an improved unequal-radius arrangement.
        hop_rng = np.random.default_rng(723941)
        for scale, spread, count in (
            (0.0025, 0.018, 4),
            (0.0060, 0.045, 6),
            (0.0120, 0.080, 6),
        ):
            for _ in range(count):
                q = np.clip(
                    best_p + hop_rng.normal(0.0, scale, size=best_p.shape),
                    0.004, 0.996
                )
                rr = best_r * np.exp(hop_rng.normal(0.0, spread, size=n))
                q, rr = make_feasible(np.r_[q.ravel(), rr])
                candidate = polish(np.r_[q.ravel(), rr])
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
