# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """Use deterministic multistart layered SLSQP searches and LP certification."""
    from scipy.optimize import minimize

    n = 26
    ii, jj = np.triu_indices(n, 1)

    def slacks(z):
        """Return wall and pairwise non-overlap margins for SLSQP."""
        p = z[:2 * n].reshape(n, 2)
        r = z[2 * n:]
        d = np.sqrt(np.sum((p[ii] - p[jj]) ** 2, axis=1))
        return np.concatenate((
            p[:, 0] - r, 1.0 - p[:, 0] - r,
            p[:, 1] - r, 1.0 - p[:, 1] - r,
            d - r[ii] - r[jj],
        ))

    def slack_jacobian(z):
        """Return exact derivatives of every wall and distance constraint."""
        p = z[:2 * n].reshape(n, 2)
        delta = p[ii] - p[jj]
        distance = np.sqrt(np.sum(delta * delta, axis=1))
        distance = np.maximum(distance, 1e-12)

        jac = np.zeros((4 * n + len(ii), 3 * n))
        q = np.arange(n)
        jac[q, 2 * q] = 1.0
        jac[q, 2 * n + q] = -1.0
        jac[n + q, 2 * q] = -1.0
        jac[n + q, 2 * n + q] = -1.0
        jac[2 * n + q, 2 * q + 1] = 1.0
        jac[2 * n + q, 2 * n + q] = -1.0
        jac[3 * n + q, 2 * q + 1] = -1.0
        jac[3 * n + q, 2 * n + q] = -1.0

        row = 4 * n + np.arange(len(ii))
        unit = delta / distance[:, None]
        jac[row, 2 * ii] = unit[:, 0]
        jac[row, 2 * ii + 1] = unit[:, 1]
        jac[row, 2 * jj] = -unit[:, 0]
        jac[row, 2 * jj + 1] = -unit[:, 1]
        jac[row, 2 * n + ii] = -1.0
        jac[row, 2 * n + jj] = -1.0
        return jac

    def layered_seed(counts, skew):
        """Create a staggered seed whose vertical scale matches its layer count."""
        points = []
        row_gap = 0.83 / (len(counts) - 1)
        middle = 0.5 * (len(counts) - 1)
        for row, count in enumerate(counts):
            y = 0.085 + row_gap * row
            offset = (0.013 if row % 2 else -0.013) + skew * (row - middle)
            xs = np.linspace(0.085, 0.915, count) + offset
            points.extend((x, y) for x in xs)
        return np.asarray(points, dtype=float)

    best_centers = None
    best_radii = None
    best_value = -np.inf
    k = np.arange(n)

    # Five-layer layouts are the primary candidates; six-layer arrangements
    # offer different wall-contact graphs and occasionally better unequal
    # radii after LP certification.
    layouts = (
        (5, 5, 6, 5, 5),
        (5, 6, 5, 5, 5),
        (5, 5, 5, 6, 5),
        (5, 6, 5, 6, 4),
        (4, 6, 6, 5, 5),
        (4, 5, 6, 6, 5),
        (4, 5, 5, 6, 6),
        (4, 4, 5, 4, 5, 4),
        (4, 5, 4, 5, 4, 4),
        (5, 4, 4, 5, 4, 4),
    )
    starts = (
        (0.0, 0.004), (1.1, 0.007), (2.3, 0.010),
        (3.7, 0.013), (5.0, 0.016), (0.6, 0.022),
        (2.9, 0.028), (4.5, 0.035),
        # Larger waves are useful for escaping the highly symmetric
        # five-row contact graph without introducing random state.
        (0.25, 0.018), (1.75, 0.024), (3.25, 0.031),
        (4.95, 0.039), (5.75, 0.044), (2.05, 0.050),
    )

    for layout in layouts:
        for skew in (-0.004, 0.0, 0.004):
            seed = layered_seed(layout, skew)
            seed_radii = compute_max_radii(seed)
            seed_value = float(seed_radii.sum())
            if seed_value > best_value:
                best_centers, best_radii, best_value = seed, seed_radii, seed_value

            for phase, amplitude in starts:
                p = seed.copy()
                # Two incommensurate waves expose asymmetric contact graphs
                # without reliance on random-number state.
                p[:, 0] += amplitude * (
                    np.sin(1.91 * k + phase) +
                    0.37 * np.cos(3.17 * k - 0.43 * phase)
                )
                p[:, 1] += amplitude * (
                    np.cos(2.37 * k + 0.71 * phase) +
                    0.31 * np.sin(4.11 * k + phase)
                )
                initial = np.concatenate((p.ravel(), compute_max_radii(p)))
                result = minimize(
                    lambda z: -np.sum(z[2 * n:]),
                    initial,
                    jac=lambda z: np.concatenate((np.zeros(2 * n), -np.ones(n))),
                    method="SLSQP",
                    bounds=[(0.0, 1.0)] * (2 * n) + [(1e-7, 0.5)] * n,
                    constraints={
                        "type": "ineq",
                        "fun": slacks,
                        "jac": slack_jacobian,
                    },
                    options={"maxiter": 550, "ftol": 1e-10, "disp": False},
                )

                # SLSQP occasionally reaches a useful stationary packing while
                # reporting a line-search status; LP certification is decisive.
                candidate = result.x[:2 * n].reshape(n, 2)
                if np.all(np.isfinite(candidate)):
                    radii = compute_max_radii(candidate)
                    value = float(radii.sum())
                    if value > best_value:
                        best_centers, best_radii, best_value = candidate, radii, value

    return best_centers, best_radii, best_value


def compute_max_radii(centers):
    """Maximize total radius for fixed centers using all linear contact bounds."""
    from scipy.optimize import linprog

    n = len(centers)
    ii, jj = np.triu_indices(n, 1)
    distances = np.sqrt(np.sum((centers[ii] - centers[jj]) ** 2, axis=1))

    # Each wall gives r_i <= wall distance; each pair gives r_i+r_j <= d.
    rows = []
    rhs = []
    for i, (x, y) in enumerate(centers):
        row = np.zeros(n)
        row[i] = 1.0
        rows.append(row)
        rhs.append(min(x, y, 1.0 - x, 1.0 - y) - 1e-10)

    for i, j, distance in zip(ii, jj, distances):
        row = np.zeros(n)
        row[i] = row[j] = 1.0
        rows.append(row)
        rhs.append(distance - 1e-10)

    solution = linprog(
        -np.ones(n),
        A_ub=np.asarray(rows),
        b_ub=np.asarray(rhs),
        bounds=[(0.0, None)] * n,
        method="highs",
    )
    if not solution.success:
        return np.zeros(n)

    # Slight shrinkage makes strict evaluator tolerance robust.
    return np.maximum(0.0, solution.x * (1.0 - 1e-9))


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
