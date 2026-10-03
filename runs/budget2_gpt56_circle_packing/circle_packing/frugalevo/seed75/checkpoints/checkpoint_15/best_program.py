# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """Search three five-layer 5-5-6-5-5 variants with SLSQP refinement."""
    from scipy.optimize import minimize

    n = 26

    def constraints(z):
        c = z[:2 * n].reshape(n, 2)
        r = z[2 * n:]
        values = [
            c[:, 0] - r,
            c[:, 1] - r,
            1.0 - c[:, 0] - r,
            1.0 - c[:, 1] - r,
        ]
        for i in range(n):
            d = c[i + 1:] - c[i]
            values.append(np.sum(d * d, axis=1) -
                          (r[i] + r[i + 1:]) ** 2)
        return np.concatenate([np.asarray(v).ravel() for v in values])

    def objective(z):
        return -np.sum(z[2 * n:])

    # Search twelve staggered triangular-lattice continuations: the six-circle
    # row occupies each interior position, both alternating phase directions
    # are tried, and two vertical spacings are used.
    topologies = (
        (6, 5, 5, 5, 5),
        (5, 6, 5, 5, 5),
        (5, 5, 6, 5, 5),
    )
    seeds = []
    for counts in topologies:
        for phase_sign in (-1.0, 1.0):
            for spacing in (0.20, 0.21):
                for phase_size in (0.025, 0.035):
                    points = []
                    y0 = 0.5 - 2.0 * spacing
                    for row, count in enumerate(counts):
                        y = y0 + spacing * row
                        alternating = 1.0 if row % 2 == 0 else -1.0
                        shift = phase_sign * phase_size * alternating
                        xs = (np.arange(count, dtype=float) + 0.5) / count
                        xs = xs + shift
                        xs -= float(np.mean(xs) - 0.5)
                        for x in xs:
                            points.append((float(x), float(y)))
                    centers = np.asarray(points, dtype=float)
                    radii = compute_max_radii(centers)
                    seeds.append(np.concatenate((centers.ravel(), radii)))

    bounds = [(0.0, 1.0)] * (2 * n) + [(0.001, 0.20)] * n
    best = None
    for x0 in seeds:
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints={"type": "ineq", "fun": constraints},
            options={"maxiter": 950, "ftol": 1e-10, "disp": False},
        )
        if result.success:
            candidate = result.x.copy()
            candidate_centers = candidate[:2 * n].reshape(n, 2)
            candidate[2 * n:] = compute_max_radii(candidate_centers)
            candidate_value = -float(np.sum(candidate[2 * n:]))
            if best is None or candidate_value < best.fun:
                result.x = candidate
                result.fun = candidate_value
                best = result

    if best is None:
        x = seeds[0]
    else:
        x = best.x
    centers = x[:2 * n].reshape(n, 2)
    radii = x[2 * n:].copy()
    radii *= 1.0 - 1e-8
    return centers, radii, float(np.sum(radii))


def compute_max_radii(centers):
    """Solve the exact fixed-center sum-radius linear program."""
    from scipy.optimize import linprog

    centers = np.asarray(centers, dtype=float)
    n = len(centers)
    rows = []
    rhs = []

    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n, dtype=float)
            row[i] = 1.0
            row[j] = 1.0
            rows.append(row)
            rhs.append(float(np.linalg.norm(centers[i] - centers[j])))

    wall_bounds = np.min(
        np.column_stack((
            centers[:, 0], centers[:, 1],
            1.0 - centers[:, 0], 1.0 - centers[:, 1],
        )),
        axis=1,
    )

    result = linprog(
        -np.ones(n, dtype=float),
        A_ub=np.asarray(rows),
        b_ub=np.asarray(rhs),
        bounds=[(0.0, float(v)) for v in wall_bounds],
        method="highs",
    )
    if not result.success:
        return np.maximum(0.0, wall_bounds) * (1.0 - 1e-8)

    return np.maximum(0.0, result.x) * (1.0 - 1e-9)


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
