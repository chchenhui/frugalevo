# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """
    Construct 26 unequal non-overlapping circles in the unit square.

    The optimization variables are all center coordinates and radii.  Several
    staggered/hexagonal seeds are used because different boundary contact
    patterns lead to distinct local optima.
    """
    from scipy.optimize import minimize

    n = 26
    rows = (5, 5, 6, 5, 5)
    pi, pj = np.triu_indices(n, 1)
    pair_count = len(pi)

    def make_rows(phase=0.0, vertical_phase=0.0):
        points = []
        for row, count in enumerate(rows):
            y = 0.10 + 0.20 * row + vertical_phase * ((row % 2) * 2 - 1)
            # phase=0 reproduces the balanced original lattice.  Nonzero
            # phase creates a more genuinely triangular alternating lattice.
            if phase == 0.0:
                xs = np.linspace(1.0 / (count + 1), count / (count + 1), count)
            else:
                step = 1.0 / count
                offset = 0.5 + phase * (1 if row % 2 else -1)
                xs = (np.arange(count) + offset) * step
                xs = np.clip(xs, 0.055, 0.945)
            points.extend((x, y) for x in xs)
        return np.asarray(points, dtype=float)

    base = make_rows()
    hex_a = make_rows(0.16)
    hex_b = make_rows(-0.16)

    def constraints(z):
        c = z[:2 * n].reshape(n, 2)
        r = z[2 * n:]
        delta = c[pi] - c[pj]
        pair_clearance = np.einsum("ij,ij->i", delta, delta) - (r[pi] + r[pj]) ** 2
        wall_clearance = np.concatenate((
            c[:, 0] - r,
            c[:, 1] - r,
            1.0 - c[:, 0] - r,
            1.0 - c[:, 1] - r,
        ))
        return np.concatenate((wall_clearance, pair_clearance))

    def constraint_jacobian(z):
        c = z[:2 * n].reshape(n, 2)
        r = z[2 * n:]
        jac = np.zeros((4 * n + pair_count, 3 * n))

        k = np.arange(n)
        jac[k, 2 * k] = 1.0
        jac[k, 2 * n + k] = -1.0
        jac[n + k, 2 * k + 1] = 1.0
        jac[n + k, 2 * n + k] = -1.0
        jac[2 * n + k, 2 * k] = -1.0
        jac[2 * n + k, 2 * n + k] = -1.0
        jac[3 * n + k, 2 * k + 1] = -1.0
        jac[3 * n + k, 2 * n + k] = -1.0

        delta = c[pi] - c[pj]
        rs = r[pi] + r[pj]
        q = 4 * n + np.arange(pair_count)
        jac[q, 2 * pi] = 2.0 * delta[:, 0]
        jac[q, 2 * pi + 1] = 2.0 * delta[:, 1]
        jac[q, 2 * pj] = -2.0 * delta[:, 0]
        jac[q, 2 * pj + 1] = -2.0 * delta[:, 1]
        jac[q, 2 * n + pi] = -2.0 * rs
        jac[q, 2 * n + pj] = -2.0 * rs
        return jac

    bounds = [(0.0, 1.0)] * (2 * n) + [(1e-8, 0.5)] * n
    con = {"type": "ineq", "fun": constraints, "jac": constraint_jacobian}
    objective_jac = np.r_[np.zeros(2 * n), -np.ones(n)]

    rng = np.random.default_rng(271828)
    starts = [base, hex_a, hex_b]

    # Small perturbations alter which circles become boundary contacts without
    # destroying the useful layered-lattice geometry.
    for seed, scale, count in ((base, 0.010, 2), (base, 0.022, 2),
                               (hex_a, 0.016, 2), (hex_b, 0.024, 2)):
        for _ in range(count):
            starts.append(np.clip(seed + rng.normal(0.0, scale, seed.shape),
                                  0.055, 0.945))

    best = None
    best_value = -np.inf
    for seed_centers in starts:
        z0 = np.concatenate((seed_centers.ravel(), np.full(n, 0.042)))
        result = minimize(
            lambda z: -np.sum(z[2 * n:]),
            z0,
            jac=lambda z: objective_jac,
            method="SLSQP",
            bounds=bounds,
            constraints=con,
            options={"maxiter": 1300, "ftol": 2e-12, "disp": False},
        )
        value = float(np.sum(result.x[2 * n:]))
        if value > best_value:
            best = result.x
            best_value = value

    # A final longer pass is inexpensive and commonly resolves the last few
    # near-active contacts after the best topology has been selected.
    polished = minimize(
        lambda z: -np.sum(z[2 * n:]),
        best,
        jac=lambda z: objective_jac,
        method="SLSQP",
        bounds=bounds,
        constraints=con,
        options={"maxiter": 2600, "ftol": 5e-13, "disp": False},
    )
    if np.sum(polished.x[2 * n:]) > np.sum(best[2 * n:]):
        best = polished.x

    centers = best[:2 * n].reshape(n, 2)
    radii = np.maximum(best[2 * n:], 0.0)

    # Convert optimizer-tolerance feasibility into strict geometric validity.
    wall_ratio = np.min(np.minimum.reduce((
        centers[:, 0], centers[:, 1],
        1.0 - centers[:, 0], 1.0 - centers[:, 1],
    )) / np.maximum(radii, 1e-300))
    distances = np.hypot(centers[pi, 0] - centers[pj, 0],
                         centers[pi, 1] - centers[pj, 1])
    pair_ratio = np.min(distances / np.maximum(radii[pi] + radii[pj], 1e-300))
    radii *= max(0.0, min(1.0, wall_ratio, pair_ratio) * (1.0 - 1e-10))

    return centers, radii, float(np.sum(radii))


def compute_max_radii(centers):
    """
    Compute the maximum possible radii for each circle position
    such that they don't overlap and stay within the unit square.
    """
    n = centers.shape[0]
    radii = np.minimum.reduce((
        centers[:, 0], centers[:, 1],
        1.0 - centers[:, 0], 1.0 - centers[:, 1],
    )).astype(float)

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.hypot(*(centers[i] - centers[j]))
            total = radii[i] + radii[j]
            if total > dist:
                scale = dist / total
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