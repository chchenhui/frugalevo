# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """Optimize all centers and unequal radii from a 5/6/5/5/5 hexagonal-row seed."""
    from scipy.optimize import minimize

    n = 26
    ii, jj = np.triu_indices(n, 1)
    pairs = len(ii)
    pair_rows = 4 * n + np.arange(pairs)

    # Search both five-row and denser six-row triangular-lattice seeds.
    # The six-row starts have a different contact graph: four-circle rows
    # create larger boundary circles while five-circle rows fill their gaps.
    # Such a graph is difficult for a local optimizer to reach by perturbing
    # only a 5/6/5/5/5 arrangement.
    seed_bank = []
    # Include every ordering of the six-row 4/4/4/4/5/5 family.
    # Although reflected orderings are equivalent in an ideal lattice, the
    # square boundary and unequal radii make their local SLSQP basins distinct.
    layouts = (
        (5, 6, 5, 5, 5), (5, 5, 6, 5, 5),
        (5, 5, 5, 6, 5), (5, 5, 5, 5, 6),
        (5, 5, 4, 4, 4, 4), (5, 4, 5, 4, 4, 4),
        (5, 4, 4, 5, 4, 4), (5, 4, 4, 4, 5, 4),
        (5, 4, 4, 4, 4, 5), (4, 5, 5, 4, 4, 4),
        (4, 5, 4, 5, 4, 4), (4, 5, 4, 4, 5, 4),
        (4, 5, 4, 4, 4, 5), (4, 4, 5, 5, 4, 4),
        (4, 4, 5, 4, 5, 4), (4, 4, 5, 4, 4, 5),
        (4, 4, 4, 5, 5, 4), (4, 4, 4, 5, 4, 5),
        (4, 4, 4, 4, 5, 5),
    )
    for counts in layouts:
        rows = len(counts)
        for stagger in (False, True):
            seed = []
            for row, count in enumerate(counts):
                y = np.linspace(0.10, 0.90, rows)[row]
                if count == 6:
                    xs = np.linspace(1 / 12, 11 / 12, 6)
                elif count == 5:
                    xs = np.linspace(0.10, 0.90, 5)
                else:
                    xs = np.linspace(0.13, 0.87, 4)
                if stagger:
                    shift = 0.018 if row % 2 else -0.018
                    xs = np.clip(xs + shift, 0.045, 0.955)
                seed.extend((x, y) for x in xs)
            seed_bank.append(np.asarray(seed, dtype=float))

    # Retain a deterministic feasible-shape fallback.
    seed_centers = seed_bank[0]

    def constraints(z):
        p = z[:2 * n].reshape(n, 2)
        r = z[2 * n:]
        delta = p[ii] - p[jj]
        dist = np.sqrt(np.sum(delta * delta, axis=1))
        return np.concatenate((
            p[:, 0] - r,
            p[:, 1] - r,
            1.0 - p[:, 0] - r,
            1.0 - p[:, 1] - r,
            dist - r[ii] - r[jj],
        ))

    def constraint_jacobian(z):
        p = z[:2 * n].reshape(n, 2)
        delta = p[ii] - p[jj]
        dist = np.sqrt(np.sum(delta * delta, axis=1))
        unit = delta / np.maximum(dist[:, None], 1e-14)
        jac = np.zeros((4 * n + pairs, 3 * n))
        k = np.arange(n)

        jac[k, 2 * k] = 1.0
        jac[k, 2 * n + k] = -1.0
        jac[n + k, 2 * k + 1] = 1.0
        jac[n + k, 2 * n + k] = -1.0
        jac[2 * n + k, 2 * k] = -1.0
        jac[2 * n + k, 2 * n + k] = -1.0
        jac[3 * n + k, 2 * k + 1] = -1.0
        jac[3 * n + k, 2 * n + k] = -1.0

        jac[pair_rows, 2 * ii] = unit[:, 0]
        jac[pair_rows, 2 * ii + 1] = unit[:, 1]
        jac[pair_rows, 2 * jj] = -unit[:, 0]
        jac[pair_rows, 2 * jj + 1] = -unit[:, 1]
        jac[pair_rows, 2 * n + ii] = -1.0
        jac[pair_rows, 2 * n + jj] = -1.0
        return jac

    objective_gradient = np.r_[np.zeros(2 * n), -np.ones(n)]
    bounds = [(1e-5, 1.0 - 1e-5)] * (2 * n) + [(1e-6, 0.5)] * n
    problem = {"type": "ineq", "fun": constraints, "jac": constraint_jacobian}
    best = None
    best_score = -np.inf

    def scaled_score(z):
        """Return the strictly feasible objective obtained by uniformly shrinking radii."""
        p = z[:2 * n].reshape(n, 2)
        r = z[2 * n:]
        border = np.min(np.minimum(p, 1.0 - p), axis=1)
        factor = np.min(border / np.maximum(r, 1e-15))
        d = np.sqrt(np.sum((p[ii] - p[jj]) ** 2, axis=1))
        factor = min(factor, np.min(d / np.maximum(r[ii] + r[jj], 1e-15)))
        return float(np.sum(r) * min(1.0, factor))

    def try_start(z0, iterations=1400):
        """Run one analytic SLSQP local optimization and retain its feasible scaled result."""
        nonlocal best, best_score
        result = minimize(
            lambda z: -np.sum(z[2 * n:]),
            z0,
            jac=lambda z: objective_gradient,
            method="SLSQP",
            bounds=bounds,
            constraints=problem,
            options={"maxiter": iterations, "ftol": 3e-12, "disp": False},
        )
        if np.min(constraints(result.x)) >= -5e-7:
            score = scaled_score(result.x)
            if score > best_score:
                best = result.x
                best_score = score

    # Broad row-topology search.  Random initial radii are important here:
    # equal-radius starts tend to preserve artificial reflection contacts,
    # whereas unequal starts permit the optimizer to select asymmetric
    # boundary circles before the contact graph becomes rigid.
    for trial in range(260):
        rng = np.random.default_rng(1709 + trial)
        template = seed_bank[trial % len(seed_bank)]
        jitter = 0.020 + 0.030 * ((trial // len(seed_bank)) % 4) / 3.0
        p0 = np.clip(template + rng.uniform(-jitter, jitter, template.shape), 0.035, 0.965)
        r0 = rng.uniform(0.018, 0.043, n)
        try_start(np.r_[p0.ravel(), r0])

    # Reconstruct nearby contact graphs from the best basin using both tiny
    # edge-preserving moves and larger moves capable of exchanging contacts.
    if best is not None:
        anchor = best.copy()
        for trial in range(180):
            rng = np.random.default_rng(90210 + trial)
            p = anchor[:2 * n].reshape(n, 2)
            amplitude = 0.002 + 0.016 * (trial % 7) / 6.0
            p0 = np.clip(p + rng.normal(0.0, amplitude, p.shape), 0.020, 0.980)
            r0 = np.clip(anchor[2 * n:] * rng.uniform(0.90, 1.025, n), 1e-5, 0.5)
            try_start(np.r_[p0.ravel(), r0], iterations=1800)

    if best is None:
        return seed_centers, np.full(n, 0.025), 0.65

    centers = best[:2 * n].reshape(n, 2)
    radii = best[2 * n:].copy()

    # Make strict feasibility robust against optimizer and evaluator rounding.
    border = np.min(np.minimum(centers, 1.0 - centers), axis=1)
    scale = np.min(border / np.maximum(radii, 1e-15))
    dist = np.sqrt(np.sum((centers[ii] - centers[jj]) ** 2, axis=1))
    scale = min(scale, np.min(dist / np.maximum(radii[ii] + radii[jj], 1e-15)))
    radii *= min(1.0, scale) * (1.0 - 1e-9)

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
