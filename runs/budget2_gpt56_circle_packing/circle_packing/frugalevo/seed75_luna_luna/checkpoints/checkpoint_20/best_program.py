# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """Optimize a staggered five-row seed with deterministic geometric multistart."""
    n = 26

    # Use a nearly symmetric five-layer hexagonal seed.  The central layer
    # contains six circles and the four boundary layers contain five each,
    # giving all sides comparable density while retaining staggered contacts.
    row_sizes = (5, 5, 6, 5, 5)
    y_levels = (0.095, 0.297, 0.500, 0.703, 0.905)
    points = []
    for iy, (count, y) in enumerate(zip(row_sizes, y_levels)):
        pitch = 0.158 if count == 6 else 0.175
        width = pitch * (count - 1)
        start = 0.5 - width / 2.0
        if iy % 2:
            start += 0.5 * pitch
        # Keep the shifted short rows away from the side walls.
        if start < 0.075:
            start = 0.075
        if start + pitch * (count - 1) > 0.925:
            start = 0.925 - pitch * (count - 1)
        for ix in range(count):
            points.append((start + ix * pitch, y))

    centers_base = np.asarray(points, dtype=float)
    radii_base = np.full(n, 0.058, dtype=float)

    try:
        from scipy.optimize import minimize

        def objective(z):
            return -np.sum(z[2 * n:])

        def constraints(z):
            c = z[:2 * n].reshape(n, 2)
            r = z[2 * n:]
            values = [
                c[:, 0] - r,
                c[:, 1] - r,
                1.0 - c[:, 0] - r,
                1.0 - c[:, 1] - r,
            ]
            for i in range(n - 1):
                delta = c[i + 1:] - c[i]
                values.append(np.sum(delta * delta, axis=1)
                              - (r[i] + r[i + 1:]) ** 2)
            return np.concatenate(values)

        best = None
        # Use several deliberately different aspect ratios and row shears.
        # These preserve the same robust five-layer topology but often lead
        # SLSQP to different active boundary/contact graphs.
        starts = []
        for yscale, xscale, r0 in (
            (0.90, 1.00, 0.060),
            (0.95, 1.00, 0.068),
            (1.00, 1.00, 0.074),
            (1.05, 1.00, 0.068),
            (1.10, 1.00, 0.060),
            (1.00, 0.975, 0.070),
            (1.00, 1.025, 0.066),
            (0.96, 0.990, 0.078),
            (1.04, 1.010, 0.078),
        ):
            seed = centers_base.copy()
            seed[:, 1] = 0.5 + yscale * (seed[:, 1] - 0.5)
            seed[:, 0] = 0.5 + xscale * (seed[:, 0] - 0.5)
            starts.append((seed, r0))

        # Alternating shear is useful for relieving congestion in the two
        # shorter rows without introducing random, potentially invalid seeds.
        for shear, r0 in ((-0.018, 0.058), (0.018, 0.058)):
            seed = centers_base.copy()
            rows = np.repeat(np.arange(5), np.asarray(row_sizes))
            seed[:, 0] += shear * (rows - 2.0)
            seed[:, 1] = 0.5 + 0.99 * (seed[:, 1] - 0.5)
            starts.append((seed, r0))

        # Deterministic locally perturbed starts help SLSQP discover distinct
        # contact graphs while retaining the favorable five-row structure.
        # The perturbations are deliberately small so that the initial
        # configuration remains close to the feasible hexagonal seed.
        rng = np.random.default_rng(26026)
        rows = np.repeat(np.arange(5), np.asarray(row_sizes))
        for trial in range(8):
            seed = centers_base.copy()
            seed[:, 0] += 0.010 * rng.standard_normal(n)
            seed[:, 1] += 0.008 * rng.standard_normal(n)
            seed[:, 0] += (trial - 3.5) * 0.0035 * (rows - 2.0)
            seed[:, 1] = 0.5 + (0.975 + 0.01 * (trial % 3)) * (
                seed[:, 1] - 0.5
            )
            seed[:, 0] = np.clip(seed[:, 0], 0.065, 0.935)
            seed[:, 1] = np.clip(seed[:, 1], 0.065, 0.935)
            starts.append((seed, 0.060 + 0.002 * (trial % 3)))

        for seed, r0 in starts:
            z0 = np.concatenate((seed.ravel(), np.full(n, r0)))
            result = minimize(
                objective,
                z0,
                method="SLSQP",
                bounds=[(0.0, 1.0)] * (2 * n) + [(0.002, 0.25)] * n,
                constraints={"type": "ineq", "fun": constraints},
                options={"maxiter": 620, "ftol": 3e-10, "disp": False},
            )
            if result.success and np.all(np.isfinite(result.x)):
                if best is None or result.fun < best.fun:
                    best = result

        if best is not None:
            centers = best.x[:2 * n].reshape(n, 2)
            radii = best.x[2 * n:].copy()
        else:
            centers, radii = centers_base, radii_base
    except Exception:
        centers, radii = centers_base, radii_base

    # Apply one uniform safety factor to eliminate optimizer roundoff
    # without changing the optimized center arrangement.
    centers = np.clip(centers, 0.0, 1.0)
    radii = np.maximum(radii, 0.001)
    margin = min(
        1.0,
        float(np.min(centers / radii[:, None])),
        float(np.min((1.0 - centers) / radii[:, None])),
    )
    for i in range(n):
        for j in range(i):
            distance = np.linalg.norm(centers[i] - centers[j])
            margin = min(margin, distance / (radii[i] + radii[j]))
    radii *= min(1.0, 0.999999 * margin)

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
