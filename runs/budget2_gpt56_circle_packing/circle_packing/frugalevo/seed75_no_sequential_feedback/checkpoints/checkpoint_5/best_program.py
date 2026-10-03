# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """Optimize from several lattice and staggered starts, returning the best feasible packing."""
    n = 26

    # All starts use small radii, hence they are strictly feasible.  The
    # differing extra-circle positions help SLSQP escape its common
    # symmetric local optimum.
    base = np.array(
        [[x, y] for y in np.linspace(.1, .9, 5)
                 for x in np.linspace(.1, .9, 5)],
        dtype=float
    )
    extra_positions = [
        (.50, .50), (.50, .60), (.40, .50),
        (.60, .50), (.50, .40), (.30, .50)
    ]
    starts = []
    for q in extra_positions:
        starts.append(np.vstack((base, q)))

    # A staggered 5-row construction supplies a less square, more
    # hexagonal contact graph as additional starting points.
    staggered = []
    row_y = (.10, .30, .50, .70, .90)
    for k, y in enumerate(row_y):
        xs = np.linspace(.10, .90, 5)
        if k in (1, 3):
            xs = xs + .04
        staggered.extend((float(x), y) for x in xs)
    staggered.append((.50, .50))
    starts.append(np.asarray(staggered, dtype=float))

    # Use a layered 5,5,6,5,5 arrangement as a stronger hexagonal start.
    # The six-circle middle row reduces the unused central area while the
    # alternating phases avoid the restrictive vertical contacts of a square
    # lattice.  Several small phase variants provide distinct active sets.
    layered_rows = (5, 5, 6, 5, 5)
    layered_y = (.095, .298, .500, .702, .905)
    layered_x = {
        5: np.linspace(.095, .905, 5),
        6: np.linspace(.050, .950, 6),
    }
    for phase in (-.012, 0.0, .012):
        layered = []
        for k, (count, y) in enumerate(zip(layered_rows, layered_y)):
            xs = layered_x[count].copy()
            if count == 5:
                xs += phase if k % 2 else -phase
            elif phase:
                # Slightly perturb only the interior middle-row positions;
                # keep the outer two circles safely inside the square.
                offsets = np.array(
                    [-phase, phase, -phase * .5, phase * .5, -phase, phase]
                )
                xs += offsets
            layered.extend((float(x), float(y)) for x in xs)
        starts.append(np.asarray(layered, dtype=float))

    try:
        from scipy.optimize import minimize

        def objective(z):
            return -np.sum(z[2 * n:])

        def constraints(z):
            """Return boundary and Euclidean separation slacks for all circles."""
            p = z[:2 * n].reshape(n, 2)
            r = z[2 * n:]
            values = [p[:, 0] - r, p[:, 1] - r,
                       1.0 - p[:, 0] - r, 1.0 - p[:, 1] - r]
            for i in range(n):
                d = p[i + 1:] - p[i]
                # Distance, rather than squared distance, has better scaling
                # when the optimizer is close to an active contact.
                values.append(np.sqrt(np.sum(d * d, axis=1) + 1e-18) -
                              (r[i] + r[i + 1:]))
            return np.concatenate(values)

        bounds = [(0.0, 1.0)] * (2 * n) + [(1e-5, .25)] * n
        best = None
        best_value = -np.inf

        for start in starts:
            # Build a strictly feasible, heterogeneous radius vector.  It
            # starts each circle at 42% of the smaller of its boundary
            # clearance and half its nearest-neighbor separation, avoiding
            # the very flat all-.012 initialization.
            delta = np.min(
                np.c_[start[:, 0], start[:, 1],
                      1.0 - start[:, 0], 1.0 - start[:, 1]], axis=1
            )
            nearest = np.full(n, np.inf, dtype=float)
            for i in range(n):
                diff = start - start[i]
                dist = np.sqrt(np.sum(diff * diff, axis=1))
                dist[i] = np.inf
                nearest[i] = np.min(dist)
            initial_radii = 0.42 * np.minimum(delta, 0.5 * nearest)
            initial_radii = np.clip(initial_radii, 0.004, 0.08)
            z0 = np.r_[start.ravel(), initial_radii]
            result = minimize(
                objective, z0, method="SLSQP",
                bounds=bounds,
                constraints={"type": "ineq", "fun": constraints},
                options={"maxiter": 1100, "ftol": 1e-10, "disp": False}
            )
            if result.success:
                violation = np.min(constraints(result.x))
                value = float(np.sum(result.x[2 * n:]))
                if violation >= -1e-7 and value > best_value:
                    best = result.x
                    best_value = value

        if best is not None:
            centers = best[:2 * n].reshape(n, 2)
            radii = best[2 * n:]
            return centers, radii, float(np.sum(radii))

    except Exception:
        pass

    # Deterministic fallback if an optimizer is unavailable.
    centers = starts[1]
    radii = compute_max_radii(centers)
    return centers, radii, float(np.sum(radii))


def compute_max_radii(centers):
    """Return conservative radii using repeated pairwise feasibility repair."""
    n = centers.shape[0]
    radii = np.min(np.c_[centers, 1.0 - centers], axis=1)

    # Repeatedly reduce the larger member of every violating pair.
    # Unlike the original one-pass scaling, this converges to a valid
    # assignment without unnecessarily scaling both circles.
    for _ in range(8):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                excess = radii[i] + radii[j] - d
                if excess > 0:
                    changed = True
                    if radii[i] >= radii[j]:
                        radii[i] = max(0.0, radii[i] - excess)
                    else:
                        radii[j] = max(0.0, radii[j] - excess)
        if not changed:
            break
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
