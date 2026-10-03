# EVOLVE-BLOCK-START
"""Layered contact-graph portfolio constructor for 26 circles in a unit square."""
import numpy as np


N = 26
SAFETY = 1.0 - 3.0e-10
II, JJ = np.triu_indices(N, 1)


def _certify(centers, radii):
    """Apply a final global shrink so every geometric constraint is strict."""
    centers = np.clip(np.asarray(centers, dtype=float), 0.0, 1.0)
    radii = np.maximum(np.asarray(radii, dtype=float), 0.0)

    wall = np.minimum.reduce((
        centers[:, 0], centers[:, 1],
        1.0 - centers[:, 0], 1.0 - centers[:, 1],
    ))
    radii = np.minimum(radii, np.maximum(wall, 0.0))

    delta = centers[II] - centers[JJ]
    distances = np.sqrt(np.einsum("ij,ij->i", delta, delta))
    sums = radii[II] + radii[JJ]

    scale = 1.0
    active = sums > 0.0
    if np.any(active):
        scale = min(scale, float(np.min(distances[active] / sums[active])))

    radii *= max(0.0, min(1.0, scale * SAFETY))
    return centers, radii


def _grid_defect(which):
    """Square scaffold with one displaced extra disk."""
    xs = np.linspace(0.10, 0.90, 5)
    ys = np.linspace(0.10, 0.90, 5)
    centers = np.array([(x, y) for y in ys for x in xs], dtype=float)
    defects = (
        (0.40, 0.40), (0.60, 0.40), (0.40, 0.60), (0.60, 0.60),
        (0.47, 0.53), (0.53, 0.47),
    )
    return np.vstack((centers, defects[which % len(defects)]))


def _layer_seed(counts, phase, mirror=False, reverse=False):
    """
    Create an asymmetric truncated triangular scaffold.

    Independent left and right margins are intentional: they allow boundary
    circles on one side to acquire a distinct contact graph from the other.
    """
    if reverse:
        counts = tuple(reversed(counts))

    rows = len(counts)
    pts = []
    for row, count in enumerate(counts):
        t = row / max(1, rows - 1)
        y = 0.082 + 0.836 * t

        # Crowded six-circle rows are narrow, while five-circle rows can
        # exploit more horizontal wall room.  The sinusoidal terms create
        # reproducible non-reflection-symmetric boundary assignments.
        if count >= 6:
            left = 0.068 + 0.010 * np.sin(phase + 1.31 * row)
            right = 0.070 + 0.011 * np.cos(phase + 1.77 * row)
        elif count == 5:
            left = 0.091 + 0.015 * np.sin(phase + 1.17 * row)
            right = 0.092 + 0.014 * np.cos(phase + 1.53 * row)
        else:
            left = 0.145 + 0.012 * np.sin(phase + row)
            right = 0.145 + 0.012 * np.cos(phase + row)

        xs = np.linspace(left, 1.0 - right, count)
        stagger = 0.015 * np.sin(phase + 2.07 * row)
        for col, x in enumerate(xs):
            # Small deterministic skew breaks the artificial aligned-column
            # stationary points without compromising the layered topology.
            dx = stagger + 0.006 * np.sin(
                phase + 1.91 * (row + 1) * (col + 1)
            )
            dy = 0.004 * np.cos(
                phase + 1.37 * (row + 2) * (col + 1)
            )
            pts.append((
                np.clip(x + dx, 0.035, 0.965),
                np.clip(y + dy, 0.035, 0.965),
            ))

    centers = np.asarray(pts, dtype=float)
    if mirror:
        centers[:, 0] = 1.0 - centers[:, 0]
    return centers


def _fallback():
    """Portable strictly feasible answer when SciPy is unavailable."""
    centers = _layer_seed((5, 6, 5, 5, 5), 0.31)
    radii = np.full(N, 0.018, dtype=float)
    return _certify(centers, radii)


def compute_max_radii(centers):
    """
    Compute high-sum feasible radii for fixed supplied centers.

    The fixed-center problem is a linear program: r_i+r_j is bounded by each
    center distance, and each individual radius is bounded by its nearest wall.
    """
    centers = np.asarray(centers, dtype=float)
    n = len(centers)
    pi, pj = np.triu_indices(n, 1)

    wall = np.minimum.reduce((
        centers[:, 0], centers[:, 1],
        1.0 - centers[:, 0], 1.0 - centers[:, 1],
    ))
    wall = np.maximum(wall, 0.0)
    distances = np.linalg.norm(centers[pi] - centers[pj], axis=1)

    try:
        from scipy.optimize import linprog
        matrix = np.zeros((len(pi), n), dtype=float)
        rows = np.arange(len(pi))
        matrix[rows, pi] = 1.0
        matrix[rows, pj] = 1.0
        result = linprog(
            c=-np.ones(n),
            A_ub=matrix,
            b_ub=distances,
            bounds=[(0.0, float(v)) for v in wall],
            method="highs",
        )
        if result.success:
            return np.maximum(0.0, result.x * SAFETY)
    except Exception:
        pass

    radii = 0.45 * wall
    for _ in range(100):
        changed = False
        for i, j, d in zip(pi, pj, distances):
            total = radii[i] + radii[j]
            if total > d and total > 0.0:
                q = d * SAFETY / total
                radii[i] *= q
                radii[j] *= q
                changed = True
        if not changed:
            break
    return radii


def construct_packing():
    """Construct and refine a portfolio of 26-disk contact graphs."""
    best_centers, best_radii = _fallback()
    best_value = float(np.sum(best_radii))

    try:
        from scipy.optimize import minimize

        m = len(II)
        variable_count = 3 * N
        gradient = np.zeros(variable_count)
        gradient[2 * N:] = -1.0

        def objective(z):
            return -float(np.sum(z[2 * N:]))

        def constraints(z):
            x = z[:N]
            y = z[N:2 * N]
            r = z[2 * N:]

            walls = np.concatenate((
                x - r, 1.0 - x - r,
                y - r, 1.0 - y - r,
            ))
            dx = x[II] - x[JJ]
            dy = y[II] - y[JJ]
            pairs = dx * dx + dy * dy - (r[II] + r[JJ]) ** 2
            return np.concatenate((walls, pairs))

        def jacobian(z):
            x = z[:N]
            y = z[N:2 * N]
            r = z[2 * N:]
            jac = np.zeros((4 * N + m, variable_count))
            k = np.arange(N)
            rr = 2 * N + k

            jac[k, k] = 1.0
            jac[k, rr] = -1.0
            jac[N + k, k] = -1.0
            jac[N + k, rr] = -1.0
            jac[2 * N + k, N + k] = 1.0
            jac[2 * N + k, rr] = -1.0
            jac[3 * N + k, N + k] = -1.0
            jac[3 * N + k, rr] = -1.0

            row = 4 * N + np.arange(m)
            dx = x[II] - x[JJ]
            dy = y[II] - y[JJ]
            sr = r[II] + r[JJ]
            jac[row, II] = 2.0 * dx
            jac[row, JJ] = -2.0 * dx
            jac[row, N + II] = 2.0 * dy
            jac[row, N + JJ] = -2.0 * dy
            jac[row, 2 * N + II] = -2.0 * sr
            jac[row, 2 * N + JJ] = -2.0 * sr
            return jac

        # Five-row forms have proved particularly useful for 26 disks: one
        # six-circle layer creates a movable defect in an otherwise broad
        # hexagonal arrangement.  Mirroring/reversal changes wall contacts.
        templates = (
            (5, 6, 5, 5, 5),
            (5, 5, 6, 5, 5),
            (5, 5, 5, 6, 5),
            (6, 5, 5, 5, 5),
            (5, 5, 5, 5, 6),
            (4, 6, 5, 6, 5),
            (5, 4, 6, 5, 6),
        )

        starts = [_grid_defect(k) for k in range(4)]
        for k, shape in enumerate(templates):
            starts.append(_layer_seed(shape, 0.29 + 0.61 * k,
                                      mirror=(k % 2 == 1),
                                      reverse=(k % 3 == 2)))
            if k < 5:
                starts.append(_layer_seed(shape, 1.31 + 0.47 * k,
                                          mirror=(k % 2 == 0),
                                          reverse=(k % 3 == 1)))

        bounds = (
            [(1.0e-6, 1.0 - 1.0e-6)] * (2 * N)
            + [(1.0e-7, 0.5)] * N
        )
        nonlinear = {"type": "ineq", "fun": constraints, "jac": jacobian}

        for c0 in starts:
            z0 = np.concatenate((
                c0[:, 0], c0[:, 1], np.full(N, 0.0025)
            ))
            result = minimize(
                objective, z0, jac=lambda z: gradient,
                method="SLSQP", bounds=bounds, constraints=nonlinear,
                options={"maxiter": 1050, "ftol": 3.0e-11, "disp": False},
            )

            z = result.x if result.x is not None else z0
            centers = np.column_stack((z[:N], z[N:2 * N]))
            radii = z[2 * N:]
            centers, radii = _certify(centers, radii)

            # A fixed-center LP supplies a balanced radius distribution for a
            # short second continuous polish, often releasing radii that the
            # nonlinear solve left unnecessarily small.
            lp_radii = compute_max_radii(centers)
            z1 = np.concatenate((centers[:, 0], centers[:, 1], lp_radii))
            result2 = minimize(
                objective, z1, jac=lambda z: gradient,
                method="SLSQP", bounds=bounds, constraints=nonlinear,
                options={"maxiter": 280, "ftol": 3.0e-11, "disp": False},
            )
            if result2.x is not None:
                z2 = result2.x
                c2 = np.column_stack((z2[:N], z2[N:2 * N]))
                c2, r2 = _certify(c2, z2[2 * N:])
                if np.sum(r2) > np.sum(radii):
                    centers, radii = c2, r2

            value = float(np.sum(radii))
            if value > best_value:
                best_centers, best_radii, best_value = centers, radii, value

    except Exception:
        pass

    return best_centers, best_radii, float(np.sum(best_radii))


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