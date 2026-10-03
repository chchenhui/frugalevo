# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """Build four staggered row layouts, LP-size their circles, then polish each with SLSQP."""
    from scipy.optimize import linprog, minimize

    n = 26
    pairs = np.array(
        [(i, j) for i in range(n) for j in range(i + 1, n)], dtype=int
    )
    best = None

    for allocation in ((6, 5, 5, 5, 5), (5, 6, 5, 5, 5),
                       (5, 5, 6, 5, 5), (5, 5, 5, 6, 5)):
        points = []
        for row, count in enumerate(allocation):
            # Stagger the interior five-circle layers strongly enough to
            # create diagonal hexagonal contacts.  On the two boundary
            # layers, move inward instead of using the alternating extreme:
            # this preserves horizontal boundary clearance for their end
            # circles while retaining the useful interior stagger.
            if count == 6:
                offset = 0.0
            elif row == 0:
                offset = 0.012
            elif row == len(allocation) - 1:
                offset = -0.012
            else:
                offset = 0.020 if row % 2 == 0 else -0.020
            for col in range(count):
                x = (col + 0.5) / count + offset
                # Give the outer layers a little more boundary clearance and
                # use the resulting nearly uniform inter-layer spacing for
                # the diagonal contacts in the middle of the stack.
                y = 0.105 + 0.1975 * row
                points.append((x, y))
        centers = np.asarray(points, dtype=float)
        if centers.shape != (n, 2):
            continue

        boundary = np.min(
            np.column_stack(
                (centers[:, 0], centers[:, 1],
                 1.0 - centers[:, 0], 1.0 - centers[:, 1])
            ), axis=1
        )
        delta = centers[pairs[:, 0]] - centers[pairs[:, 1]]
        distances = np.sqrt(np.sum(delta * delta, axis=1))

        """Refine each row seed by alternating radius LP and center polishing."""
        A = np.zeros((len(pairs), n), dtype=float)
        for k, (i, j) in enumerate(pairs):
            A[k, i] = 1.0
            A[k, j] = 1.0

        def radius_lp(p):
            """Solve the exact maximum-sum radius LP for fixed centers."""
            bnd = np.min(
                np.column_stack(
                    (p[:, 0], p[:, 1], 1.0 - p[:, 0], 1.0 - p[:, 1])
                ), axis=1
            )
            d = p[pairs[:, 0]] - p[pairs[:, 1]]
            dist = np.sqrt(np.sum(d * d, axis=1))
            return linprog(
                np.full(n, -1.0), A_ub=A, b_ub=dist,
                bounds=[(1e-7, float(bnd[i])) for i in range(n)],
                method="highs",
            )

        def center_constraints(x, radii):
            """Return boundary and non-overlap slacks for fixed radii."""
            p = x.reshape(n, 2)
            edge = np.column_stack(
                (p[:, 0] - radii, p[:, 1] - radii,
                 1.0 - p[:, 0] - radii, 1.0 - p[:, 1] - radii)
            ).ravel()
            d = p[pairs[:, 0]] - p[pairs[:, 1]]
            sep = np.sqrt(np.sum(d * d, axis=1)) - (
                radii[pairs[:, 0]] + radii[pairs[:, 1]]
            )
            return np.concatenate((edge, sep))

        p = centers.copy()
        lp = radius_lp(p)
        if not lp.success:
            continue
        r = np.asarray(lp.x, dtype=float).copy()

        for _ in range(3):
            old_p, old_r = p.copy(), r.copy()

            # Maximize total geometric clearance while keeping the current
            # radii feasible; the subsequent LP can then reallocate radii.
            def center_objective(x):
                s = center_constraints(x, r)
                return -float(np.sum(np.log(np.maximum(s, 1e-5))))

            cr = minimize(
                center_objective, p.ravel(), method="SLSQP",
                bounds=[(float(r[i]), float(1.0 - r[i]))
                        for i in range(n) for _ in (0, 1)],
                constraints={"type": "ineq",
                             "fun": lambda x: center_constraints(x, r)},
                options={"maxiter": 180, "ftol": 2e-8, "disp": False},
            )
            if cr.success and np.all(center_constraints(cr.x, r) >= -2e-6):
                p = cr.x.reshape(n, 2)
            lp = radius_lp(p)
            if lp.success:
                r = np.asarray(lp.x, dtype=float).copy()
            else:
                p, r = old_p, old_r

        z0 = np.concatenate((p.ravel(), r))

        def unpack(z):
            return z[:2 * n].reshape(n, 2), z[2 * n:]

        def objective(z):
            return -np.sum(z[2 * n:])

        def constraints(z):
            p, r = unpack(z)
            edge = np.column_stack(
                (p[:, 0] - r, p[:, 1] - r,
                 1.0 - p[:, 0] - r, 1.0 - p[:, 1] - r)
            ).ravel()
            d = p[pairs[:, 0]] - p[pairs[:, 1]]
            sep = np.sqrt(np.sum(d * d, axis=1)) - (
                r[pairs[:, 0]] + r[pairs[:, 1]]
            )
            return np.concatenate((edge, sep))

        result = minimize(
            objective, z0, method="SLSQP",
            bounds=[(0.0, 1.0)] * (2 * n) + [(1e-7, 0.25)] * n,
            constraints={"type": "ineq", "fun": constraints},
            options={"maxiter": 300, "ftol": 2e-8, "disp": False},
        )
        candidate = result.x if (
            result.success or np.all(constraints(result.x) >= -2e-6)
        ) else z0
        if np.min(constraints(candidate)) < -2e-6:
            candidate = z0
        if best is None or np.sum(candidate[2 * n:]) > np.sum(best[2 * n:]):
            best = candidate

    if best is None:
        centers = np.array(
            [[0.1 + 0.2 * (i % 5), 0.1 + 0.2 * (i // 5)]
             for i in range(25)] + [[0.5, 0.3]], dtype=float
        )
        radii = np.full(n, 0.075, dtype=float)
        radii[-1] = 0.02
    else:
        centers, radii = best[:2 * n].reshape(n, 2), best[2 * n:]
    centers = np.clip(centers, 0.0, 1.0)
    radii = np.maximum(0.0, radii) * (1.0 - 2e-8)
    return centers, radii, float(np.sum(radii))


def compute_max_radii(centers):
    """Return conservative radii from boundary and pairwise nearest-neighbor limits."""
    n = centers.shape[0]
    radii = np.min(
        np.column_stack(
            (centers[:, 0], centers[:, 1],
             1.0 - centers[:, 0], 1.0 - centers[:, 1])
        ),
        axis=1,
    )
    for i in range(n):
        for j in range(i):
            d = np.linalg.norm(centers[i] - centers[j])
            if radii[i] + radii[j] > d:
                # Reduce only the newer circle, preserving earlier choices.
                radii[i] = min(radii[i], max(0.0, d - radii[j]))
    return np.maximum(0.0, radii) * (1.0 - 1e-9)


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
