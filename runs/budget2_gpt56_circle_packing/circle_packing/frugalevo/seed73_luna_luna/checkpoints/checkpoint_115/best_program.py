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

        """Optimize six adjacent two-row subsystems with global frozen obstacles."""
        row_ids = []
        cursor = 0
        for count in allocation:
            row_ids.append(np.arange(cursor, cursor + count, dtype=int))
            cursor += count

        # Three adjacent pairs, visited twice: exactly six local solves.
        for _sweep in range(2):
            for first in (0, 1, 2):
                local = np.concatenate((row_ids[first], row_ids[first + 1]))
                m = len(local)
                old_p = p.copy()
                old_r = r.copy()
                old_sum = float(np.sum(old_r))

                def local_unpack(z):
                    """Split local center and radius variables into writable arrays."""
                    return z[:2 * m].reshape(m, 2), z[2 * m:].copy()

                def local_constraints(z):
                    """Enforce boundaries and separation from local and frozen circles."""
                    q, rr = local_unpack(z)
                    trial_p = p.copy()
                    trial_r = r.copy()
                    trial_p[local] = q
                    trial_r[local] = rr
                    return center_constraints(trial_p.ravel(), trial_r)

                def local_objective(z):
                    """Maximize the radius sum of the selected two-row subsystem."""
                    return -float(np.sum(z[2 * m:]))

                z0 = np.concatenate((p[local].ravel(), r[local]))
                result = minimize(
                    local_objective,
                    z0,
                    method="SLSQP",
                    bounds=[(1e-8, 1.0 - 1e-8)] * (2 * m)
                    + [(1e-7, 0.25)] * m,
                    constraints={"type": "ineq", "fun": local_constraints},
                    options={"maxiter": 45, "ftol": 2e-8, "disp": False},
                )

                trial_q, trial_local_r = local_unpack(result.x)
                trial_p = p.copy()
                trial_p[local] = trial_q
                local_r = r.copy()
                local_r[local] = trial_local_r
                slack = center_constraints(trial_p.ravel(), local_r)

                # Re-size all circles globally, but never allow a local move
                # to replace the incumbent with a lower-valued LP solution.
                if np.min(slack) >= -5e-7:
                    trial_lp = radius_lp(trial_p)
                    if trial_lp.success:
                        trial_r = np.asarray(trial_lp.x, dtype=float).copy()
                        trial_slack = center_constraints(
                            trial_p.ravel(), trial_r
                        )
                        trial_sum = float(np.sum(trial_r))
                        if (
                            np.min(trial_slack) >= -5e-7
                            and trial_sum >= old_sum - 1e-9
                        ):
                            p, r = trial_p, trial_r
                        else:
                            p, r = old_p, old_r
                    else:
                        p, r = old_p, old_r
                else:
                    p, r = old_p, old_r

        # Continue from a deliberately relaxed radius vector to the LP
        # optimum, allowing centers and radii to move together at every stage.
        relaxed = np.maximum(1e-7, 0.72 * r).astype(float, copy=True)
        z0 = np.concatenate((p.ravel(), relaxed))
        candidate = z0.copy()

        def unpack(z):
            """Split a joint center/radius vector into writable arrays."""
            return z[:2 * n].reshape(n, 2), z[2 * n:]

        def continuation_constraints(z, t):
            """Enforce exact packing constraints for interpolated radii."""
            q = (1.0 - t) * relaxed + t * z[2 * n:]
            cp = z[:2 * n].reshape(n, 2)
            edge = np.column_stack(
                (cp[:, 0] - q, cp[:, 1] - q,
                 1.0 - cp[:, 0] - q, 1.0 - cp[:, 1] - q)
            ).ravel()
            d = cp[pairs[:, 0]] - cp[pairs[:, 1]]
            sep = np.sqrt(np.sum(d * d, axis=1)) - (
                q[pairs[:, 0]] + q[pairs[:, 1]]
            )
            return np.concatenate((edge, sep))

        def continuation_objective(z, t):
            """Maximize the interpolated radius sum at continuation stage t."""
            return -float(np.sum((1.0 - t) * relaxed + t * z[2 * n:]))

        for t in (0.25, 0.50, 0.75, 1.0):
            result = minimize(
                lambda z, stage=t: continuation_objective(z, stage),
                candidate, method="SLSQP",
                bounds=[(0.0, 1.0)] * (2 * n) + [(1e-7, 0.25)] * n,
                constraints={
                    "type": "ineq",
                    "fun": lambda z, stage=t:
                        continuation_constraints(z, stage),
                },
                options={"maxiter": 220, "ftol": 2e-8, "disp": False},
            )
            trial = result.x
            if np.all(continuation_constraints(trial, t) >= -2e-6):
                candidate = trial.copy()

        # At the final stage the interpolated radii are the actual radii.
        if np.min(continuation_constraints(candidate, 1.0)) < -2e-6:
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
