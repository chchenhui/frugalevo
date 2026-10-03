# EVOLVE-BLOCK-START
"""Unequal-radius layered circle packing constructor for n=26."""
import numpy as np


def construct_packing():
    """
    Construct 26 non-overlapping circles in the unit square while maximizing
    the sum of their radii.  A common-radius continuation finds stable dense
    layer structures, then several feasible asymmetric radius perturbations
    are optimized jointly with all center coordinates.
    """
    n = 26
    ii, jj = np.triu_indices(n, 1)
    pair_count = len(ii)
    indices = np.arange(n)

    # All layouts contain 26 points.  They represent distinct square-adapted
    # hexagonal layer contact graphs rather than merely random perturbations.
    layouts = (
        (5, 6, 5, 5, 5),
        (5, 6, 5, 6, 4),
        (6, 5, 5, 5, 5),
        (5, 5, 6, 5, 5),
        (4, 5, 6, 5, 6),
        (5, 5, 5, 6, 5),
    )

    def layered_seed(counts, stagger):
        points = []
        rows = len(counts)
        for row, count in enumerate(counts):
            y = (row + 0.5) / rows
            # Alternating displacement supplies diagonal contacts typical of
            # triangular packing while leaving room for square-boundary flaws.
            shift = stagger * (0.5 if row % 2 else -0.5)
            for col in range(count):
                x = (col + 0.5 + shift) / count
                points.append((x, y))
        return np.clip(np.asarray(points, dtype=float), 0.012, 0.988)

    def full_constraints(z):
        c = z[:2 * n].reshape(n, 2)
        r = z[2 * n:]
        dx = c[ii, 0] - c[jj, 0]
        dy = c[ii, 1] - c[jj, 1]
        distances = np.sqrt(dx * dx + dy * dy)
        return np.concatenate((
            c[:, 0] - r,
            c[:, 1] - r,
            1.0 - c[:, 0] - r,
            1.0 - c[:, 1] - r,
            distances - r[ii] - r[jj],
        ))

    def full_jacobian(z):
        c = z[:2 * n].reshape(n, 2)
        jac = np.zeros((4 * n + pair_count, 3 * n), dtype=float)

        jac[indices, 2 * indices] = 1.0
        jac[indices, 2 * n + indices] = -1.0
        jac[n + indices, 2 * indices + 1] = 1.0
        jac[n + indices, 2 * n + indices] = -1.0
        jac[2 * n + indices, 2 * indices] = -1.0
        jac[2 * n + indices, 2 * n + indices] = -1.0
        jac[3 * n + indices, 2 * indices + 1] = -1.0
        jac[3 * n + indices, 2 * n + indices] = -1.0

        dx = c[ii, 0] - c[jj, 0]
        dy = c[ii, 1] - c[jj, 1]
        d = np.maximum(np.sqrt(dx * dx + dy * dy), 1.0e-13)
        rows = 4 * n + np.arange(pair_count)

        jac[rows, 2 * ii] = dx / d
        jac[rows, 2 * ii + 1] = dy / d
        jac[rows, 2 * jj] = -dx / d
        jac[rows, 2 * jj + 1] = -dy / d
        jac[rows, 2 * n + ii] = -1.0
        jac[rows, 2 * n + jj] = -1.0
        return jac

    def common_constraints(z):
        c = z[:2 * n].reshape(n, 2)
        q = z[-1]
        dx = c[ii, 0] - c[jj, 0]
        dy = c[ii, 1] - c[jj, 1]
        return np.concatenate((
            c[:, 0] - q,
            c[:, 1] - q,
            1.0 - c[:, 0] - q,
            1.0 - c[:, 1] - q,
            np.sqrt(dx * dx + dy * dy) - 2.0 * q,
        ))

    def common_jacobian(z):
        c = z[:2 * n].reshape(n, 2)
        jac = np.zeros((4 * n + pair_count, 2 * n + 1), dtype=float)

        jac[indices, 2 * indices] = 1.0
        jac[indices, -1] = -1.0
        jac[n + indices, 2 * indices + 1] = 1.0
        jac[n + indices, -1] = -1.0
        jac[2 * n + indices, 2 * indices] = -1.0
        jac[2 * n + indices, -1] = -1.0
        jac[3 * n + indices, 2 * indices + 1] = -1.0
        jac[3 * n + indices, -1] = -1.0

        dx = c[ii, 0] - c[jj, 0]
        dy = c[ii, 1] - c[jj, 1]
        d = np.maximum(np.sqrt(dx * dx + dy * dy), 1.0e-13)
        rows = 4 * n + np.arange(pair_count)

        jac[rows, 2 * ii] = dx / d
        jac[rows, 2 * ii + 1] = dy / d
        jac[rows, 2 * jj] = -dx / d
        jac[rows, 2 * jj + 1] = -dy / d
        jac[rows, -1] = -2.0
        return jac

    def unequal_starts(centers, q, phase):
        """
        Produce buffered feasible radius perturbations from a q-feasible
        common-radius packing.  The 0.90 buffer dominates the maximum 8%
        profile increase, retaining feasibility even at common contacts.
        """
        border = np.minimum.reduce((
            centers[:, 0], centers[:, 1],
            1.0 - centers[:, 0], 1.0 - centers[:, 1],
        ))
        interior = np.clip((border / max(q, 1.0e-12) - 1.0) / 0.35, 0.0, 1.0)
        wave = np.sin(2.31 * indices + phase)
        checker = np.cos(3.17 * indices - 0.7 * phase)

        # Larger initial radii are preferentially assigned to circles away
        # from walls; a small alternating component breaks remaining symmetry.
        profile_a = 0.055 * (interior - np.mean(interior)) + 0.025 * wave
        profile_b = 0.050 * (interior - np.mean(interior)) + 0.030 * checker
        profile_c = -0.040 * (interior - np.mean(interior)) - 0.028 * wave

        starts = []
        for profile in (profile_a, profile_b, profile_c):
            profile = np.clip(profile, -0.08, 0.08)
            radii = 0.90 * q * (1.0 + profile)
            starts.append(np.concatenate((centers.ravel(), radii)))
        return starts

    fallback_centers = layered_seed(layouts[0], 0.18)
    fallback_radii = np.full(n, 0.02)
    best = np.concatenate((fallback_centers.ravel(), fallback_radii))
    best_value = float(np.sum(fallback_radii))

    try:
        from scipy.optimize import minimize

        full_obj_jac = np.zeros(3 * n)
        full_obj_jac[2 * n:] = -1.0
        common_obj_jac = np.zeros(2 * n + 1)
        common_obj_jac[-1] = -1.0

        full_bounds = [(0.0, 1.0)] * (2 * n) + [(1.0e-7, 0.5)] * n
        common_bounds = [(0.0, 1.0)] * (2 * n) + [(1.0e-6, 0.25)]

        rng = np.random.default_rng(26031991)
        starts = []

        for k, layout in enumerate(layouts):
            seed = layered_seed(layout, 0.16 + 0.025 * (k % 3))
            if k:
                seed = np.clip(
                    seed + rng.uniform(-0.0065, 0.0065, seed.shape),
                    0.012, 0.988
                )

            common_initial = np.concatenate((seed.ravel(), [0.018]))
            common = minimize(
                lambda z: -z[-1],
                common_initial,
                jac=lambda z: common_obj_jac,
                method="SLSQP",
                bounds=common_bounds,
                constraints={
                    "type": "ineq",
                    "fun": common_constraints,
                    "jac": common_jacobian,
                },
                options={"maxiter": 700, "ftol": 1.0e-12, "disp": False},
            )

            if (np.all(np.isfinite(common.x)) and
                    np.min(common_constraints(common.x)) >= -1.0e-7):
                common_centers = common.x[:2 * n].reshape(n, 2)
                starts.extend(unequal_starts(
                    common_centers, common.x[-1], 0.81 * (k + 1)
                ))

                # A nearly uniform continuation remains useful for contact
                # graphs whose optimum is only weakly asymmetric.
                starts.append(np.concatenate((
                    common_centers.ravel(),
                    np.full(n, 0.965 * common.x[-1])
                ))

            # Direct low-radius starts remain an escape route when common
            # radius optimization selected an overly rigid contact network.
            starts.append(np.concatenate((seed.ravel(), np.full(n, 0.023))))

        constraint = {
            "type": "ineq",
            "fun": full_constraints,
            "jac": full_jacobian,
        }

        for start in starts:
            result = minimize(
                lambda z: -np.sum(z[2 * n:]),
                start,
                jac=lambda z: full_obj_jac,
                method="SLSQP",
                bounds=full_bounds,
                constraints=constraint,
                options={"maxiter": 1850, "ftol": 5.0e-12, "disp": False},
            )
            proposal = result.x
            if (np.all(np.isfinite(proposal)) and
                    np.min(full_constraints(proposal)) >= -1.0e-7):
                value = float(np.sum(proposal[2 * n:]))
                if value > best_value:
                    best = proposal
                    best_value = value

        # Final local solve uses a tighter stopping threshold only once, so
        # broad topology exploration remains inexpensive.
        polish = minimize(
            lambda z: -np.sum(z[2 * n:]),
            best,
            jac=lambda z: full_obj_jac,
            method="SLSQP",
            bounds=full_bounds,
            constraints=constraint,
            options={"maxiter": 5000, "ftol": 5.0e-14, "disp": False},
        )
        if (np.all(np.isfinite(polish.x)) and
                np.min(full_constraints(polish.x)) >= -2.0e-8):
            value = float(np.sum(polish.x[2 * n:]))
            if value > best_value:
                best = polish.x
                best_value = value
    except Exception:
        pass

    centers = np.clip(best[:2 * n].reshape(n, 2), 0.0, 1.0)
    radii = np.maximum(best[2 * n:], 0.0)

    # Strictly scale radii after optimization to eliminate roundoff-level
    # wall or contact violations while preserving optimized proportions.
    border = np.minimum.reduce((
        centers[:, 0], centers[:, 1],
        1.0 - centers[:, 0], 1.0 - centers[:, 1],
    ))
    scale = np.min(border / np.maximum(radii, 1.0e-15))

    dx = centers[ii, 0] - centers[jj, 0]
    dy = centers[ii, 1] - centers[jj, 1]
    distances = np.sqrt(dx * dx + dy * dy)
    pair_scale = np.min(
        distances / np.maximum(radii[ii] + radii[jj], 1.0e-15)
    )
    scale = min(1.0, scale, pair_scale)
    radii *= max(0.0, scale) * (1.0 - 1.0e-10)

    return centers, radii, float(np.sum(radii))


def compute_max_radii(centers):
    """Compute conservative feasible radii for a fixed set of centers."""
    n = centers.shape[0]
    radii = np.minimum.reduce((
        centers[:, 0], centers[:, 1],
        1.0 - centers[:, 0], 1.0 - centers[:, 1],
    )).copy()

    for i in range(n):
        for j in range(i + 1, n):
            distance = np.linalg.norm(centers[i] - centers[j])
            if radii[i] + radii[j] > distance:
                factor = distance / (radii[i] + radii[j])
                radii[i] *= factor
                radii[j] *= factor
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