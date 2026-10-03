"""Deterministic multistart nonlinear packing construction for 26 circles."""
import numpy as np


def construct_packing():
    """Optimize layered seeds, then continue from the active-contact incumbent."""
    from scipy.optimize import minimize

    n = 26
    pairs = np.array([(i, j) for i in range(n) for j in range(i + 1, n)],
                     dtype=int)
    m = len(pairs)

    def make_seed(variant):
        """Create a feasible 16-circle wall ring with asymmetric 3--4--3 core.

        Variant zero exactly preserves the established translated-wall seed.
        The other seven starts independently phase the four walls, producing
        different corner-adjacent and wall/interior contact graphs while using
        the same 16 perimeter circles, 10 interior circles, and fixed budget.
        """
        wall_r = 0.035
        inner_r = 0.041
        corner = np.array([[wall_r, wall_r], [1.0-wall_r, wall_r],
                           [1.0-wall_r, 1.0-wall_r], [wall_r, 1.0-wall_r]],
                          dtype=float)
        base = np.array([0.255, 0.500, 0.745], dtype=float)

        # Preserve the prior best-known basin exactly as start zero.  Subsequent
        # starts move opposite walls independently; all tangential coordinates
        # remain well separated both from corners and from their wall neighbors.
        if variant == 0:
            bottom = top = left = right = base - 0.042
        else:
            bottom = base + (0.036 if variant & 1 else -0.036)
            top = base + (0.030 if variant & 2 else -0.030)
            left = base + (0.040 if variant & 4 else -0.040)
            right = base + (-0.028 if variant & 1 else 0.028)
        edge = np.vstack((
            np.c_[bottom, np.full(3, wall_r)],
            np.c_[top, np.full(3, 1.0-wall_r)],
            np.c_[np.full(3, wall_r), left],
            np.c_[np.full(3, 1.0-wall_r), right],
        ))

        phase = -1.0 if variant & 1 else 1.0
        drift = (variant // 2 - 1.5) * 0.008
        offsets = np.array([-0.034, 0.020, -0.034], dtype=float) * phase + drift
        interior = np.vstack((
            np.c_[np.array([0.250, 0.500, 0.750]) + offsets[0],
                  np.full(3, 0.280)],
            np.c_[np.array([0.150, 0.383, 0.617, 0.850]) + offsets[1],
                  np.full(4, 0.500)],
            np.c_[np.array([0.250, 0.500, 0.750]) + offsets[2],
                  np.full(3, 0.720)],
        ))
        c = np.vstack((corner, edge, interior)).astype(float, copy=False)
        assert c.shape == (n, 2)
        r = np.r_[np.full(16, wall_r, dtype=float),
                  np.full(10, inner_r, dtype=float)]
        return np.r_[c.ravel(), r]

    def con(z):
        c = z[:2*n].reshape(n, 2)
        r = z[2*n:]
        d = c[pairs[:, 0]] - c[pairs[:, 1]]
        return np.r_[c[:, 0] - r, 1. - c[:, 0] - r,
                     c[:, 1] - r, 1. - c[:, 1] - r,
                     np.einsum("ij,ij->i", d, d) -
                     (r[pairs[:, 0]] + r[pairs[:, 1]]) ** 2]

    def jac(z):
        c = z[:2*n].reshape(n, 2)
        r = z[2*n:]
        a = np.zeros((4*n + m, 3*n))
        ii = np.arange(n)
        a[ii, 2*ii] = 1.
        a[ii, 2*n + ii] = -1.
        a[n + ii, 2*ii] = -1.
        a[n + ii, 2*n + ii] = -1.
        a[2*n + ii, 2*ii + 1] = 1.
        a[2*n + ii, 2*n + ii] = -1.
        a[3*n + ii, 2*ii + 1] = -1.
        a[3*n + ii, 2*n + ii] = -1.
        for k, (i, j) in enumerate(pairs):
            q = 4*n + k
            dx, dy = c[i] - c[j]
            a[q, 2*i:2*i+2] = (2.*dx, 2.*dy)
            a[q, 2*j:2*j+2] = (-2.*dx, -2.*dy)
            a[q, 2*n+i] = a[q, 2*n+j] = -2.*(r[i] + r[j])
        return a

    def safely_shrink(z):
        """Convert small solver residuals into strict direct feasibility."""
        c = z[:2*n].reshape(n, 2).copy()
        r = z[2*n:].copy()
        wall = np.min(np.c_[c[:, 0] - r, 1-c[:, 0] - r,
                            c[:, 1] - r, 1-c[:, 1] - r])
        d = c[pairs[:, 0]] - c[pairs[:, 1]]
        dist = np.sqrt(np.einsum("ij,ij->i", d, d))
        pair_slack = np.min(dist - r[pairs[:, 0]] - r[pairs[:, 1]])
        # A common subtraction repairs wall error delta and pair error 2 delta.
        delta = max(0.0, -wall, -0.5 * pair_slack) + 2e-8
        r = np.maximum(0.0, r - delta)
        return c, r

    # Eight perimeter-supported variants explore distinct wall/interior contact
    # graphs without adding an unbounded random-search component.
    seeds = [make_seed(variant) for variant in range(8)]

    best_c = best_r = None
    best_value = -np.inf
    objective_jac = np.r_[np.zeros(2*n), -np.ones(n)]
    for z0 in seeds:
        res = minimize(
            lambda z: -np.sum(z[2*n:]), z0, jac=lambda z: objective_jac,
            method="SLSQP",
            bounds=[(0., 1.)] * (2*n) + [(1e-6, .5)] * n,
            constraints={"type": "ineq", "fun": con, "jac": jac},
            options={"maxiter": 900, "ftol": 2e-12, "disp": False},
        )
        if np.all(np.isfinite(res.x)):
            c, r = safely_shrink(res.x)
            d = c[pairs[:, 0]] - c[pairs[:, 1]]
            valid = (np.min(c[:, 0]-r) >= -1e-10 and
                     np.min(1-c[:, 0]-r) >= -1e-10 and
                     np.min(c[:, 1]-r) >= -1e-10 and
                     np.min(1-c[:, 1]-r) >= -1e-10 and
                     np.min(np.sqrt(np.einsum("ij,ij->i", d, d)) -
                            r[pairs[:, 0]] - r[pairs[:, 1]]) >= -1e-10)
            if valid and np.sum(r) > best_value:
                best_c, best_r, best_value = c, r, float(np.sum(r))

    # Release the incumbent contact framework and deliberately perturb only
    # circles supported by nearly tight contacts.  The six directions include
    # axial and diagonal motions, allowing an active diagonal to be replaced
    # rather than simply re-solving the identical contact graph.
    if best_c is not None:
        base_c = best_c.copy()
        base_r = best_r.copy()
        base_d = base_c[pairs[:, 0]] - base_c[pairs[:, 1]]
        base_pair_slack = (np.sqrt(np.einsum("ij,ij->i", base_d, base_d)) -
                           base_r[pairs[:, 0]] - base_r[pairs[:, 1]])
        base_wall_slack = np.c_[base_c[:, 0] - base_r,
                                1.0 - base_c[:, 0] - base_r,
                                base_c[:, 1] - base_r,
                                1.0 - base_c[:, 1] - base_r]

        contact_pairs = pairs[base_pair_slack <= 4e-4]
        active = np.zeros(n, dtype=bool)
        if len(contact_pairs):
            active[contact_pairs.ravel()] = True
        active[np.any(base_wall_slack <= 4e-4, axis=1)] = True

        # Degree parity provides a second deterministic displacement pattern;
        # it changes which local gaps open along otherwise similar directions.
        degree = np.zeros(n, dtype=int)
        if len(contact_pairs):
            np.add.at(degree, contact_pairs[:, 0], 1)
            np.add.at(degree, contact_pairs[:, 1], 1)
        alternating = np.where(np.arange(n) & 1, 1.0, -1.0)
        degree_sign = np.where(degree & 1, 1.0, -1.0)
        directions = np.array([
            [1.0, 0.0], [0.0, 1.0],
            [1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)],
            [1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0)],
            [-1.0, 0.0], [0.0, -1.0],
        ], dtype=float)

        for k, direction in enumerate(directions):
            trial_c = base_c.copy()
            signs = alternating if k < 3 else alternating * degree_sign
            trial_c[active] += 8e-4 * signs[active, None] * direction
            trial_r = np.maximum(1e-6, base_r - 2.5e-4)
            z0 = np.r_[trial_c.ravel(), trial_r]
            res = minimize(
                lambda z: -np.sum(z[2*n:]), z0, jac=lambda z: objective_jac,
                method="SLSQP",
                bounds=[(0., 1.)] * (2*n) + [(1e-6, .5)] * n,
                constraints={"type": "ineq", "fun": con, "jac": jac},
                options={"maxiter": 600, "ftol": 2e-12, "disp": False},
            )
            if np.all(np.isfinite(res.x)):
                c, r = safely_shrink(res.x)
                d = c[pairs[:, 0]] - c[pairs[:, 1]]
                valid = (np.min(c[:, 0]-r) >= -1e-10 and
                         np.min(1-c[:, 0]-r) >= -1e-10 and
                         np.min(c[:, 1]-r) >= -1e-10 and
                         np.min(1-c[:, 1]-r) >= -1e-10 and
                         np.min(np.sqrt(np.einsum("ij,ij->i", d, d)) -
                                r[pairs[:, 0]] - r[pairs[:, 1]]) >= -1e-10)
                if valid and np.sum(r) > best_value:
                    best_c, best_r, best_value = c, r, float(np.sum(r))

    # All supplied starts are feasible enough for SLSQP; this is a defensive
    # fallback preserving the public contract if a platform solver fails.
    if best_c is None:
        z = seeds[0]
        best_c, best_r = safely_shrink(z)
        best_value = float(np.sum(best_r))
    return best_c, best_r, best_value


def compute_max_radii(centers):
    n = len(centers)
    radii = np.minimum.reduce(
        [centers[:, 0], centers[:, 1], 1.0-centers[:, 0], 1.0-centers[:, 1]]
    )
    for _ in range(3):
        for i in range(n):
            for j in range(i + 1, n):
                distance = np.linalg.norm(centers[i] - centers[j])
                total = radii[i] + radii[j]
                if total > distance:
                    scale = distance / total
                    radii[i] *= scale
                    radii[j] *= scale
    return radii


def run_packing():
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii


def visualize(centers, radii):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True)
    for i, (center, radius) in enumerate(zip(centers, radii)):
        ax.add_patch(Circle(center, radius, alpha=0.5))
        ax.text(center[0], center[1], str(i), ha="center", va="center")
    plt.title(f"Circle Packing (n={len(centers)}, sum={sum(radii):.6f})")
    plt.show()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii}")