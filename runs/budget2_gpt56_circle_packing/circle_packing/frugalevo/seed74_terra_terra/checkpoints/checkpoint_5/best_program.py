"""Deterministic multistart nonlinear packing construction for 26 circles."""
import numpy as np


def construct_packing():
    """Optimize layered seeds, then continue from the active-contact incumbent."""
    from scipy.optimize import minimize

    n = 26
    pairs = np.array([(i, j) for i in range(n) for j in range(i + 1, n)],
                     dtype=int)
    m = len(pairs)

    def make_seed(counts, dy, phase, radius=0.040):
        """Create independently centered rows with quarter-pitch staggering."""
        pts = []
        y0 = 0.5 - dy * (len(counts) - 1) / 2.0
        for row, count in enumerate(counts):
            # Six-row constructions need broad sparse rows, while the five-row
            # defect layouts begin nearer the standard triangular lattice.
            pitch = {4: 0.190, 5: 0.164, 6: 0.145}[count]
            xs = 0.5 + pitch * (np.arange(count) - (count - 1) / 2.0)
            offset = pitch / 4.0 if ((row + phase) & 1) else -pitch / 4.0
            for x in xs + offset:
                pts.append((x, y0 + row * dy))
        c = np.asarray(pts, dtype=float)
        assert len(c) == n
        return np.r_[c.ravel(), np.full(n, radius, dtype=float)]

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

    # Six-layer alternations change the contact graph substantially.  Their
    # vertical reflections move the defect to the opposite boundary.  The final
    # four starts are five-layer, two-wide-row defect arrangements.
    layer_specs = [
        ([4, 5, 4, 5, 4, 4], .120, 0),
        ([4, 4, 5, 4, 5, 4], .120, 1),
        ([5, 4, 4, 5, 4, 4], .120, 0),
        ([4, 4, 5, 4, 5, 4], .120, 0),
        ([4, 4, 5, 4, 5, 4], .120, 1),
        ([4, 4, 5, 4, 4, 5], .120, 1),
        ([4, 6, 6, 5, 5], .145, 0),
        ([5, 5, 6, 6, 4], .145, 1),
        ([5, 6, 6, 5, 4], .145, 0),
        ([4, 5, 6, 6, 5], .145, 1),
    ]
    seeds = [make_seed(counts, dy, phase) for counts, dy, phase in layer_specs]

    best_c = best_r = None
    best_value = -np.inf
    objective_jac = np.r_[np.zeros(2*n), -np.ones(n)]
    for z0 in seeds:
        res = minimize(
            lambda z: -np.sum(z[2*n:]), z0, jac=lambda z: objective_jac,
            method="SLSQP",
            bounds=[(0., 1.)] * (2*n) + [(1e-6, .5)] * n,
            constraints={"type": "ineq", "fun": con, "jac": jac},
            options={"maxiter": 800, "ftol": 2e-12, "disp": False},
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

    # Continue from the best topology after identifying its direct-contact
    # graph.  A small common radius release and alternating displacement of
    # contact-incident circles lets SLSQP change the active contact pattern.
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
        least_slack = min(float(np.min(base_pair_slack)),
                          float(np.min(base_wall_slack)))
        active = np.zeros(n, dtype=bool)
        active[pairs[base_pair_slack <= least_slack + 5e-5].ravel()] = True
        active[np.any(base_wall_slack <= least_slack + 5e-5, axis=1)] = True

        # The unperturbed incumbent plus these three deterministic directions
        # form the bounded four-start active-contact continuation.
        directions = np.array([[0.0, 0.0],
                               [1.0, 0.0],
                               [0.0, 1.0],
                               [1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0)]])
        signs = np.where(np.arange(n) & 1, 1.0, -1.0)
        for direction in directions:
            trial_c = base_c.copy()
            if np.any(direction):
                trial_c[active] += (7e-4 * signs[active, None] * direction)
            trial_r = np.maximum(1e-6, base_r - 2e-5)
            z0 = np.r_[trial_c.ravel(), trial_r]
            res = minimize(
                lambda z: -np.sum(z[2*n:]), z0, jac=lambda z: objective_jac,
                method="SLSQP",
                bounds=[(0., 1.)] * (2*n) + [(1e-6, .5)] * n,
                constraints={"type": "ineq", "fun": con, "jac": jac},
                options={"maxiter": 500, "ftol": 2e-12, "disp": False},
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