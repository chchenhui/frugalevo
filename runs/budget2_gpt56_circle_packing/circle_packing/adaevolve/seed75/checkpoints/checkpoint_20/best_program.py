# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """Optimize 26 unequal circles using grid, staggered-hexagonal, and polished feasible starts."""
    n = 26
    grid = np.array([[.1 + .2 * i, .1 + .2 * j]
                     for j in range(5) for i in range(5)], float)
    base_c = np.vstack((grid, [[.4, .4]]))
    base_r = np.r_[np.full(25, .1), np.sqrt(.02) - .1]
    pairs = [(i, j) for i in range(n) for j in range(i)]
    pi = np.array([p[0] for p in pairs])
    pj = np.array([p[1] for p in pairs])

    def repair(c, r):
        """Clip to walls and uniformly shrink radii to strict feasibility."""
        c = np.clip(np.asarray(c, float), 0., 1.)
        r = np.maximum(np.asarray(r, float), 0.)
        r = np.minimum(r, np.minimum.reduce(
            (c[:, 0], c[:, 1], 1. - c[:, 0], 1. - c[:, 1])))
        d = np.linalg.norm(c[pi] - c[pj], axis=1)
        s = r[pi] + r[pj]
        if np.any(s > 0):
            r *= min(1., (1. - 2e-9) * np.min(d[s > 0] / s[s > 0]))
        return c, r

    best_c, best_r = repair(base_c, base_r)
    best = best_r.sum()
    try:
        from scipy.optimize import minimize

        def con(z):
            """Return wall and squared pairwise non-overlap slacks."""
            c, r = z[:2 * n].reshape(n, 2), z[2 * n:]
            q = c[pi] - c[pj]
            return np.r_[c[:, 0] - r, c[:, 1] - r,
                         1. - c[:, 0] - r, 1. - c[:, 1] - r,
                         np.einsum("ij,ij->i", q, q) - (r[pi] + r[pj]) ** 2]

        def jac(z):
            """Provide exact derivatives of every packing inequality."""
            c, r = z[:2 * n].reshape(n, 2), z[2 * n:]
            a = np.zeros((4 * n + len(pi), 3 * n))
            k = np.arange(n)
            a[k, 2*k], a[k, 2*n+k] = 1., -1.
            a[n+k, 2*k+1], a[n+k, 2*n+k] = 1., -1.
            a[2*n+k, 2*k], a[2*n+k, 2*n+k] = -1., -1.
            a[3*n+k, 2*k+1], a[3*n+k, 2*n+k] = -1., -1.
            row, q = 4*n + np.arange(len(pi)), c[pi] - c[pj]
            a[row, 2*pi], a[row, 2*pi+1] = 2*q[:, 0], 2*q[:, 1]
            a[row, 2*pj], a[row, 2*pj+1] = -2*q[:, 0], -2*q[:, 1]
            a[row, 2*n+pi] = -2*(r[pi] + r[pj])
            a[row, 2*n+pj] = -2*(r[pi] + r[pj])
            return a

        rng = np.random.default_rng(26026)
        starts = [np.r_[base_c.ravel(), base_r]]
        hex_c = []
        for row in range(6):
            for col in range(5 if row % 2 == 0 else 4):
                hex_c.append((.085 + .085*(row % 2) + .17*col,
                              .085 + np.sqrt(3)*.085*row))
        hex_c = np.asarray(hex_c[:n])
        starts.append(np.r_[hex_c.ravel(), np.full(n, .085)])

        for scale in (.008, .016, .028, .045, .065):
            c = np.clip(base_c + rng.normal(0., scale, base_c.shape), .015, .985)
            starts.append(np.r_[c.ravel(), np.full(n, .065)])
        for _ in range(3):
            c = base_c.copy()
            c[:25] += rng.uniform(-.055, .055, (25, 2))
            c[-1] = rng.uniform(.18, .82, 2)
            starts.append(np.r_[np.clip(c, .015, .985).ravel(), np.full(n, .052)])

        # The raw staggered arrangement occupies only one side of the square.
        # Recentered variants expose boundary contacts and unequal-radius
        # contact graphs which are inaccessible to the grid perturbations.
        hex_rng = np.random.default_rng(726026)
        centered_hex = hex_c.copy()
        centered_hex[:, 0] = .5 + .94 * (centered_hex[:, 0] - centered_hex[:, 0].mean())
        centered_hex[:, 1] = .5 + .94 * (centered_hex[:, 1] - centered_hex[:, 1].mean())
        for mirror_x, mirror_y in ((False, False), (True, False), (False, True)):
            c = centered_hex.copy()
            if mirror_x:
                c[:, 0] = 1. - c[:, 0]
            if mirror_y:
                c[:, 1] = 1. - c[:, 1]
            c += hex_rng.normal(0., .014, c.shape)
            starts.append(np.r_[np.clip(c, .015, .985).ravel(), np.full(n, .068)])

        # Larger asymmetric displacements of the same lattice often settle
        # into a different diagonal-contact basin than small Gaussian noise.
        for scale in (.026, .042):
            c = centered_hex + hex_rng.uniform(-scale, scale, centered_hex.shape)
            starts.append(np.r_[np.clip(c, .015, .985).ravel(), np.full(n, .062)])

        bounds = [(0., 1.)] * (2*n) + [(1e-7, .5)] * n
        objective_jac = np.r_[np.zeros(2*n), -np.ones(n)]
        for start in starts:
            result = minimize(
                lambda z: -z[2*n:].sum(), start, method="SLSQP",
                jac=lambda z: objective_jac, bounds=bounds,
                constraints={"type": "ineq", "fun": con, "jac": jac},
                options={"maxiter": 1200, "ftol": 3e-12, "disp": False})
            if np.all(np.isfinite(result.x)):
                c, r = repair(result.x[:2*n].reshape(n, 2), result.x[2*n:])

                # Restarting from a strictly feasible point removes small
                # terminal constraint violations without sacrificing the
                # local contact graph found by the first solve.
                if r.sum() > best - .02:
                    polished = minimize(
                        lambda z: -z[2*n:].sum(), np.r_[c.ravel(), r],
                        method="SLSQP", jac=lambda z: objective_jac,
                        bounds=bounds,
                        constraints={"type": "ineq", "fun": con, "jac": jac},
                        options={"maxiter": 500, "ftol": 2e-12, "disp": False})
                    if np.all(np.isfinite(polished.x)):
                        pc, pr = repair(polished.x[:2*n].reshape(n, 2),
                                        polished.x[2*n:])
                        if pr.sum() > r.sum():
                            c, r = pc, pr
                if r.sum() > best:
                    best_c, best_r, best = c, r, r.sum()

        # A final algebraic polish works on the jammed contact graph rather
        # than repeatedly optimizing all 325 separation inequalities.  The
        # graph is read from the best local packing: wall/pair constraints
        # which are numerically tangent become equations, and nonnegative KKT
        # multipliers enforce stationarity of the radius-sum objective.
        #
        # This is deliberately retained as a separate least-squares stage:
        # SLSQP is useful for discovering a contact topology, whereas the
        # tangency/KKT equations remove its last small constraint residuals.
        try:
            from scipy.optimize import least_squares, nnls

            z0 = np.r_[best_c.ravel(), best_r]
            slack0 = con(z0)

            # The preceding safety contraction changes exact contacts by only
            # O(1e-10); the looser threshold also retains contacts that were
            # active immediately before SLSQP's final feasibility repair.
            active = np.flatnonzero(slack0 < 2.5e-4)

            # A viable jammed graph needs enough equations to constrain the
            # 78 geometric variables.  Otherwise leave the already feasible
            # SLSQP answer untouched.
            if len(active) >= 70:
                j0 = jac(z0)[active]
                lam0 = nnls(j0.T, objective_jac)[0]
                u0 = np.r_[z0, np.sqrt(np.maximum(lam0, 1e-18))]

                def contact_kkt(u):
                    """Tangencies, wall contacts, and KKT stationarity."""
                    z = u[:3*n]
                    lam = u[3*n:] ** 2
                    # Weight geometric equalities moderately more strongly:
                    # this makes the solved contact graph accurately tangent
                    # before the independent full-pair feasibility check.
                    geometric = 12.0 * con(z)[active]
                    stationary = jac(z)[active].T.dot(lam) - objective_jac
                    return np.r_[geometric, stationary]

                lo = np.r_[np.zeros(2*n), np.full(n, 1e-9),
                           np.zeros(len(active))]
                hi = np.r_[np.ones(2*n), np.full(n, .5),
                           np.full(len(active), np.inf)]
                algebraic = least_squares(
                    contact_kkt, u0, bounds=(lo, hi), method="trf",
                    x_scale="jac", ftol=2e-13, xtol=2e-13, gtol=2e-13,
                    max_nfev=3500)

                if np.all(np.isfinite(algebraic.x)):
                    c, r = repair(algebraic.x[:2*n].reshape(n, 2),
                                  algebraic.x[2*n:3*n])
                    # repair is a full vectorized certificate over every
                    # pair, not merely the selected tangency graph.
                    if r.sum() > best:
                        best_c, best_r, best = c, r, r.sum()
        except Exception:
            pass
    except Exception:
        pass
    return best_c, best_r, float(best)


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
