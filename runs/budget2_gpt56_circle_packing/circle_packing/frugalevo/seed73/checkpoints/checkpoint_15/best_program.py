# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """Alternate row-shape SLSQP geometry with an exact linear radius program."""
    from scipy.optimize import minimize, linprog

    n = 26
    patterns = ((5, 5, 6, 5, 5), (5, 6, 5, 5, 5),
                (5, 5, 5, 6, 5))
    best_c = best_r = None
    best_value = -1.0

    def make_centers(q, counts):
        """Convert five row ordinates, endpoints, and shears into centers."""
        out = []
        for k, m in enumerate(counts):
            y, left, right, shear = q[4*k:4*k+4]
            xs = np.linspace(left, right, m) + shear * (k - 2)
            out.extend((x, y) for x in xs)
        return np.asarray(out, dtype=float)

    def radius_lp(c):
        """Maximize the sum of radii for fixed centers by linear programming."""
        wall = np.min(np.column_stack((c[:, 0], c[:, 1],
                                       1-c[:, 0], 1-c[:, 1])), axis=1)
        aa, bb = [], []
        for i in range(n):
            row = np.zeros(n)
            row[i] = 1
            aa.append(row)
            bb.append(wall[i])
            for j in range(i):
                row = np.zeros(n)
                row[i] = row[j] = 1
                aa.append(row)
                bb.append(np.linalg.norm(c[i] - c[j]))
        ans = linprog(-np.ones(n), A_ub=np.asarray(aa),
                      b_ub=np.asarray(bb), bounds=[(0.002, 0.2)] * n,
                      method="highs")
        return (ans.x if ans.success else np.full(n, 0.08))

    def feasible(c, r):
        """Return all wall and pair nonoverlap margins."""
        v = [c[:, 0]-r, c[:, 1]-r, 1-c[:, 0]-r, 1-c[:, 1]-r]
        for i in range(n):
            d = c[i+1:] - c[i]
            v.append(np.sum(d*d, axis=1) - (r[i]+r[i+1:])**2)
        return np.concatenate(v)

    starts = []
    for si, counts in enumerate(patterns):
        for phase in (0, 1, 2):
            q = []
            for k, m in enumerate(counts):
                y = 0.101 + 0.198*k + 0.002*(phase-1)*(k-2)
                left, right = ((0.095, 0.905) if m == 5
                               else (1/12, 11/12))
                if (k + phase) % 2:
                    left, right = left + 0.006, right - 0.006
                q.extend((y, left, right, 0.003*((k+phase) % 2)))
            starts.append((np.asarray(q, dtype=float), counts))

    for attempt, (q, counts) in enumerate(starts):
        """Optimize overlapping adjacent row bands with local circle variables."""
        c = make_centers(q, counts).astype(float, copy=True)
        r = radius_lp(c).astype(float, copy=True)
        rows = []
        pos = 0
        for m in counts:
            rows.append(np.arange(pos, pos + m, dtype=int))
            pos += m

        # Two passes allow a displaced contact in one band to be used by the
        # neighboring band on the return sweep.
        for direction in (range(4), range(3, -1, -1)):
            for band in direction:
                active = np.concatenate((rows[band], rows[band + 1]))
                active = np.asarray(active, dtype=int)
                fixed = np.ones(n, dtype=bool)
                fixed[active] = False
                old = np.concatenate((c[active].ravel(), r[active]))
                bounds = [(0.035, 0.965)] * (2 * len(active))
                bounds += [(0.002, 0.2)] * len(active)

                def unpack(v):
                    cc = c.copy()
                    rr = r.copy()
                    cc[active] = v[:2 * len(active)].reshape(-1, 2)
                    rr[active] = v[2 * len(active):]
                    return cc, rr

                def band_constraints(v):
                    """Enforce global walls, contacts, and row ordering."""
                    cc, rr = unpack(v)
                    value = list(feasible(cc, rr))
                    off = 0
                    for row in (rows[band], rows[band + 1]):
                        for a, b in zip(row[:-1], row[1:]):
                            if a in active and b in active:
                                value.append(cc[b, 0] - cc[a, 0] - 0.01)
                            off += 1
                    return np.concatenate((np.asarray(value, dtype=float),))

                def band_objective(v):
                    """Maximize active radii while retaining slack in fixed rows."""
                    cc, rr = unpack(v)
                    slack = np.maximum(feasible(cc, rr), 0.0)
                    return (-float(np.sum(rr[active])
                                  + 0.015 * np.sum(rr[fixed]))
                            - 1.e-5 * float(np.sum(slack)))

                trial = minimize(band_objective, old, method="SLSQP",
                                 bounds=bounds,
                                 constraints={"type": "ineq",
                                              "fun": band_constraints},
                                 options={"maxiter": 140, "ftol": 2e-9})
                if np.all(np.isfinite(trial.x)):
                    tc, tr = unpack(trial.x)
                    if np.min(feasible(tc, tr)) >= -2e-7:
                        c, r = tc, radius_lp(tc).astype(float, copy=True)

        z = np.concatenate((c.ravel(), r))

        def all_constraints(x):
            return feasible(x[:2*n].reshape(n, 2), x[2*n:])

        # Release the active contacts before the final radius maximization.
        # Three continuation stages let centers change contact topology while
        # preserving a quadratic preference for the preceding radii.
        x0 = np.asarray(z, dtype=float).copy()
        polished = None
        for release, strength in ((0.992, 2e-2), (0.996, 5e-3),
                                  (1.0, 0.0)):
            if release != 1.0:
                x0[2*n:] *= release
            previous = x0[2*n:].copy()

            def continuation_objective(x):
                """Maximize total radius while softly retaining the prior stage."""
                delta = x[2*n:] - previous
                return -float(np.sum(x[2*n:])) + strength * float(np.dot(delta, delta))

            trial = minimize(continuation_objective, x0, method="SLSQP",
                             bounds=[(0, 1)]*(2*n)+[(0.002, 0.2)]*n,
                             constraints={"type": "ineq",
                                          "fun": all_constraints},
                             options={"maxiter": 320, "ftol": 2e-10})
            if np.all(np.isfinite(trial.x)):
                tc = trial.x[:2*n].reshape(n, 2)
                tr = trial.x[2*n:]
                if np.min(feasible(tc, tr)) >= -2e-7:
                    x0 = np.asarray(trial.x, dtype=float).copy()
                    polished = trial

        if polished is not None and np.all(np.isfinite(polished.x)):
            pc = polished.x[:2*n].reshape(n, 2)
            pr = polished.x[2*n:]
            if np.min(feasible(pc, pr)) >= -2e-7 and np.sum(pr) > best_value:
                best_c, best_r, best_value = pc.copy(), pr.copy(), float(np.sum(pr))

    if best_c is None:
        q, counts = starts[0]
        best_c = make_centers(q, counts)
        best_r = radius_lp(best_c)
    best_r = np.asarray(best_r, dtype=float) * 0.999995
    return best_c, best_r, float(np.sum(best_r))


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
