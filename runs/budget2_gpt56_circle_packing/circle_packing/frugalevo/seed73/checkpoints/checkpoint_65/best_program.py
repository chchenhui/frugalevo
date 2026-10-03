"""Constructor-based circle packing for n=26 circles."""
import numpy as np


def construct_packing():
    from scipy.optimize import minimize, linprog

    n = 26
    counts_list = (
        (5, 5, 6, 5, 5),
        (5, 6, 5, 5, 5),
        (5, 5, 5, 6, 5),
    )

    def make_centers(counts, phase):
        pts = []
        for k, m in enumerate(counts):
            y = 0.101 + 0.198 * k + 0.002 * (phase - 1) * (k - 2)
            if m == 5:
                left, right = 0.095, 0.905
            else:
                left, right = 1.0 / 12.0, 11.0 / 12.0
            if (k + phase) & 1:
                left += 0.006
                right -= 0.006
            xs = np.linspace(left, right, m)
            xs += 0.003 * ((k + phase) & 1) * (k - 2)
            pts.extend((x, y) for x in xs)
        return np.asarray(pts, dtype=float)

    def lp_radii(c, relax=0.0, wall_margin=0.0):
        wall = np.minimum.reduce((c[:, 0], c[:, 1],
                                  1.0 - c[:, 0], 1.0 - c[:, 1]))
        A, b = [], []
        for i in range(n):
            z = np.zeros(n)
            z[i] = 1.0
            A.append(z)
            b.append(wall[i] - wall_margin)
            for j in range(i):
                z = np.zeros(n)
                z[i] = z[j] = 1.0
                A.append(z)
                b.append(np.linalg.norm(c[i] - c[j]) + relax)
        ans = linprog(-np.ones(n), A_ub=np.asarray(A), b_ub=np.asarray(b),
                      bounds=[(0.002, 0.20)] * n, method="highs")
        return ans.x if ans.success else None

    def feasible(c, r):
        z = [c[:, 0] - r, c[:, 1] - r,
             1.0 - c[:, 0] - r, 1.0 - c[:, 1] - r]
        for i in range(n - 1):
            d = c[i + 1:] - c[i]
            z.append(np.einsum("ij,ij->i", d, d)
                     - (r[i] + r[i + 1:]) ** 2)
        return np.concatenate(z)

    def clearances(c, r):
        wall = float(np.min(np.minimum.reduce((
            c[:, 0] - r, c[:, 1] - r,
            1.0 - c[:, 0] - r, 1.0 - c[:, 1] - r))))
        pair = np.inf
        for i in range(n - 1):
            d = np.linalg.norm(c[i + 1:] - c[i], axis=1)
            pair = min(pair, float(np.min(d - r[i] - r[i + 1:])))
        return wall, pair

    best_c = best_r = None
    best_sum = -np.inf

    # Deterministic structured starts, followed by full feasible polishing.
    for counts in counts_list:
        for phase in (0, 1, 2):
            c0 = make_centers(counts, phase)
            r0 = lp_radii(c0)
            if r0 is None:
                continue

            # Slightly deflate before nonlinear movement, giving SLSQP
            # room to choose a better contact graph without initial conflicts.
            x0 = np.r_[c0.ravel(), 0.992 * r0]

            def con(x):
                return feasible(x[:2 * n].reshape(n, 2), x[2 * n:])

            previous = x0[2 * n:].copy()

            def objective(x):
                rr = x[2 * n:]
                return -np.sum(rr) + 0.004 * np.dot(rr - previous, rr - previous)

            sol = minimize(
                objective, x0, method="SLSQP",
                bounds=[(0.0, 1.0)] * (2 * n) + [(0.002, 0.20)] * n,
                constraints={"type": "ineq", "fun": con},
                options={"maxiter": 520, "ftol": 2e-10, "disp": False},
            )
            if not np.all(np.isfinite(sol.x)):
                continue

            c = sol.x[:2 * n].reshape(n, 2)
            if np.min(feasible(c, sol.x[2 * n:])) < -2e-7:
                continue
            r = lp_radii(c)
            if r is None or np.min(feasible(c, r)) < -2e-7:
                continue

            # A final unregularized pass from the recovered radius optimum.
            x1 = np.r_[c.ravel(), 0.996 * r]
            sol2 = minimize(
                lambda x: -np.sum(x[2 * n:]),
                x1, method="SLSQP",
                bounds=[(0.0, 1.0)] * (2 * n) + [(0.002, 0.20)] * n,
                constraints={"type": "ineq", "fun": con},
                options={"maxiter": 360, "ftol": 2e-10, "disp": False},
            )
            if np.all(np.isfinite(sol2.x)):
                cc = sol2.x[:2 * n].reshape(n, 2)
                rr = sol2.x[2 * n:]
                if np.min(feasible(cc, rr)) >= -2e-7:
                    c, r = cc, rr

            rr = lp_radii(c, relax=9.5e-7, wall_margin=2e-9)
            if rr is not None:
                wm, pm = clearances(c, rr)
                if wm >= -1e-8 and pm >= -9.7e-7:
                    r = rr

            if np.sum(r) > best_sum:
                best_c, best_r, best_sum = c.copy(), r.copy(), float(np.sum(r))

    if best_c is None:
        best_c = make_centers(counts_list[0], 1)
        best_r = lp_radii(best_c)

    # Final direct safety verification.  The permitted pair tolerance is used
    # conservatively; walls remain strictly contained.
    wm, pm = clearances(best_c, best_r)
    if wm < -1e-8 or pm < -9.7e-7:
        best_r = best_r * 0.999995

    return best_c, best_r, float(np.sum(best_r))


def compute_max_radii(centers):
    n = centers.shape[0]
    radii = np.minimum.reduce(
        (centers[:, 0], centers[:, 1], 1 - centers[:, 0], 1 - centers[:, 1]))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if radii[i] + radii[j] > d:
                s = d / (radii[i] + radii[j])
                radii[i] *= s
                radii[j] *= s
    return radii


def run_packing():
    return construct_packing()


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
    visualize(centers, radii)