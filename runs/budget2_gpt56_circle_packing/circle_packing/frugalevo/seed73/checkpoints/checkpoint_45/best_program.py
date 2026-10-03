"""Constructor-based circle packing for n=26 circles."""
import numpy as np


def construct_packing():
    from scipy.optimize import minimize, linprog

    n = 26
    patterns = ((5, 5, 6, 5, 5), (5, 6, 5, 5, 5), (5, 5, 5, 6, 5))
    best_c = best_r = None
    best_value = -np.inf

    def make_centers(q, counts):
        out = []
        for k, m in enumerate(counts):
            y, left, right, shear = q[4 * k:4 * k + 4]
            xs = np.linspace(left, right, m) + shear * (k - 2)
            out.extend((x, y) for x in xs)
        return np.asarray(out, dtype=float)

    def radius_lp(c):
        wall = np.min(np.column_stack(
            (c[:, 0], c[:, 1], 1 - c[:, 0], 1 - c[:, 1])), axis=1)
        aa, bb = [], []
        for i in range(n):
            z = np.zeros(n)
            z[i] = 1.0
            aa.append(z)
            bb.append(wall[i])
            for j in range(i):
                z = np.zeros(n)
                z[i] = z[j] = 1.0
                aa.append(z)
                bb.append(np.linalg.norm(c[i] - c[j]))
        ans = linprog(-np.ones(n), A_ub=np.asarray(aa), b_ub=np.asarray(bb),
                      bounds=[(0.002, 0.2)] * n, method="highs")
        return ans.x if ans.success else np.full(n, 0.075)

    def feasible(c, r):
        out = [c[:, 0] - r, c[:, 1] - r,
               1 - c[:, 0] - r, 1 - c[:, 1] - r]
        for i in range(n):
            d = c[i + 1:] - c[i]
            out.append(np.sum(d * d, axis=1) - (r[i] + r[i + 1:]) ** 2)
        return np.concatenate(out)

    def linear_clearances(c, r):
        wall = np.min(np.column_stack((
            c[:, 0] - r, c[:, 1] - r,
            1.0 - c[:, 0] - r, 1.0 - c[:, 1] - r)))
        pair = np.inf
        for i in range(n):
            d = c[i + 1:] - c[i]
            if len(d):
                pair = min(pair, float(np.min(
                    np.linalg.norm(d, axis=1) - r[i] - r[i + 1:])))
        return float(wall), float(pair)

    starts = []
    for counts in patterns:
        for phase in (0, 1, 2):
            q = []
            for k, m in enumerate(counts):
                y = 0.101 + 0.198 * k + 0.002 * (phase - 1) * (k - 2)
                left, right = ((0.095, 0.905) if m == 5 else
                               (1 / 12, 11 / 12))
                if (k + phase) % 2:
                    left, right = left + 0.006, right - 0.006
                q.extend((y, left, right, 0.003 * ((k + phase) % 2)))
            starts.append((np.asarray(q), counts))

    for q, counts in starts:
        c = make_centers(q, counts)
        r = radius_lp(c)
        rows, p = [], 0
        for m in counts:
            rows.append(np.arange(p, p + m))
            p += m

        for direction in (range(4), range(3, -1, -1)):
            for band in direction:
                active = np.concatenate((rows[band], rows[band + 1]))
                fixed = np.ones(n, dtype=bool)
                fixed[active] = False
                old = np.r_[c[active].ravel(), r[active]]
                na = len(active)

                def unpack(v):
                    cc, rr = c.copy(), r.copy()
                    cc[active] = v[:2 * na].reshape(-1, 2)
                    rr[active] = v[2 * na:]
                    return cc, rr

                def cons(v):
                    cc, rr = unpack(v)
                    z = list(feasible(cc, rr))
                    for row in (rows[band], rows[band + 1]):
                        for a, b in zip(row[:-1], row[1:]):
                            z.append(cc[b, 0] - cc[a, 0] - 0.01)
                    return np.asarray(z)

                def obj(v):
                    cc, rr = unpack(v)
                    return -np.sum(rr[active]) - .015 * np.sum(rr[fixed])

                t = minimize(
                    obj, old, method="SLSQP",
                    bounds=[(.035, .965)] * (2 * na) + [(.002, .2)] * na,
                    constraints={"type": "ineq", "fun": cons},
                    options={"maxiter": 140, "ftol": 2e-9})
                if np.all(np.isfinite(t.x)):
                    tc, tr = unpack(t.x)
                    if np.min(feasible(tc, tr)) >= -2e-7:
                        c, r = tc, radius_lp(tc)

        x0 = np.r_[c.ravel(), r]
        polished = None

        def all_cons(x):
            return feasible(x[:2 * n].reshape(n, 2), x[2 * n:])

        for release, strength in ((.992, 2e-2), (.996, 5e-3), (1.0, 0.0)):
            if release != 1.0:
                x0[2 * n:] *= release
            previous = x0[2 * n:].copy()

            def obj(x):
                d = x[2 * n:] - previous
                return -np.sum(x[2 * n:]) + strength * np.dot(d, d)

            t = minimize(
                obj, x0, method="SLSQP",
                bounds=[(0, 1)] * (2 * n) + [(.002, .2)] * n,
                constraints={"type": "ineq", "fun": all_cons},
                options={"maxiter": 320, "ftol": 2e-10})
            if np.all(np.isfinite(t.x)):
                tc = t.x[:2 * n].reshape(n, 2)
                tr = t.x[2 * n:]
                if np.min(feasible(tc, tr)) >= -2e-7:
                    x0, polished = t.x.copy(), t

        if polished is not None:
            pc = polished.x[:2 * n].reshape(n, 2)
            pr = polished.x[2 * n:]
            if np.min(feasible(pc, pr)) >= -2e-7 and np.sum(pr) > best_value:
                best_c, best_r = pc.copy(), pr.copy()
                best_value = float(np.sum(pr))

    if best_c is None:
        q, counts = starts[0]
        best_c = make_centers(q, counts)
        best_r = radius_lp(best_c)

    # First create the incumbent's conservative strict-feasibility candidate.
    candidate = np.asarray(best_r, dtype=float).copy() * 0.9999985
    wm, pm = linear_clearances(best_c, candidate)

    # If numerical SLSQP noise survived scaling, use the prior conservative
    # fallback before extracting a fresh clearance budget.
    if wm < 0.0 or pm < 0.0:
        candidate = np.asarray(best_r, dtype=float).copy() * 0.999995
        wm, pm = linear_clearances(best_c, candidate)

    # A common additive lift consumes one unit of wall clearance and two units
    # of pair clearance.  Leave five percent of that measured clearance unused.
    lift = 0.95 * max(0.0, min(wm, 0.5 * pm))
    if np.isfinite(lift) and lift > 0.0:
        candidate += lift

    # Last deterministic strict check; only reduce, never risk invalid output.
    wm, pm = linear_clearances(best_c, candidate)
    if wm < 0.0 or pm < 0.0:
        candidate *= 0.999999

    return best_c, candidate, float(np.sum(candidate))


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