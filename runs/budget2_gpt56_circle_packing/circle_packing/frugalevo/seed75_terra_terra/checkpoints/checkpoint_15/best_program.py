"""Constructor-based circle packing for 26 circles."""
import numpy as np


def construct_packing():
    """Build 26 circles with multistart elite contact-graph crossover polishing."""
    from scipy.optimize import minimize

    n = 26
    nv = 3 * n
    pairs = np.array([(i, j) for i in range(n) for j in range(i + 1, n)],
                     dtype=int)
    ii, jj = pairs[:, 0], pairs[:, 1]

    corners = np.array(
        [[.055, .055], [.945, .055], [.945, .945], [.055, .945]],
        dtype=float)
    sides = np.array(
        [[x, .055] for x in (.25, .50, .75)] +
        [[x, .945] for x in (.25, .50, .75)] +
        [[.055, y] for y in (.25, .50, .75)] +
        [[.945, y] for y in (.25, .50, .75)],
        dtype=float)
    core = np.array(
        [[x, .27] for x in (.20, .40, .60, .80)] +
        [[x, .50] for x in (.27, .50, .73)] +
        [[x, .73] for x in (.27, .50, .73)],
        dtype=float)

    centers0 = np.vstack((corners, sides, core))
    base = np.r_[centers0.ravel(), np.full(n, .035)]

    def clearance(z):
        c = z[:2 * n].reshape(n, 2)
        r = z[2 * n:]
        d = np.sqrt(np.sum((c[ii] - c[jj]) ** 2, axis=1))
        return np.r_[c[:, 0] - r, 1.0 - c[:, 0] - r,
                     c[:, 1] - r, 1.0 - c[:, 1] - r,
                     d - r[ii] - r[jj]]

    def clearance_jacobian(z):
        c = z[:2 * n].reshape(n, 2)
        jac = np.zeros((4 * n + len(ii), nv))
        k = np.arange(n)

        jac[k, 2 * k] = 1.0
        jac[k, 2 * n + k] = -1.0
        jac[n + k, 2 * k] = -1.0
        jac[n + k, 2 * n + k] = -1.0
        jac[2 * n + k, 2 * k + 1] = 1.0
        jac[2 * n + k, 2 * n + k] = -1.0
        jac[3 * n + k, 2 * k + 1] = -1.0
        jac[3 * n + k, 2 * n + k] = -1.0

        delta = c[ii] - c[jj]
        d = np.sqrt(np.sum(delta * delta, axis=1))
        u = delta / np.maximum(d[:, None], 1e-14)
        row = 4 * n + np.arange(len(ii))
        jac[row, 2 * ii] = u[:, 0]
        jac[row, 2 * ii + 1] = u[:, 1]
        jac[row, 2 * jj] = -u[:, 0]
        jac[row, 2 * jj + 1] = -u[:, 1]
        jac[row, 2 * n + ii] = -1.0
        jac[row, 2 * n + jj] = -1.0
        return jac

    def make_valid(z):
        c = np.clip(z[:2 * n].reshape(n, 2), 0.0, 1.0).copy()
        r = np.maximum(z[2 * n:].copy(), 1e-10)
        edge = np.minimum.reduce((c[:, 0], c[:, 1],
                                  1.0 - c[:, 0], 1.0 - c[:, 1]))
        scale = min(1.0, float(np.min(edge / r)))
        d = np.sqrt(np.sum((c[ii] - c[jj]) ** 2, axis=1))
        scale = min(scale, float(np.min(d / (r[ii] + r[jj]))))
        return c, r * max(0.0, scale * (1.0 - 2e-10))

    def polish(z0, iterations):
        c0, r0 = make_valid(z0)
        result = minimize(
            lambda z: -np.sum(z[2 * n:]),
            np.r_[c0.ravel(), r0],
            jac=lambda z: objective_jac,
            method="SLSQP",
            bounds=bounds,
            constraints=constraint,
            options={"maxiter": iterations, "ftol": 2e-12, "disp": False},
        )
        return make_valid(result.x)

    best_c, best_r = make_valid(base)
    best_value = float(best_r.sum())

    bounds = [(0.0, 1.0)] * (2 * n) + [(1e-10, .25)] * n
    objective_jac = np.r_[np.zeros(2 * n), -np.ones(n)]
    constraint = {"type": "ineq", "fun": clearance, "jac": clearance_jacobian}
    rng = np.random.default_rng(26031991)
    elites = []

    def add_elite(c, r, value):
        """Keep eight spatially distinct feasible local optima for recombination."""
        nonlocal elites
        ordered = c[np.lexsort((c[:, 1], c[:, 0]))]
        key = np.round(ordered / 2e-3).astype(np.int16).ravel()
        for q, (_, _, old_value, old_key) in enumerate(elites):
            if np.array_equal(key, old_key):
                if value > old_value:
                    elites[q] = (c.copy(), r.copy(), value, key.copy())
                break
        else:
            elites.append((c.copy(), r.copy(), value, key.copy()))
        elites.sort(key=lambda item: item[2], reverse=True)
        del elites[8:]

    # The original successful boundary annulus retains several alternate local
    # contact graphs, rather than discarding all but one incumbent.
    for trial in range(180):
        z0 = base.copy()
        if trial:
            z0[:2 * n] += rng.uniform(-.026, .026, size=2 * n)
            z0[:2 * n] = np.clip(z0[:2 * n], .0005, .9995)
        z0[2 * n:] *= rng.uniform(.52, .91, size=n)
        c, r = polish(z0, 800)
        value = float(r.sum())
        add_elite(c, r, value)
        if value > best_value:
            best_c, best_r, best_value = c, r, value

    # A bounded deletion/reinsertion phase supplies a real contact-graph escape.
    # Only candidates that survive exact constrained polishing can replace best.
    grid = np.linspace(.035, .965, 31)
    for trial in range(36):
        c = best_c.copy()
        r = best_r.copy()
        dmat = np.sqrt(np.sum((c[:, None, :] - c[None, :, :]) ** 2, axis=2))
        np.fill_diagonal(dmat, np.inf)
        slack = np.min(dmat - r[:, None] - r[None, :], axis=1)
        candidates = np.argsort(r + 0.25 * slack)[:12]
        victim = int(candidates[trial % len(candidates)])

        keep = np.ones(n, dtype=bool)
        keep[victim] = False
        ck, rk = c[keep], r[keep]
        xx, yy = np.meshgrid(grid, grid, indexing="ij")
        pts = np.column_stack((xx.ravel(), yy.ravel()))
        border = np.minimum.reduce((pts[:, 0], pts[:, 1],
                                    1.0 - pts[:, 0], 1.0 - pts[:, 1]))
        dist = np.sqrt(np.sum((pts[:, None, :] - ck[None, :, :]) ** 2,
                              axis=2))
        score = np.minimum(border, np.min(dist - rk[None, :], axis=1))
        p = pts[int(np.argmax(score))]
        c[victim] = p
        r[victim] = max(1e-5, .55 * float(np.max(score)))

        z0 = np.r_[c.ravel(), np.maximum(1e-9, r - 4e-4)]
        cp, rp = polish(z0, 650)
        value = float(rp.sum())
        add_elite(cp, rp, value)
        if value > best_value:
            best_c, best_r, best_value = cp, rp, value

    # Use both parent directions.  The earlier one-direction splice could only
    # retain A's left/lower chain and B's right/upper chain, missing the
    # complementary contact-graph exchange.  Six pairs times two axes times two
    # directions gives exactly 24 bounded crossover optimizations.
    pair_count = 0
    for left in range(len(elites)):
        for right in range(left + 1, len(elites)):
            ca, ra, _, _ = elites[left]
            cb, rb, _, _ = elites[right]
            for axis in (0, 1):
                order_a = np.argsort(ca[:, axis], kind="stable")
                order_b = np.argsort(cb[:, axis], kind="stable")
                for reverse in (False, True):
                    if reverse:
                        take_a, take_b = order_a[n // 2:], order_b[:n // 2]
                    else:
                        take_a, take_b = order_a[:n // 2], order_b[n // 2:]

                    child_c = np.vstack((ca[take_a], cb[take_b])).copy()
                    child_r = np.r_[ra[take_a], rb[take_b]].copy()
                    cc, rr = make_valid(
                        np.r_[child_c.ravel(),
                              np.maximum(1e-10, child_r - 1e-3)])
                    cp, rp = polish(np.r_[cc.ravel(), rr], 700)
                    value = float(rp.sum())
                    if value > best_value:
                        best_c, best_r, best_value = cp, rp, value
            pair_count += 1
            if pair_count >= 6:
                break
        if pair_count >= 6:
            break

    # Final exact polish is deliberately started from the selected incumbent.
    c, r = polish(np.r_[best_c.ravel(), best_r], 900)
    if r.sum() > best_value:
        best_c, best_r, best_value = c, r, float(r.sum())

    best_c, best_r = make_valid(np.r_[best_c.ravel(), best_r])
    return best_c, best_r, float(best_r.sum())


def compute_max_radii(centers):
    n = centers.shape[0]
    radii = np.ones(n)
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(x, y, 1 - x, 1 - y)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if radii[i] + radii[j] > dist:
                scale = dist / (radii[i] + radii[j])
                radii[i] *= scale
                radii[j] *= scale
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
        ax.add_patch(Circle(center, radius, alpha=.5))
        ax.text(center[0], center[1], str(i), ha="center", va="center")
    plt.title(f"Circle Packing (n={len(centers)}, sum={sum(radii):.6f})")
    plt.show()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii}")