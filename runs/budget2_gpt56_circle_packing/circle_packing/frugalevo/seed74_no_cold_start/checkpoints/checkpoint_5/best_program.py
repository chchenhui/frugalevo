"""Constructor-based circle packing for n=26 circles."""
import numpy as np


def compute_max_radii(centers):
    n = centers.shape[0]
    radii = np.minimum.reduce(
        [centers[:, 0], centers[:, 1], 1.0 - centers[:, 0], 1.0 - centers[:, 1]]
    ).copy()
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(centers[i] - centers[j]))
            if radii[i] + radii[j] > d:
                q = d / (radii[i] + radii[j])
                radii[i] *= q
                radii[j] *= q
    return radii


def _incumbent():
    r5 = 0.10
    r6 = 1.0 / 12.0
    x5 = np.linspace(r5, 1.0 - r5, 5)
    x6 = np.linspace(r6, 1.0 - r6, 6)
    rows = [
        (0.10, x5, r5),
        (0.31666666666666665, x5, r5),
        (0.50, x6, r6),
        (0.6833333333333333, x5, r5),
        (0.90, x5, r5),
    ]
    c = np.array([[x, y] for y, xs, _ in rows for x in xs], dtype=float)
    r = np.array([rr for _, xs, rr in rows for _ in xs], dtype=float)
    r *= 1.0 - 1e-10
    return c, r


def _repair(centers, radii, margin=2e-10):
    """Project radii inward repeatedly while preserving the optimized centers."""
    c = np.asarray(centers, dtype=float).copy()
    r = np.maximum(np.asarray(radii, dtype=float).copy(), 0.0)
    border = np.minimum.reduce([c[:, 0], c[:, 1], 1.0 - c[:, 0], 1.0 - c[:, 1]])
    r = np.minimum(r, np.maximum(0.0, border - margin))
    n = len(r)
    # Repeated sweeps handle chains of contacts without materially reducing
    # otherwise feasible circles.
    for _ in range(6):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.linalg.norm(c[i] - c[j]))
                allowed = max(0.0, d - margin)
                s = r[i] + r[j]
                if s > allowed and s > 0.0:
                    q = allowed / s
                    r[i] *= q
                    r[j] *= q
                    changed = True
        if not changed:
            break
    return c, r


def _zipper_seed():
    counts = [5, 6, 4, 6, 5]
    ys = [0.10, 0.30, 0.50, 0.70, 0.90]
    centers = []
    radii = []
    for k, (m, y) in enumerate(zip(counts, ys)):
        if m == 5:
            xs = np.linspace(0.10, 0.90, 5)
            rr = 0.075 if k in (0, 4) else 0.040
        elif m == 6:
            xs = np.linspace(1.0 / 12.0, 11.0 / 12.0, 6)
            rr = 0.040
        else:
            xs = np.linspace(0.20, 0.80, 4)
            rr = 0.040
        centers.extend((x, y) for x in xs)
        radii.extend([rr] * m)
    return np.asarray(centers), np.asarray(radii)


def _polish(centers, radii, maxiter=550):
    try:
        from scipy.optimize import minimize
    except Exception:
        return centers, radii

    n = 26
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m = 4 * n + len(pairs)

    z0 = np.concatenate([centers.ravel(), radii])

    def fun(z):
        return -float(np.sum(z[2 * n:]))

    def jacfun(z):
        g = np.zeros(3 * n)
        g[2 * n:] = -1.0
        return g

    def con(z):
        xy = z[:2 * n].reshape(n, 2)
        rr = z[2 * n:]
        out = np.empty(m)
        out[0:n] = xy[:, 0] - rr
        out[n:2*n] = xy[:, 1] - rr
        out[2*n:3*n] = 1.0 - xy[:, 0] - rr
        out[3*n:4*n] = 1.0 - xy[:, 1] - rr
        for q, (i, j) in enumerate(pairs):
            d = xy[i] - xy[j]
            out[4*n + q] = np.dot(d, d) - (rr[i] + rr[j]) ** 2
        return out

    def jaccon(z):
        xy = z[:2 * n].reshape(n, 2)
        rr = z[2 * n:]
        a = np.zeros((m, 3 * n))
        for i in range(n):
            a[i, 2*i] = 1.0
            a[i, 2*n+i] = -1.0
            a[n+i, 2*i+1] = 1.0
            a[n+i, 2*n+i] = -1.0
            a[2*n+i, 2*i] = -1.0
            a[2*n+i, 2*n+i] = -1.0
            a[3*n+i, 2*i+1] = -1.0
            a[3*n+i, 2*n+i] = -1.0
        for q, (i, j) in enumerate(pairs):
            d = xy[i] - xy[j]
            row = 4*n + q
            a[row, 2*i:2*i+2] = 2.0 * d
            a[row, 2*j:2*j+2] = -2.0 * d
            a[row, 2*n+i] = -2.0 * (rr[i] + rr[j])
            a[row, 2*n+j] = -2.0 * (rr[i] + rr[j])
        return a

    bounds = [(0.0, 1.0)] * (2 * n) + [(1e-7, 0.35)] * n
    result = minimize(
        fun, z0, jac=jacfun,
        constraints={"type": "ineq", "fun": con, "jac": jaccon},
        bounds=bounds, method="SLSQP",
        options={"maxiter": maxiter, "ftol": 1e-10, "disp": False},
    )
    if not np.all(np.isfinite(result.x)):
        return centers, radii
    return result.x[:2*n].reshape(n, 2), result.x[2*n:]


def construct_packing():
    """Jointly polish two feasible row layouts with free centers and radii."""
    best_c, best_r = _incumbent()
    best_sum = float(np.sum(best_r))

    # The first run releases the incumbent's fixed radii, heights, and
    # horizontal coordinates simultaneously.
    c0, r0 = _incumbent()
    c, r = _polish(c0, r0, maxiter=500)
    c, r = _repair(c, r)
    s = float(np.sum(r))
    if s > best_sum:
        best_c, best_r, best_sum = c, r, s

    # The second run exposes the complementary nonsymmetric contact pattern.
    c1, r1 = _incumbent()
    middle = slice(10, 16)
    c1[middle, 0] += 0.25 / 6.0
    c1[middle, 0] = np.minimum(c1[middle, 0], 1.0 - r1[middle])
    r1 *= 0.97
    c, r = _polish(c1, r1, maxiter=500)
    c, r = _repair(c, r)
    s = float(np.sum(r))
    if s > best_sum:
        best_c, best_r, best_sum = c, r, s

    return best_c, best_r, best_sum


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