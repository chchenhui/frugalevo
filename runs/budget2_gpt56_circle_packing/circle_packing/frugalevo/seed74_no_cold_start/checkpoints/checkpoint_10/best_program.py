"""Deterministic multistart constructor-based circle packing for n=26 circles."""
import numpy as np


def compute_max_radii(centers):
    n = centers.shape[0]
    radii = np.minimum.reduce(
        [centers[:, 0], centers[:, 1], 1.0 - centers[:, 0], 1.0 - centers[:, 1]]
    ).copy()
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(centers[i] - centers[j]))
            s = radii[i] + radii[j]
            if s > d and s > 0.0:
                q = d / s
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
    centers = np.array([[x, y] for y, xs, _ in rows for x in xs], dtype=float)
    radii = np.array([rr for _, xs, rr in rows for _ in xs], dtype=float)
    radii *= 1.0 - 1e-9
    return centers, radii


def _repair(centers, radii, margin=3e-8):
    """Return a conservatively feasible packing without moving centers."""
    c = np.asarray(centers, dtype=float).copy()
    c = np.clip(c, 0.0, 1.0)
    r = np.maximum(np.asarray(radii, dtype=float).copy(), 0.0)
    border = np.minimum.reduce([c[:, 0], c[:, 1], 1.0 - c[:, 0], 1.0 - c[:, 1]])
    r = np.minimum(r, np.maximum(0.0, border - margin))
    n = len(r)
    # Later reductions can only improve already visited constraints, so a
    # handful of sweeps is ample and also protects against roundoff chains.
    for _ in range(4):
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


def _row_seed(counts, reflected=False):
    """A small-radius, feasible alternating-layer seed."""
    ys = np.linspace(0.075, 0.925, len(counts))
    centers = []
    for k, (m, y) in enumerate(zip(counts, ys)):
        # The alternating offsets provide diagonal, rather than rectangular,
        # neighbors; clipping keeps the initial seed comfortably in the box.
        spacing = 0.86 / max(1, m - 1)
        offset = 0.5 * spacing if (k & 1) else 0.0
        xs = np.linspace(0.07, 0.93, m) + offset
        xs = np.clip(xs, 0.07, 0.93)
        if reflected:
            xs = 1.0 - xs[::-1]
        centers.extend((x, y) for x in xs)
    c = np.asarray(centers, dtype=float)
    return c, np.full(26, 0.025, dtype=float)


def _polish(centers, radii, maxiter=850):
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

    bounds = [(0.0, 1.0)] * (2 * n) + [(1e-8, 0.35)] * n
    result = minimize(
        fun,
        z0,
        jac=jacfun,
        constraints={"type": "ineq", "fun": con, "jac": jaccon},
        bounds=bounds,
        method="SLSQP",
        options={"maxiter": maxiter, "ftol": 2e-10, "disp": False},
    )
    if not np.all(np.isfinite(result.x)):
        return centers, radii
    return result.x[:2*n].reshape(n, 2), result.x[2*n:]


def construct_packing():
    """Run a bounded deterministic collection of distinct packing starts."""
    best_c, best_r = _incumbent()
    best_c, best_r = _repair(best_c, best_r)
    best_sum = float(np.sum(best_r))

    starts = []

    c, r = _incumbent()
    starts.append((c, r))

    c, r = _incumbent()
    c[10:16, 0] += 0.25 / 6.0
    c[:, 0] = np.clip(c[:, 0], 0.02, 0.98)
    starts.append((c, r * 0.96))

    # Contact-breaking starts near the strong row construction.
    rng = np.random.default_rng(371942)
    base_c, _ = _incumbent()
    for k in range(8):
        delta = rng.uniform(-0.018, 0.018, size=(26, 2))
        # Keep outer wall contacts from all moving in the same direction.
        delta[:, 1] *= 0.85
        if k & 1:
            delta[:, 0] *= -1.0
        c = np.clip(base_c + delta, 0.025, 0.975)
        starts.append((c, np.full(26, 0.055, dtype=float)))

    # Two genuinely different six-layer defect topologies and reflections.
    for counts, reflected in [
        ([4, 5, 4, 5, 4, 4], False),
        ([4, 5, 4, 5, 4, 4], True),
        ([4, 5, 4, 4, 5, 4], False),
        ([4, 5, 4, 4, 5, 4], True),
    ]:
        starts.append(_row_seed(counts, reflected))

    # The first two starts are already good; give them the full budget.
    for q, (c0, r0) in enumerate(starts):
        c0, r0 = _repair(c0, r0)
        c, r = _polish(c0, r0, maxiter=900 if q < 2 else 780)
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