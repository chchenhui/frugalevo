"""Direct constrained constructor for 26 nonoverlapping circles."""
import numpy as np
from scipy.optimize import minimize


def construct_packing():
    """Optimize centers and radii jointly with continuation on contact margins."""
    n = 26
    # Symmetric five-layer seed: the dense six-circle layer is central,
    # while the four five-circle layers use the available side clearance.
    rows = (5, 5, 6, 5, 5)
    rng = np.random.default_rng(26026)

    def seed(j):
        """Create a bounded, staggered five-layer starting configuration."""
        ys = np.array([.105, .302, .500, .698, .895], dtype=float)
        ys += rng.normal(0.0, .0035, 5)
        out = []
        for k, m in enumerate(rows):
            lo, hi = ((.08, .92) if m == 6 else
                      (.095, .905) if m == 5 else
                      (.185, .815))
            x = np.linspace(lo, hi, m)
            if (k + j) & 1:
                x += .5 * (x[1] - x[0])
            x += rng.normal(0.0, .0025, m)
            # Keep shifted six-point rows inside the SLSQP variable bounds.
            x = np.clip(x, .02, .98)
            out.extend(zip(x, np.full(m, ys[k])))
        return np.asarray(out, dtype=float)

    def feasible_radii(p):
        d = np.sqrt(np.maximum(((p[:, None] - p[None, :]) ** 2).sum(2),
                               1e-30))
        np.fill_diagonal(d, np.inf)
        b = np.minimum.reduce((p[:, 0], p[:, 1], 1-p[:, 0], 1-p[:, 1]))
        # Start near the natural scale of the five-layer lattice rather than
        # imposing a small artificial radius cap.
        r = np.minimum(b, .12)
        for _ in range(35):
            for i in range(n):
                r[i] = max(1e-10, min(b[i], np.min(d[i] - r)))
        return np.maximum(r * (1 - 2e-6), 1e-9)

    ii, jj = np.triu_indices(n, 1)

    def unpack(z):
        return z[:2*n].reshape(n, 2), z[2*n:]

    def constraints(z, margin):
        p, r = unpack(z)
        delta = p[ii] - p[jj]
        d = np.sqrt(np.maximum((delta * delta).sum(axis=1), 1e-24))
        return np.r_[p[:, 0] - r, p[:, 1] - r,
                     1 - p[:, 0] - r, 1 - p[:, 1] - r,
                     d - r[ii] - r[jj] - margin]

    bounds = [(1e-5, 1 - 1e-5)] * (2*n) + [(1e-9, .25)] * n
    best = None
    bestv = -np.inf

    for start in range(3):
        p = seed(start)
        # Retain a small feasibility cushion, but avoid unnecessarily
        # shrinking the initial objective before active-set optimization.
        r = feasible_radii(p) * (1 - .001)
        z = np.r_[p.ravel(), r]
        ok = True
        for margin in (2e-5, 1e-5, 1e-7):
            res = minimize(
                lambda q: -float(np.sum(q[2*n:])),
                z, method="SLSQP", bounds=bounds,
                constraints={"type": "ineq",
                             "fun": lambda q, m=margin:
                             constraints(q, m)},
                options={"maxiter": 350, "ftol": 5e-10, "disp": False})
            if not np.all(np.isfinite(res.x)):
                ok = False
                break
            z = res.x

        if ok:
            p, r = unpack(z)
            if np.all(constraints(z, 2e-8) >= -2e-7):
                val = float(np.sum(r))
                if val > bestv:
                    bestv = val
                    best = (p.copy(), r.copy())

    if best is None:
        p = seed(0)
        r = feasible_radii(p)
        return p, r, float(r.sum())

    # A short final active-set polish removes continuation slack.
    p, r = best
    z = np.r_[p.ravel(), r]
    res = minimize(
        lambda q: -float(np.sum(q[2*n:])), z, method="SLSQP",
        bounds=bounds,
        constraints={"type": "ineq",
                     "fun": lambda q: constraints(q, 2e-8)},
        options={"maxiter": 80, "ftol": 1e-10, "disp": False})
    if np.all(np.isfinite(res.x)):
        polished_p, polished_r = unpack(res.x)
        polished_value = float(np.sum(polished_r))
        incumbent_value = float(np.sum(best[1]))
        if (np.all(constraints(res.x, 0.0) >= -1e-7) and
                np.isfinite(polished_value) and
                polished_value >= incumbent_value):
            best = (polished_p.copy(), polished_r.copy())

    p, r = best
    p = np.clip(p, 1e-8, 1 - 1e-8)
    r = np.minimum(r, np.minimum.reduce(
        (p[:, 0], p[:, 1], 1-p[:, 0], 1-p[:, 1])))
    d = np.sqrt(np.maximum(((p[:, None] - p[None, :]) ** 2).sum(2), 1e-30))
    np.fill_diagonal(d, np.inf)
    for _ in range(3):
        for i, j in zip(ii, jj):
            excess = r[i] + r[j] + 2e-9 - d[i, j]
            if excess > 0:
                if r[i] >= r[j]:
                    r[i] = max(1e-10, r[i] - excess)
                else:
                    r[j] = max(1e-10, r[j] - excess)
    r *= 1 - 1e-8
    return p, r, float(np.sum(r))


def compute_max_radii(centers):
    n = centers.shape[0]
    r = np.minimum.reduce((centers[:,0], centers[:,1],
                           1-centers[:,0], 1-centers[:,1]))
    d = np.sqrt(np.maximum(((centers[:,None]-centers[None,:])**2).sum(2), 1e-30))
    np.fill_diagonal(d, np.inf)
    for i in range(n):
        for j in range(i+1, n):
            if r[i] + r[j] > d[i,j]:
                q = d[i,j] / (r[i]+r[j])
                r[i] *= q
                r[j] *= q
    return r


def run_packing():
    return construct_packing()


def visualize(centers, radii):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_aspect("equal"); ax.grid(True)
    for i, (c, r) in enumerate(zip(centers, radii)):
        ax.add_patch(Circle(c, r, alpha=.5))
        ax.text(c[0], c[1], str(i), ha="center", va="center")
    plt.show()


if __name__ == "__main__":
    c, r, s = run_packing()
    print("Sum of radii:", s)