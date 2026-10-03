"""Deterministic multi-topology constrained constructor for 26 circles."""
import numpy as np
from scipy.optimize import minimize


def _make_seed(rows, variant):
    ys = np.array([.105, .302, .500, .698, .895], dtype=float)
    if variant & 1:
        ys = ys[::-1].copy()
    if variant & 2:
        ys += np.array([-.006, .004, 0., -.004, .006])
    out = []
    for k, m in enumerate(rows):
        if m == 6:
            lo, hi = .075, .925
        elif m == 5:
            lo, hi = .095, .905
        else:
            lo, hi = .18, .82
        x = np.linspace(lo, hi, m)
        if (k + variant) & 1:
            x += .5 * (x[1] - x[0])
        if variant & 2:
            x += (.004 if k % 2 else -.004)
        out.extend(zip(x, np.full(m, ys[k])))
    return np.asarray(out, dtype=float)


def _initial_radii(p):
    n = len(p)
    d = np.sqrt(np.maximum(((p[:, None] - p[None, :]) ** 2).sum(2), 1e-30))
    np.fill_diagonal(d, np.inf)
    b = np.minimum.reduce((p[:, 0], p[:, 1], 1-p[:, 0], 1-p[:, 1]))
    r = np.minimum(b, .14)
    for _ in range(45):
        for i in range(n):
            r[i] = max(1e-10, min(b[i], np.min(d[i] - r)))
    return np.maximum(r * (1 - 2e-5), 1e-9)


def construct_packing():
    n = 26
    ii, jj = np.triu_indices(n, 1)

    def unpack(z):
        return z[:2*n].reshape(n, 2), z[2*n:]

    def cons(z, margin):
        p, r = unpack(z)
        q = p[ii] - p[jj]
        d = np.sqrt(np.maximum((q*q).sum(axis=1), 1e-24))
        return np.r_[p[:, 0]-r, p[:, 1]-r,
                     1-p[:, 0]-r, 1-p[:, 1]-r,
                     d-r[ii]-r[jj]-margin]

    bounds = [(1e-5, 1-1e-5)]*(2*n) + [(1e-9, .25)]*n
    topologies = ((5,5,6,5,5), (4,6,6,6,4),
                  (5,6,5,6,4), (4,6,5,6,5))
    best = None
    best_sum = -np.inf

    # Use four deterministic layer topologies and an auxiliary clearance
    # variable, first finding a robust contact topology before enlarging radii.
    for ti, rows in enumerate(topologies):
        p = _make_seed(rows, ti)
        r = _initial_radii(p)
        z0 = np.r_[p.ravel(), r, 1e-7]

        def unpack_margin(q):
            return q[:2*n].reshape(n, 2), q[2*n:3*n], q[-1]

        def margin_cons(q):
            qq = q[:-1]
            return np.r_[cons(qq, 0.), q[-1]]

        fixed_bounds = ([(1e-5, 1-1e-5)]*(2*n) +
                        [(float(x), float(x)) for x in r] + [(0., .02)])
        res = minimize(
            lambda q: -q[-1], z0, method="SLSQP", bounds=fixed_bounds,
            constraints={"type": "ineq", "fun": margin_cons},
            options={"maxiter": 260, "ftol": 2e-9, "disp": False})
        if not np.all(np.isfinite(res.x)):
            continue

        # Release radii after the topology has acquired positive clearance.
        z = res.x.copy()
        z[2*n:3*n] = np.maximum(z[2*n:3*n], 1e-9)
        free_bounds = bounds + [(0., .02)]
        for objective, margin, iters in (
                (lambda q: -float(np.sum(q[2*n:3*n])) -
                 2e-3*q[-1], 1e-7, 260),
                (lambda q: -float(np.sum(q[2*n:3*n])), 2e-8, 260)):
            def phase_cons(q, m=margin):
                return np.r_[cons(q[:-1], m), q[-1] - m]
            res = minimize(
                objective, z, method="SLSQP", bounds=free_bounds,
                constraints={"type": "ineq", "fun": phase_cons},
                options={"maxiter": iters, "ftol": 2e-9, "disp": False})
            if not np.all(np.isfinite(res.x)):
                break
            z = res.x
        else:
            p1, r1, _ = unpack_margin(z)
            if np.all(cons(z[:-1], 0.) >= -3e-7):
                val = float(r1.sum())
                if val > best_sum:
                    best_sum = val
                    best = (p1.copy(), r1.copy())

    if best is None:
        p = _make_seed((5,5,6,5,5), 0)
        r = _initial_radii(p)
        return p, r, float(r.sum())

    p, r = best
    z = np.r_[p.ravel(), r, 1e-8]
    polish_bounds = bounds + [(0., .02)]
    res = minimize(
        lambda q: -float(np.sum(q[2*n:3*n])) - 1e-3*q[-1],
        z, method="SLSQP", bounds=polish_bounds,
        constraints={"type": "ineq",
                     "fun": lambda q: np.r_[cons(q[:-1], 1e-8),
                                             q[-1] - 1e-8]},
        options={"maxiter": 100, "ftol": 1e-10, "disp": False})
    if (np.all(np.isfinite(res.x)) and
            np.all(cons(res.x[:-1], 0.) >= -2e-7)):
        pp, rr = unpack(res.x[:-1])
        if rr.sum() >= r.sum():
            p, r = pp.copy(), rr.copy()

    p = np.clip(p, 1e-8, 1-1e-8)
    b = np.minimum.reduce((p[:,0], p[:,1], 1-p[:,0], 1-p[:,1]))
    r = np.minimum(r, b)
    d = np.sqrt(np.maximum(((p[:,None]-p[None,:])**2).sum(2), 1e-30))
    np.fill_diagonal(d, np.inf)
    for i in range(n):
        for j in range(i+1, n):
            excess = r[i] + r[j] + 2e-9 - d[i,j]
            if excess > 0:
                if r[i] >= r[j]:
                    r[i] = max(1e-10, r[i]-excess)
                else:
                    r[j] = max(1e-10, r[j]-excess)
    r *= 1 - 1e-8
    return p, r, float(r.sum())


def compute_max_radii(centers):
    n = centers.shape[0]
    r = np.minimum.reduce((centers[:,0], centers[:,1],
                           1-centers[:,0], 1-centers[:,1]))
    d = np.sqrt(np.maximum(((centers[:,None]-centers[None,:])**2).sum(2), 1e-30))
    np.fill_diagonal(d, np.inf)
    for i in range(n):
        for j in range(i+1,n):
            if r[i]+r[j] > d[i,j]:
                q = d[i,j]/(r[i]+r[j])
                r[i] *= q
                r[j] *= q
    return r


def run_packing():
    return construct_packing()


def visualize(centers, radii):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_aspect("equal"); ax.grid(True)
    for i,(c,r) in enumerate(zip(centers,radii)):
        ax.add_patch(Circle(c,r,alpha=.5))
        ax.text(c[0],c[1],str(i),ha="center",va="center")
    plt.show()


if __name__ == "__main__":
    c,r,s = run_packing()
    print("Sum of radii:", s)