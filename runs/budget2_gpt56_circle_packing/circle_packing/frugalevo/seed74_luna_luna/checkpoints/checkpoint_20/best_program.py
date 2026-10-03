"""Deterministic multi-topology constrained constructor for 26 circles."""
import numpy as np
from scipy.optimize import minimize


def _make_seed(rows, variant):
    """Construct a 26-point asymmetric seed by sequential grid clearance growth."""
    n = int(sum(rows))
    side = 55
    axis = np.linspace(.012, .988, side, dtype=float)
    xx, yy = np.meshgrid(axis, axis)
    grid = np.column_stack((xx.ravel(), yy.ravel()))
    wall = np.minimum.reduce((grid[:, 0], grid[:, 1],
                              1.0-grid[:, 0], 1.0-grid[:, 1]))
    used = np.zeros(len(grid), dtype=bool)
    centers = []
    radii = []

    policy = int(variant % 3)
    for k in range(n):
        if not centers:
            target = np.array((.5, .5), dtype=float)
            score = -np.sum((grid-target)**2, axis=1)
        else:
            c = np.asarray(centers, dtype=float)
            oldr = np.asarray(radii, dtype=float)
            dist = np.sqrt(((grid[:, None]-c[None, :])**2).sum(axis=2))
            clearance = np.min(dist-oldr[None, :], axis=1)
            clearance = np.minimum(clearance, wall)

            # Boundary-first and interior-first policies alter the growth
            # objective rather than merely changing a random seed.
            if policy == 1:
                weight = .16 if k < 10 else -.035
                score = clearance + weight*wall
            elif policy == 2:
                weight = .11 if k < 12 else -.045
                score = clearance + weight*(1.0-wall)
            else:
                score = clearance + .018*wall
        score[used] = -np.inf
        pick = int(np.argmax(score))
        used[pick] = True
        centers.append(grid[pick].copy())

        if len(centers) == 1:
            rr = wall[pick]
        else:
            prev = np.asarray(centers[:-1], dtype=float)
            nearest = float(np.min(np.sqrt(((grid[pick]-prev)**2).sum(axis=1))))
            rr = min(float(wall[pick]), nearest)
        radii.append(max(.006, .46*rr))

    p = np.asarray(centers, dtype=float)

    # Replace grid quantization by a conservative local clearance centroid.
    # A move is accepted only when it increases clearance from all other
    # centers, so this stage cannot destroy the sequential topology.
    for i in range(n):
        other = np.delete(p, i, axis=0)
        dist = np.sqrt(((grid[:, None]-other[None, :])**2).sum(axis=2))
        clear = np.minimum(wall, np.min(dist, axis=1))
        clear[used] = -np.inf
        top = np.argpartition(clear, -4)[-4:]
        cand = grid[top].mean(axis=0)
        old = min(np.min(np.sqrt(((p[i]-other)**2).sum(axis=1))),
                  min(cand[0], cand[1], 1-cand[0], 1-cand[1]))
        new = min(np.min(np.sqrt(((cand-other)**2).sum(axis=1))),
                  min(cand[0], cand[1], 1-cand[0], 1-cand[1]))
        if new > old + 1e-5:
            p[i] = cand
    return p


def _initial_radii(p):
    """Compute feasible radii by fixed-center clearance propagation."""
    n = len(p)
    d = np.sqrt(np.maximum(((p[:, None]-p[None, :])**2).sum(2), 1e-30))
    np.fill_diagonal(d, np.inf)
    b = np.minimum.reduce((p[:, 0], p[:, 1], 1-p[:, 0], 1-p[:, 1]))
    r = np.minimum(b, .16).astype(float, copy=True)
    for _ in range(60):
        for i in range(n):
            r[i] = max(1e-10, min(b[i], np.min(d[i]-r)))
    return np.maximum(r*(1-2e-5), 1e-9)


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
        # Keep the sequential seed strictly inside its clearance envelope
        # during the fixed-radius margin-growth phase.
        r = np.maximum(r*.97, 1e-9)
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

        # Solve the fixed-center radius problem as a bounded linear program.
        # Its active set is more reliable than a simultaneous SLSQP radius
        # release, particularly for nearly equal neighboring circles.
        z = res.x.copy()
        p0 = z[:2*n].reshape(n, 2).copy()
        r0 = np.maximum(z[2*n:3*n].copy(), 1e-9)
        b0 = np.minimum.reduce((p0[:, 0], p0[:, 1],
                                1-p0[:, 0], 1-p0[:, 1]))
        d0 = np.sqrt(np.maximum(
            ((p0[:, None] - p0[None, :]) ** 2).sum(axis=2), 1e-24))
        np.fill_diagonal(d0, np.inf)

        def fixed_radius_problem(rr):
            """Return fixed-center boundary and pairwise clearance."""
            return np.r_[b0 - rr - 2e-8,
                         d0[ii, jj] - rr[ii] - rr[jj] - 2e-8]

        # This is the exact LP: maximize the radius sum subject to every
        # boundary and pairwise inequality, with a strict numerical buffer.
        from scipy.optimize import linprog
        A = np.zeros((4*n + len(ii), n), dtype=float)
        u = np.empty(4*n + len(ii), dtype=float)
        for k in range(n):
            A[4*k, k] = 1.
            A[4*k+1, k] = 1.
            A[4*k+2, k] = 1.
            A[4*k+3, k] = 1.
            u[4*k:4*k+4] = b0[k] - 2e-8
        A[4*n:, ii] = 1.
        A[4*n:, jj] = 1.
        u[4*n:] = d0[ii, jj] - 2e-8
        fixed = linprog(
            -np.ones(n), A_ub=A, b_ub=u,
            bounds=[(1e-9, float(min(.25, x))) for x in b0],
            method="highs")
        if fixed.success and np.all(np.isfinite(fixed.x)):
            candidate_r = np.maximum(fixed.x.astype(float, copy=True), 1e-9)
            if np.all(fixed_radius_problem(candidate_r) >= -3e-8):
                r0 = candidate_r

        # Restore center freedom only after the fixed-center active set has
        # been identified.  One bounded polish avoids duplicating the parent
        # joint-radius search while retaining the LP's improved warm start.
        z = np.r_[p0.ravel(), r0, 2e-8]
        free_bounds = bounds + [(0., .02)]

        def polish_constraints(q):
            """Require positive boundary, pairwise, and margin slack."""
            return np.r_[cons(q[:-1], 2e-8), q[-1] - 2e-8]

        polished = minimize(
            lambda q: -float(np.sum(q[2*n:3*n])) - 1e-4*q[-1],
            z, method="SLSQP", bounds=free_bounds,
            constraints={"type": "ineq", "fun": polish_constraints},
            options={"maxiter": 120, "ftol": 5e-10, "disp": False})
        if np.all(np.isfinite(polished.x)):
            z = polished.x.copy()
            p1, r1, _ = unpack_margin(z)
            if np.all(cons(z[:-1], 0.) >= -2e-7):
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