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
        # Start with an explicit positive clearance target.  This keeps the
        # sequential disk-growth topology away from numerically tangent
        # contacts while SLSQP adjusts the centers.
        z0 = np.r_[p.ravel(), r, 5e-7]

        def unpack_margin(q):
            """Unpack centers, fixed seed radii, and the clearance margin."""
            return q[:2*n].reshape(n, 2), q[2*n:3*n], q[-1]

        def margin_cons(q):
            """Require every boundary and pairwise clearance to dominate q[-1]."""
            qq = q[:-1]
            return np.r_[cons(qq, q[-1]), q[-1]]

        fixed_bounds = ([(1e-5, 1-1e-5)]*(2*n) +
                        [(float(x), float(x)) for x in r] + [(5e-7, .02)])
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

        # Reoptimize the incumbent's contact graph with centers and radii
        # free simultaneously.  The auxiliary m variable prevents SLSQP
        # from trading numerical feasibility for an apparently larger sum.
        def direct_constraints(q, margin):
            """Require every wall and pair clearance to exceed margin."""
            return cons(q[:-1], margin)[:-1]

        def direct_margin_constraints(q):
            """Enforce positive minimum clearance while optimizing all variables."""
            return np.r_[direct_constraints(q, q[-1]), q[-1] - 2e-7]

        free_bounds = bounds + [(2e-7, .01)]
        start = np.r_[p0.ravel(), r0, 2e-7]

        # First establish a well-separated center configuration with radii
        # fixed at the LP solution, then release the radii for joint growth.
        fixed_bounds = ([(1e-5, 1-1e-5)]*(2*n) +
                        [(float(x), float(x)) for x in r0] +
                        [(2e-7, .01)])
        fixed_start = start.copy()
        fixed_phase = minimize(
            lambda q: -q[-1], fixed_start, method="SLSQP",
            bounds=fixed_bounds,
            constraints={"type": "ineq", "fun": direct_margin_constraints},
            options={"maxiter": 25, "ftol": 2e-9, "disp": False})
        if np.all(np.isfinite(fixed_phase.x)):
            start = fixed_phase.x.copy()

        polished = minimize(
            lambda q: -float(np.sum(q[2*n:3*n])) - 1e-3*q[-1],
            start, method="SLSQP", bounds=free_bounds,
            constraints={"type": "ineq", "fun": direct_margin_constraints},
            options={"maxiter": 155, "ftol": 5e-10, "disp": False})
        if np.all(np.isfinite(polished.x)):
            z = polished.x.copy()
            p1, r1, _ = unpack_margin(z)
            if np.all(cons(z[:-1], 1e-7) >= -1e-8):
                val = float(r1.sum())
                if val > best_sum:
                    best_sum = val
                    best = (p1.copy(), r1.copy())

    if best is None:
        p = _make_seed((5,5,6,5,5), 0)
        r = _initial_radii(p)
        return p, r, float(r.sum())

    p, r = best

    def weighted_cell_relaxation(p0, r0):
        """Move low-radius circles using weighted local slack, then optimize all radii."""
        p0 = np.asarray(p0, dtype=float).copy()
        r0 = np.asarray(r0, dtype=float).copy()
        b = np.minimum.reduce((p0[:, 0], p0[:, 1],
                               1-p0[:, 0], 1-p0[:, 1]))
        dd = np.sqrt(np.maximum(
            ((p0[:, None] - p0[None, :]) ** 2).sum(axis=2), 1e-24))
        np.fill_diagonal(dd, np.inf)
        nearest = np.min(dd - r0[None, :], axis=1)
        slack = np.maximum(1e-5, np.minimum(b-r0, nearest-r0))
        median = float(np.median(r0))
        movable = r0 < median - 1e-4
        weights = 1.0 / np.maximum(slack, 1e-4)
        start = np.r_[p0.ravel(), r0, 2e-7]
        wbounds = [(1e-5, 1-1e-5)]*(2*n) + [(2e-7, .25)]*n + [(2e-7, .01)]
        for i in range(n):
            if not movable[i]:
                wbounds[2*i] = (float(p0[i, 0]), float(p0[i, 0]))
                wbounds[2*i+1] = (float(p0[i, 1]), float(p0[i, 1]))
        anchor = p0.ravel().copy()

        def weighted_objective(q):
            """Maximize radii while penalizing displacement from weighted cells."""
            displacement = q[:2*n] - anchor
            penalty = 2e-4 * float(np.sum(weights.repeat(2) * displacement**2))
            return -float(np.sum(q[2*n:3*n])) - 1e-3*q[-1] + penalty

        def safe_constraints(q):
            """Enforce walls, pair separation, and positive numerical clearance."""
            return np.r_[cons(q[:-1], 1e-8), q[-1] - 2e-7]

        moved = minimize(
            weighted_objective, start, method="SLSQP", bounds=wbounds,
            constraints={"type": "ineq", "fun": safe_constraints},
            options={"maxiter": 80, "ftol": 5e-10, "disp": False})
        if not np.all(np.isfinite(moved.x)):
            return p0, r0
        return moved.x[:2*n].reshape(n, 2), moved.x[2*n:3*n]

    # Alternating-radius transport: repeatedly reallocate the exact
    # fixed-center LP radii, transport the most constrained centers through
    # locally promising clearance directions, and then release all variables.
    def transport_radii(pp):
        """Optimize fixed-center radii and transport eight constrained centers."""
        pp = np.asarray(pp, dtype=float).copy()
        directions = np.column_stack((
            np.cos(np.arange(16, dtype=float) * np.pi / 8.0),
            np.sin(np.arange(16, dtype=float) * np.pi / 8.0)))
        best_value = -np.inf
        best_p = pp.copy()
        best_r = np.zeros(n, dtype=float)

        def radius_lp(cp):
            """Solve the exact fixed-center linear radius allocation problem."""
            bb = np.minimum.reduce((cp[:, 0], cp[:, 1],
                                    1-cp[:, 0], 1-cp[:, 1]))
            dd = np.sqrt(np.maximum(
                ((cp[:, None] - cp[None, :]) ** 2).sum(axis=2), 1e-30))
            np.fill_diagonal(dd, np.inf)
            aa = np.zeros((4*n + len(ii), n), dtype=float)
            uu = np.empty(4*n + len(ii), dtype=float)
            for kk in range(n):
                aa[4*kk:4*kk+4, kk] = 1.0
                uu[4*kk:4*kk+4] = bb[kk] - 2e-8
            aa[4*n:, ii] = 1.0
            aa[4*n:, jj] = 1.0
            uu[4*n:] = dd[ii, jj] - 2e-8
            ans = linprog(
                -np.ones(n), A_ub=aa, b_ub=uu,
                bounds=[(1e-9, float(min(.25, x))) for x in bb],
                method="highs")
            if not ans.success or not np.all(np.isfinite(ans.x)):
                return None
            return np.maximum(ans.x.astype(float, copy=True), 1e-9)

        # Distinct deterministic starts emphasize different transport orders.
        starts = (
            pp.copy(),
            pp[np.argsort(r)].copy(),
            pp[np.argsort(-r)].copy())
        for start in starts:
            cp = start.copy()
            cr = radius_lp(cp)
            if cr is None:
                continue
            current = float(cr.sum())

            for _ in range(35):
                wall_slack = np.minimum.reduce((
                    cp[:, 0] - cr, cp[:, 1] - cr,
                    1-cp[:, 0] - cr, 1-cp[:, 1] - cr))
                dd = np.sqrt(np.maximum(
                    ((cp[:, None] - cp[None, :]) ** 2).sum(axis=2),
                    1e-30))
                np.fill_diagonal(dd, np.inf)
                pair_slack = np.min(dd - cr[None, :] - cr[:, None], axis=1)
                tight = np.minimum(wall_slack, pair_slack)
                order = np.argsort(tight)[:8]
                changed = False

                # Transport complementary bottleneck pairs.  The pair
                # separation normal is always retained; a tangent component
                # is selected from the locally tight constraints so that
                # motions are not restricted to the old angular directions.
                def active_direction(index):
                    """Build a normalized direction from tight wall and pair slacks."""
                    value = np.zeros(2, dtype=float)
                    wall_gap = np.array((
                        cp[index, 0] - cr[index],
                        cp[index, 1] - cr[index],
                        1.0 - cp[index, 0] - cr[index],
                        1.0 - cp[index, 1] - cr[index]), dtype=float)
                    wall_normals = np.array((
                        (1.0, 0.0), (0.0, 1.0),
                        (-1.0, 0.0), (0.0, -1.0)), dtype=float)
                    value += wall_normals[int(np.argmin(wall_gap))]

                    delta = cp[index] - cp
                    distance = np.sqrt(np.maximum(
                        (delta * delta).sum(axis=1), 1e-30))
                    distance[index] = np.inf
                    for q in np.argsort(distance)[:3]:
                        gap = distance[q] - cr[index] - cr[q]
                        if gap <= pair_slack[index] + .006:
                            value += delta[q] / distance[q]
                    length = float(np.linalg.norm(value))
                    return value / max(length, 1e-12)

                pair_candidates = []
                for k in order:
                    near = np.argsort(dd[k])[:3]
                    for j in near:
                        if j != k and np.isfinite(dd[k, j]):
                            pair_candidates.append((int(k), int(j)))

                seen_pairs = set()
                for k, j in pair_candidates:
                    key = tuple(sorted((k, j)))
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)

                    separation = cp[k] - cp[j]
                    length = float(np.linalg.norm(separation))
                    if length < 1e-12:
                        continue
                    separation /= length

                    # Use the component perpendicular to the pair axis as a
                    # second degree of freedom, weighted by active slacks.
                    nk = active_direction(k)
                    nj = active_direction(j)
                    tangent = nk - nj
                    tangent -= separation * float(np.dot(tangent, separation))
                    tlength = float(np.linalg.norm(tangent))
                    if tlength > 1e-12:
                        tangent /= tlength
                    else:
                        tangent = np.array((-separation[1], separation[0]))

                    best_trial = None
                    best_radii = None
                    best_value = current
                    for axis in (separation, tangent,
                                 separation + .65*tangent):
                        axis_length = float(np.linalg.norm(axis))
                        axis = axis / max(axis_length, 1e-12)
                        for step in (.003, .007, .014):
                            trial = cp.copy()
                            trial[k] += step * axis
                            trial[j] -= step * axis
                            trial = np.clip(trial, 1e-5, 1.0-1e-5)
                            nr = radius_lp(trial)
                            if nr is None:
                                continue
                            value = float(nr.sum())
                            if value > best_value + 2e-8:
                                best_value = value
                                best_trial = trial
                                best_radii = nr

                    if best_trial is not None:
                        cp = best_trial
                        cr = best_radii
                        current = best_value
                        changed = True

                nr = radius_lp(cp)
                if nr is None or float(nr.sum()) <= current + 1e-7:
                    if nr is not None and float(nr.sum()) > current:
                        cr, current = nr, float(nr.sum())
                    break
                cr, current = nr, float(nr.sum())

            if current > best_value:
                best_value = current
                best_p = cp.copy()
                best_r = cr.copy()

        return best_p, best_r

    transport_starts = (
        p.copy(),
        p[np.argsort(r)].copy(),
        p[np.argsort(-r)].copy())
    transported = []
    for candidate in transport_starts:
        tp, tr = transport_radii(candidate)
        transported.append((float(tr.sum()), tp, tr))
    _, p, r = max(transported, key=lambda item: item[0])

    z = np.r_[p.ravel(), r, 2e-7]
    polish_bounds = bounds + [(2e-7, .01)]
    res = minimize(
        lambda q: -float(np.sum(q[2*n:3*n])) - 1e-3*q[-1],
        z, method="SLSQP", bounds=polish_bounds,
        constraints={"type": "ineq",
                     "fun": lambda q: np.r_[cons(q[:-1], 1e-8),
                                             q[-1] - 2e-7]},
        options={"maxiter": 70, "ftol": 5e-10, "disp": False})
    if (np.all(np.isfinite(res.x)) and
            np.all(cons(res.x[:-1], 0.) >= -2e-7)):
        pp, rr = unpack(res.x[:-1])
        if rr.sum() > r.sum():
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