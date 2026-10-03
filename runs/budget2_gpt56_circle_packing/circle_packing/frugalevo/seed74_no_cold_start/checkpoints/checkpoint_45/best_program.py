"""Deterministic multistart constructor for 26 unequal circles in a unit square."""
import numpy as np


def compute_max_radii(centers):
    c = np.asarray(centers, dtype=float)
    n = len(c)
    r = np.minimum.reduce([c[:, 0], c[:, 1], 1 - c[:, 0], 1 - c[:, 1]]).copy()
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(c[i] - c[j]))
            if r[i] + r[j] > d and r[i] + r[j] > 0:
                q = d / (r[i] + r[j])
                r[i] *= q
                r[j] *= q
    return r


def _incumbent():
    x5 = np.linspace(.1, .9, 5)
    x6 = np.linspace(1 / 12, 11 / 12, 6)
    rows = [(.1, x5, .1), (.31666666666666665, x5, .1),
            (.5, x6, 1 / 12), (.6833333333333333, x5, .1), (.9, x5, .1)]
    c = np.array([(x, y) for y, xs, _ in rows for x in xs], dtype=float)
    r = np.array([rr for _, xs, rr in rows for _ in xs], dtype=float)
    return c, r * (1 - 1e-9)


def _repair(centers, radii, margin=3e-8):
    c = np.clip(np.asarray(centers, float).copy(), 0.0, 1.0)
    r = np.maximum(np.asarray(radii, float).copy(), 0.0)
    wall = np.minimum.reduce([c[:, 0], c[:, 1], 1-c[:, 0], 1-c[:, 1]])
    r = np.minimum(r, np.maximum(0.0, wall - margin))
    for _ in range(5):
        changed = False
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = float(np.linalg.norm(c[i] - c[j]))
                a = max(0.0, d - margin)
                s = r[i] + r[j]
                if s > a and s > 0:
                    q = a / s
                    r[i] *= q
                    r[j] *= q
                    changed = True
        if not changed:
            break
    return c, r


def _row_seed(counts=None, reflected=False, shear=0.0):
    """Construct a repaired corner-petal layout with four corners and a six-disk spine."""
    # Four corner disks, each tangent to the two nearby walls.
    pts = [
        (.060, .060), (.940, .060), (.060, .940), (.940, .940),

        # Two wall-adjacent petals around each corner.
        (.178, .064), (.064, .178),
        (.822, .064), (.936, .178),
        (.064, .822), (.178, .936),
        (.936, .822), (.822, .936),

        # Eight side/interior transition disks.
        (.285, .080), (.500, .090), (.715, .080),
        (.080, .315), (.920, .315),
        (.080, .685), (.920, .685),
        (.285, .920),

        # Six staggered central disks.
        (.335, .285), (.665, .285),
        (.275, .500), (.500, .500), (.725, .500),
        (.335, .715),
    ]
    c = np.asarray(pts, dtype=float)
    if shear:
        c[20:, 1] += shear * np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    if reflected:
        c[:, 0] = 1.0 - c[::-1, 0]

    # Unequal initial radii preserve the intended corner/petal/transition
    # hierarchy while allowing _joint to redistribute them.
    r = np.empty(26, dtype=float)
    r[:4] = .060
    r[4:12] = .070
    r[12:20] = .080
    r[20:] = .075
    return c.copy(), r.copy()


def _joint(centers, radii, maxiter=850):
    try:
        from scipy.optimize import minimize
    except Exception:
        return centers, radii
    n = 26
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m = 4*n + len(pairs)
    z0 = np.r_[np.asarray(centers, float).ravel(), np.asarray(radii, float)]

    def con(z):
        xy, rr = z[:2*n].reshape(n, 2), z[2*n:]
        out = np.empty(m)
        out[:n] = xy[:, 0] - rr
        out[n:2*n] = xy[:, 1] - rr
        out[2*n:3*n] = 1 - xy[:, 0] - rr
        out[3*n:4*n] = 1 - xy[:, 1] - rr
        for k, (i, j) in enumerate(pairs):
            d = xy[i] - xy[j]
            out[4*n+k] = d.dot(d) - (rr[i] + rr[j])**2
        return out

    def jac(z):
        xy, rr = z[:2*n].reshape(n, 2), z[2*n:]
        a = np.zeros((m, 3*n))
        for i in range(n):
            a[i, 2*i] = 1; a[i, 2*n+i] = -1
            a[n+i, 2*i+1] = 1; a[n+i, 2*n+i] = -1
            a[2*n+i, 2*i] = -1; a[2*n+i, 2*n+i] = -1
            a[3*n+i, 2*i+1] = -1; a[3*n+i, 2*n+i] = -1
        for k, (i, j) in enumerate(pairs):
            d = xy[i] - xy[j]
            row = 4*n+k
            a[row, 2*i:2*i+2] = 2*d
            a[row, 2*j:2*j+2] = -2*d
            a[row, 2*n+i] = a[row, 2*n+j] = -2*(rr[i]+rr[j])
        return a

    ans = minimize(
        lambda z: -float(np.sum(z[2*n:])), z0,
        jac=lambda z: np.r_[np.zeros(2*n), -np.ones(n)],
        constraints={"type": "ineq", "fun": con, "jac": jac},
        bounds=[(0, 1)]*(2*n) + [(1e-8, .35)]*n,
        method="SLSQP",
        options={"maxiter": maxiter, "ftol": 2e-10, "disp": False},
    )
    if not np.all(np.isfinite(ans.x)):
        return centers, radii
    return ans.x[:2*n].reshape(n, 2), ans.x[2*n:]


def _polish(centers, radii, maxiter=850):
    try:
        from scipy.optimize import minimize
    except Exception:
        return centers, radii
    n = 26
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    xy0 = np.asarray(centers, float).copy()
    wall = float(np.min(np.minimum.reduce(
        [xy0[:, 0], xy0[:, 1], 1-xy0[:, 0], 1-xy0[:, 1]])))
    pair = min(.5*float(np.linalg.norm(xy0[i]-xy0[j])) for i, j in pairs)
    z0 = np.r_[xy0.ravel(), .97*min(wall, pair)]

    def common_con(z):
        xy, q = z[:2*n].reshape(n, 2), z[-1]
        out = np.empty(4*n + len(pairs))
        out[:n] = xy[:, 0]-q
        out[n:2*n] = xy[:, 1]-q
        out[2*n:3*n] = 1-xy[:, 0]-q
        out[3*n:4*n] = 1-xy[:, 1]-q
        for k, (i, j) in enumerate(pairs):
            d = xy[i]-xy[j]
            out[4*n+k] = d.dot(d)-(2*q)**2
        return out

    ans = minimize(
        lambda z: -z[-1], z0, method="SLSQP",
        constraints={"type": "ineq", "fun": common_con},
        bounds=[(0, 1)]*(2*n)+[(1e-8, .25)],
        options={"maxiter": 380, "ftol": 2e-9, "disp": False},
    )
    if np.all(np.isfinite(ans.x)):
        centers = ans.x[:2*n].reshape(n, 2)
        radii = np.full(n, max(1e-8, ans.x[-1]))
    return _joint(centers, radii, maxiter)


def _contact_escape(c, r):
    """Generate six center-and-radius active-contact deflation directions."""
    n = 26
    nv = 3*n
    rows = []
    for i in range(n):
        for axis in range(2):
            for sign, val in (
                (1.0, c[i, axis]-r[i]),
                (-1.0, 1-c[i, axis]-r[i]),
            ):
                if val < 2e-4:
                    row = np.zeros(nv, dtype=float)
                    row[2*i+axis] = sign
                    rows.append(row)
    for i in range(n):
        for j in range(i+1, n):
            d = c[i] - c[j]
            norm = float(np.linalg.norm(d))
            gap = norm - float(r[i] + r[j])
            if gap < 2e-4 and norm > 1e-10:
                u = d / norm
                row = np.zeros(nv, dtype=float)
                row[2*i:2*i+2] = u
                row[2*j:2*j+2] = -u
                row[2*n+i] = -1.0
                row[2*n+j] = -1.0
                rows.append(row)
    if len(rows) < 2:
        return []

    a = np.asarray(rows, dtype=float)
    _, _, vh = np.linalg.svd(a, full_matrices=True)

    # Remove the uniform radius-shrinkage component from every mode.
    shrink = np.zeros(nv, dtype=float)
    shrink[2*n:] = 1.0
    shrink /= np.linalg.norm(shrink)
    modes = []
    for raw in vh[-min(3, len(vh)):]:
        v = np.asarray(raw, dtype=float).copy()
        v -= shrink * float(np.dot(v, shrink))
        nr = float(np.linalg.norm(v))
        if nr > 1e-10:
            modes.append(v / nr)

    out = []
    for mode in modes:
        center_mode = mode[:2*n].reshape(n, 2)
        radius_mode = mode[2*n:].copy()
        radius_mode -= np.mean(radius_mode)
        for sign in (-1.0, 1.0):
            for amplitude in (.0015, .0030):
                dc = center_mode * (amplitude / max(
                    1e-12, float(np.max(np.abs(center_mode)))))
                dr = radius_mode * (amplitude / max(
                    1e-12, float(np.max(np.abs(radius_mode)))))
                cc = np.clip(np.asarray(c, dtype=float) + sign*dc, .01, .99)
                rr = np.maximum(1e-8, np.asarray(r, dtype=float) *
                                .985 + sign*dr)
                out.append((cc, rr))
                if len(out) >= 6:
                    return out
    return out


def construct_packing():
    best_c, best_r = _repair(*_incumbent())
    best_sum = float(np.sum(best_r))
    starts = [_incumbent()]
    c, r = _incumbent()
    c[10:16, 0] += .25/6
    starts.append((np.clip(c, .02, .98), r*.96))

    # Search coherent layered layouts instead of applying independent
    # coordinate noise to the incumbent.
    try:
        from scipy.optimize import differential_evolution, linprog

        def structural_layout(p):
            """Build a perturbed 5,5,6,5,5 layout from 14 structural parameters."""
            p = np.asarray(p, dtype=float)
            ys = np.array([.105, .305, .500, .695, .895], dtype=float) + p[:5]
            x5 = np.linspace(.105, .895, 5)
            x6 = np.linspace(1.0 / 12.0, 11.0 / 12.0, 6)
            scale = 1.0 + p[10]
            pts = []
            for k, y in enumerate(ys):
                xs = x6 if k == 2 else x5
                xs = .5 + (xs - .5) * scale + p[5 + k]
                if k == 2:
                    xs = xs + p[11] * np.linspace(-1.0, 1.0, 6)
                elif k == 1:
                    xs = xs + p[12] * np.linspace(-1.0, 1.0, 5)
                elif k == 3:
                    xs = xs + p[13] * np.linspace(1.0, -1.0, 5)
                pts.extend((float(x), float(y)) for x in xs)
            return np.asarray(pts, dtype=float)

        def fixed_radii(c):
            """Maximize the radius sum for fixed centers using a linear program."""
            n = len(c)
            aub = []
            bub = []
            for i in range(n):
                row = np.zeros(n, dtype=float)
                row[i] = 1.0
                aub.append(row)
                bub.append(float(min(c[i, 0], c[i, 1],
                                     1.0-c[i, 0], 1.0-c[i, 1])))
            for i in range(n):
                for j in range(i + 1, n):
                    row = np.zeros(n, dtype=float)
                    row[i] = row[j] = 1.0
                    aub.append(row)
                    bub.append(float(np.linalg.norm(c[i] - c[j])))
            ans = linprog(
                -np.ones(n), A_ub=np.asarray(aub), b_ub=np.asarray(bub),
                bounds=[(0.0, None)] * n, method="highs",
            )
            return np.maximum(ans.x, 0.0) if ans.success else np.zeros(n)

        def structural_objective(p):
            """Score a structural layout by feasible fixed-center radius sum."""
            c = structural_layout(p)
            violation = np.maximum(.012 - c, 0.0) + np.maximum(c - .988, 0.0)
            if np.any(violation):
                return 1000.0 + 100.0 * float(np.sum(violation))
            return -float(np.sum(fixed_radii(c)))

        bounds = [
            (-.025, .025), (-.025, .025), (-.025, .025),
            (-.025, .025), (-.025, .025),
            (-.035, .035), (-.035, .035), (-.035, .035),
            (-.035, .035), (-.035, .035),
            (-.055, .055), (-.025, .025), (-.025, .025),
            (-.025, .025),
        ]
        result = differential_evolution(
            structural_objective, bounds, seed=371942, popsize=8,
            maxiter=35, workers=1, polish=False, updating="immediate",
        )

        # Release the best three evolved individuals, rather than merely
        # reflecting one winner; this preserves distinct contact topologies.
        population = [np.asarray(result.x, dtype=float)]
        population.extend(np.asarray(q, dtype=float) for q in result.population)
        ranked = []
        for p in population:
            value = structural_objective(p)
            if np.isfinite(value):
                ranked.append((value, p))
        ranked.sort(key=lambda item: item[0])
        chosen = []
        for _, p in ranked:
            if all(float(np.max(np.abs(p-q))) > 1e-5 for q in chosen):
                chosen.append(p)
            if len(chosen) == 3:
                break
        for p in chosen:
            c = structural_layout(p)
            r = fixed_radii(c)
            starts.append(_repair(c, r))

    except Exception:
        # Keep the deterministic incumbent and the alternate geometric seeds
        # usable when SciPy's global or linear optimizer is unavailable.
        pass

    # Corner-petal starts expose corner voids that the horizontal-row seeds
    # cannot use efficiently. Reflection and small spine shears provide four
    # distinct contact orderings without increasing the optimizer workload.
    for reflection, shear in [
        (False, 0.000),
        (True, 0.000),
        (False, .014),
        (True, -.014),
    ]:
        starts.append(_row_seed(None, reflection, shear))

    for q, (c0, r0) in enumerate(starts):
        c0, r0 = _repair(c0, r0)
        c1, r1 = _polish(c0, r0, 900 if q < 2 else 780)
        c1, r1 = _repair(c1, r1)
        if float(np.sum(r1)) > best_sum:
            best_c, best_r, best_sum = c1, r1, float(np.sum(r1))

    # Remove disks from crowded cells and reinsert them into independently
    # discovered cavities before the final joint optimization.
    def _cavity_candidates(c, r):
        """Relocate eight crowded disks by grid insertion and bounded SLSQP."""
        try:
            from scipy.optimize import minimize
        except Exception:
            return []

        c = np.asarray(c, dtype=float).copy()
        r = np.asarray(r, dtype=float).copy()
        n = len(r)
        clearance = np.empty(n, dtype=float)

        # A disk is a useful removal candidate when its radius consumes a
        # large fraction of its smallest wall or pair clearance.
        for i in range(n):
            wall = min(c[i, 0], c[i, 1], 1.0-c[i, 0], 1.0-c[i, 1])
            pair = min(
                float(np.linalg.norm(c[i] - c[j])) - r[i] - r[j]
                for j in range(n) if j != i
            )
            available = max(1e-9, min(wall, pair + r[i]))
            clearance[i] = r[i] / available

        removed = np.argsort(-clearance)[:8]
        candidates = []
        safety = 2e-7
        grid = np.linspace(.02, .98, 31)

        for removed_i in removed:
            keep = np.arange(n) != int(removed_i)
            fixed_c = c[keep]
            fixed_r = r[keep]

            def insertion_radius(x):
                """Return the safe radius available at one trial center."""
                wall = min(x[0], x[1], 1.0-x[0], 1.0-x[1])
                if len(fixed_c):
                    gap = np.linalg.norm(fixed_c-x, axis=1) - fixed_r
                    wall = min(wall, float(np.min(gap)))
                return max(0.0, wall-safety)

            best_x = np.array([.5, .5], dtype=float)
            best_q = -1.0
            for gy in grid:
                for gx in grid:
                    x = np.array([gx, gy], dtype=float)
                    q = insertion_radius(x)
                    if q > best_q:
                        best_x, best_q = x, q

            if best_q <= 1e-8:
                continue

            def constraints(z):
                """Enforce wall and squared non-overlap inequalities."""
                x = z[:2]
                q = z[2]
                out = [
                    x[0] - q - safety,
                    x[1] - q - safety,
                    1.0 - x[0] - q - safety,
                    1.0 - x[1] - q - safety,
                ]
                if len(fixed_c):
                    d = fixed_c - x
                    out.extend(
                        np.einsum("ij,ij->i", d, d) -
                        (fixed_r + q + safety)**2
                    )
                return np.asarray(out, dtype=float)

            z0 = np.r_[best_x, best_q]
            ans = minimize(
                lambda z: -float(z[2]),
                z0,
                method="SLSQP",
                constraints={"type": "ineq", "fun": constraints},
                bounds=[(safety, 1.0-safety),
                        (safety, 1.0-safety), (1e-8, .35)],
                options={"maxiter": 120, "ftol": 2e-10, "disp": False},
            )

            # The grid point is already feasible; retain it if the local
            # optimizer reports failure or produces an invalid point.
            if np.all(np.isfinite(ans.x)) and np.all(constraints(ans.x) >= -1e-8):
                trial_x = ans.x[:2].copy()
                trial_q = max(1e-8, float(ans.x[2]))
            else:
                trial_x = best_x
                trial_q = best_q

            cc = c.copy()
            rr = r.copy()
            cc[int(removed_i)] = trial_x
            rr[int(removed_i)] = trial_q
            cc, rr = _repair(cc, rr)
            candidates.append((float(np.sum(rr)), cc, rr))

        candidates.sort(key=lambda item: item[0], reverse=True)
        return [(cc, rr) for _, cc, rr in candidates[:3]]

    cavity_starts = _cavity_candidates(best_c, best_r)
    for c0, r0 in cavity_starts:
        c0, r0 = _repair(c0, r0)
        c1, r1 = _joint(c0, r0, 650)
        c1, r1 = _repair(c1, r1)
        candidate_sum = float(np.sum(r1))
        if candidate_sum > best_sum:
            best_c, best_r, best_sum = c1, r1, candidate_sum
    return best_c, best_r, best_sum


def run_packing():
    return construct_packing()


def visualize(centers, radii):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    for i, (c, r) in enumerate(zip(centers, radii)):
        ax.add_patch(Circle(c, r, alpha=.5))
        ax.text(c[0], c[1], str(i), ha="center", va="center")
    plt.title(f"Circle Packing (n={len(centers)}, sum={sum(radii):.6f})")
    plt.show()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii}")