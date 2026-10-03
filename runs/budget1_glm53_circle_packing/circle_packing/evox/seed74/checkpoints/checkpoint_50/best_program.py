# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """
    Construct a specific arrangement of 26 circles in a unit square
    that attempts to maximize the sum of their radii.

    Returns:
        Tuple of (centers, radii, sum_of_radii)
        centers: np.array of shape (26, 2) with (x, y) coordinates
        radii: np.array of shape (26) with radius of each circle
        sum_of_radii: Sum of all radii
    """
    # Hexagonal lattice packing: 6 staggered rows with widths 5,4,5,4,5,3 (=26).
    # All circles share radius r. Vertical constraint is binding:
    #   2r + 5*(sqrt(3)*r) = 1  =>  r = 1/(2 + 5*sqrt(3)) ~ 0.09381
    # Adjacent rows are staggered horizontally by r so that neighbor
    # center-to-center distances are exactly 2r (hexagonal contacts).
    # Sum of radii = 26 * r ~ 2.439.
    # Approach: seed with a hexagonal-lattice layout (guaranteed feasible,
    # sum = 26*r ~ 2.439), then improve by hill-climbing the centers.
    # For fixed centers, the optimal radii come from a linear program:
    #   maximize sum(r) s.t. r_i + r_j <= dist(i,j), 0 <= r_i <= wall cap_i.
    # Random single-center perturbations are accepted only when the LP
    # objective improves, so the result is always valid and never worse
    # than the seed. This escapes the equal-radius hexagonal optimum by
    # letting circles grow unevenly into wall gaps.
    n = 26

    def hex_seed(row_widths):
        """Staggered hexagonal-lattice seed: rows of the given widths with
        uniform radius r = 1/(2 + (rows-1)*sqrt(3)) so it fits vertically;
        adjacent rows are staggered by r for hexagonal contacts."""
        C = np.zeros((n, 2))
        nr = len(row_widths)
        r = 1.0 / (2.0 + (nr - 1) * np.sqrt(3.0))
        dy = np.sqrt(3.0) * r
        idx = 0
        for row, w in enumerate(row_widths):
            y = r + row * dy
            # offset parity keeps each row centered and staggered
            c = w - 1 if (w - 1) % 2 == row % 2 else w
            for k in range(w):
                C[idx] = [0.5 + (2 * k - c) * r, y]
                idx += 1
        return C, r

    def corner_seed(a, inner_rows):
        """Seed with 4 large corner circles of radius a plus an interior
        staggered hex pattern of the given row widths. Corner circles sit
        at (a,a), (1-a,a), (a,1-a), (1-a,1-a); the interior block is
        centered in the remaining square of side 1-2a and staggered so
        adjacent rows touch hexagonally. LP-based growth then lets the
        corner circles dominate, matching the structure of good n=26
        packings that uniform hex seeds cannot reach."""
        C = np.zeros((n, 2))
        C[0] = [a, a]
        C[1] = [1.0 - a, a]
        C[2] = [a, 1.0 - a]
        C[3] = [1.0 - a, 1.0 - a]
        m = n - 4
        w = max(inner_rows)
        nr = len(inner_rows)
        # Interior radius fits vertically in (1-2a) and horizontally in
        # (1-2a) for the widest row: use the smaller of the two limits.
        r_v = (1.0 - 2.0 * a) / (2.0 + (nr - 1) * np.sqrt(3.0))
        r_h = (1.0 - 2.0 * a) / (2.0 * w)
        r = min(r_v, r_h)
        dy = np.sqrt(3.0) * r
        idx = 4
        y0 = 0.5 - (nr - 1) * dy / 2.0
        for row, ww in enumerate(inner_rows):
            y = y0 + row * dy
            c = ww - 1 if (ww - 1) % 2 == row % 2 else ww
            for k in range(ww):
                C[idx] = [0.5 + (2 * k - c) * r, y]
                idx += 1
        return C

    def edge_seed(a, inner_rows):
        """Seed with 4 corner circles of radius a, 4 edge-midpoint circles
        of radius a (one per wall), plus an interior staggered hex block
        shrunk by 10% for a safety margin. Wall-hugging large circles plus
        a smaller interior hex core matches the structure of the best
        known square packings for n around 26, a basin the pure hex and
        corner seeds cannot reach."""
        C = np.zeros((n, 2))
        pos = [(a, a), (1 - a, a), (a, 1 - a), (1 - a, 1 - a),
               (0.5, a), (1 - a, 0.5), (0.5, 1 - a), (a, 0.5)]
        for t, p in enumerate(pos):
            C[t] = p
        nr = len(inner_rows)
        w = max(inner_rows)
        r_v = (1.0 - 2.0 * a) / (2.0 + (nr - 1) * np.sqrt(3.0))
        r_h = (1.0 - 2.0 * a) / (2.0 * w)
        r = 0.9 * min(r_v, r_h)
        dy = np.sqrt(3.0) * r
        idx = 8
        y0 = 0.5 - (nr - 1) * dy / 2.0
        for row, ww in enumerate(inner_rows):
            y = y0 + row * dy
            c = ww - 1 if (ww - 1) % 2 == row % 2 else ww
            for k in range(ww):
                C[idx] = [0.5 + (2 * k - c) * r, y]
                idx += 1
        return C

    try:
        from scipy.optimize import linprog
        from scipy.sparse import csr_matrix
    except Exception:
        C, r = hex_seed([5, 4, 5, 4, 5, 3])
        return C, np.full(n, r), float(n * r)

    # Constant upper-triangular pair index lists.
    ii, jj = np.triu_indices(n, 1)

    def solve_radii(C):
        """LP: max sum(r) s.t. r_i + r_j <= d_ij, 0 <= r_i <= wall cap_i.
        Pairs with d_ij >= cap_i + cap_j can never bind (the bounds already
        imply r_i + r_j <= d_ij), so they are dropped exactly -- the LP stays
        tiny and fast. r_i = 0 is always feasible."""
        d = np.hypot(C[ii, 0] - C[jj, 0], C[ii, 1] - C[jj, 1])
        caps = np.minimum(np.minimum(C[:, 0], C[:, 1]),
                          np.minimum(1.0 - C[:, 0], 1.0 - C[:, 1]))
        keep = d < (caps[ii] + caps[jj])
        ik, jk, dk = ii[keep], jj[keep], d[keep]
        m = len(dk)
        if m == 0:
            return caps, float(caps.sum())
        rows = np.concatenate([np.arange(m), np.arange(m)])
        cols = np.concatenate([ik, jk])
        A = csr_matrix((np.ones(2 * m), (rows, cols)), shape=(m, n))
        res = linprog(c=-np.ones(n), A_ub=A, b_ub=dk,
                      bounds=np.stack([np.zeros(n), caps], axis=1),
                      method="highs")
        if not res.success:
            return None, -np.inf
        return res.x, -res.fun

    import time
    t0 = time.time()
    deadline = t0 + 345.0  # wall-clock budget, inside the 360s limit
    rng = np.random.default_rng(12345)

    def climb(C, sub_deadline, scale0=0.03):
        """Hill-climb of centers under the LP objective. Improving moves
        are always accepted; when stagnant, slightly worsening moves are
        allowed to wander between basins while the best configuration
        ever seen is tracked and returned. Every 4th move is a
        contact-relief step: a random binding circle pair (r_i + r_j =
        d_ij) is pushed apart along its center line, or, if no pair
        binds, the most wall-tight circle is pushed off its nearest
        wall -- the productive directions for growing radii. All moves
        are validated by the LP, so the returned packing is feasible."""
        rad, s = solve_radii(C)
        if rad is None:
            return -np.inf, C, None
        cur_sum, cur_rad = s, rad
        best_sum, best_C, best_rad = cur_sum, C.copy(), rad
        scale = scale0
        it = 0
        stuck = 0

        def evaluate(C2, sideways_tol=2e-3, eager=False):
            nonlocal C, cur_sum, cur_rad, stuck
            nonlocal best_sum, best_C, best_rad
            rad2, s2 = solve_radii(C2)
            if rad2 is None:
                stuck += 1
                return
            if s2 > cur_sum + 1e-9:
                C, cur_sum, cur_rad = C2, s2, rad2
                stuck = 0
            elif s2 > cur_sum - sideways_tol and (eager or stuck > 120):
                # Damped sideways move: cross narrow ridges between
                # basins without ever losing the best-ever configuration.
                C, cur_sum, cur_rad = C2, s2, rad2
                stuck = 0
            else:
                stuck += 1
            if s2 > best_sum + 1e-12:
                best_sum, best_C, best_rad = s2, C2.copy(), rad2

        while time.time() < sub_deadline:
            it += 1
            if it % 150 == 0:
                scale = max(scale * 0.9, min(0.004, scale0))
            if stuck > 250:
                # Reheat + multi-circle shake to escape local optima.
                stuck = 0
                reheat_cap = 0.03 if scale0 >= 0.02 else max(4.0 * scale0, 0.004)
                scale = min(scale * 2.5, reheat_cap)
                C2 = C.copy()
                for i in rng.choice(n, size=3, replace=False):
                    C2[i] += rng.normal(0.0, scale, 2)
                    C2[i] = np.clip(C2[i], 0.02, 0.98)
                evaluate(C2)
                continue
            if it % 10 == 0 and cur_rad is not None:
                # Force-directed relaxation: push every circle away from
                # its near-binding neighbors and tight walls at once.
                # Weights decay with slack so only pressing contacts
                # contribute. This global move reshapes the whole layout
                # toward the expanded configuration far faster than
                # single-circle jiggles, and is LP-validated like all
                # other moves (best-ever is snapshotted, so it is safe).
                C2 = C.copy()
                d = np.hypot(C[ii, 0] - C[jj, 0], C[ii, 1] - C[jj, 1])
                slack = d - (cur_rad[ii] + cur_rad[jj])
                w = np.clip(1.0 - slack / max(3.0 * scale, 1e-4), 0.0, 1.0)
                if w.sum() > 0.0:
                    inv = 1.0 / np.maximum(d, 1e-12)
                    dxu = (C[ii, 0] - C[jj, 0]) * inv
                    dyu = (C[ii, 1] - C[jj, 1]) * inv
                    fx = np.zeros(n)
                    fy = np.zeros(n)
                    np.add.at(fx, ii, w * dxu)
                    np.add.at(fx, jj, -w * dxu)
                    np.add.at(fy, ii, w * dyu)
                    np.add.at(fy, jj, -w * dyu)
                    caps = np.minimum(np.minimum(C[:, 0], C[:, 1]),
                                      np.minimum(1.0 - C[:, 0], C[:, 1]))
                    wt = np.clip(1.0 - (caps - cur_rad) / max(3.0 * scale, 1e-4),
                                 0.0, 1.0)
                    # Push wall-tight circles away from their tight wall.
                    fx = fx - wt * np.sign(C[:, 0] - 0.5)
                    fy = fy - wt * np.sign(C[:, 1] - 0.5)
                    fn = np.hypot(fx, fy)
                    act = fn > 1e-9
                    C2[act, 0] += scale * fx[act] / fn[act]
                    C2[act, 1] += scale * fy[act] / fn[act]
                    C2 = np.clip(C2, 0.02, 0.98)
                    evaluate(C2, 1e-3, True)
                    continue
            if it % 4 == 0 and cur_rad is not None:
                # Contact-relief move (see docstring).
                C2 = C.copy()
                d = np.hypot(C[ii, 0] - C[jj, 0], C[ii, 1] - C[jj, 1])
                slack = d - (cur_rad[ii] + cur_rad[jj])
                tight = np.where(slack < 1e-7)[0]
                if len(tight) > 0:
                    k = tight[rng.integers(len(tight))]
                    i, j = int(ii[k]), int(jj[k])
                    dirv = C[i] - C[j]
                    nrm = np.hypot(dirv[0], dirv[1])
                    if nrm > 1e-12:
                        u = dirv / nrm
                        C2[i] += u * (0.5 * scale)
                        C2[j] -= u * (0.5 * scale)
                        C2 = np.clip(C2, 0.02, 0.98)
                        evaluate(C2)
                        continue
                else:
                    caps = np.minimum(np.minimum(C[:, 0], C[:, 1]),
                                      np.minimum(1.0 - C[:, 0], 1.0 - C[:, 1]))
                    i = int(np.argmin(caps - cur_rad))
                    if caps[i] - cur_rad[i] < 1e-7:
                        w = [C[i, 0], C[i, 1], 1.0 - C[i, 0], 1.0 - C[i, 1]]
                        wd = int(np.argmin(w))
                        if wd == 0:
                            C2[i, 0] += scale
                        elif wd == 1:
                            C2[i, 1] += scale
                        elif wd == 2:
                            C2[i, 0] -= scale
                        else:
                            C2[i, 1] -= scale
                        C2[i] = np.clip(C2[i], 0.02, 0.98)
                        evaluate(C2)
                        continue
            C2 = C.copy()
            i = rng.integers(n)
            C2[i] += rng.normal(0.0, scale, 2)
            C2[i] = np.clip(C2[i], 0.02, 0.98)
            evaluate(C2)
        return best_sum, best_C, best_rad

    def slsqq_polish_placeholder():
        pass

    def slsqp_polish(C, rad, sub_deadline):
        """Joint gradient-based polish of centers AND radii with SLSQP.
        Variables: 52 center coords + 26 radii. Constraints (all 'ineq',
        i.e. fun >= 0): pairwise d_ij - r_i - r_j, and per-circle wall
        margins x-r, y-r, 1-x-r, 1-y-r (4 smooth constraints per circle
        instead of a kinked min). Bounds keep centers in [0,1] and radii
        in [0, 0.5]. A callback aborts on the sub-deadline. The result is
        re-validated through the LP oracle (which can only improve the
        radii for the given centers), so the returned packing is always
        feasible and at least as good as the LP value of the input."""
        try:
            from scipy.optimize import minimize
        except Exception:
            return -np.inf, C, rad
        x0 = np.concatenate([C.ravel(), rad])

        def unpack(x):
            return x[:2 * n].reshape(n, 2), x[2 * n:]

        def obj(x):
            return -float(x[2 * n:].sum())

        def c_pair(x):
            Cx, r = unpack(x)
            d = np.hypot(Cx[ii, 0] - Cx[jj, 0], Cx[ii, 1] - Cx[jj, 1])
            return d - r[ii] - r[jj]

        def c_wall(x):
            Cx, r = unpack(x)
            return np.concatenate([Cx[:, 0] - r, Cx[:, 1] - r,
                                   1.0 - Cx[:, 0] - r, 1.0 - Cx[:, 1] - r])

        cons = ({'type': 'ineq', 'fun': c_pair},
                {'type': 'ineq', 'fun': c_wall})
        bnds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n
        stopped = []

        def cb(xk):
            if time.time() >= sub_deadline:
                stopped.append(True)
                return False
            return True

        try:
            res = minimize(obj, x0, method='SLSQP', constraints=cons,
                           bounds=bnds, callback=cb,
                           options={'maxiter': 400, 'ftol': 1e-10})
            xs = res.x if res.x is not None else x0
        except Exception:
            xs = x0
        Cx, _ = unpack(xs)
        Cx = np.clip(Cx, 0.0, 1.0)
        rad2, s2 = solve_radii(Cx)
        if rad2 is None:
            return -np.inf, C, rad
        return s2, Cx, rad2

    # Multi-start over hexagonal row patterns (each sums to 26) and
    # corner-circle seeds (4 big corner circles + 22 interior hex circles),
    # which give the climb access to the corner-dominated basin that the
    # best known n=26 packings live in.
    hex_seeds = [[5, 4, 5, 4, 5, 3],
                 [4, 5, 4, 5, 4, 4],
                 [5, 5, 4, 4, 5, 3],
                 [3, 5, 5, 4, 5, 4],
                 [5, 4, 4, 5, 4, 4],
                 [4, 4, 4, 4, 4, 3, 3]]
    seed_fns = [(lambda sw=sw: hex_seed(sw)[0]) for sw in hex_seeds]
    for a in (0.11, 0.14, 0.17, 0.20, 0.23):
        for rows in ([4, 5, 4, 5, 4], [5, 4, 5, 4, 4], [6, 5, 5, 6]):
            if sum(rows) == n - 4:
                seed_fns.append(lambda a=a, rows=rows: corner_seed(a, rows))
    for a in (0.10, 0.13):
        for rows in ([5, 4, 5, 4], [4, 5, 4, 5], [5, 5, 4, 4]):
            if sum(rows) == n - 8:
                seed_fns.append(lambda a=a, rows=rows: edge_seed(a, rows))
    n_seeds = len(seed_fns)
    best_sum, best_C, best_rad = -np.inf, None, None
    # Multi-start gets ~55% of the budget; the rest is reserved for
    # coarse-to-fine polish of the winner and shaken restarts (the old
    # schedule let multi-start consume the whole budget, so the polish
    # and restart phases never actually ran).
    span = deadline - t0
    ms_deadline = t0 + 0.50 * span
    for si, fn in enumerate(seed_fns):
        sub_dl = min(t0 + (si + 1) * (ms_deadline - t0) / n_seeds, ms_deadline)
        if time.time() >= sub_dl:
            break
        C0 = fn()
        s, C, rad = climb(C0, sub_dl)
        if s > best_sum:
            best_sum, best_C, best_rad = s, C, rad

    # Coarse-to-fine polish of the best configuration: progressively
    # smaller step sizes settle the layout into its exact local optimum.
    if best_C is not None:
        for sc, frac in ((0.012, 0.55), (0.005, 0.62), (0.002, 0.67)):
            sub_dl = t0 + frac * span
            if time.time() >= sub_dl:
                break
            s, C, rad = climb(best_C, sub_dl, sc)
            if s > best_sum:
                best_sum, best_C, best_rad = s, C, rad
        # Gradient-based joint (centers, radii) polish: settles the
        # winner into its exact local optimum in seconds.
        if time.time() < t0 + 0.72 * span:
            s, C, rad = slsqp_polish(best_C, best_rad, t0 + 0.72 * span)
            if s > best_sum:
                best_sum, best_C, best_rad = s, C, rad

    # Extra random restarts from the best packing (shake all centers) to
    # escape the local optimum, using the remaining time in equal slices.
    ri = 0
    while time.time() < deadline - 2.0 and best_C is not None:
        ri += 1
        sub_dl = min(t0 + (0.75 + 0.05 * ri) * span, deadline)
        C0 = best_C.copy()
        if ri % 3 == 0:
            # Occasional full shake to jump between distant basins.
            C0 += rng.normal(0.0, 0.05, (n, 2))
        else:
            # Large-neighborhood search: displace only a few circles
            # strongly while keeping the rest of the good layout intact.
            # This locally rebuilds weak spots instead of destroying the
            # whole packing, which full shakes almost never recover from.
            k = int(rng.integers(3, 9))
            sel = rng.choice(n, size=k, replace=False)
            C0[sel] += rng.normal(0.0, 0.09, (k, 2))
        C0 = np.clip(C0, 0.02, 0.98)
        s, C, rad = climb(C0, sub_dl, 0.012)
        if s > best_sum:
            best_sum, best_C, best_rad = s, C, rad

    if best_C is None:
        C, r = hex_seed([5, 4, 5, 4, 5, 3])
        return C, np.full(n, r), float(n * r)

    # Final SLSQP polish on the best-ever configuration with whatever
    # time remains (at least a 2s slice), squeezing the last few tenths
    # of a percent out of the layout before returning.
    if time.time() < deadline - 2.0:
        s, C, rad = slsqp_polish(best_C, best_rad, deadline - 1.0)
        if s > best_sum:
            best_sum, best_C, best_rad = s, C, rad

    return best_C, best_rad, float(best_sum)





# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii


def visualize(centers, radii):
    """
    Visualize the circle packing

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radius of each circle
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw unit square
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True)

    # Draw circles
    for i, (center, radius) in enumerate(zip(centers, radii)):
        circle = Circle(center, radius, alpha=0.5)
        ax.add_patch(circle)
        ax.text(center[0], center[1], str(i), ha="center", va="center")

    plt.title(f"Circle Packing (n={len(centers)}, sum={sum(radii):.6f})")
    plt.show()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii}")
    # AlphaEvolve improved this to 2.635

    # Uncomment to visualize:
    visualize(centers, radii)
