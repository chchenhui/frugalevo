# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations
from scipy.optimize import minimize


def heilbronn_triangle11() -> np.ndarray:
    """
    Maximize the minimum normalized area over all C(11,3)=165 triangles with
    vertices inside the equilateral triangle (0,0), (1,0), (0.5, sqrt(3)/2).

    Approach:
    - Barycentric coordinates via softmax of free variables: every point is
      strictly feasible by construction.
    - Vectorized computation of all 165 triangle areas.
    - Log-sum-exp soft-min surrogate with temperature continuation
      (smooth -> sharp) optimized with SLSQP from deterministic multistarts.
    - Final polish at very high temperature to align with the exact min.
    - Best valid incumbent is always retained and returned; a uniform
      barycentric fallback guarantees a valid finite output.
    """
    rng = np.random.default_rng(12345)
    n = 11
    V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]])
    tri_area = np.sqrt(3.0) / 4.0

    idx = np.array(list(combinations(range(n), 3)))
    i0, i1, i2 = idx[:, 0], idx[:, 1], idx[:, 2]

    def unpack(z):
        B = np.exp(z.reshape(n, 3) - z.reshape(n, 3).max(axis=1, keepdims=True))
        return B / B.sum(axis=1, keepdims=True)

    def all_areas(B):
        P = B @ V
        a, b, c = P[i0], P[i1], P[i2]
        return 0.5 * np.abs(
            (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
            - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        ) / tri_area

    def softmin(vals, t):
        m = vals.min()
        e = np.exp(-t * (vals - m))
        return m - np.log(e.sum()) / t

    DEGEN = 1e-6  # min-area threshold below which a config is degenerate

    def valid(B):
        P = B @ V
        d = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        return d.min() > 1e-5

    def repair(B):
        # Deterministically nudge near-coincident points apart.
        P = B @ V
        for _ in range(20):
            d = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
            np.fill_diagonal(d, np.inf)
            i, j = np.unravel_index(np.argmin(d), d.shape)
            if d[i, j] > 1e-4:
                break
            mid = (B[i] + B[j]) / 2.0
            push = np.array([1.0, -1.0, 0.0]) * 1e-3
            B[i] = np.clip(mid + push, 1e-6, 1.0)
            B[j] = np.clip(mid - push, 1e-6, 1.0)
            B[i] /= B[i].sum(); B[j] /= B[j].sum()
            P = B @ V
        return B

    def refine(z, t_sched):
        best = (-1.0, None)
        for t in t_sched:
            def obj(zz):
                return -softmin(all_areas(unpack(zz)), t)
            try:
                res = minimize(obj, z, method="SLSQP",
                               options={"maxiter": 200, "ftol": 1e-12})
                z = res.x
            except Exception:
                pass
            B = unpack(z)
            ma = all_areas(B).min()
            if ma > best[0] and valid(B):
                best = (ma, B.copy())
        # Active-set polish on the EXACT minimum area (nonsmooth but
        # derivative-free), bounded iterations, incumbent kept on failure.
        if best[1] is not None:
            zb = np.log(np.clip(best[1], 1e-12, None)).ravel()
            def exact_obj(zz):
                return -all_areas(unpack(zz)).min()
            try:
                res = minimize(exact_obj, zb, method="Powell",
                               options={"maxiter": 400, "xtol": 1e-9,
                                        "ftol": 1e-14})
                Bp = unpack(res.x)
                map_ = all_areas(Bp).min()
                if map_ > best[0] and valid(Bp):
                    best = (map_, Bp.copy())
                    zb = res.x
            except Exception:
                pass
            # Second Powell pass from its own output (nonsmooth polish).
            try:
                res = minimize(exact_obj, zb, method="Powell",
                               options={"maxiter": 400, "xtol": 1e-9,
                                        "ftol": 1e-14})
                Bp = unpack(res.x)
                map_ = all_areas(Bp).min()
                if map_ > best[0] and valid(Bp):
                    best = (map_, Bp.copy())
            except Exception:
                pass
        return best

    # --- cevian-radial seed (replaces the boundary-lattice enumeration) ---
    # Points lie on the three medians (cevians through the centroid): a
    # point on median m has barycentric coords with component m = 1-2r and
    # the other two components equal to r, r in (0, 0.5) -- strictly
    # interior by construction, so no boundary or duplicate risk.  Splits:
    # 4 radial fractions on median 0, 4 on median 1, and either 3 on
    # median 2 (11 points) or 2 on median 2 plus the centroid as the 11th
    # point.  Because the areas are invariant to which index receives
    # which fraction along a median, only strictly ordered fraction
    # SUBSETS of a coarse grid are enumerated (not permutations), keeping
    # the closed-form enumeration at a few thousand exact 165-triangle
    # minimum-area evaluations.  The best configuration becomes the seed
    # B0; the existing rotation seeds + continuation + active-set +
    # swap-descent pipeline refines it unchanged.
    def edge_bary(edge, u):
        if edge == 0:   # V0-V1, barycentric component 2 is zero
            return np.array([1.0 - u, u, 0.0])
        if edge == 1:   # V1-V2, component 0 is zero
            return np.array([0.0, 1.0 - u, u])
        return np.array([u, 0.0, 1.0 - u])  # V2-V0, component 1 is zero

    grid = np.array([0.08, 0.14, 0.20, 0.26, 0.32, 0.38])

    def med_pt(m, r):
        # point on the median from vertex m: component m = 1-2r, others r
        b = np.full(3, float(r))
        b[m] = 1.0 - 2.0 * float(r)
        return b

    from itertools import combinations as _combs
    subs4 = list(_combs(range(len(grid)), 4))
    subs3 = list(_combs(range(len(grid)), 3))
    subs2 = list(_combs(range(len(grid)), 2))
    centroid = np.full(3, 1.0 / 3.0)

    best_ma, best_cfg = -1.0, None
    for s0 in subs4:
        for s1 in subs4:
            for s2, with_c in ([(s, False) for s in subs3]
                               + [(s, True) for s in subs2]):
                B = np.empty((n, 3))
                for k, gi in enumerate(s0):
                    B[k] = med_pt(0, grid[gi])
                for k, gi in enumerate(s1):
                    B[4 + k] = med_pt(1, grid[gi])
                for k, gi in enumerate(s2):
                    B[8 + k] = med_pt(2, grid[gi])
                if with_c:
                    B[10] = centroid
                ma = all_areas(B).min()
                if ma > best_ma:
                    best_ma, best_cfg = ma, B.copy()
    B0 = best_cfg

    # Stage 2: score the three cyclic rotations of the lattice seed in closed
    # form (threefold symmetry of the domain), warm-start the continuation
    # only from the best rotation (single optimizer run, per plan budget).
    def rot(B):
        # cyclic permutation of barycentric coords = 120-degree rotation
        return B[:, [2, 0, 1]]

    seeds = [B0]
    R = B0.copy()
    for _ in range(2):
        R = rot(R)
        seeds.append(R.copy())
    base_ma = max(all_areas(S).min() for S in seeds)
    S0 = max(seeds, key=lambda S: all_areas(S).min())

    best_ma, best_B = base_ma, repair(S0.copy())
    # Warm-start the continuation from each cyclic rotation of the lattice:
    # the domain's threefold symmetry makes these distinct optimizer basins,
    # and only 3 bounded runs are used (within the plan's compute budget).
    for S in seeds:
        z0 = np.log(np.clip(S, 1e-9, None)).ravel()
        try:
            ma, B = refine(z0, (10.0, 40.0, 160.0, 640.0, 2500.0, 10000.0))
            if B is not None and ma > best_ma:
                best_ma, best_B = ma, B.copy()
        except Exception:
            pass
    # Active-set refinement of the best incumbent (replaces the 6-image
    # dihedral multistart, which consumed most of the runtime with no gain):
    # repeatedly (i) find the tight triples whose area is within a factor of
    # the current minimum, (ii) solve the smooth constrained problem
    # max t  s.t.  area_i(B(z)) - t >= 0  for active i  via SLSQP, and
    # (iii) recompute the active set.  This directly optimizes the exact
    # max-min objective, closing the softmin-vs-exact gap.  A single
    # reflected-image warm start (one symmetry-distinct basin) is kept,
    # with hard iteration caps and accept-only-if-better guards throughout.
    if best_B is not None:
        def refl(B):
            return B[:, [0, 2, 1]]

        try:
            z0 = np.log(np.clip(refl(best_B).copy(), 1e-12, None)).ravel()
            ma, B = refine(z0, (40.0, 400.0, 4000.0, 20000.0))
            if B is not None and ma > best_ma:
                best_ma, best_B = ma, B.copy()
        except Exception:
            pass

        # --- augmented-Lagrangian epigraph continuation ---
        # Replaces the hard-threshold active-set SLSQP loop. Variables are
        # (z, t) with ALL 165 constraints area_i(B(z)) - t >= 0 handled via
        # the quadratic ALM penalty; the multiplier vector lam_i discovers
        # the active set continuously (lam_i -> 0 for inactive triples) and
        # survives active-set reshuffling that blocks a fixed threshold.
        # Penalty schedule rho = 1e2..1e5, warm-started, <=10 multiplier
        # updates per round, maxiter 250 per SLSQP solve. Incumbent is only
        # accepted on strict exact-objective improvement and validity.
        zb = np.log(np.clip(best_B, 1e-12, None)).ravel()
        lam = np.zeros(len(idx))
        t_cur = float(all_areas(unpack(zb)).min())
        for rho in (1e2, 1e3, 1e4, 1e5):
            for _inner in range(10):
                def alm_obj(zz, rho=rho, lam=lam):
                    t = zz[-1]
                    g = all_areas(unpack(zz[:-1])) - t
                    viol = np.maximum(0.0, lam - rho * g)
                    return -t + 0.5 * float(viol @ viol) / rho
                z_aug = np.concatenate([zb, [t_cur]])
                try:
                    res = minimize(alm_obj, z_aug, method="SLSQP",
                                   options={"maxiter": 250, "ftol": 1e-14})
                    if not np.all(np.isfinite(res.x)):
                        break
                    z_aug = res.x
                except Exception:
                    break
                zb = z_aug[:-1]
                t_cur = float(z_aug[-1])
                g = all_areas(unpack(zb)) - t_cur
                lam = np.maximum(0.0, lam - rho * g)
                Bp = unpack(zb)
                mp = all_areas(Bp).min()
                if mp > best_ma + 1e-12 and valid(Bp):
                    best_ma, best_B = mp, Bp.copy()
                if np.max(np.maximum(0.0, -g)) < 1e-9:
                    break
        # Final exact Powell polish from the active-set incumbent.
        try:
            res = minimize(lambda zz: -all_areas(unpack(zz)).min(), zb,
                           method="Powell",
                           options={"maxiter": 600, "xtol": 1e-10,
                                    "ftol": 1e-15})
            Bp = unpack(res.x)
            if all_areas(Bp).min() > best_ma and valid(Bp):
                best_ma, best_B = all_areas(Bp).min(), Bp.copy()
        except Exception:
            pass

    # --- symmetry-snap-relax: the optimizer's iterates drift off the exact
    # C3 orbits the seed started on; small asymmetric residuals pin the
    # active-set heuristics to a slightly off-center basin.  Here we
    # (1) detect near-orbit triples under the 120-degree barycentric
    # rotation [2,0,1] and near-centroid fixed points, (2) snap each orbit
    # to its exact C3-average orbit and fixed points to the exact centroid,
    # (3) verify the snapped config loses <5% of the incumbent min area,
    # then (4) re-anneal FULLY UNCONSTRAINED from the symmetric point with
    # the existing refine continuation plus one active-set epigraph round.
    # Symmetry is used only as a repair/warm-start; accept only on strict
    # exact-objective improvement.
    if best_B is not None:
        def rotc(b):
            return b[[2, 0, 1]]

        tol = 0.05
        used = [False] * n
        snapped = best_B.copy()
        orbits = []
        for k in range(n):
            if used[k]:
                continue
            # near-centroid fixed point?
            if np.linalg.norm(best_B[k] - np.full(3, 1.0 / 3.0)) < tol:
                snapped[k] = np.full(3, 1.0 / 3.0)
                used[k] = True
                continue
            # find j such that B[j] ~ rot(B[k])
            grp = [k]
            rk = rotc(best_B[k])
            for j in range(n):
                if j != k and not used[j] and \
                        np.linalg.norm(best_B[j] - rk) < tol:
                    grp.append(j)
                    break
            if len(grp) == 2:
                r2 = rotc(rotc(best_B[grp[0]]))
                for j in range(n):
                    if j not in grp and not used[j] and \
                            np.linalg.norm(best_B[j] - r2) < tol:
                        grp.append(j)
                        break
            if len(grp) == 3:
                # exact C3-average: average of the three rotated copies
                avg = (best_B[grp[0]]
                       + rotc(best_B[grp[0]])
                       + rotc(rotc(best_B[grp[0]]))) / 3.0
                avg = avg / avg.sum()
                for m in grp:
                    snapped[m] = rotc(avg) if m != grp[0] else avg
                    # assign consistent orbit members
                snapped[grp[1]] = rotc(avg)
                snapped[grp[2]] = rotc(rotc(avg))
                for m in grp:
                    used[m] = True
                orbits.append(grp)
            else:
                used[k] = True
        snapped = np.clip(snapped, 1e-9, None)
        snapped /= snapped.sum(axis=1, keepdims=True)
        ma_snap = all_areas(snapped).min()
        if ma_snap >= 0.95 * best_ma and valid(snapped):
            zs = np.log(snapped).ravel()
            try:
                ma_r, B_r = refine(zs, (40.0, 400.0, 4000.0, 20000.0))
                if B_r is not None and ma_r > best_ma:
                    best_ma, best_B = ma_r, B_r.copy()
            except Exception:
                pass
            # one active-set epigraph round from the snapped/re-annealed point
            if B_r is not None:
                zb2 = np.log(np.clip(best_B, 1e-12, None)).ravel()
                Bc = unpack(zb2)
                areas = all_areas(Bc)
                amin = areas.min()
                act = np.where(areas <= amin * 1.6 + 1e-12)[0]
                a0, b0, c0 = i0[act], i1[act], i2[act]

                def aobj2(zz):
                    return -zz[-1]

                def acon2(zz):
                    Bp = unpack(zz[:-1])
                    P = Bp @ V
                    pa, pb, pc = P[a0], P[b0], P[c0]
                    return 0.5 * np.abs(
                        (pb[:, 0] - pa[:, 0]) * (pc[:, 1] - pa[:, 1])
                        - (pb[:, 1] - pa[:, 1]) * (pc[:, 0] - pa[:, 0])
                    ) / tri_area - zz[-1]

                try:
                    res = minimize(aobj2,
                                   np.concatenate([zb2, [amin]]),
                                   method="SLSQP",
                                   constraints=[{"type": "ineq",
                                                 "fun": acon2}],
                                   options={"maxiter": 150, "ftol": 1e-14})
                    Bp = unpack(res.x[:-1])
                    map_ = all_areas(Bp).min()
                    if map_ > best_ma + 1e-12 and valid(Bp):
                        best_ma, best_B = map_, Bp.copy()
                except Exception:
                    pass

    # --- leave-one-out reinsertion refinement (replaces pair-swap-descent) ---
    # For each of the 11 points k: delete k, re-optimize the remaining
    # 10-point max-min subproblem (C(10,3)=120 triples, fewer active
    # constraints) via a 4-temperature softmin continuation (<=150 SLSQP
    # iters each) plus one active-set epigraph polish on the exact
    # sub-areas; then reinsert k at the best of the top-3 cells of a
    # 21x21 barycentric grid scan (u, w >= 0.02, u+w <= 0.98), polishing
    # each candidate's 2 free coords with a bounded Powell, followed by
    # one full 33-variable softmin polish (<=200 SLSQP iters).  Each
    # round that strictly improves the exact 165-triangle minimum UPDATES
    # the running base, so later rounds re-optimize from the improved
    # layout (compounding gains).  A wall-clock guard (~140 s) returns
    # the incumbent on expiry; the incumbent is never degraded.
    if best_B is not None:
        import time
        _t0 = time.time()
        _TIME_LIMIT = 140.0
        idx10 = np.array(list(combinations(range(10), 3)))
        j0, j1, j2 = idx10[:, 0], idx10[:, 1], idx10[:, 2]

        def areas10(Bb):
            P = Bb @ V
            a, b, c = P[j0], P[j1], P[j2]
            return 0.5 * np.abs(
                (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
            ) / tri_area

        def unpack10(z):
            E = np.exp(z.reshape(10, 3)
                       - z.reshape(10, 3).max(axis=1, keepdims=True))
            return E / E.sum(axis=1, keepdims=True)

        base_B = best_B.copy()
        base_ma = best_ma
        for k in range(11):
            if time.time() - _t0 > _TIME_LIMIT:
                break
            order = [j for j in range(11) if j != k]
            B10 = np.clip(base_B[order].copy(), 1e-9, None)

            # stage 1: 10-point softmin continuation
            z10 = np.log(B10).ravel()
            for t in (40.0, 400.0, 4000.0, 20000.0):
                try:
                    res = minimize(
                        lambda zz, tt=t: -softmin(areas10(unpack10(zz)), tt),
                        z10, method="SLSQP",
                        options={"maxiter": 150, "ftol": 1e-12})
                    if np.all(np.isfinite(res.x)):
                        z10 = res.x
                except Exception:
                    pass
            B10 = unpack10(z10)

            # stage 2: active-set epigraph polish on exact sub-areas
            try:
                ar = areas10(B10)
                am10 = float(ar.min())
                act = np.where(ar <= am10 * 1.6 + 1e-12)[0]
                a0, b0, c0 = j0[act], j1[act], j2[act]

                def eobj(v):
                    return -v[-1]

                def econ(v):
                    Bp = unpack10(v[:-1])
                    P = Bp @ V
                    pa, pb, pc = P[a0], P[b0], P[c0]
                    return 0.5 * np.abs(
                        (pb[:, 0] - pa[:, 0]) * (pc[:, 1] - pa[:, 1])
                        - (pb[:, 1] - pa[:, 1]) * (pc[:, 0] - pa[:, 0])
                    ) / tri_area - v[-1]

                res = minimize(
                    eobj,
                    np.concatenate([np.log(np.clip(B10, 1e-12, None)).ravel(),
                                    [am10]]),
                    method="SLSQP",
                    constraints=[{"type": "ineq", "fun": econ}],
                    options={"maxiter": 150, "ftol": 1e-14})
                Bt = unpack10(res.x[:-1])
                if areas10(Bt).min() > areas10(B10).min():
                    B10 = Bt
            except Exception:
                pass

            # stage 3: grid reinsertion of the removed point k (top-3 cells)
            Bfull = np.empty((11, 3))
            Bfull[order] = B10
            gs = np.linspace(0.02, 0.98, 21)
            cells = []
            for u in gs:
                for w in gs:
                    bv = 1.0 - u - w
                    if bv < 0.02:
                        continue
                    Bfull[k] = np.array([u, bv, w])
                    cells.append((all_areas(Bfull).min(), u, bv, w))
            cells.sort(key=lambda c: -c[0])
            round_best = (all_areas(Bfull).min(), Bfull.copy())
            for _, u, bv, w in cells[:3]:
                if time.time() - _t0 > _TIME_LIMIT:
                    break
                def pt_obj(qq):
                    Bp = Bfull.copy()
                    e = np.exp(qq - qq.max())
                    Bp[k] = e / e.sum()
                    return -all_areas(Bp).min()
                q0 = np.log(np.array([u, bv, w]) + 1e-12)
                try:
                    res = minimize(pt_obj, q0, method="Powell",
                                   options={"maxiter": 120, "xtol": 1e-9,
                                            "ftol": 1e-14})
                    q = np.clip(res.x, -30.0, 30.0)
                    e = np.exp(q - q.max())
                    Bfull[k] = e / e.sum()
                except Exception:
                    Bfull[k] = np.array([u, bv, w])
                # stage 4: full 33-variable softmin polish
                zf = np.log(np.clip(Bfull, 1e-12, None)).ravel()
                try:
                    res = minimize(
                        lambda zz: -softmin(all_areas(unpack(zz)), 5000.0),
                        zf, method="SLSQP",
                        options={"maxiter": 200, "ftol": 1e-14})
                    if np.all(np.isfinite(res.x)):
                        Bp = unpack(res.x)
                        mp = all_areas(Bp).min()
                        if (mp > round_best[0] and valid(Bp)
                                and time.time() - _t0 <= _TIME_LIMIT + 10.0):
                            round_best = (mp, Bp.copy())
                except Exception:
                    pass
            if round_best[0] > base_ma + 1e-13 and valid(round_best[1]):
                base_ma = round_best[0]
                base_B = round_best[1].copy()
                # Epigraph polish on the exact objective from the accepted
                # reinsertion point: directly optimizes max-min on the
                # current active set so gains compound across rounds.
                # Bounded, wall-clock-guarded, accept-only-if-better.
                try:
                    if time.time() - _t0 <= _TIME_LIMIT + 10.0:
                        Bc = np.clip(base_B.copy(), 1e-12, None)
                        ar = all_areas(Bc)
                        am = float(ar.min())
                        act = np.where(ar <= am * 1.6 + 1e-12)[0]
                        a0, b0, c0 = i0[act], i1[act], i2[act]

                        def eobj(v):
                            return -v[-1]

                        def econ(v):
                            Bp2 = unpack(v[:-1])
                            P2 = Bp2 @ V
                            pa, pb, pc = P2[a0], P2[b0], P2[c0]
                            return 0.5 * np.abs(
                                (pb[:, 0] - pa[:, 0]) * (pc[:, 1] - pa[:, 1])
                                - (pb[:, 1] - pa[:, 1]) * (pc[:, 0] - pa[:, 0])
                            ) / tri_area - v[-1]

                        zc = np.log(Bc).ravel()
                        res = minimize(
                            eobj, np.concatenate([zc, [am]]), method="SLSQP",
                            constraints=[{"type": "ineq", "fun": econ}],
                            options={"maxiter": 150, "ftol": 1e-14})
                        if np.all(np.isfinite(res.x)):
                            Bp2 = unpack(res.x[:-1])
                            mp2 = all_areas(Bp2).min()
                            if (mp2 > base_ma + 1e-13 and valid(Bp2)
                                    and time.time() - _t0
                                    <= _TIME_LIMIT + 10.0):
                                base_ma, base_B = mp2, Bp2.copy()
                except Exception:
                    pass

        if base_ma > best_ma and valid(base_B):
            best_ma, best_B = base_ma, base_B.copy()

        # Final exact Powell polish from the leave-one-out incumbent.
        try:
            zf = np.log(np.clip(best_B, 1e-12, None)).ravel()
            res = minimize(lambda zz: -all_areas(unpack(zz)).min(), zf,
                           method="Powell",
                           options={"maxiter": 400, "xtol": 1e-10,
                                    "ftol": 1e-15})
            Bp = unpack(res.x)
            if all_areas(Bp).min() > best_ma and valid(Bp):
                best_ma, best_B = all_areas(Bp).min(), Bp.copy()
        except Exception:
            pass

    if best_B is None or best_ma < 1e-6:
        best_B = np.full((n, 3), 1.0 / 3.0)
    best_B = repair(best_B.copy())
    return np.clip(best_B, 0.0, 1.0) @ V


# EVOLVE-BLOCK-END
