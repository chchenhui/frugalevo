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

        # --- active-set refinement loop on the exact objective ---
        zb = np.log(np.clip(best_B, 1e-12, None)).ravel()
        for _round in range(15):
            Bc = unpack(zb)
            areas = all_areas(Bc)
            amin = areas.min()
            act = np.where(areas <= amin * 1.6 + 1e-12)[0]
            a0, b0, c0 = i0[act], i1[act], i2[act]

            def aobj(zz):
                return -zz[-1]

            def acon(zz):
                Bp = unpack(zz[:-1])
                P = Bp @ V
                pa, pb, pc = P[a0], P[b0], P[c0]
                ar = 0.5 * np.abs(
                    (pb[:, 0] - pa[:, 0]) * (pc[:, 1] - pa[:, 1])
                    - (pb[:, 1] - pa[:, 1]) * (pc[:, 0] - pa[:, 0])
                ) / tri_area
                return ar - zz[-1]

            z_aug = np.concatenate([zb, [amin]])
            try:
                res = minimize(aobj, z_aug, method="SLSQP",
                               constraints=[{"type": "ineq", "fun": acon}],
                               options={"maxiter": 150, "ftol": 1e-14})
                z_new = res.x[:-1]
                Bp = unpack(z_new)
                map_ = all_areas(Bp).min()
                if map_ > best_ma + 1e-12 and valid(Bp):
                    best_ma, best_B = map_, Bp.copy()
                    zb = z_new
                else:
                    # keep moving even without strict improvement so the
                    # active set can reshuffle; only accept non-degrading
                    if map_ >= amin - 1e-12 and valid(Bp):
                        zb = z_new
            except Exception:
                pass
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

    # --- pair-swap-descent: mixed discrete-continuous refinement appended
    # after the active-set/Powell polish.  Discrete moves change the
    # point-to-edge/interior assignment (edge-edge swaps carrying each
    # point's u-coordinate from its original to its destination edge,
    # edge<->interior swaps, edge->interior migration, interior->edge
    # migration toward the edge with the largest empty arc).  Each move is
    # re-polished in two coupled stages (moved points only, then all
    # variables) and accepted only if the exact 165-triangle minimum area
    # strictly improves; otherwise the incumbent is reverted.  Budget:
    # at most 40 discrete moves, each with bounded SLSQP work.
    if best_B is not None:
        def classify(B):
            labels = []
            for k in range(n):
                b = B[k]
                if b.min() < 5e-3:
                    labels.append(int(np.argmin(b)))
                else:
                    labels.append(3)
            return labels

        def proj_interior(b):
            w = np.clip(np.asarray(b, dtype=float), 1e-4, None)
            return w / w.sum()

        def repolish_moved(Bc, free_idx, t=500.0):
            Bw = Bc.copy()
            nf = len(free_idx)

            def to_B(x):
                E = np.exp(x.reshape(nf, 3)
                           - x.reshape(nf, 3).max(axis=1, keepdims=True))
                return E / E.sum(axis=1, keepdims=True)

            zf = np.log(np.clip(Bw[free_idx], 1e-9, None)).ravel()

            def obj(x):
                Bt = Bw.copy()
                Bt[free_idx] = to_B(x)
                return -softmin(all_areas(Bt), t)

            try:
                res = minimize(obj, zf, method="SLSQP",
                               options={"maxiter": 150, "ftol": 1e-14})
                Bt = Bw.copy()
                Bt[free_idx] = to_B(res.x)
                return Bt
            except Exception:
                return Bc

        def repolish_full(Bc, t=2000.0):
            zf = np.log(np.clip(Bc, 1e-12, None)).ravel()
            try:
                res = minimize(lambda zz: -softmin(all_areas(unpack(zz)), t),
                               zf, method="SLSQP",
                               options={"maxiter": 200, "ftol": 1e-14})
                return unpack(res.x)
            except Exception:
                return Bc

        def try_move(Bc, Bn, moved):
            ma0 = all_areas(Bc).min()
            Bp = repolish_moved(Bn, moved)
            mp = all_areas(Bp).min()
            if mp > ma0 + 1e-12:
                Bp = repolish_full(Bp)
                mp = all_areas(Bp).min()
                if mp > ma0 + 1e-12 and valid(Bp):
                    return Bp, mp
            return None, ma0

        moves_used = 0
        improved_any = True
        while improved_any and moves_used < 40:
            improved_any = False
            labels = classify(best_B)
            edge_pts = {e: [k for k in range(n) if labels[k] == e]
                        for e in range(3)}
            int_pts = [k for k in range(n) if labels[k] == 3]

            def accept(Bn, moved):
                nonlocal best_B, best_ma, moves_used, improved_any
                nb, nm = try_move(best_B, Bn, moved)
                moves_used += 1
                if nb is not None:
                    best_B, best_ma, improved_any = nb, nm, True

            # (i) edge-edge swaps: u carried original -> destination edge
            done = False
            for e1 in range(3):
                for e2 in range(e1 + 1, 3):
                    for p1 in edge_pts[e1]:
                        for p2 in edge_pts[e2]:
                            if moves_used >= 40:
                                done = True
                                break
                            u1 = float(np.clip(best_B[p1][e1], 0.02, 0.98))
                            u2 = float(np.clip(best_B[p2][e2], 0.02, 0.98))
                            Bn = best_B.copy()
                            Bn[p1] = edge_bary(e1, u2)
                            Bn[p2] = edge_bary(e2, u1)
                            accept(Bn, [p1, p2])
                            if improved_any:
                                done = True
                                break
                        if done or moves_used >= 40:
                            break
                    if done or moves_used >= 40:
                        break
                if done or moves_used >= 40:
                    break
            if improved_any or moves_used >= 40:
                continue

            # (ii) edge <-> interior swaps
            done = False
            for e in range(3):
                for p in edge_pts[e]:
                    for q in int_pts:
                        if moves_used >= 40:
                            done = True
                            break
                        ue = float(np.clip(best_B[p][e], 0.02, 0.98))
                        Bn = best_B.copy()
                        Bn[p] = proj_interior(best_B[q])
                        Bn[q] = edge_bary(e, ue)
                        accept(Bn, [p, q])
                        if improved_any:
                            done = True
                            break
                    if done or moves_used >= 40:
                        break
                if done or moves_used >= 40:
                    break
            if improved_any or moves_used >= 40:
                continue

            # (iii) edge -> interior migration (blend toward centroid)
            done = False
            for e in range(3):
                for p in edge_pts[e]:
                    if moves_used >= 40:
                        done = True
                        break
                    Bn = best_B.copy()
                    Bn[p] = proj_interior(
                        0.5 * best_B[p] + 0.5 * np.array([1/3, 1/3, 1/3]))
                    accept(Bn, [p])
                    if improved_any:
                        done = True
                        break
                if done or moves_used >= 40:
                    break
            if improved_any or moves_used >= 40:
                continue

            # (iv) interior -> edge with the largest empty arc
            if int_pts:
                gaps = []
                for e in range(3):
                    us = sorted(best_B[k][e] for k in edge_pts[e])
                    if not us:
                        gaps.append(1.0)
                    else:
                        gaps.append(max([us[0]]
                                        + [us[j + 1] - us[j]
                                           for j in range(len(us) - 1)]
                                        + [1.0 - us[-1]]))
                e_star = int(np.argmax(gaps))
                for p in int_pts:
                    if moves_used >= 40:
                        break
                    Bn = best_B.copy()
                    Bn[p] = edge_bary(e_star, 0.5)
                    accept(Bn, [p])
                    if improved_any:
                        break

        # Final exact Powell polish from the swap-descent incumbent.
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
