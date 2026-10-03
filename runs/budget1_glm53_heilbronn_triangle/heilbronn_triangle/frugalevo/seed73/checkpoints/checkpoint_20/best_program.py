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

    # --- deterministic boundary-lattice seed with closed-form enumeration ---
    # 9 boundary points (3 per edge), each parameterized by a single edge
    # coordinate u in [0.02, 0.98], plus 2 interior points.  Following the
    # plan, candidate u-triples, per-edge orderings of the u values, and
    # interior-point candidates are enumerated and each full configuration is
    # scored in closed form via the exact 165-triangle minimum area; the best
    # non-degenerate configuration becomes the seed.  This guarantees a
    # strictly positive, directly computable starting min area.
    def edge_bary(edge, u):
        if edge == 0:   # V0-V1, barycentric component 2 is zero
            return np.array([1.0 - u, u, 0.0])
        if edge == 1:   # V1-V2, component 0 is zero
            return np.array([0.0, 1.0 - u, u])
        return np.array([u, 0.0, 1.0 - u])  # V2-V0, component 1 is zero

    from itertools import permutations as _perms

    u_sets = [(1.0 / 3.0, 0.5, 2.0 / 3.0),
              (0.25, 0.5, 0.75),
              (1.0 / 6.0, 0.5, 5.0 / 6.0),
              (0.2, 0.5, 0.8)]
    int_base = [np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]),
                np.array([0.5, 0.25, 0.25]),
                np.array([0.4, 0.4, 0.2]),
                np.array([0.6, 0.2, 0.2])]
    int_cands = []
    for b in int_base:
        r = b
        for _ in range(3):  # include the three 120-degree rotations
            int_cands.append(r.copy())
            r = r[[2, 0, 1]]

    best_ma, best_cfg = -1.0, None
    for us in u_sets:
        for o0 in _perms(range(3)):
            for o1 in _perms(range(3)):
                for o2 in _perms(range(3)):
                    for ic in int_cands:
                        for jc in int_cands:
                            B = np.empty((n, 3))
                            for e, order in enumerate((o0, o1, o2)):
                                for k in range(3):
                                    B[3 * e + k] = edge_bary(e, us[order[k]])
                            B[9], B[10] = ic, jc
                            ma = all_areas(B).min()
                            if ma > best_ma:
                                best_ma, best_cfg = ma, B.copy()
    B0 = best_cfg

    # Stage 1: refine only the 9 edge coordinates (interior fixed), so the
    # optimizer cannot collapse boundary points onto each other.
    def edge_obj(uv, t):
        B = B0.copy()
        for j, u in enumerate(uv):
            B[j] = edge_bary(j // 3, u)
        return -softmin(all_areas(B), t)

    uv = np.array([1.0 / 3.0, 0.5, 2.0 / 3.0] * 3)
    ma_seed = all_areas(B0).min()
    for t in (10.0, 100.0, 1000.0):
        try:
            res = minimize(edge_obj, uv, args=(t,), method="SLSQP",
                           bounds=[(0.02, 0.98)] * 9,
                           options={"maxiter": 300, "ftol": 1e-14})
            cand = np.clip(res.x, 0.02, 0.98)
            # Accept a temperature level only if the exact min area of the
            # resulting lattice does not degrade.
            Bc = B0.copy()
            for j, u in enumerate(cand):
                Bc[j] = edge_bary(j // 3, u)
            if all_areas(Bc).min() >= ma_seed:
                uv = cand
                ma_seed = all_areas(Bc).min()
        except Exception:
            pass
    for j, u in enumerate(uv):
        B0[j] = edge_bary(j // 3, u)

    # Interior subproblem: refine the 2 interior points (6 barycentric
    # variables) with the (now optimized) boundary lattice held fixed.
    def interior_obj(w, t):
        B = B0.copy()
        B[9] = np.clip(np.array([w[0], w[1], w[2]]), 1e-4, None)
        B[9] /= B[9].sum()
        B[10] = np.clip(np.array([w[3], w[4], w[5]]), 1e-4, None)
        B[10] /= B[10].sum()
        return -softmin(all_areas(B), t)

    w0 = np.concatenate([B0[9], B0[10]])
    for t in (100.0, 1000.0):
        try:
            res = minimize(interior_obj, w0, args=(t,), method="SLSQP",
                           options={"maxiter": 200, "ftol": 1e-14})
            Bc = B0.copy()
            Bc[9] = np.clip(res.x[0:3], 1e-4, None)
            Bc[9] /= Bc[9].sum()
            Bc[10] = np.clip(res.x[3:6], 1e-4, None)
            Bc[10] /= Bc[10].sum()
            if all_areas(Bc).min() >= all_areas(B0).min():
                B0 = Bc
                w0 = res.x
        except Exception:
            pass

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

    if best_B is None or best_ma < 1e-6:
        best_B = np.full((n, 3), 1.0 / 3.0)
    best_B = repair(best_B.copy())
    return np.clip(best_B, 0.0, 1.0) @ V


# EVOLVE-BLOCK-END
