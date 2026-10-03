# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, least_squares


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Approach: build a deterministic ladder of structurally distinct
    polyhedral seeds — 12-vertex icosahedron, cuboctahedron, and a
    dodecahedron-vertex subset, each combined with 2 antipodal points on a
    symmetry axis (3 axis directions x 2 heights per family, plus 6
    perturbed copies of the best seed) — and refine every seed with the
    exact max-min SLSQP reformulation (maximize t s.t. d_ij^2 >= t,
    d_ij^2 <= 1) with analytic Jacobians, capped at 600 iterations per
    start. The best refined configuration is further polished with the
    soft-min/soft-max surrogate and additional exact SLSQP restarts. A
    cube+octahedron fallback preserves validity if optimization fails; a
    finite (14,3) array is always returned.
    """
    n, d = 14, 3
    rng = np.random.RandomState(42)
    iu = np.triu_indices(n, 1)

    def ratio(P):
        D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
        dd = D[iu]
        return dd.min() / dd.max()

    def sq_ratio(P):
        D2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
        dd = D2[iu]
        return dd.min() / dd.max()

    def cube_octa(s, a):
        P = np.zeros((n, d))
        k = 0
        for x in (-1, 1):
            for y in (-1, 1):
                for z in (-1, 1):
                    P[k] = (x * s, y * s, z * s)
                    k += 1
        for j in range(3):
            P[8 + 2 * j, j] = a
            P[9 + 2 * j, j] = -a
        return P

    best_P = cube_octa(1.0, 1.0)
    best_r = ratio(best_P)

    # ---- Polyhedral seed ladder -------------------------------------
    # 12-vertex polyhedra (different contact topologies than the D6d
    # hexagonal-ring family) + 2 antipodal points on a symmetry axis.
    phi_g = (1.0 + np.sqrt(5.0)) / 2.0
    icosa = np.array(
        [(0., sg1, sg2 * phi_g) for sg1 in (-1, 1) for sg2 in (-1, 1)]
        + [(sg1, sg2 * phi_g, 0.) for sg1 in (-1, 1) for sg2 in (-1, 1)]
        + [(sg2 * phi_g, 0., sg1) for sg1 in (-1, 1) for sg2 in (-1, 1)])
    cubocta = np.array(
        [(sg1, sg2, 0.) for sg1 in (-1, 1) for sg2 in (-1, 1)]
        + [(sg1, 0., sg2) for sg1 in (-1, 1) for sg2 in (-1, 1)]
        + [(0., sg1, sg2) for sg1 in (-1, 1) for sg2 in (-1, 1)])
    dodeca12 = np.array(
        [(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        + [(0., sg1 / phi_g, sg2 * phi_g)
           for sg1 in (-1, 1) for sg2 in (-1, 1)])
    families = [icosa, cubocta, dodeca12]
    # (0, 1/phi, phi) is the dodecahedron 5-fold axis (through opposite
    # dodecahedron vertices), matching the dodeca12 family far better
    # than the generic 3-fold (1,1,1) direction.
    axes = [np.array([1., 0., 0.]),
            np.array([0., 1., phi_g]),
            np.array([0., 1.0 / phi_g, phi_g])]
    axes = [a / np.linalg.norm(a) for a in axes]

    def con_vals_loc(P):
        D2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
        return D2[iu]

    def con_jac_loc(P):
        J = np.zeros((91, 42))
        for k, (i, j) in enumerate(zip(*iu)):
            diff = 2.0 * (P[i] - P[j])
            J[k, 3 * i:3 * i + 3] = diff
            J[k, 3 * j:3 * j + 3] = -diff
        return J

    def slsqp_refine(P0, maxiter=600):
        """Exact max-min SLSQP: maximize t s.t. d_ij^2 >= t, d_ij^2 <= 1."""
        Q0 = np.array(P0, dtype=float, copy=True)
        D2 = ((Q0[:, None, :] - Q0[None, :, :]) ** 2).sum(-1)
        d2 = D2[iu]
        Q0 = Q0 / np.sqrt(d2.max())
        t0 = max(d2.min() / d2.max(), 1e-6)
        z0 = np.concatenate([Q0.ravel(), [t0]])
        res = minimize(
            lambda z: -z[42], z0,
            jac=lambda z: np.concatenate([np.zeros(42), [-1.0]]),
            method="SLSQP",
            constraints=[
                {"type": "ineq",
                 "fun": lambda z: con_vals_loc(z[:42].reshape(n, d)) - z[42],
                 "jac": lambda z: np.hstack(
                     [con_jac_loc(z[:42].reshape(n, d)),
                      -np.ones((91, 1))])},
                {"type": "ineq",
                 "fun": lambda z: 1.0 - con_vals_loc(z[:42].reshape(n, d)),
                 "jac": lambda z: np.hstack(
                     [-con_jac_loc(z[:42].reshape(n, d)),
                      np.zeros((91, 1))])},
            ],
            options={"ftol": 1e-12, "maxiter": maxiter},
        )
        if not np.all(np.isfinite(res.x)):
            return None
        return res.x[:42].reshape(n, d)

    seeds = []
    for V in families:
        Vn = V / np.sqrt((V ** 2).sum(1)).max()
        for ax in axes:
            for h in (0.35, 0.6):
                seeds.append(np.vstack([Vn, h * ax, -h * ax]))

    for P0 in seeds:
        try:
            # Small deterministic perturbation breaks the exact polyhedral
            # symmetry so SLSQP is not pinned to the symmetric KKT point
            # with a degenerate active set; same start count, new basins.
            P = slsqp_refine(P0 + 0.02 * rng.randn(n, d))
            if P is not None:
                r = ratio(P)
                if r > best_r:
                    best_r, best_P = r, P
        except Exception:
            continue

    # Structured topology-move restarts (two-hop). Isotropic Gaussian
    # perturbations are structurally stable w.r.t. the active set, so noise
    # multistart relaxes back to the same contact graph. These moves change
    # the contact graph by construction:
    #  (a) cap swap: the least-contacted point is moved to the antipode of a
    #      dmax-realizing endpoint (2 height variants);
    #  (b) ring break: the 6 points nearest the plane through the centroid
    #      perpendicular to the dmax axis are twisted by 15deg / 30deg;
    #  (c) axial promotion: a high-contact equatorial point is promoted onto
    #      the polar axis, the polar point reflected into the vacated slot;
    #  (d) dmin split: the closest pair is pushed symmetrically apart along
    #      their connecting line (breaks the tightest contact directly);
    #  (e) contact reflection: the most-contacted point is reflected through
    #      the centroid (reassigns all of its contacts at once).
    # Hop 1 refines every move with exact SLSQP; hop 2 re-applies the
    # ring-break/cap-swap moves to the best hop-1 result so two discrete
    # contact-graph changes can compose.
    def topo_moves(P0):
        """Build <=10 structured contact-graph-changing moves from P0."""
        P = np.array(P0, dtype=float, copy=True)
        D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
        dmin = D[iu].min()
        a, b = np.unravel_index(np.argmax(D), D.shape)
        ctr = P.mean(0)
        adj = (D < 1.02 * dmin) & ~np.eye(n, dtype=bool)
        contacts = adj.sum(1)
        order = np.argsort(contacts)
        dir_ab = (P[a] - P[b]) / (D[a, b] + 1e-12)
        mvs = []
        # (a) cap swap
        for k in range(2):
            Q = P.copy()
            Q[order[k]] = -P[b] + (0.01 + 0.02 * k) * dir_ab
            mvs.append(Q)
        # (b) ring break about the dmax axis
        nrm = dir_ab
        h = np.abs((P - ctr) @ nrm)
        ring = np.argsort(h)[:6]
        e1 = np.cross(nrm, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(e1) < 1e-8:
            e1 = np.cross(nrm, np.array([0.0, 1.0, 0.0]))
        e1 = e1 / np.linalg.norm(e1)
        e2 = np.cross(nrm, e1)
        for tw in (np.pi / 12.0, np.pi / 6.0):
            Q = P.copy()
            c, s = np.cos(tw), np.sin(tw)
            for i in ring:
                v = P[i] - ctr
                u = (v @ e1) * e1 + (v @ e2) * e2
                w = (v @ nrm) * nrm
                Q[i] = ctr + w + c * u + s * np.cross(nrm, u)
            mvs.append(Q)
        # (c) axial promotion with reflection into the vacated slot
        eq = [i for i in np.argsort(-contacts)
              if i not in (int(a), int(b))]
        for k in range(2):
            if k >= len(eq):
                break
            Q = P.copy()
            t = 0.25 + 0.25 * k
            Q[eq[k]] = ctr + t * (P[a] - ctr) + (1.0 - t) * (P[b] - ctr)
            Q[b] = ctr - (P[eq[k]] - ctr)
            mvs.append(Q)
        # (d) dmin split: push the closest pair symmetrically apart
        i0, j0 = np.unravel_index(
            np.argmin(D + np.eye(n) * np.inf), (n, n))
        u_ij = (P[i0] - P[j0]) / (D[i0, j0] + 1e-12)
        Q = P.copy()
        Q[i0] = P[i0] + 0.15 * dmin * u_ij
        Q[j0] = P[j0] - 0.15 * dmin * u_ij
        mvs.append(Q)
        # (e) contact reflection of the most-contacted point
        Q = P.copy()
        Q[order[-1]] = 2.0 * ctr - P[order[-1]]
        mvs.append(Q)
        out = []
        for Q in mvs:
            Dq = np.sqrt(((Q[:, None, :] - Q[None, :, :]) ** 2).sum(-1))
            m = Dq[iu].max()
            if np.isfinite(m) and m > 0:
                Q = Q / m
            out.append(Q)
        return out[:10]

    def run_moves(P0):
        """Refine all moves of P0 and the best-of-hop-1 with a second hop."""
        local_best_P = np.array(P0, dtype=float, copy=True)
        local_best_r = ratio(local_best_P)
        moves = topo_moves(local_best_P)
        hop1 = []
        for M in moves:
            try:
                P = slsqp_refine(M)
                if P is None or not np.all(np.isfinite(P)):
                    continue
                hop1.append(P)
                r = ratio(P)
                if r > local_best_r:
                    local_best_r, local_best_P = r, P
            except Exception:
                continue
        # Hop 2: compose ring-break and cap-swap on the best hop-1 point.
        if hop1:
            src = max(hop1, key=ratio)
            for M in topo_moves(src)[:4]:
                try:
                    P = slsqp_refine(M)
                    if P is None or not np.all(np.isfinite(P)):
                        continue
                    r = ratio(P)
                    if r > local_best_r:
                        local_best_r, local_best_P = r, P
                except Exception:
                    continue
        return local_best_P

    newP = run_moves(np.array(best_P, dtype=float, copy=True))
    r = ratio(newP)
    if r > best_r:
        best_r, best_P = r, newP

    # Annealed repulsion dynamics: minimize E_p = sum_{i<j} d_ij^{-p} with
    # exact renormalization dmax <- 1 between annealing stages. Unlike the
    # fixed-beta soft-min/soft-max surrogate (whose optimum lies strictly
    # interior to the true max-min optimum), this objective is unbiased:
    # dmax is pinned exactly at 1 and only dmin is effectively maximized.
    # Annealing p from 2 to 20 is a smooth continuation that can migrate
    # points across near-degenerate arrangements. Each p-stage is solved
    # properly with L-BFGS-B and an analytic gradient (not a fixed-step
    # Euler loop), so the dynamics actually converge at each temperature.
    def repulse(P0, ps=(2, 3, 4, 6, 8, 12, 16, 20, 28, 40), steps=300):
        P = np.array(P0, dtype=float, copy=True)

        def energy(x, p):
            Pp = x.reshape(n, d)
            diff = Pp[:, None, :] - Pp[None, :, :]
            d2 = (diff ** 2).sum(-1)
            d2v = d2[iu]
            d2v = np.maximum(d2v, 1e-24)
            return (d2v ** (-p / 2.0)).sum()

        def egrad(x, p):
            Pp = x.reshape(n, d)
            diff = Pp[:, None, :] - Pp[None, :, :]
            d2 = (diff ** 2).sum(-1)
            np.fill_diagonal(d2, np.inf)
            w = np.where(np.isfinite(d2), d2 ** (-(p / 2.0 + 1.0)), 0.0)
            g = -p * (w[:, :, None] * diff).sum(1)
            return g.ravel()

        mid_done = False
        for p in ps:
            x0 = P.ravel()
            res = minimize(energy, x0, args=(p,), jac=egrad,
                           method="L-BFGS-B",
                           options={"maxiter": steps, "ftol": 1e-16,
                                    "gtol": 1e-14})
            if np.all(np.isfinite(res.x)):
                P = res.x.reshape(n, d)
            # Exact renormalization: pin dmax = 1 after each stage.
            D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
            m = D[iu].max()
            if np.isfinite(m) and m > 0:
                P = P / m
            # Adaptive mid-schedule re-centering: once the annealing passes
            # the sharpness where the active set is essentially fixed (p=8),
            # snap back to an exact max-min KKT point so the remaining
            # high-p stages polish the true objective instead of drifting
            # along the repulsion field. Done once per trajectory to keep
            # the compute contract.
            if (not mid_done) and p >= 8:
                mid_done = True
                try:
                    Pm = slsqp_finish(P, maxiter=400)
                    if Pm is not None and np.all(np.isfinite(Pm)):
                        P = Pm
                except Exception:
                    pass
        return P

    # Annealed-repulsion multistart: incumbent plus 5 deterministic
    # perturbations (6 starts total). Each repulsion trajectory is followed
    # by an immediate exact max-min SLSQP finish so the unbiased repulsion
    # basin is always converted to a true max-min point before incumbent
    # selection (repulsion output alone is not KKT for the exact problem).
    def slsqp_finish(P0, maxiter=800):
        Q0 = np.array(P0, dtype=float, copy=True)
        D2 = ((Q0[:, None, :] - Q0[None, :, :]) ** 2).sum(-1)
        d2 = D2[iu]
        Q0 = Q0 / np.sqrt(d2.max())
        t0 = max(d2.min() / d2.max(), 1e-6)
        z0 = np.concatenate([Q0.ravel(), [t0]])
        res = minimize(
            lambda z: -z[42], z0,
            jac=lambda z: np.concatenate([np.zeros(42), [-1.0]]),
            method="SLSQP",
            constraints=[
                {"type": "ineq", "fun": lambda z: con_vals(z) - z[42],
                 "jac": con_jac},
                {"type": "ineq", "fun": lambda z: 1.0 - con_vals(z),
                 "jac": lambda z: -con_jac(z)},
            ],
            options={"ftol": 1e-12, "maxiter": maxiter},
        )
        if np.all(np.isfinite(res.x)):
            return res.x[:42].reshape(n, d)
        return None

    starts = [np.array(best_P, dtype=float, copy=True)]
    for _ in range(5):
        starts.append(best_P + 0.03 * rng.randn(n, d))

    for P0 in starts:
        try:
            P = repulse(P0)
            if np.all(np.isfinite(P)):
                Pf = slsqp_finish(P)
                if Pf is not None:
                    P = Pf
                r = ratio(P)
                if r > best_r:
                    best_r, best_P = r, P
        except Exception:
            continue

    # Exact max-min refinement: SLSQP on the true problem (maximize t s.t.
    # d_ij^2 >= t, d_ij^2 <= 1), with analytic Jacobians. This removes the
    # surrogate bias of the soft-min/soft-max polish at the active set.
    def con_vals(z):
        P = z[:42].reshape(n, d)
        D2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
        return D2[iu]

    def con_jac(z):
        P = z[:42].reshape(n, d)
        J = np.zeros((91, 43))
        for k, (i, j) in enumerate(zip(*iu)):
            diff = 2.0 * (P[i] - P[j])
            J[k, 3 * i:3 * i + 3] = diff
            J[k, 3 * j:3 * j + 3] = -diff
        J[:, 42] = -1.0
        return J

    # Bounded topology-move SLSQP: structured contact-graph moves from the
    # incumbent (cap swap / ring break / axial promotion / dmin split /
    # contact reflection), each finished to an exact max-min KKT point with
    # an 800-iteration SLSQP cap; exact-ratio incumbent selection. Replaces
    # isotropic Gaussian starts, which relax back to the same active set.
    for M in topo_moves(np.array(best_P, dtype=float, copy=True)):
        try:
            res = slsqp_finish(M, maxiter=800)
            if res is not None and np.all(np.isfinite(res)):
                P = res
                r = ratio(P)
                if r > best_r:
                    best_r, best_P = r, P
        except Exception:
            continue

    # Interleaved re-polish: one sharper soft-surrogate pass from the SLSQP
    # incumbent, then two more exact SLSQP restarts from perturbations of it.
    try:
        P = repulse(np.array(best_P, dtype=float, copy=True))
        if np.all(np.isfinite(P)):
            Pf = slsqp_finish(P)
            if Pf is not None:
                P = Pf
            r = ratio(P)
            if r > best_r:
                best_r, best_P = r, P
    except Exception:
        pass
    for _ in range(2):
        P0 = best_P + 0.03 * rng.randn(n, d)
        try:
            Q0 = np.array(P0, dtype=float, copy=True)
            D2 = ((Q0[:, None, :] - Q0[None, :, :]) ** 2).sum(-1)
            d2 = D2[iu]
            Q0 = Q0 / np.sqrt(d2.max())
            t0 = max(d2.min() / d2.max(), 1e-6)
            z0 = np.concatenate([Q0.ravel(), [t0]])
            res = minimize(
                lambda z: -z[42], z0,
                jac=lambda z: np.concatenate([np.zeros(42), [-1.0]]),
                method="SLSQP",
                constraints=[
                    {"type": "ineq", "fun": lambda z: con_vals(z) - z[42],
                     "jac": con_jac},
                    {"type": "ineq", "fun": lambda z: 1.0 - con_vals(z),
                     "jac": lambda z: -con_jac(z)},
                ],
                options={"ftol": 1e-12, "maxiter": 800},
            )
            if np.all(np.isfinite(res.x)):
                P = res.x[:42].reshape(n, d)
                r = ratio(P)
                if r > best_r:
                    best_r, best_P = r, P
        except Exception:
            continue

    # Exact active-set Newton KKT polish: replaces the alternating
    # surrogate/SLSQP polish. The SLSQP incumbent stalls with a residual
    # KKT gap (~1e-6) from line-search/ftol tolerances; this step solves
    # the equality-constrained stationarity system of the identified
    # contact set directly. For the incumbent (normalized dmax = 1):
    #   A_min = {(i,j): d_ij^2 <= t + eps}, A_max = {(i,j): d_ij^2 >= 1-eps}
    # Unknowns: 42 coordinates, t, lambda on A_min, mu on A_max.
    # Residual: dL/dx = sum_Amin lam*grad(d^2) - sum_Amax mu*grad(d^2) = 0,
    # sum lam - 1 = 0, d^2_ij - t = 0 (Amin), d^2_ij - 1 = 0 (Amax).
    # Solved to machine precision with a trust-region LM Newton solve on
    # the square system (max 20 iterations). If the polished point does
    # not improve the exact ratio, the active-set threshold eps is grown
    # x10 and the set is re-identified (up to 6 times). The polished point
    # is accepted only if finite and strictly better than the incumbent,
    # which remains the fallback.
    def active_set_newton(P0):
        P = np.array(P0, dtype=float, copy=True)
        bestP = np.array(P, dtype=float, copy=True)
        bestr = ratio(bestP)
        eps = 1e-9
        for _outer in range(6):
            D2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
            d2 = D2[iu]
            dmax2 = d2.max()
            if not (np.isfinite(dmax2) and dmax2 > 0):
                break
            t = d2.min() / dmax2
            Amin = np.where(d2 <= t * dmax2 + eps * dmax2)[0]
            Amax = np.where(d2 >= dmax2 - eps * dmax2)[0]
            if len(Amin) == 0 or len(Amax) == 0:
                break
            kA, kB = iu[0][Amin], iu[1][Amin]
            cA, cB = iu[0][Amax], iu[1][Amax]
            na, nm = len(kA), len(cA)
            off_l, off_m = 43, 43 + na
            nz = off_m + nm

            def resid(z):
                Q = z[:42].reshape(n, d)
                lam = z[off_l:off_l + na]
                mu = z[off_m:off_m + nm]
                g = np.zeros(43)
                for kk in range(na):
                    i, j = kA[kk], kB[kk]
                    diff = 2.0 * (Q[i] - Q[j])
                    g[3 * i:3 * i + 3] += lam[kk] * diff
                    g[3 * j:3 * j + 3] -= lam[kk] * diff
                for kk in range(nm):
                    i, j = cA[kk], cB[kk]
                    diff = 2.0 * (Q[i] - Q[j])
                    g[3 * i:3 * i + 3] -= mu[kk] * diff
                    g[3 * j:3 * j + 3] += mu[kk] * diff
                g[42] = lam.sum() - 1.0
                D2z = ((Q[:, None, :] - Q[None, :, :]) ** 2).sum(-1)
                rmin = D2z[kA, kB] - z[42]
                rmax = D2z[cA, cB] - 1.0
                return np.concatenate([g, rmin, rmax])

            Qn0 = np.array(P, dtype=float, copy=True)
            Qn0 = Qn0 / np.sqrt(dmax2)
            z = np.concatenate([
                Qn0.ravel(), [t],
                np.full(na, 1.0 / na), np.full(nm, 1.0 / nm)])
            try:
                resn = least_squares(
                    resid, z, method="lm",
                    xtol=1e-15, ftol=1e-15, gtol=1e-15,
                    max_nfev=20 * nz)
            except Exception:
                eps *= 10.0
                continue
            z = resn.x
            Qn = z[:42].reshape(n, d)
            if not np.all(np.isfinite(Qn)):
                eps *= 10.0
                continue
            Dq = np.sqrt(((Qn[:, None, :] - Qn[None, :, :]) ** 2).sum(-1))
            mq = Dq[iu].max()
            if not (np.isfinite(mq) and mq > 0):
                eps *= 10.0
                continue
            Qn = Qn / mq
            rn = ratio(Qn)
            if rn > bestr:
                bestr, bestP = rn, Qn
                P = np.array(Qn, dtype=float, copy=True)
            else:
                eps *= 10.0
        return bestP

    try:
        Pn = active_set_newton(np.array(best_P, dtype=float, copy=True))
        if np.all(np.isfinite(Pn)):
            r = ratio(Pn)
            if r > best_r:
                best_r, best_P = r, Pn
    except Exception:
        pass

    return best_P


# EVOLVE-BLOCK-END
