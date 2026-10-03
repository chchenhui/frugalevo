# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

_A = np.array([0.0, 0.0])
_B = np.array([1.0, 0.0])
_C = np.array([0.5, np.sqrt(3.0) / 2.0])
_TRI = np.vstack([_A, _B, _C])
_CEN = _TRI.mean(axis=0)

# All 165 index triples (i, j, k) for n = 11, precomputed once.
_IJK = np.array([(i, j, k)
                 for i in range(11) for j in range(i + 1, 11)
                 for k in range(j + 1, 11)], dtype=np.intp)


def _areas(P):
    """Exact (unnormalized) areas of all 165 triangles, vectorized."""
    Pi = P[_IJK[:, 0]]
    Pj = P[_IJK[:, 1]]
    Pk = P[_IJK[:, 2]]
    return 0.5 * np.abs(
        (Pj[:, 0] - Pi[:, 0]) * (Pk[:, 1] - Pi[:, 1])
        - (Pj[:, 1] - Pi[:, 1]) * (Pk[:, 0] - Pi[:, 0]))


def _exact_min(P):
    """Minimum triangle area normalized by the container area sqrt(3)/2."""
    return _areas(P).min() / (np.sqrt(3.0) / 2.0)


def _to_xy(z):
    """(22,) raw params in [0,1] -> 11 feasible points via barycentric map.

    a = u, b = v*(1-u), c = 1-a-b >= 0 ensures every point lies in the
    triangle while the optimizer sees a plain box (bounds always match x0).
    """
    z = np.clip(z, 0.0, 1.0)
    a = z[0::2]
    b = z[1::2] * (1.0 - a)
    Bary = np.column_stack([a, b, 1.0 - a - b])
    return Bary @ _TRI


def _to_z(P):
    """Inverse of _to_xy for points already inside the triangle."""
    M = np.vstack([_B - _A, _C - _A]).T
    z = np.empty((len(P), 2))
    for i, p in enumerate(P):
        ab, *_ = np.linalg.lstsq(M, p - _A, rcond=None)
        a = min(max(ab[0], 0.0), 1.0 - 1e-9)
        b = min(max(ab[1], 0.0), 1.0 - a)
        z[i] = [a, b / (1.0 - a)]
    return np.clip(z.ravel(), 0.0, 1.0)


def _to_xy_uv(z):
    """Barycentric map returning the points plus the raw (u, v) parameters."""
    z = np.clip(z, 0.0, 1.0)
    u = z[0::2]
    v = z[1::2]
    a = u
    b = v * (1.0 - u)
    Bary = np.column_stack([a, b, 1.0 - a - b])
    return Bary @ _TRI, u, v


def _areas_grad(z):
    """Areas of all 165 triples and their exact gradient wrt the 22 params."""
    P, u, v = _to_xy_uv(z)
    idx = _IJK
    Pi, Pj, Pk = P[idx[:, 0]], P[idx[:, 1]], P[idx[:, 2]]
    d1 = Pj - Pi
    d2 = Pk - Pi
    s = 0.5 * (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
    areas = np.abs(s)
    sgn = np.where(s >= 0.0, 1.0, -1.0)[:, None]
    rows = np.arange(len(idx))
    gxy = np.zeros((len(idx), 11, 2))
    gxy[rows, idx[:, 1]] = 0.5 * sgn * np.column_stack([-d2[:, 1], d2[:, 0]])
    gxy[rows, idx[:, 2]] = -0.5 * sgn * np.column_stack([-d1[:, 1], d1[:, 0]])
    gxy[rows, idx[:, 0]] = -(gxy[rows, idx[:, 1]] + gxy[rows, idx[:, 2]])
    # chain rule through the barycentric parameterization
    dPdu = _A[None, :] - v[:, None] * _B[None, :] - (1.0 - v)[:, None] * _C[None, :]
    dPdv = (1.0 - u)[:, None] * (_B - _C)[None, :]
    gz = np.empty((len(idx), 22))
    gz[:, 0::2] = np.einsum('tpi,pi->tp', gxy, dPdu)
    gz[:, 1::2] = np.einsum('tpi,pi->tp', gxy, dPdv)
    return areas, gz


def _soft_obj(z, tau):
    """Smooth minimum of the 165 areas (log-sum-exp); minimize this."""
    a = _areas(_to_xy(z))
    m = a.min()
    return m - np.log(np.sum(np.exp(-tau * (a - m)))) / tau


def _soft_obj_grad(z, tau):
    """Exact gradient of _soft_obj: weighted mean of the area gradients."""
    a, gz = _areas_grad(z)
    e = np.exp(-tau * (a - a.min()))
    w = e / e.sum()
    return gz.T @ w


def _newton_round(z, best_z, best_a, iters=30):
    """Active-set equality-constrained Newton polish (replaces SLSQP epigraph).

    At a max-min optimum the tied minimal triples A satisfy area_k(z) = t.
    Form J = [d area_k / dz , -1] (na x 23, exact gradients from
    _areas_grad). The projected-ascent direction d_free = N e_t with
    nullspace projector N = I - J^T (J J^T)^-1 J raises all active areas
    equally while remaining on the manifold; a least-norm correction
    J^+ (m - areas_A) re-feasibilizes. Backtracking line search accepts
    only strict improvements of the exact 165-triple min, with box
    clipping, so (best_z, best_a) never regresses; feasibility is
    structural via the barycentric map. Active set refreshes each
    iteration; a vanishing nullspace component signals KKT stationarity.
    Bounded to `iters` iterations of small dense solves (na <= 40);
    fully exception-guarded so the incumbent is always returned.
    """
    try:
        z = np.clip(np.asarray(z, dtype=float).copy(), 0.0, 1.0)
        for _ in range(iters):
            areas = _areas(_to_xy(z))
            m = float(areas.min())
            if m > best_a:
                best_a, best_z = m, z.copy()
            tol = max(1e-8 * max(m, 1e-12), 1e-12)
            act = areas <= m + tol
            na = int(act.sum())
            if na == 0 or na > 40:
                break
            _, gz = _areas_grad(z)
            J = np.hstack([gz[act], -np.ones((na, 1))])
            JT_K_inv = J.T @ np.linalg.inv(J @ J.T + 1e-12 * np.eye(na))
            N = np.eye(23) - JT_K_inv @ J
            g = np.zeros(23)
            g[-1] = 1.0  # ascent on t
            d_free = N @ g
            nf = float(np.linalg.norm(d_free))
            if not np.isfinite(nf) or nf < 1e-13:
                break  # KKT stationary for the current active set
            d_free /= nf
            d_corr = JT_K_inv @ (m - areas[act])
            found = False
            alpha = 0.05
            for _bt in range(10):
                zt = np.clip(z + alpha * d_free[:22] + d_corr[:22], 0.0, 1.0)
                v = float(_areas(_to_xy(zt)).min())
                if v > m + 1e-15:
                    z, found = zt, True
                    break
                alpha *= 0.5
            if not found:
                break
        a = _exact_min(_to_xy(z))
        if a > best_a:
            best_a, best_z = a, z.copy()
    except Exception:
        pass
    return best_z, best_a


def _asan_round(z, best_z, best_a, iters=30):
    """Active-set Newton ascent on the tied minimal triples (exact balancing).

    At a max-min optimum the minimal triples form a small active set A. The
    min-norm solution of J dz = dt * 1 (J = exact gradient rows of active
    areas from _areas_grad) is d ∝ J^T (J J^T + epsI)^-1 1, which raises ALL
    active areas equally — a true ascent direction for the exact min area.
    An aggressive backtracking line search on the exact min area (keep the
    best iterate, break on no improvement) with box clipping guarantees the
    incumbent never regresses; feasibility is structural (barycentric map).
    The active set refreshes each iteration from all 165 areas. Bounded to
    `iters` iterations; each solve is a small dense system (few ms).
    """
    try:
        z = np.clip(np.asarray(z, dtype=float).copy(), 0.0, 1.0)
        for _ in range(iters):
            areas = _areas(_to_xy(z))
            m = float(areas.min())
            if m > best_a:
                best_a, best_z = m, z.copy()
            act = areas <= m + max(1e-7 * max(m, 1e-12), 1e-12)
            na = int(act.sum())
            if na == 0 or na > 60:
                break
            _, gz = _areas_grad(z)
            J = np.hstack([gz[act], -np.ones((na, 1))])[:, :22]
            K = J @ J.T + 1e-10 * np.eye(na)
            d = J.T @ np.linalg.solve(K, np.ones(na))
            nd = np.max(np.abs(d))
            if not np.isfinite(nd) or nd <= 0.0:
                break
            d *= 0.05 / nd  # cap the raw step
            # backtracking line search on the exact min over all 165 triples
            best_step_z, best_step_val, found = None, m, False
            step = 1.0
            for _bt in range(14):
                zt = np.clip(z + step * d, 0.0, 1.0)
                v = float(_areas(_to_xy(zt)).min())
                if v > best_step_val + 1e-15:
                    best_step_z, best_step_val, found = zt, v, True
                    step *= 2.0  # expand while improving
                else:
                    break
            if not found:
                break
            z = best_step_z
        a = _exact_min(_to_xy(z))
        if a > best_a:
            best_a, best_z = a, z.copy()
    except Exception:
        pass
    return best_z, best_a





def _refine(z0):
    """Shared pipeline: analytic-gradient soft-min continuation (L-BFGS-B)
    followed by exact epigraph SLSQP refinement (max t s.t. area_k >= t),
    plus deterministic structured perturbation restarts around the incumbent
    (symmetry release; no RNG, fixed cosine/sine offsets of decreasing size).

    Returns best feasible point set by exact min-area, never worse than seed.
    """
    z = _to_z(_to_xy(z0))  # project seed into feasible box
    best_z, best_a = z.copy(), _exact_min(_to_xy(z))
    bnds22 = [(0.0, 1.0)] * 22
    for tau in (20.0, 60.0, 200.0, 800.0, 3000.0):
        r = minimize(_soft_obj, z, args=(tau,), jac=_soft_obj_grad,
                     method="L-BFGS-B", bounds=bnds22,
                     options={"maxiter": 400, "ftol": 1e-16})
        z = np.clip(r.x, 0.0, 1.0)
        a = _exact_min(_to_xy(z))
        if a > best_a:
            best_a, best_z = a, z.copy()
    best_z, best_a = _newton_round(z, best_z, best_a, iters=15)
    best_z, best_a = _asan_round(z if best_a <= _exact_min(_to_xy(z)) else best_z,
                                 best_z, best_a, iters=25)
    # deterministic perturbation restarts: fixed structured offsets (no RNG).
    # Reinvests the budget freed by replacing the SLSQP epigraph stage with
    # the millisecond active-set Newton stage: more restarts, slower amp
    # decay, deeper incumbent polish. All monotone, bounded, deterministic.
    # Reinvested compute: the active-set Newton stage replaced the slow
    # SLSQP epigraph path (~ms per polish), so the freed seconds go into
    # more deterministic restarts, coarser exploration amplitudes, and
    # deeper incumbent polish. Every accept is monotone on the exact
    # 165-triple min, so the incumbent can never regress.
    # Budget reinvestment: the active-set Newton stage runs in milliseconds,
    # so the freed compute goes into ~2x more deterministic restarts with a
    # slower amplitude decay (longer coarse exploration) and deeper
    # incumbent re-polish per trial. Every accept in _newton_round and
    # _asan_round is monotone on the exact 165-triple min, so the
    # incumbent can never regress regardless of trial count.
    amp = 0.018
    phases = np.arange(22) * (np.pi / 7.0)
    # Perturbation-restart budget halved (170 -> 85 trials): the freed
    # compute funds the alternating leave-one-out reinsertion stage on the
    # final incumbent (see heilbronn_triangle11), which supplies the
    # discontinuous single-point relocation the coupled 22-D solvers
    # cannot express. Every accept remains monotone on the exact min.
    for trial in range(85):
        # alternate two structured kick families: sinusoidal offsets and
        # fixed-sign cosine offsets (deterministic, no RNG)
        if trial % 2 == 0:
            kick = amp * np.sin(phases + 1.3 * trial)
        else:
            kick = amp * np.cos(phases + 0.7 * trial)
        zp = np.clip(best_z + kick, 0.0, 1.0)
        for tau in (200.0, 3000.0):
            r = minimize(_soft_obj, zp, args=(tau,), jac=_soft_obj_grad,
                         method="L-BFGS-B", bounds=bnds22,
                         options={"maxiter": 300, "ftol": 1e-16})
            zp = np.clip(r.x, 0.0, 1.0)
        best_z, best_a = _newton_round(zp, best_z, best_a, iters=12)
        best_z, best_a = _asan_round(zp, best_z, best_a, iters=45)
        # re-polish the incumbent itself so active-set gains accumulate
        # across restarts instead of applying only to the perturbed point
        best_z, best_a = _asan_round(best_z, best_z, best_a, iters=60)
        best_z, best_a = _newton_round(best_z, best_z, best_a, iters=8)
        # mid-search basin hop: lateral one-point reinsertion unlocks a
        # discontinuous relocation the coupled 22-D solvers cannot take;
        # _reinsert is strictly monotone, so best can only improve. The
        # point set is materialized from the incumbent parameters (the
        # incumbent lives in z-space inside _refine). Triggered every 8
        # trials (was 15) with 2 cycles: the active-set Newton stages run
        # in milliseconds, so the freed budget funds more frequent and
        # deeper single-point relocation passes — more discontinuous
        # basin-jump opportunities per unit compute.
        if trial % 8 == 7:
            try:
                Pb = _to_xy(best_z)
                Pk, ak = _reinsert(Pb, cycles=2)
                if np.isfinite(ak) and ak > best_a:
                    best_a = float(ak)
                    best_z = _to_z(np.array(Pk, dtype=float))
            except Exception:
                pass
        amp *= 0.975  # slow decay: explore coarser offsets much longer
    # incumbent can never regress: every accept is monotone on exact min
    best_z, best_a = _asan_round(best_z, best_z, best_a, iters=120)
    best_z, best_a = _newton_round(best_z, best_z, best_a, iters=40)
    return _to_xy(best_z), best_a


def _reinsert(P, cycles=2, grid_n=17, tau=400.0):
    """Leave-one-out cyclic one-point re-placement with LATERAL acceptance.

    For each point i: freeze the other 10 points, so the 165 triples split
    into 120 frozen triples (constant) plus the 45 triples containing i,
    whose areas are functions of the single barycentric (u, v) of point i.
    Maximize a log-sum-exp soft-min of those 45 areas over [0,1]^2 (coarse
    17x17 grid seed + analytic-gradient L-BFGS-B).  KEY MECHANISM: accept
    the relocation whenever the 45-area subproblem min does NOT degrade
    (lateral / basin-hopping move), even if the full 165-triple min does not
    yet rise — a discontinuous single-point jump across active-set basins
    that the coupled 22-D solvers provably cannot express (their line
    searches reject any infinitesimal coupled move).  After each full cycle
    the best recorded point and the lateral chain endpoint are polished
    with the existing _asan_round/_newton_round active-set solvers.  The
    return is strictly monotone: (improved P, exact min) or the original
    input unchanged.  Fully exception-guarded, bounded compute (a handful
    of milliseconds per 2-D subproblem).
    """
    P0 = np.array(P, dtype=float, copy=True)
    a0 = _exact_min(P0)
    bestP, bestA = P0.copy(), a0
    jj, kk = np.triu_indices(10, k=1)  # the 45 pairs completing a triple with i
    A, B = _A, _B
    try:
        W = P0.copy()
        us = np.linspace(0.0, 1.0, grid_n)
        UU, VV = np.meshgrid(us, us)
        grid_uv = np.column_stack([UU.ravel(), VV.ravel()])
        gp = np.array([u * A + v * (1.0 - u) * B + (1.0 - u - v * (1.0 - u)) * _C
                       for u, v in grid_uv])
        for _c in range(cycles):
            moved = False
            for i in range(11):
                others = np.delete(W, i, axis=0)
                Oj, Ok = others[jj], others[kk]
                d1o, d2o = Oj - W[i], Ok - W[i]
                old45 = float((0.5 * np.abs(d1o[:, 0] * d2o[:, 1]
                                            - d1o[:, 1] * d2o[:, 0])).min())

                def a45(p, Oj=Oj, Ok=Ok):
                    d1, d2 = Oj - p, Ok - p
                    return 0.5 * np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])

                def soft(uv, Oj=Oj, Ok=Ok):
                    u, v = float(uv[0]), float(uv[1])
                    b = v * (1.0 - u)
                    p = u * A + b * B + (1.0 - u - b) * _C
                    a = a45(p, Oj, Ok)
                    m = a.min()
                    e = np.exp(-tau * (a - m))
                    return -(m - np.log(e.sum() + 1e-300) / tau)

                def sgrad(uv, Oj=Oj, Ok=Ok):
                    u, v = float(uv[0]), float(uv[1])
                    b = v * (1.0 - u)
                    p = u * A + b * B + (1.0 - u - b) * _C
                    d1, d2 = Oj - p, Ok - p
                    s = 0.5 * (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
                    sgn = np.where(s >= 0.0, 1.0, -1.0)
                    a = np.abs(s)
                    m = a.min()
                    e = np.exp(-tau * (a - m))
                    w = e / e.sum()
                    gx = 0.5 * sgn * (d1[:, 1] - d2[:, 1])
                    gy = 0.5 * sgn * (d2[:, 0] - d1[:, 0])
                    du = A - v * B - (1.0 - v) * _C
                    dv = (1.0 - u) * (B - _C)
                    return -np.array([w @ gx * du[0] + w @ gy * du[1],
                                      w @ gx * dv[0] + w @ gy * dv[1]])

                vals = np.array([a45(p) .min() for p in gp])
                uv0 = grid_uv[int(np.argmax(vals))]
                try:
                    r = minimize(soft, uv0, jac=sgrad, method="L-BFGS-B",
                                 bounds=[(0.0, 1.0), (0.0, 1.0)],
                                 options={"maxiter": 200, "ftol": 1e-16})
                    uv = np.clip(r.x, 0.0, 1.0)
                except Exception:
                    uv = np.clip(uv0, 0.0, 1.0)
                u, v = float(uv[0]), float(uv[1])
                pnew = u * A + v * (1.0 - u) * B + (1.0 - u - v * (1.0 - u)) * _C
                # lateral acceptance: subproblem min must not degrade
                if float(a45(pnew).min()) >= old45 - 1e-12:
                    Wt = W.copy()
                    Wt[i] = pnew
                    W = Wt
                    moved = True
                    at = _exact_min(W)
                    if np.isfinite(at) and at > bestA + 1e-15:
                        bestA, bestP = at, W.copy()
            if not moved:
                break
        # polish the best recorded point and the lateral chain endpoint
        for cand in (bestP, W):
            try:
                zc = _to_z(cand)
                a_c = _exact_min(_to_xy(zc))
                zc, ac = _asan_round(zc, zc, a_c, iters=80)
                zc, ac2 = _newton_round(zc, zc, ac, iters=30)
                af = max(ac, ac2, _exact_min(_to_xy(zc)))
                if np.isfinite(af) and af > bestA + 1e-15:
                    bestA, bestP = af, _to_xy(zc)
            except Exception:
                pass
    except Exception:
        return P0, a0
    if not (np.isfinite(bestA) and bestA > a0):
        return P0, a0
    return np.array(bestP, dtype=float), float(bestA)


def _orbit(r, phase, n=3):
    """n points on a circle of radius r around the centroid."""
    return [_CEN + r * np.array([np.cos(phase + 2 * np.pi * k / n),
                                 np.sin(phase + 2 * np.pi * k / n)])
            for k in range(n)]


def _edge_pts(t1, t2):
    """Two points at fractions t1, t2 along each of the three edges."""
    return [t1 * _A + (1 - t1) * _B, t2 * _A + (1 - t2) * _B,
            t1 * _A + (1 - t1) * _C, t2 * _A + (1 - t2) * _C,
            t1 * _B + (1 - t1) * _C, t2 * _B + (1 - t2) * _C]


def _seeds():
    """12 deterministic seeds over three distinct contact topologies.

    Family A: boundary-heavy (6 edge points) + two concentric 3-orbits.
    Family B: interior-hex (centroid + two interleaved 3-orbits + 4 inner).
    Family C: C3-symmetric warm start with an explicit symmetry-breaking
              shift (delta = 0.05) releasing degenerate equal-area ties.
    Variants within a family rotate orbits/scales radii slightly.
    """
    seeds = []
    for vi in range(12):
        d = 0.06 * vi  # deterministic variant offset
        # A: boundary-ring
        P = _edge_pts(1 / 3 + d * 0.3, 2 / 3 - d * 0.3)
        P += _orbit(0.16 + d, np.pi / 2)
        P += _orbit(0.30 + d, np.pi / 2 + np.pi / 3)
        seeds.append(np.array(P[:11]))
        # B: interior-hex
        P = [_CEN] + _orbit(0.20 + d, 0.0) + _orbit(0.36 + d, np.pi / 3)
        P += _orbit(0.09 + d * 0.5, np.pi / 4, n=4)
        seeds.append(np.array(P[:11]))
        # C: symmetry-broken
        P = [_CEN] + _orbit(0.18 + d, np.pi / 2) + _orbit(0.34 + d, np.pi / 2)
        P = [np.array(p) for p in P[:7]]
        P[1:4] = P[1:4] + np.array([0.05, 0.0])  # release C3 symmetry
        P += [(1 - t) * _A + (t / 2) * _B + (t / 2) * _C
              for t in (0.25, 0.40, 0.55, 0.70)]
        seeds.append(np.array(P[:11]))
    return seeds


def heilbronn_triangle11() -> np.ndarray:
    """
     Deterministic multistart Heilbronn n=11 optimizer: 12 seeds spanning
     boundary-ring / interior-hex / symmetry-broken contact topologies, each
     refined by a shared soft-min L-BFGS-B continuation + epigraph SLSQP
    polish with deterministic perturbation restarts, in a box-constrained
    barycentric parameterization (22 vars, 22 matching bounds). Best
    exactly-verified incumbent is returned; valid fallback on any failure.
     """
    try:
        best_P, best_a = None, -np.inf
        for s in _seeds():
            if s.shape != (11, 2):
                continue
            P, a = _refine(s.ravel())
            if np.isfinite(a) and a > best_a:
                best_a, best_P = a, np.array(P, dtype=float)
        if best_P is not None and np.isfinite(best_a) and best_a > 0.0:
            # final deep global active-set Newton polish of the overall
            # incumbent (exact balancing of tied minimal triples); the
            # incumbent can only improve (monotone accept inside _asan_round)
            try:
                zb, ab = _asan_round(_to_z(best_P), _to_z(best_P),
                                     best_a, iters=250)
                # alternate the two active-set solvers: Newton's projected
                # step and the min-norm balancing step unlock different
                # contact topologies of the tied minimal triples
                if np.isfinite(ab) and ab > best_a:
                    best_a, best_P = ab, np.array(_to_xy(zb), dtype=float)
                zn, an = _newton_round(_to_z(best_P), _to_z(best_P),
                                       best_a, iters=40)
                if np.isfinite(an) and an > best_a:
                    best_a = an
                    zb, ab = _asan_round(zn, zn, best_a, iters=120)
                    if np.isfinite(ab) and ab > best_a:
                        best_a = ab
                        best_P = np.array(_to_xy(zb), dtype=float)
            except Exception:
                pass
            # Final alternating leave-one-out reinsertion stage on the
            # normal return path: each round runs cyclic exact one-point
            # re-placement with lateral acceptance (a discontinuous
            # single-point jump across active-set basins, unreachable by
            # the coupled 22-D solvers), followed by an active-set
            # _asan_round rebalance so the remaining 10 points adjust to
            # the relocated point. Both operators are strictly monotone
            # on the exact 165-triple min, so best_P can only improve.
            # Bounded compute: 6 rounds x (3 cycles x 11 subproblems +
            # one 80-iter Newton polish); each 2-D subproblem is
            # milliseconds, so total added time stays well under budget.
            # Every accept is monotone on the exact 165-triple min, so
            # best_P can only improve across the extra rounds.
            for _round in range(6):
                try:
                    Pr, ar = _reinsert(best_P, cycles=3)
                    if np.isfinite(ar) and ar > best_a:
                        best_a = float(ar)
                        best_P = np.array(Pr, dtype=float)
                except Exception:
                    pass
                try:
                    zr, a2 = _asan_round(_to_z(best_P), _to_z(best_P),
                                         best_a, iters=80)
                    if np.isfinite(a2) and a2 > best_a:
                        best_a = float(a2)
                        best_P = np.array(_to_xy(zr), dtype=float)
                except Exception:
                    pass
            return best_P
    except Exception:
        pass
    # valid deterministic fallback: boundary-ring seed configuration
    return np.array(_seeds()[0][:11], dtype=float)


# EVOLVE-BLOCK-END
