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


def _bary(P):
    """Cartesian -> exact barycentric coords w.r.t. the container triangle.

    p = wA*A + wB*B + wC*C with wA+wB+wC=1. With M = [B-A, C-A] as columns,
    (wB, wC) = M^{-1}(p - A) and wA = 1 - wB - wC. Small negative entries
    from round-off are clamped to 0 and the triple renormalized, so the
    result is always a valid convex combination (feasibility is exact).
    """
    M = np.column_stack([_B - _A, _C - _A])
    ab = np.linalg.solve(M, (np.asarray(P, dtype=float) - _A).T).T
    w = np.column_stack([1.0 - ab[:, 0] - ab[:, 1], ab])
    w = np.clip(w, 0.0, None)
    w /= w.sum(axis=1, keepdims=True)
    return w


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





def _epigraph_slsqp(z, best_z, best_a, maxiter=250, restarts=3):
    """Dense epigraph SLSQP polish: maximize t s.t. area_k(z) - t >= 0 for
    all 165 triples in the 22-D barycentric box, with the exact analytic
    constraint Jacobian from _areas_grad. Each SQP step solves the full
    inequality-constrained QP (23 vars, 165 constraints), handling
    near-active triples that block crease ascent — structure the
    equality-only active-set Newton/min-norm rounds ignore. Restarts
    re-warm from the solution with a tiny deterministic sawtooth offset
    (amplitude 1e-7) so repeated QP solves are reconditioned rather than
    identical. Every candidate is verified by the exact 165-triple min and
    the incumbent (best_z, best_a) is updated monotonically, so the caller's
    incumbent can never regress. Bounded to (restarts+1) dense SLSQP
    solves; fully exception-guarded.
    """
    try:
        z = np.clip(np.asarray(z, dtype=float).copy(), 0.0, 1.0)

        def con(x):
            a, _ = _areas_grad(x[:22])
            return a - x[-1]

        def jcon(x):
            _, g = _areas_grad(x[:22])
            return np.hstack([g, -np.ones((len(g), 1))])

        for r in range(restarts + 1):
            t0 = float(_areas(_to_xy(z)).min())
            if not np.isfinite(t0) or t0 <= 0.0:
                break
            x0 = np.concatenate([z, [t0 * 1.5]])
            bnds = ([(0.0, 1.0)] * 22
                    + [(t0, max(t0 * 6.0, 1e-3))])
            res = minimize(lambda x: -x[-1], x0,
                           jac=lambda x: np.array([-1.0]),
                           method="SLSQP", bounds=bnds,
                           constraints=[{"type": "ineq",
                                         "fun": con, "jac": jcon}],
                           options={"maxiter": maxiter, "ftol": 1e-14})
            xt = np.asarray(res.x, dtype=float).copy()
            xt[:22] = np.clip(xt[:22], 0.0, 1.0)
            a = float(_exact_min(_to_xy(xt[:22])))
            if np.isfinite(a) and a > best_a:
                best_a, best_z = a, xt[:22].copy()
            z = xt[:22]
            if r < restarts:
                z = np.clip(z + 1e-7 * (np.arange(22) % 7 - 3.0), 0.0, 1.0)
        return best_z, best_a
    except Exception:
        return best_z, best_a


def _pattern_polish(z, best_z, best_a, max_levels=15, sweeps=2):
    """Hooke-Jeeves compass search directly on the EXACT min over all 165
    triangle areas (nonsmooth objective, no surrogate).

    At the incumbent the smooth machinery (log-sum-exp soft-min L-BFGS-B and
    active-set Newton) stalls on creases of the piecewise-smooth exact min
    where no smooth gradient / nullspace direction exists. This polish takes
    coordinate steps in the raw 22-D barycentric box parameters z (point
    feasibility is structural via _to_xy), plus 6 mixed directions built
    from exact triangle symmetries (barycentric permutations of subsets),
    accepting a step only when the exact 165-triple min strictly improves —
    so (best_z, best_a) is monotonically non-decreasing and the incumbent
    can never regress. Step size halves on level failure: 3e-3 -> 1e-5.
    Bounded: <= 9 levels x 2 sweeps x (22 + 6) directions x backtracking
    trials of one vectorized _areas call each (< 1 s total). Fully
    exception-guarded; always returns a valid (best_z, best_a).
    """
    try:
        z = np.clip(np.asarray(z, dtype=float).copy(), 0.0, 1.0)
        best = float(_areas(_to_xy(z)).min())
        if best > best_a:
            best_a, best_z = best, z.copy()
        perms = [(1, 2, 0), (2, 0, 1), (1, 0, 2), (0, 2, 1), (2, 1, 0)]
        step = 3e-3
        for _lvl in range(max_levels):
            improved = False
            for _s in range(sweeps):
                # 22 coordinate directions
                dirs = [np.eye(22)[j] for j in range(22)]
                # 11 per-point diagonal directions: coupled (u, v) moves of a
                # single point. Pure coordinate axes cannot follow a crease
                # that requires both barycentric parameters of one point to
                # change together; the 4 sign combinations per point are
                # covered by the existing +/- trial logic on d.
                eye22 = np.eye(22)
                for i in range(11):
                    for sg in ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0),
                               (-1.0, -1.0)):
                        d = sg[0] * eye22[2 * i] + sg[1] * eye22[2 * i + 1]
                        dirs.append(d / np.sqrt(2.0))
                # 6 symmetry-mixed directions: barycentric permutation of
                # the 3 points most involved in near-minimal triples,
                # expressed as a finite delta in z-space (feasible target
                # reprojected to z, direction = target - current).
                P = _to_xy(z)
                areas = _areas(P)
                m = float(areas.min())
                tol = max(1e-8 * max(m, 1e-12), 1e-12)
                near = areas <= m + tol
                cnt = np.zeros(11)
                for tc in range(3):
                    cnt += np.bincount(_IJK[near, tc], minlength=11).astype(float)
                sub = np.argsort(-cnt, kind='stable')[:3]
                Br = _bary(P)
                for perm in perms:
                    Br2 = np.array(Br, dtype=float, copy=True)
                    Br2[sub] = Br[sub][:, list(perm)]
                    Br2 = np.clip(Br2, 0.0, None)
                    Br2 /= Br2.sum(axis=1, keepdims=True)
                    d = (_to_z(np.asarray(Br2 @ _TRI, dtype=float))
                         - z)
                    nd = float(np.linalg.norm(d))
                    if np.isfinite(nd) and nd > 1e-14:
                        dirs.append(step * d / nd)
                for d in dirs:
                    d = np.asarray(d, dtype=float)
                    found = False
                    trial = step
                    for _bt in range(6):
                        zt = np.clip(z + trial * d, 0.0, 1.0)
                        v = float(_areas(_to_xy(zt)).min())
                        if v > best + 1e-15:
                            z, best, improved, found = zt, v, True, True
                            if v > best_a:
                                best_a, best_z = v, z.copy()
                            break
                        trial *= 0.5
                    if not found:
                        # try the opposite direction once
                        for _bt in range(4):
                            zt = np.clip(z - trial * d, 0.0, 1.0)
                            v = float(_areas(_to_xy(zt)).min())
                            if v > best + 1e-15:
                                z, best, improved = zt, v, True
                                if v > best_a:
                                    best_a, best_z = v, z.copy()
                                break
                            trial *= 0.5
                if not improved:
                    break
            if improved:
                # adaptive re-expansion: after a productive level, grow the
                # step back toward the initial scale (capped) so the compass
                # search can traverse larger crease segments without paying
                # extra levels; strictly bounded by max_levels.
                step = min(step * 1.5, 3e-3)
            else:
                step *= 0.5
                if step < 1e-7:
                    break
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
    # Partial-symmetry-subset kicks: instead of small isotropic parameter
    # noise (which stays inside one contact topology), apply exact triangle
    # symmetries — barycentric coordinate permutations. The two cyclic
    # permutations are the 120-degree rotations about the centroid and the
    # three transpositions are the reflections about the medians; each maps
    # the equilateral triangle exactly onto itself, so every kicked point
    # stays inside the container (verified convex barycentric combo from
    # _bary) and the reprojection via _to_z is always in-box — no explicit
    # matrix reflections, so no boundary leaks like the earlier attempt.
    # The 165-triple objective is invariant under GLOBAL symmetry but NOT
    # under partial symmetry, so teleporting only a SUBSET of k points
    # (k in {2,3,4}, chosen deterministically as those participating in the
    # most near-minimal triples) yields genuinely new contact topologies —
    # finite structured jumps the decaying sinusoidal kicks could not take.
    # 5 symmetries x 3 subset sizes = 15 kicks per _refine call (same cost
    # class as the replaced loop); each is polished by the existing
    # two-tau L-BFGS-B continuation and the millisecond active-set rounds.
    # Every accept is monotone on the exact 165-triple min, so the
    # incumbent can never regress.
    perms = [(1, 2, 0), (2, 0, 1), (1, 0, 2), (0, 2, 1), (2, 1, 0)]
    for pi, perm in enumerate(perms):
        try:
            Pb = _to_xy(best_z)
            Br = _bary(Pb)
            areas = _areas(Pb)
            m = float(areas.min())
            tol = max(1e-8 * max(m, 1e-12), 1e-12)
            near = areas <= m + tol
            cnt = np.zeros(11)
            for tcol in range(3):
                cnt += np.bincount(_IJK[near, tcol], minlength=11).astype(float)
            order = np.argsort(-cnt, kind='stable')
            for k in (2, 3, 4):
                sub = order[:k]
                Br2 = np.array(Br, dtype=float, copy=True)
                Br2[sub] = Br[sub][:, list(perm)]
                Br2 = np.clip(Br2, 0.0, None)
                Br2 /= Br2.sum(axis=1, keepdims=True)  # exact feasibility
                zp = _to_z(np.asarray(Br2 @ _TRI, dtype=float))
                for tau in (200.0, 3000.0):
                    r = minimize(_soft_obj, zp, args=(tau,), jac=_soft_obj_grad,
                                 method="L-BFGS-B", bounds=bnds22,
                                 options={"maxiter": 300, "ftol": 1e-16})
                    zp = np.clip(r.x, 0.0, 1.0)
                best_z, best_a = _newton_round(zp, best_z, best_a, iters=12)
                best_z, best_a = _asan_round(zp, best_z, best_a, iters=45)
                # re-polish the incumbent so active-set gains accumulate
                best_z, best_a = _asan_round(best_z, best_z, best_a, iters=60)
                best_z, best_a = _newton_round(best_z, best_z, best_a, iters=8)
        except Exception:
            pass
        # periodic lateral one-point reinsertion (strictly monotone):
        # a discontinuous single-point basin jump the coupled 22-D
        # solvers cannot express.
        if pi % 2 == 1:
            try:
                Pk2, ak = _reinsert(_to_xy(best_z), cycles=2)
                if np.isfinite(ak) and ak > best_a:
                    best_a = float(ak)
                    best_z = _to_z(np.array(Pk2, dtype=float))
            except Exception:
                pass
    # incumbent can never regress: every accept is monotone on exact min.
    # Final polish per seed: one active-set rebalance, then the dense
    # epigraph SLSQP whose inequality-aware SQP steps resolve creases the
    # equality-only active-set rounds stall on.
    best_z, best_a = _asan_round(best_z, best_z, best_a, iters=60)
    best_z, best_a = _epigraph_slsqp(best_z, best_z, best_a,
                                     maxiter=250, restarts=2)
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


def _pair_swap_stage(P, best_a):
    """Refined combinatorial reindexing escape with deep re-polish.

    The area objective depends only on the SET of 11 points, but the
    continuous optimizers are index-locked. This stage enumerates three
    discrete move families no continuous step can express:
      (a) all C(11,2)=55 position pair swaps,
      (b) 3-cycles of point positions on triples drawn from the points
          participating most in near-minimal triangles,
      (c) cyclic single-point barycentric permutations.
    Candidates are screened by exact 165-triple min within 20% of the
    incumbent, deduplicated by rounded coordinates, and the top ones get
    the full polish chain (active-set rounds; top 3 additionally crease
    walk + epigraph SLSQP; ranks 3-5 a short active-set rebalance).
    Returns the best result; the monotone guard means it is never worse
    than the input. Bounded: ~150 microsecond screenings plus at most
    10 polished candidates. Fully exception-guarded, ASCII-only.
    """
    P0 = np.array(P, dtype=float, copy=True)
    a_in = float(_exact_min(P0))
    bestP, bestA = P0.copy(), a_in
    try:
        cands = []
        # (a) pair swaps
        for i in range(11):
            for j in range(i + 1, 11):
                Q = P0.copy()
                Q[[i, j]] = Q[[j, i]]
                a = float(_exact_min(Q))
                if np.isfinite(a) and a >= 0.8 * a_in:
                    cands.append((a, Q))
        # (b) 3-cycles on the points most involved in near-minimal areas:
        # a genuinely different discrete move than a pair swap.
        areas0 = _areas(P0)
        m0 = float(areas0.min())
        tol0 = max(1e-8 * max(m0, 1e-12), 1e-12)
        near0 = areas0 <= m0 + tol0
        cnt = np.zeros(11)
        for tcol in range(3):
            cnt += np.bincount(_IJK[near0, tcol], minlength=11).astype(float)
        hot = np.argsort(-cnt, kind='stable')[:6]
        for ai in range(len(hot)):
            for bi in range(ai + 1, len(hot)):
                for ci in range(bi + 1, len(hot)):
                    i3 = [int(hot[ai]), int(hot[bi]), int(hot[ci])]
                    Q = P0.copy()
                    Q[i3] = Q[[i3[1], i3[2], i3[0]]]
                    a = float(_exact_min(Q))
                    if np.isfinite(a) and a >= 0.8 * a_in:
                        cands.append((a, Q))
        # (c) single-point barycentric permutations
        Br0 = _bary(P0)
        for i in range(11):
            for perm in ((1, 2, 0), (2, 0, 1)):
                Br2 = np.array(Br0, dtype=float, copy=True)
                Br2[i] = Br2[i][list(perm)]
                Br2 = np.clip(Br2, 0.0, None)
                Br2 /= Br2.sum(axis=1, keepdims=True)
                Q = np.asarray(Br2 @ _TRI, dtype=float)
                a = float(_exact_min(Q))
                if np.isfinite(a) and a >= 0.8 * a_in:
                    cands.append((a, Q))
        # dedupe candidates by rounded coordinates
        uniq, seen = [], set()
        for a, Q in sorted(cands, key=lambda t: -t[0]):
            key = tuple(np.round(Q.ravel(), 9))
            if key not in seen:
                seen.add(key)
                uniq.append((a, Q))
        for rank, (a0, Q) in enumerate(uniq[:10]):
            try:
                zq = _to_z(Q)
                aq = float(_exact_min(_to_xy(zq)))
                zq, ap1 = _asan_round(zq, zq, aq, iters=40)
                zq, ap2 = _newton_round(zq, zq, ap1, iters=15)
                af = max(ap1, ap2, float(_exact_min(_to_xy(zq))))
                if rank < 3:
                    zq, ap3 = _pattern_polish(zq, zq, af)
                    af = max(ap3, float(_exact_min(_to_xy(zq))))
                    zq, ap4 = _epigraph_slsqp(zq, zq, af,
                                              maxiter=150, restarts=1)
                    af = max(ap4, float(_exact_min(_to_xy(zq))))
                elif rank < 6:
                    zq, ap3 = _asan_round(zq, zq, af, iters=60)
                    af = max(ap3, float(_exact_min(_to_xy(zq))))
                if np.isfinite(af) and af > bestA + 1e-15:
                    bestA = float(af)
                    bestP = np.array(_to_xy(zq), dtype=float)
            except Exception:
                pass
    except Exception:
        return P0, a_in
    if not (np.isfinite(bestA) and bestA > a_in):
        return P0, a_in
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


def _rigid_sweep_stage(best_P, best_a):
    """Global rigid-motion sweep continuation over the two degrees of freedom
    no incumbent operator (per-point moves, pair swaps, single-point
    reinsertion) can express: a coordinated rotation θ about the centroid and
    uniform rescale s of all 11 points simultaneously.

    The container triangle is only invariant under 120° rotations, so any
    intermediate θ changes every one of the 165 areas and lands in a
    genuinely different contact topology after local rebalancing. Scaling
    about the centroid pushes points toward/away from the boundary, flipping
    which triples are active. For each (θ, s) on a 24 x 4 deterministic grid
    (θ over [0, 2π/3), s in {0.90, 0.95, 1.00, 1.05} = 96 copies), the
    transformed set P' = CEN + s R(θ)(P - CEN) is projected through the
    convex-combination barycentric map (structural feasibility — no boundary
    leak possible), converted to box parameters with _to_z, and polished by
    one short _asan_round + _newton_round pair (milliseconds each). The best
    copy by exact 165-triple min then receives the full polish chain once.
    Strict monotone acceptance on the exact min guarantees the incumbent
    never regresses. Bounded, fully exception-guarded, ASCII-only.
    """
    try:
        P0 = np.array(best_P, dtype=float, copy=True)
        a_in = float(best_a)
        best_z, bestA = _to_z(P0), a_in
        cands = []
        # Densified sweep: 36 rotations (5-degree steps over the distinct
        # [0, 120) range) x 7 scales (0.80..1.10) = 252 screened copies.
        # Each screening is one vectorized _areas call plus one cheap
        # barycentric projection; only the top few get any polish.
        for ti in range(36):
            th = ti * (2.0 * np.pi / 3.0) / 36.0
            c, si = np.cos(th), np.sin(th)
            R = np.array([[c, -si], [si, c]])
            for s in (0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10):
                Pt = _CEN + s * (P0 - _CEN) @ R.T
                Br = _bary(Pt)                      # structural feasibility
                Br = np.clip(Br, 0.0, None)
                Br /= Br.sum(axis=1, keepdims=True)
                zq = _to_z(np.asarray(Br @ _TRI, dtype=float))
                aq = float(_exact_min(_to_xy(zq)))
                if np.isfinite(aq) and aq >= 0.70 * a_in:
                    cands.append((aq, zq))
        for rank, (aq, zq) in enumerate(
                sorted(cands, key=lambda t: -t[0])[:12]):
            try:
                zq, ap1 = _asan_round(zq, zq, aq, iters=30)
                zq, ap2 = _newton_round(zq, zq, ap1, iters=12)
                af = max(ap1, ap2, float(_exact_min(_to_xy(zq))))
                if rank < 3:
                    # full rebalance chain for the best rotated topologies:
                    # crease walk + bounded epigraph SLSQP so the new
                    # contact topology gets a genuine chance before
                    # acceptance (still only 3 such chains per call).
                    zq, ap3 = _pattern_polish(zq, zq, af,
                                              max_levels=8, sweeps=1)
                    af = max(ap3, float(_exact_min(_to_xy(zq))))
                    zq, ap4 = _epigraph_slsqp(zq, zq, af,
                                              maxiter=120, restarts=1)
                    af = max(ap4, float(_exact_min(_to_xy(zq))))
                if np.isfinite(af) and af > bestA + 1e-15:
                    bestA, best_z = float(af), np.array(zq, copy=True)
            except Exception:
                pass
        if not (np.isfinite(bestA) and bestA > a_in):
            return P0, a_in
        return np.array(_to_xy(best_z), dtype=float), float(bestA)
    except Exception:
        return np.array(best_P, dtype=float, copy=True), float(best_a)


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
            # rigid-motion-sweep-continuation stage: rotate+rescale the whole
            # 11-point set about the centroid over a deterministic (theta,
            # scale) grid — a coordinated global motion no per-point, pair,
            # or reinsertion operator can express (their line searches reject
            # any step that initially lowers the min). Feasibility is
            # structural via the barycentric map; acceptance is strictly
            # monotone on the exact 165-triple min, so best_P never regresses.
            try:
                Pr, ar = _rigid_sweep_stage(best_P, best_a)
                if np.isfinite(ar) and ar > best_a:
                    best_a = float(ar)
                    best_P = np.array(Pr, dtype=float)
            except Exception:
                pass
            # dense epigraph SLSQP polish of the overall incumbent BEFORE
            # reinsertion: the SQP QP step handles inequality-active
            # (near-minimal) triples the equality-only rounds ignore;
            # monotone accept keeps best_P non-regressing.
            try:
                ze, ae = _epigraph_slsqp(_to_z(best_P), _to_z(best_P),
                                         best_a, maxiter=250, restarts=3)
                if np.isfinite(ae) and ae > best_a:
                    best_a = float(ae)
                    best_P = np.array(_to_xy(ze), dtype=float)
            except Exception:
                pass
            # primary new mechanism: Hooke-Jeeves compass search directly on
            # the nonsmooth exact min — walks creases where the smooth
            # surrogates and active-set nullspace steps stall (monotone,
            # so best_P/best_a can only improve)
            try:
                zp, ap = _pattern_polish(_to_z(best_P),
                                         _to_z(best_P), best_a)
                if np.isfinite(ap) and ap > best_a:
                    best_a = float(ap)
                    best_P = np.array(_to_xy(zp), dtype=float)
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
                    # crease-walking compass search on the exact min after
                    # each reinsertion (monotone, incumbent-safe)
                    zp, ap = _pattern_polish(_to_z(best_P),
                                             _to_z(best_P), best_a)
                    if np.isfinite(ap) and ap > best_a:
                        best_a = float(ap)
                        best_P = np.array(_to_xy(zp), dtype=float)
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
            # combinatorial pair-swap stage BEFORE the terminal polish:
            # discrete reindexing escape (exchange positions of two points)
            # plus cyclic single-point barycentric permutations. Screened
            # candidates get deep polish (active-set + crease-walk + SLSQP
            # for the top few), and any improved basin is then finalized by
            # the full terminal sequence below. Monotone accept keeps
            # best_P non-regressing.
            try:
                Ps, a_s = _pair_swap_stage(best_P, best_a)
                if np.isfinite(a_s) and a_s > best_a:
                    best_a = float(a_s)
                    best_P = np.array(Ps, dtype=float)
            except Exception:
                pass
            # final crease-walk, then the TERMINAL epigraph SLSQP on the
            # fully polished incumbent: alternated with one short active-set
            # rebalance and a second SQP pass so newly balanced points get
            # re-optimized under the full inequality constraints. Monotone
            # accept throughout; bounded to ~3 additional dense solves.
            try:
                zf, af = _pattern_polish(_to_z(best_P),
                                         _to_z(best_P), best_a)
                if np.isfinite(af) and af > best_a:
                    best_a = float(af)
                    best_P = np.array(_to_xy(zf), dtype=float)
            except Exception:
                pass
            try:
                ze, ae = _epigraph_slsqp(_to_z(best_P), _to_z(best_P),
                                         best_a, maxiter=250, restarts=3)
                if np.isfinite(ae) and ae > best_a:
                    best_a = float(ae)
                    best_P = np.array(_to_xy(ze), dtype=float)
                za, aa = _asan_round(_to_z(best_P), _to_z(best_P),
                                     best_a, iters=50)
                if np.isfinite(aa) and aa > best_a:
                    best_a = float(aa)
                    best_P = np.array(_to_xy(za), dtype=float)
                ze2, ae2 = _epigraph_slsqp(_to_z(best_P), _to_z(best_P),
                                           best_a, maxiter=250, restarts=2)
                if np.isfinite(ae2) and ae2 > best_a:
                    best_a = float(ae2)
                    best_P = np.array(_to_xy(ze2), dtype=float)
            except Exception:
                pass
            return best_P
    except Exception:
        pass
    # valid deterministic fallback: boundary-ring seed configuration
    return np.array(_seeds()[0][:11], dtype=float)


# EVOLVE-BLOCK-END
