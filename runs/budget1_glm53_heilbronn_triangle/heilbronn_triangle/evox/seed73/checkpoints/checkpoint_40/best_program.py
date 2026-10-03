# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
from scipy.optimize import minimize

_S3 = np.sqrt(3.0) / 2.0
_IDX = np.array(list(itertools.combinations(range(11), 3)))
_CACHE = None


def _areas_all(P):
    """Areas of all C(11,3)=165 triplets for one configuration."""
    a = P[_IDX[:, 0]]
    b = P[_IDX[:, 1]]
    c = P[_IDX[:, 2]]
    return 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                        - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))


def _signed_area(P, i, j, k):
    """Signed area of triplet (i, j, k)."""
    a, b, c = P[i], P[j], P[k]
    return 0.5 * ((b[0] - a[0]) * (c[1] - a[1])
                  - (b[1] - a[1]) * (c[0] - a[0]))


def _lift(P, deadline, restarts=4):
    """Exact max-min refinement via SLSQP on the NLP:

        maximize t  s.t.  area_i(z) >= t  (all 165 triplets),
                          points inside the triangle.

    Variables z = [t, x0, y0, ..., x10, y10]. All constraint Jacobians are
    analytic (signed-area vertex gradients), so each solve is fast. Runs
    a few restarts from slightly perturbed copies to escape local
    convergence. Returns (P, min_area).
    """
    s = np.sqrt(3.0)
    P0 = _project_inside(np.asarray(P, dtype=float))
    A0 = float(_areas_all(P0).min())
    best_P, best_A = P0, A0
    I = _IDX
    m = len(I)
    rng = np.random.default_rng(777)

    def obj(z):
        return -z[0]

    def obj_jac(z):
        g = np.zeros(23)
        g[0] = -1.0
        return g

    def cons_area(z):
        Q = z[1:].reshape(11, 2)
        return _areas_all(Q) - z[0]

    def cons_area_jac(z):
        Q = z[1:].reshape(11, 2)
        a = Q[I[:, 0]]; b = Q[I[:, 1]]; c = Q[I[:, 2]]
        S = 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                   - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
        sgn = np.where(S >= 0.0, 1.0, -1.0)
        J = np.zeros((m, 23))
        J[:, 0] = -1.0
        for r in range(m):
            i, j, k = I[r]
            ar, br, cr = a[r], b[r], c[r]
            J[r, 1 + 2 * i] += sgn[r] * 0.5 * (br[1] - cr[1])
            J[r, 2 + 2 * i] += sgn[r] * 0.5 * (cr[0] - br[0])
            J[r, 1 + 2 * j] += sgn[r] * 0.5 * (cr[1] - ar[1])
            J[r, 2 + 2 * j] += sgn[r] * 0.5 * (ar[0] - cr[0])
            J[r, 1 + 2 * k] += sgn[r] * 0.5 * (ar[1] - br[1])
            J[r, 2 + 2 * k] += sgn[r] * 0.5 * (br[0] - ar[0])
        return J

    def cons_box(z):
        Q = z[1:].reshape(11, 2)
        return np.concatenate([
            Q[:, 0],
            1.0 - Q[:, 0],
            Q[:, 1],
            s * np.minimum(Q[:, 0], 1.0 - Q[:, 0]) - Q[:, 1],
        ])

    def cons_box_jac(z):
        Q = z[1:].reshape(11, 2)
        J = np.zeros((44, 23))
        for p in range(11):
            J[p, 1 + 2 * p] = 1.0
            J[11 + p, 1 + 2 * p] = -1.0
            J[22 + p, 2 + 2 * p] = 1.0
            w = 1.0 if Q[p, 0] <= 0.5 else -1.0
            J[33 + p, 1 + 2 * p] = s * w
            J[33 + p, 2 + 2 * p] = -1.0
        return J

    cons = [
        {"type": "ineq", "fun": cons_area, "jac": cons_area_jac},
        {"type": "ineq", "fun": cons_box, "jac": cons_box_jac},
    ]

    for r in range(restarts):
        if time.time() > deadline:
            break
        if r == 0:
            Pt = best_P
        else:
            Pt = _project_inside(best_P + rng.normal(0.0, 0.01, best_P.shape))
        z0 = np.concatenate([[best_A], Pt.ravel()])
        try:
            res = minimize(obj, z0, jac=obj_jac, method="SLSQP",
                           constraints=cons,
                           options={"maxiter": 400, "ftol": 1e-14})
            Q = _project_inside(res.x[1:].reshape(11, 2))
            A = float(_areas_all(Q).min())
            if A > best_A + 1e-15:
                best_A, best_P = A, Q
        except Exception:
            continue
    return best_P, best_A


def _grad_step(P, A, h, deadline):
    """Direct gradient ascent on the bottleneck triplet's area.

    Computes the analytic gradient of the (signed) bottleneck triangle
    area with respect to its three vertices, tries a small
    backtracking/advancing line search along the ascent direction
    (projected back into the triangle), and accepts any strict
    improvement of the global min area. Returns (P, A, success).
    """
    ar = _areas_all(P)
    t = int(np.argmin(ar))
    i, j, k = _IDX[t]
    S = _signed_area(P, i, j, k)
    if abs(S) < 1e-15:
        return P, A, False
    sgn = 1.0 if S > 0 else -1.0
    a, b, c = P[i], P[j], P[k]
    ga = sgn * 0.5 * np.array([b[1] - c[1], c[0] - b[0]])
    gb = sgn * 0.5 * np.array([c[1] - a[1], a[0] - c[0]])
    gc = sgn * 0.5 * np.array([a[1] - b[1], b[0] - a[0]])
    for hh in (h, 2 * h, 4 * h, 0.5 * h):
        Q = P.copy()
        Q[i] += hh * ga
        Q[j] += hh * gb
        Q[k] += hh * gc
        Q = _project_inside(Q)
        A2 = float(_areas_all(Q).min())
        if A2 > A + 1e-15:
            return Q, A2, True
    return P, A, False





def _polish(P, deadline, rounds=4):
    """SLSQP polish on a soft-min (log-sum-exp) surrogate of all 165 areas.

    The soft-min temperature tau is annealed across rounds so early rounds
    pull many near-bottleneck triplets up together and later rounds focus
    on the true bottleneck. A quadratic barrier keeps points inside the
    triangle; the result is projected exactly and kept only if the exact
    minimum area strictly improves. Returns (P, min_area).
    """
    s = np.sqrt(3.0)
    best_P = _project_inside(np.asarray(P, dtype=float))
    best_A = float(_areas_all(best_P).min())
    for r in range(rounds):
        if time.time() > deadline:
            break
        tau = max(1e-5, best_A * 0.08 / (r + 1))

        def f(z):
            Q = z.reshape(11, 2)
            ar = _areas_all(Q)
            softmin = -tau * np.log(np.sum(np.exp(-ar / tau)))
            pen = (np.sum(np.maximum(Q[:, 0] - 1.0, 0.0) ** 2)
                   + np.sum(np.maximum(-Q[:, 0], 0.0) ** 2)
                   + np.sum(np.maximum(-Q[:, 1], 0.0) ** 2)
                   + np.sum(np.maximum(
                       Q[:, 1] - s * np.minimum(Q[:, 0], 1.0 - Q[:, 0]),
                       0.0) ** 2))
            return -softmin + 1e4 * pen

        try:
            res = minimize(f, best_P.ravel(), method="SLSQP",
                           options={"maxiter": 300, "ftol": 1e-14})
            Q = _project_inside(res.x.reshape(11, 2))
            A = float(_areas_all(Q).min())
            if A > best_A + 1e-15:
                best_A, best_P = A, Q
        except Exception:
            break
    return best_P, best_A


def _areas_batch(PB):
    """Vectorized min-area over a batch of B configurations (B,11,2)."""
    a = PB[:, _IDX[:, 0], :]
    b = PB[:, _IDX[:, 1], :]
    c = PB[:, _IDX[:, 2], :]
    ar = 0.5 * np.abs((b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1])
                      - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0]))
    return ar.min(axis=1)


def _project_inside(P):
    """Project points back into the unit equilateral triangle."""
    Q = np.array(P, dtype=float, copy=True)
    Q[..., 0] = np.clip(Q[..., 0], 0.0, 1.0)
    Q[..., 1] = np.clip(Q[..., 1], 0.0,
                        np.sqrt(3) * np.minimum(Q[..., 0], 1.0 - Q[..., 0]))
    return Q


def _boundary_split(k0, k1, k2, jitter=0.0, seed=0):
    """k0/k1/k2 points arclength-uniform on the three edges (vertices shared).

    Boundary-dominant configurations are known to be strong for the
    Heilbronn problem; splitting points across edges with uniform arclength
    spacing gives high-quality structured seeds. Optional small jitter
    (fixed seed) diversifies the basin.
    """
    s = _S3
    rng = np.random.default_rng(seed)
    pts = []
    # edge 1: (0,0)->(1,0)
    for j in range(k0):
        t = (j + 0.5) / k0
        pts.append((t, 0.0))
    # edge 2: (1,0)->(0.5,s)
    for j in range(k1):
        t = (j + 0.5) / k1
        pts.append((1.0 - 0.5 * t, s * t))
    # edge 3: (0.5,s)->(0,0)
    for j in range(k2):
        t = (j + 0.5) / k2
        pts.append((0.5 * (1.0 - t), s * (1.0 - t)))
    P = np.array(pts, dtype=float)
    if jitter > 0:
        P = P + rng.normal(0.0, jitter, P.shape)
    return _project_inside(P)


def _initial_configs():
    """Deterministic structured seeds: lattices, boundary ring, boundary splits, jittered."""
    s = _S3
    cfgs = []
    # Boundary-split seeds: points distributed across the three edges.
    # Known-good Heilbronn structure for n=11 uses mostly boundary points.
    for (k0, k1, k2) in [(4, 4, 3), (4, 3, 4), (3, 4, 4), (5, 3, 3),
                         (3, 5, 3), (3, 3, 5), (6, 3, 2), (2, 6, 3)]:
        cfgs.append(_boundary_split(k0, k1, k2))
    # Jittered boundary splits to diversify basins near the structured optima
    for sd, (k0, k1, k2) in enumerate([(4, 4, 3), (3, 4, 4), (5, 3, 3)]):
        for _ in range(2):
            cfgs.append(_boundary_split(k0, k1, k2, jitter=0.01, seed=100 + sd))
    # Triangular lattice rows (rows of 1,2,3,... points), several densities
    for denom in (4, 5, 6):
        P = []
        for r in range(denom + 1):
            y = s * r / denom
            m = r + 1
            for j in range(m):
                x = 0.5 + (j - (m - 1) / 2.0) / denom
                P.append((x, y))
        cfgs.append(_project_inside(np.array(P[:11])))
    # Boundary ring: 11 points arclength-uniform around perimeter
    ts = np.linspace(0.0, 3.0, 12)[:-1]
    pts = []
    for t in ts:
        if t < 1.0:
            pts.append((t, 0.0))
        elif t < 2.0:
            u = t - 1.0
            pts.append((1.0 - 0.5 * u, s * u))
        else:
            u = t - 2.0
            pts.append((0.5 * (1.0 - u), s * (1.0 - u)))
    cfgs.append(np.array(pts))
    # Jittered random, fixed seed (more restarts for wider exploration)
    rng = np.random.default_rng(12345)
    for _ in range(8):
        x = rng.random(11)
        y = rng.random(11) * s
        cfgs.append(_project_inside(np.stack([x, y], axis=1)))
    # Slightly-perturbed lattice seeds (good basins near structured optima)
    rng2 = np.random.default_rng(999)
    for base in list(cfgs[:3]):
        for _ in range(3):
            cfgs.append(_project_inside(base + rng2.normal(0.0, 0.02, base.shape)))
    return cfgs


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct 11 points inside the unit equilateral triangle maximizing the
    smallest triangle area.

    Approach: multi-start local search combining (a) exact gradient ascent
    on the bottleneck triplet's area (analytic gradient + line search over
    the three bottleneck vertices, projected into the triangle) and (b) a
    vectorized random-batch fallback that jitters the two worst triplets to
    escape constrained/degenerate local optima. Fully deterministic
    (fixed seeds), time-budgeted, result cached across calls.
    """
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    n = 11
    rng = np.random.default_rng(20240711)
    deadline = time.time() + 280.0

    best_P, best_A = None, -1.0
    cands_all = []

    for P0 in _initial_configs():
        if time.time() > deadline:
            break
        P = _project_inside(np.asarray(P0, dtype=float)[:n])
        A = _areas_all(P).min()
        h = 0.05          # gradient line-search scale
        step = 0.05       # random-batch scale
        stall = 0
        while time.time() < deadline:
            # Phase 1: gradient ascent on the bottleneck (cheap, directed)
            P, A, ok = _grad_step(P, A, h, deadline)
            if ok:
                h = min(h * 1.5, 0.2)
                stall = 0
                continue
            # Phase 2: random batch on two worst triplets
            ar = _areas_all(P)
            order = np.argsort(ar)
            tri = _IDX[int(order[0])]
            tri2 = _IDX[int(order[1])]
            B = 256
            cands = np.repeat(P[None, :, :], B, axis=0)
            noise = rng.normal(0.0, step, (B, 3, 2))
            cands[:, tri[0], :] += noise[:, 0, :]
            cands[:, tri[1], :] += noise[:, 1, :]
            cands[:, tri[2], :] += noise[:, 2, :]
            half = B // 2
            noise2 = rng.normal(0.0, step, (half, 3, 2))
            cands[:half, tri2[0], :] += noise2[:, 0, :]
            cands[:half, tri2[1], :] += noise2[:, 1, :]
            cands[:half, tri2[2], :] += noise2[:, 2, :]
            if stall > 3:  # occasional full perturbation to escape
                cands += rng.normal(0.0, step * 0.5, (B, n, 2))
            cands = _project_inside(cands.reshape(-1, 2)).reshape(B, n, 2)
            vals = _areas_batch(cands)
            j = int(np.argmax(vals))
            # accept strict improvements; ties too (plateau drift helps lift)
            if vals[j] > A + 1e-15 or (vals[j] > A - 1e-13 and stall < 3):
                P, A = cands[j], max(float(vals[j]), A)
                stall = 0
            else:
                stall += 1
                step *= 0.7
                h *= 0.5
                if step < 1e-7 and h < 1e-9:
                    break                # converged for this start
                if stall > 6:
                    step = 0.02
                    h = 0.02
                    stall = 0
        if A > best_A:
            best_A, best_P = A, P
        cands_all.append((A, P))

    # Final refinement: exact max-min lift on the top-k end configurations,
    # followed by the soft-min polish as a secondary pass.
    try:
        cands_all.sort(key=lambda t: -t[0])
        for A0, P0 in cands_all[:8]:
            if time.time() > deadline:
                break
            Qp, Ap = _lift(P0, deadline)
            if Ap > best_A:
                best_A, best_P = Ap, Qp
            Qp, Ap = _polish(Qp, deadline, rounds=8)
            if Ap > best_A:
                best_A, best_P = Ap, Qp
    except Exception:
        pass

    # Basin hopping with reheating: perturb the best configuration with a
    # cycling schedule of noise scales, re-optimize with the fast local
    # search plus the exact max-min lift, and keep strict improvements.
    # On success, reheat from the smallest scale (fine tuning); on
    # failure, escalate the scale (explore a different basin).
    try:
        rng3 = np.random.default_rng(31337)
        scales = [0.004, 0.01, 0.02, 0.04, 0.08]
        si = 0
        while time.time() < deadline - 3.0:
            hop_end = min(deadline, time.time() + 15.0)
            Q = _project_inside(
                best_P + rng3.normal(0.0, scales[si], best_P.shape))
            A = float(_areas_all(Q).min())
            h = 0.02
            step = 0.02
            stall = 0
            while time.time() < hop_end and step > 1e-6:
                Q, A, ok = _grad_step(Q, A, h, deadline)
                if ok:
                    h = min(h * 1.5, 0.2)
                    stall = 0
                    continue
                ar = _areas_all(Q)
                tri = _IDX[int(np.argmin(ar))]
                B = 128
                cands = np.repeat(Q[None, :, :], B, axis=0)
                noise = rng3.normal(0.0, step, (B, 3, 2))
                cands[:, tri[0], :] += noise[:, 0, :]
                cands[:, tri[1], :] += noise[:, 1, :]
                cands[:, tri[2], :] += noise[:, 2, :]
                cands = _project_inside(cands.reshape(-1, 2)).reshape(B, 11, 2)
                vals = _areas_batch(cands)
                j = int(np.argmax(vals))
                if vals[j] > A + 1e-15:
                    Q, A = cands[j], float(vals[j])
                    stall = 0
                else:
                    stall += 1
                    step *= 0.7
                    h *= 0.5
                    if stall > 6:
                        step = 0.02
                        h = 0.02
                        stall = 0
            Qp, Ap = _lift(Q, deadline, restarts=2)
            if Ap > A:
                Q, A = Qp, Ap
            Qp, Ap = _polish(Q, deadline, rounds=3)
            if Ap > A:
                Q, A = Qp, Ap
            if A > best_A + 1e-15:
                best_A, best_P = A, Q
                si = 0          # reheat: fine-tune the new best
            else:
                si = (si + 1) % len(scales)
    except Exception:
        pass

    if best_P is None:
        # Fallback: simple triangular-lattice configuration (always valid)
        P = []
        for r in range(5):
            y = _S3 * r / 4.0
            for j in range(r + 1):
                P.append((0.5 + (j - r / 2.0) / 4.0, y))
        best_P = _project_inside(np.array(P[:n]))

    _CACHE = best_P
    return best_P


# EVOLVE-BLOCK-END
