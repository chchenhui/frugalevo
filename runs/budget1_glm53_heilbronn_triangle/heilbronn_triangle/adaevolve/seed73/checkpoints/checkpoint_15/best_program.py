# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

try:
    from scipy.optimize import minimize as _scipy_minimize
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

_V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]])
_H = np.sqrt(3.0) / 2.0
_N = 11
_IDX = np.array(list(combinations(range(_N), 3)), dtype=np.intp)
_II, _JJ, _KK = _IDX[:, 0], _IDX[:, 1], _IDX[:, 2]


def _areas(P):
    """Areas of all C(11,3)=165 triangles formed by rows of P."""
    pa, pb, pc = P[_II], P[_JJ], P[_KK]
    return 0.5 * np.abs((pb[:, 0] - pa[:, 0]) * (pc[:, 1] - pa[:, 1]) -
                        (pb[:, 1] - pa[:, 1]) * (pc[:, 0] - pa[:, 0]))


def _neg_softmin_grad(z_flat, t):
    """-(softmin of areas) and its analytic gradient wrt unconstrained
    barycentric logits z (rows softmaxed into the triangle)."""
    z = z_flat.reshape(_N, 3)
    e = np.exp(z - z.max(axis=1, keepdims=True))
    B = e / e.sum(axis=1, keepdims=True)
    P = B @ _V
    pa, pb, pc = P[_II], P[_JJ], P[_KK]
    s = 0.5 * ((pb[:, 0] - pa[:, 0]) * (pc[:, 1] - pa[:, 1]) -
               (pb[:, 1] - pa[:, 1]) * (pc[:, 0] - pa[:, 0]))
    a = np.abs(s)
    m = a.min()
    w = np.exp(-(a - m) / t)
    S = w.sum()
    F = m - t * np.log(S)          # smooth lower bound on the min area
    g = (w / S) * np.sign(s)       # dF/ds per triplet
    X, Y = P[:, 0], P[:, 1]
    grad = np.zeros((_N, 2))
    gi = np.stack([0.5 * (Y[_JJ] - Y[_KK]), 0.5 * (X[_KK] - X[_JJ])], axis=1)
    gj = np.stack([0.5 * (Y[_KK] - Y[_II]), 0.5 * (X[_II] - X[_KK])], axis=1)
    gk = np.stack([0.5 * (Y[_II] - Y[_JJ]), 0.5 * (X[_JJ] - X[_II])], axis=1)
    np.add.at(grad, _II, g[:, None] * gi)
    np.add.at(grad, _JJ, g[:, None] * gj)
    np.add.at(grad, _KK, g[:, None] * gk)
    gradB = grad @ _V.T            # chain: P = B @ V
    dot = (gradB * B).sum(axis=1, keepdims=True)
    gradz = B * (gradB - dot)       # chain: B = softmax(z)
    return -F, -gradz.ravel()


def _anneal(B0, schedule):
    """Gradient ascent on the softmin objective with decreasing temperature."""
    B0 = np.clip(B0, 1e-12, None)
    B0 = B0 / B0.sum(axis=1, keepdims=True)
    z = np.log(B0).ravel()
    for t in schedule:
        try:
            res = _scipy_minimize(_neg_softmin_grad, z, args=(t,), jac=True,
                                  method='L-BFGS-B',
                                  options={'maxiter': 600, 'maxfun': 4000,
                                           'ftol': 1e-14})
            if res.x is not None and np.all(np.isfinite(res.x)):
                z = res.x
        except Exception:
            break
    z = z.reshape(_N, 3)
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _xy_to_bary(p):
    b2 = p[1] / _H
    b1 = p[0] - 0.5 * b2
    return np.array([1.0 - b1 - b2, b1, b2])


def _polish(B, rng, deadline):
    """Exact min-area hill climbing with geometric 'escape' moves perpendicular
    to the lines of the currently-tight (smallest) triangles, edge projections,
    barycentric axis moves and random moves; step shrinks on plateaus."""
    P = B @ _V
    cur = _areas(P).min()
    step = 0.02
    while step > 1e-6 and time.time() < deadline:
        improved = False
        areas = _areas(P)
        thr = cur * 1.4 + 1e-12
        hot = np.zeros(_N, dtype=bool)
        hot[_IDX[areas <= thr].ravel()] = True
        order = list(np.where(hot)[0]) + list(np.where(~hot)[0])
        tight = _IDX[areas <= thr][:12]
        for pi in order:
            cands = []
            for ax in range(3):
                for sgn in (-1.0, 1.0):
                    c = B[pi].copy()
                    c[ax] += sgn * step
                    c = np.clip(c, 0.0, None)
                    s = c.sum()
                    if s > 1e-12:
                        cands.append(c / s)
            for ax in range(3):
                c = B[pi].copy()
                c[ax] = 0.0
                s = c.sum()
                if s > 1e-12:
                    cands.append(c / s)
            for tri in tight:
                if pi in tri:
                    j, k = [q for q in tri if q != pi]
                    d = P[k] - P[j]
                    nrm = np.array([-d[1], d[0]])
                    ln = np.linalg.norm(nrm)
                    if ln < 1e-12:
                        continue
                    nrm /= ln
                    cr = d[0] * (P[pi, 1] - P[j, 1]) - d[1] * (P[pi, 0] - P[j, 0])
                    direction = nrm if cr >= 0 else -nrm
                    for mag in (step, 3.0 * step):
                        c = _xy_to_bary(P[pi] + mag * direction)
                        c = np.clip(c, 0.0, None)
                        s = c.sum()
                        if s > 1e-12:
                            cands.append(c / s)
            for _ in range(6):
                c = np.clip(B[pi] + rng.normal(size=3) * step * 0.7, 0.0, None)
                s = c.sum()
                if s > 1e-12:
                    cands.append(c / s)
            for c in cands:
                trialB = B.copy()
                trialB[pi] = c
                trialP = trialB @ _V
                v = _areas(trialP).min()
                if v > cur + 1e-14:
                    B, P, cur = trialB, trialP, v
                    improved = True
        if not improved:
            step *= 0.4
    return B, cur


def _fallback():
    return np.array([
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        [0.66, 0.34, 0.0], [0.33, 0.67, 0.0],
        [0.0, 0.66, 0.34], [0.0, 0.33, 0.67],
        [0.34, 0.0, 0.66], [0.67, 0.0, 0.33],
        [0.45, 0.27, 0.28], [0.25, 0.45, 0.30],
    ])


def _make_starts(rng):
    starts = [_fallback()]
    starts.append(np.array([
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        [0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5],
        [0.5, 0.25, 0.25], [0.25, 0.5, 0.25], [0.25, 0.25, 0.5],
        [0.6, 0.2, 0.2], [0.2, 0.6, 0.2]]))
    for f in (0.3, 0.7):
        starts.append(np.array([
            [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
            [1 - f, f, 0.0], [f, 1 - f, 0.0],
            [0.0, 1 - f, f], [0.0, f, 1 - f],
            [f, 0.0, 1 - f], [1 - f, 0.0, f],
            [0.45, 0.3, 0.25], [0.25, 0.45, 0.3]]))
    # Systematic edge-lattice starts: 3 vertices + all ways to split the
    # remaining 8 points across the three edges at evenly spaced fractions.
    # Good n=11 Heilbronn configs are boundary-heavy and near-regular, so
    # these give the annealer many high-quality basins to descend from.
    def _edge_start(split):
        rows = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        for e, cnt in enumerate(split):
            for j in range(1, cnt + 1):
                f = j / (cnt + 1)
                c = [0.0, 0.0, 0.0]
                c[e] = 1.0 - f
                c[(e + 1) % 3] = f
                rows.append(c)
        return np.array(rows)

    for a in range(0, 9):
        for b in range(0, 9 - a):
            c = 8 - a - b
            starts.append(_edge_start((a, b, c)))
    # near-regular boundary configs with one interior point
    for e in range(3):
        rows = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        for ee, cnt in enumerate(((3, 3, 2) if e == 0 else
                                  (3, 2, 3) if e == 1 else (2, 3, 3))):
            for j in range(1, cnt + 1):
                f = j / (cnt + 1)
                cc = [0.0, 0.0, 0.0]
                cc[ee] = 1.0 - f
                cc[(ee + 1) % 3] = f
                rows.append(cc)
        rows.append([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
        starts.append(np.array(rows))
    for _ in range(10):
        starts.append(rng.dirichlet(np.ones(3) * 1.2, size=_N))
    for _ in range(6):
        starts.append(rng.dirichlet(np.ones(3) * 0.5, size=_N))
    # boundary-heavy starts: many points on edges (good Heilbronn configs
    # for n=11 tend to place most points on the boundary)
    for _ in range(6):
        Bs = rng.dirichlet(np.ones(3) * 0.15, size=_N)
        starts.append(Bs)
    return starts


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 11.

    Approach: points are parameterized by unconstrained barycentric logits
    (softmax), which guarantees containment in the equilateral triangle.
    (1) Multi-start: deterministic boundary/lattice configurations plus seeded
        Dirichlet samples.
    (2) Annealing: L-BFGS-B on a smooth log-sum-exp lower bound of the minimum
        triangle area, using its exact analytic gradient, with a decreasing
        temperature schedule (softmin -> hard min).
    (3) Exact polish: hill climbing on the true minimum area, with moves
        perpendicular to the lines of currently-tight triangles (the exact
        direction that grows a small triangle), edge projections, barycentric
        axis moves, and random moves, with a shrinking step.
    (4) A final fine-grained re-anneal + re-polish of the best configuration.
    Fully deterministic (fixed seed); time-budgeted; falls back to a fixed
    valid configuration on any failure.
    """
    t0 = time.time()
    deadline = t0 + 130.0
    rng = np.random.default_rng(20250611)

    try:
        starts = _make_starts(rng)
        results = []
        if _HAVE_SCIPY:
            schedule = [0.02, 0.005, 0.001, 2e-4, 5e-5]
            for B0 in starts:
                if time.time() > t0 + 75.0:
                    break
                try:
                    B = _anneal(B0, schedule)
                except Exception:
                    B = B0 / B0.sum(axis=1, keepdims=True)
                results.append((_areas(B @ _V).min(), B))
        else:
            for B0 in starts:
                results.append((_areas(B0 @ _V).min(), B0))
        results.sort(key=lambda r: -r[0])
        if not results:
            results = [(_areas(_fallback() @ _V).min(), _fallback())]

        best_val, best_B = results[0]
        # Polish top anneal results plus small jittered restarts around the
        # best few (cheap extra diversity to escape local optima).
        polish_list = [B for _, B in results[:12]]
        for _, B in results[:5]:
            for scale in (0.005, 0.02, 0.05):
                Bj = np.clip(B + rng.normal(size=B.shape) * scale, 1e-9, None)
                polish_list.append(Bj / Bj.sum(axis=1, keepdims=True))
        for B in polish_list:
            if time.time() > deadline:
                break
            Bp, vp = _polish(B, rng, deadline)
            if vp > best_val:
                best_val, best_B = vp, Bp

        # Final refinement round: fine re-anneal + re-polish of the best.
        if _HAVE_SCIPY and time.time() < deadline - 10.0:
            fine = [1e-3, 2e-4, 5e-5, 1e-5, 2e-6]
            Bf = _anneal(best_B, fine)
            Bf, vf = _polish(Bf, rng, deadline)
            if vf > best_val:
                best_val, best_B = vf, Bf

        pts = best_B @ _V
        if not np.all(np.isfinite(pts)):
            raise ValueError("non-finite result")
    except Exception:
        pts = _fallback() @ _V

    # Numerical safety clamp into the triangle.
    pts = np.asarray(pts, dtype=float)
    pts[:, 0] = np.clip(pts[:, 0], 0.0, 1.0)
    pts[:, 1] = np.clip(pts[:, 1], 0.0, _H)
    ymax = _H * (1.0 - np.abs(pts[:, 0] - 0.5))
    pts[:, 1] = np.minimum(pts[:, 1], ymax)
    return pts


# EVOLVE-BLOCK-END
