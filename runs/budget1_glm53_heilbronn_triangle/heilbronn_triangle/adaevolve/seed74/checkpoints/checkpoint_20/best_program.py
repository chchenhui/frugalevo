# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations

_N = 11
_S3 = np.sqrt(3) / 2.0
_V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, _S3]])
_TRI_IDX = np.array(list(combinations(range(_N), 3)))
_REGION_AREA = np.sqrt(3) / 4.0


def _min_area(P):
    """Vectorized smallest normalized triangle area over all triplets."""
    T = P[_TRI_IDX]
    a2 = np.abs((T[:, 1, 0] - T[:, 0, 0]) * (T[:, 2, 1] - T[:, 0, 1])
               - (T[:, 2, 0] - T[:, 0, 0]) * (T[:, 1, 1] - T[:, 0, 1]))
    i = int(a2.argmin())
    return 0.5 * a2[i] / _REGION_AREA, _TRI_IDX[i]


def _clip(P):
    """Barycentric clamp all points into the triangle."""
    A, B, C = _V
    v0, v1 = B - A, C - A
    d = v0[0] * v1[1] - v0[1] * v1[0]
    w = np.empty((len(P), 3))
    w[:, 0] = ((P[:, 0] - A[0]) * v1[1] - (P[:, 1] - A[1]) * v1[0]) / d
    w[:, 1] = (v0[0] * (P[:, 1] - A[1]) - v0[1] * (P[:, 0] - A[0])) / d
    w[:, 2] = 1.0 - w[:, 0] - w[:, 1]
    w = np.maximum(w, 0.0)
    w /= w.sum(axis=1, keepdims=True)
    return w[:, :1] * A + w[:, 1:2] * B + w[:, 2:3] * C


def _areas(P):
    """Vectorized double-areas (unnormalized) for all triplets."""
    T = P[_TRI_IDX]
    return np.abs((T[:, 1, 0] - T[:, 0, 0]) * (T[:, 2, 1] - T[:, 0, 1])
                  - (T[:, 2, 0] - T[:, 0, 0]) * (T[:, 1, 1] - T[:, 0, 1]))


def _soft_score(a2, k=6):
    """Soft-min: mean of the k smallest double-areas (smooth guidance)."""
    return float(np.partition(a2, k)[:k].mean())


def _greedy_polish(P, step0=0.012, rounds=40):
    """Deterministic coordinate descent on the hard objective: for each
    point, try moves in 12 directions at three magnitudes; accept only
    strict improvements of the true minimum area; shrink step on stall."""
    P = _clip(P.copy())
    best = _min_area(P)[0]
    ang = np.linspace(0.0, 2.0 * np.pi, 13)[:-1]
    dirs = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    step = step0
    for _ in range(rounds):
        improved = False
        for j in range(_N):
            for mag in (1.0, 0.4, 0.15):
                for d in dirs:
                    Q = P.copy()
                    Q[j] += step * mag * d
                    Q = _clip(Q)
                    a = _min_area(Q)[0]
                    if a > best + 1e-12:
                        best, P, improved = a, Q, True
        if not improved:
            step *= 0.5
            if step < 1e-5:
                break
    return P, _min_area(P)[0]


def _targeted_polish(P, rounds=200, step0=0.01):
    """Deterministic gradient-directed polish: repeatedly find the current
    worst triangle and try moving each of its vertices along the
    perpendicular to the opposite side (the exact gradient direction for
    increasing that triangle's area), at several magnitudes including a
    small negative one (sometimes shrinking frees room for others)."""
    P = _clip(P.copy())
    best = _min_area(P)[0]
    step = step0
    for _ in range(rounds):
        a2 = _areas(P)
        tri = _TRI_IDX[int(a2.argmin())]
        T = P[tri]
        cen = T.mean(axis=0)
        improved = False
        for k in range(3):
            o1, o2 = tri[(k + 1) % 3], tri[(k + 2) % 3]
            seg = P[o2] - P[o1]
            nrm = np.hypot(seg[0], seg[1])
            if nrm < 1e-12:
                continue
            nvec = np.array([-seg[1], seg[0]]) / nrm
            if np.dot(nvec, P[tri[k]] - cen) < 0.0:
                nvec = -nvec
            for mag in (1.0, 0.5, 0.25, -0.5):
                Q = P.copy()
                Q[tri[k]] += nvec * step * mag
                Q = _clip(Q)
                a = _min_area(Q)[0]
                if a > best + 1e-12:
                    best, P, improved = a, Q, True
        if not improved:
            step *= 0.6
            if step < 1e-5:
                break
    return P, _min_area(P)[0]


def _optimize(P, rng, iters, step0):
    """Annealed local search attacking near-worst triangles, accepted on a
    soft-min criterion over the 6 worst triangles (smoother hill-climbing
    than the hard minimum, which tends to cycle on a single bottleneck)."""
    P = _clip(P.copy())
    a2 = _areas(P)
    best_a, worst = _min_area(P)
    best_P = P.copy()
    step = step0
    for it in range(iters):
        Q = P.copy()
        r = rng.random()
        if r < 0.20:
            j = int(rng.integers(0, _N))
            Q[j] += rng.normal(0.0, step, size=2)
        else:
            # attack one of the 6 currently worst triangles
            cand = np.argsort(a2)[:6]
            tri = _TRI_IDX[int(rng.choice(cand))]
            T = P[tri]
            if r < 0.55:
                # perpendicular lift: move one vertex of the attacked
                # triangle along the normal to the opposite side — the
                # exact gradient direction for increasing its area.
                k = int(rng.integers(0, 3))
                o1, o2 = tri[(k + 1) % 3], tri[(k + 2) % 3]
                seg = P[o2] - P[o1]
                nrm = np.hypot(seg[0], seg[1])
                if nrm > 1e-12:
                    nvec = np.array([-seg[1], seg[0]]) / nrm
                    cen = T.mean(axis=0)
                    if np.dot(nvec, P[tri[k]] - cen) < 0.0:
                        nvec = -nvec
                    Q[tri[k]] += nvec * step * rng.uniform(0.5, 2.0)
                else:
                    Q[tri[k]] += rng.normal(0.0, step, size=2)
            else:
                # radial push: expand the attacked triangle from its
                # centroid
                cen = T.mean(axis=0)
                for k in range(3):
                    d = T[k] - cen
                    nrm = np.linalg.norm(d)
                    if nrm > 1e-12:
                        Q[tri[k]] += (d / nrm) * step * rng.uniform(0.5, 1.5)
                    else:
                        Q[tri[k]] += rng.normal(0.0, step, size=2)
        Q = _clip(Q)
        a2q = _areas(Q)
        a, w = _min_area(Q)
        # accept on soft-min improvement (with rare uphill moves)
        if _soft_score(a2q) >= _soft_score(a2) or rng.random() < 0.01:
            P, a2 = Q, a2q
            if a > best_a:
                best_a, best_P = a, Q.copy()
        if it % 500 == 499:
            step *= 0.85
            P = best_P.copy()
            a2 = _areas(P)
            best_a, worst = _min_area(P)
    return best_P, _min_area(best_P)[0]


def heilbronn_triangle11() -> np.ndarray:
    """
    Arrange 11 points in the unit equilateral triangle to maximize the
    minimum triangle area over all triplets.

    Approach: deterministic multi-start annealed local search. Several
    structured seeds (vertices + edge points + interior points, with
    seeded jitter) are optimized by perturbing the points of the current
    worst triangle; the best result is returned. Falls back to a fixed
    structured configuration on any failure.
    """
    try:
        rng = np.random.default_rng(20240607)
        seeds = []
        # Seed 1: 3 vertices, 2 per edge, 2 interior
        p = [v.copy() for v in _V]
        for t in (1.0 / 3.0, 2.0 / 3.0):
            p.append(_V[0] + t * (_V[1] - _V[0]))
            p.append(_V[1] + t * (_V[2] - _V[1]))
            p.append(_V[2] + t * (_V[0] - _V[2]))
        p.append(np.array([0.5, _S3 / 3.0]))
        p.append(np.array([0.5, 2.0 * _S3 / 3.0]))
        seeds.append(np.array(p))
        # Seed 2: jittered seed 1
        seeds.append(_clip(seeds[0] + rng.normal(0.0, 0.02, size=(_N, 2))))
        # Seed 3: uniform random
        w = rng.dirichlet(np.ones(3), size=_N)
        seeds.append(w[:, :1] * _V[0] + w[:, 1:2] * _V[1] + w[:, 2:3] * _V[2])
        # Seeds 4-5: more jitter scales for diversity
        seeds.append(_clip(seeds[0] + rng.normal(0.0, 0.05, size=(_N, 2))))
        seeds.append(_clip(seeds[0] + rng.normal(0.0, 0.10, size=(_N, 2))))
        # Seed 6: edge-ring — all 11 points on the boundary at staggered
        # arc-length positions (boundary-heavy layouts often beat interior
        # ones for Heilbronn-type objectives).
        ts = (np.arange(_N) + 0.5) / _N
        pts = []
        for t in ts:
            u = 3.0 * t
            e = int(u)
            f = u - e
            pts.append(_V[e % 3] + f * (_V[(e + 1) % 3] - _V[e % 3]))
        seeds.append(np.array(pts))
        # Seed 7: another uniform random draw
        w = rng.dirichlet(np.ones(3), size=_N)
        seeds.append(w[:, :1] * _V[0] + w[:, 1:2] * _V[1] + w[:, 2:3] * _V[2])

        best_P, best_a = None, -1.0
        for si, S in enumerate(seeds):
            R = np.random.default_rng(1000 + si)
            P, a = _optimize(S, R, 5000, 0.04)
            if a > best_a:
                best_a, best_P = a, P
        # Multi-stage polish from best: annealed (medium then fine steps)
        # followed by a deterministic greedy hill-climb on the hard
        # objective after each anneal stage.
        for st, sd in ((0.01, 2001), (0.003, 3001)):
            R = np.random.default_rng(sd)
            P, a = _optimize(best_P, R, 5000, st)
            if a > best_a:
                best_a, best_P = a, P
            P, a = _targeted_polish(best_P)
            if a > best_a:
                best_a, best_P = a, P
            P, a = _greedy_polish(best_P)
            if a > best_a:
                best_a, best_P = a, P
        # Final micro-polish: targeted gradient polish then tiny-step greedy
        P, a = _targeted_polish(best_P, step0=0.004)
        if a > best_a:
            best_a, best_P = a, P
        P, a = _greedy_polish(best_P, step0=0.004, rounds=30)
        if a > best_a:
            best_a, best_P = a, P
        return np.asarray(best_P, dtype=float)
    except Exception:
        # Deterministic fallback: vertices + edge points + interior
        p = [v.copy() for v in _V]
        for t in (1.0 / 3.0, 2.0 / 3.0):
            p.append(_V[0] + t * (_V[1] - _V[0]))
            p.append(_V[1] + t * (_V[2] - _V[1]))
            p.append(_V[2] + t * (_V[0] - _V[2]))
        p.append(np.array([0.5, _S3 / 3.0]))
        p.append(np.array([0.5, 2.0 * _S3 / 3.0]))
        return np.array(p)


# EVOLVE-BLOCK-END
