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


def _optimize(P, rng, iters, step0):
    """Annealed local search: perturb worst-triangle points or random points."""
    P = _clip(P.copy())
    best_a, worst = _min_area(P)
    best_P = P.copy()
    step = step0
    for it in range(iters):
        Q = P.copy()
        if it % 4 == 3:
            j = int(rng.integers(0, _N))
            Q[j] += rng.normal(0.0, step, size=2)
        else:
            T = P[worst]
            cen = T.mean(axis=0)
            for k in range(3):
                d = T[k] - cen
                nrm = np.linalg.norm(d)
                if nrm > 1e-12:
                    Q[worst[k]] += (d / nrm) * step * rng.uniform(0.5, 1.5)
                else:
                    Q[worst[k]] += rng.normal(0.0, step, size=2)
        Q = _clip(Q)
        a, w = _min_area(Q)
        if a >= best_a or rng.random() < 0.01:
            best_a, worst, P = a, w, Q
            if a >= _min_area(best_P)[0]:
                best_P = Q.copy()
        if it % 500 == 499:
            step *= 0.85
            P = best_P.copy()
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
        # Seed 4: another jittered variant
        seeds.append(_clip(seeds[0] + rng.normal(0.0, 0.05, size=(_N, 2))))

        best_P, best_a = None, -1.0
        for si, S in enumerate(seeds):
            R = np.random.default_rng(1000 + si)
            P, a = _optimize(S, R, 6000, 0.04)
            if a > best_a:
                best_a, best_P = a, P
        # Final polish from best
        R = np.random.default_rng(999)
        P, a = _optimize(best_P, R, 6000, 0.008)
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
