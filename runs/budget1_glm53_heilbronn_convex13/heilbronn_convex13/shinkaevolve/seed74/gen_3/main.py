# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    rng = np.random.default_rng(seed=42)
    points = rng.random((n, 2))
    # ---- Deterministic refinement stage (Heilbronn local optimization) ----
    import numpy as _np

    _pts = _np.asarray(points, dtype=float)
    _n = _pts.shape[0]

    if _n == 13 and _pts.shape == (13, 2) and _np.all(_np.isfinite(_pts)):
        # Precompute all triangle index triples (C(13,3) = 286)
        _idx = _np.array([(i, j, k)
                           for i in range(_n)
                           for j in range(i + 1, _n)
                           for k in range(j + 1, _n)], dtype=int)

        def _hull_ids(P):
            """Monotone chain convex hull, returns hull vertex indices (CCW)."""
            order = _np.lexsort((P[:, 1], P[:, 0]))
            hull = []
            for chain in (order, order[::-1]):
                start = len(hull)
                for oi in chain:
                    while len(hull) - start >= 2:
                        o, a = P[hull[-2]], P[hull[-1]]
                        if (a[0] - o[0]) * (P[oi][1] - o[1]) - \
                           (a[1] - o[1]) * (P[oi][0] - o[0]) <= 1e-14:
                            hull.pop()
                        else:
                            break
                    hull.append(int(oi))
                hull.pop()
            return _np.array(hull[:len(hull)] if hull else [0])

        def _score(P):
            """Return (min triangle area / hull area, min index) or -inf."""
            try:
                a = P[_idx[:, 0]]
                b = P[_idx[:, 1]]
                c = P[_idx[:, 2]]
                areas = 0.5 * _np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) -
                                      (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
                h = P[_hull_ids(P)]
                if len(h) < 3:
                    return -_np.inf, 0
                A = 0.5 * abs(_np.sum(h[:, 0] * _np.roll(h[:, 1], -1) -
                                      h[:, 1] * _np.roll(h[:, 0], -1)))
                if A < 1e-16:
                    return -_np.inf, 0
                m = areas.min()
                return m / A, int(areas.argmin())
            except Exception:
                return -_np.inf, 0

        rng = _np.random.default_rng(42)
        best_P = _pts.copy()
        best_s, _ = _score(best_P)

        if _np.isfinite(best_s):
            cur_P = best_P.copy()
            cur_s = best_s
            # Multi-scale hill climbing with deterministic restarts
            rounds = [
                (0.30, 3000),   # coarse exploration
                (0.10, 3000),   # medium
                (0.03, 3000),   # fine
                (0.01, 3000),   # polish
                (0.003, 2000),  # final polish
            ]
            total = sum(c for _, c in rounds)
            done = 0
            for sigma, count in rounds:
                for _ in range(count):
                    done += 1
                    i = int(rng.integers(0, _n))
                    Q = cur_P.copy()
                    Q[i] = Q[i] + rng.normal(0.0, sigma, size=2) * 2.0
                    s, _ = _score(Q)
                    if s > cur_s:
                        cur_P, cur_s = Q, s
                        if s > best_s:
                            best_P, best_s = Q.copy(), s
                    # occasional deterministic restart around the incumbent
                    if done % 1500 == 0:
                        cur_P = best_P + rng.normal(0.0, 0.05, size=(_n, 2))
                        cur_s, _ = _score(cur_P)
                        if cur_s < best_s * 0.5:
                            cur_P, cur_s = best_P.copy(), best_s
                # re-anchor to best between scales
                if best_s > cur_s:
                    cur_P, cur_s = best_P.copy(), best_s

            points = best_P

    return points


# EVOLVE-BLOCK-END