# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations

SQRT3 = np.sqrt(3.0)
VERTS = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, SQRT3 / 2.0]])


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct 11 points inside/on the equilateral triangle (0,0),(1,0),(0.5,sqrt(3)/2)
    maximizing the minimum triangle area over all C(11,3)=165 triplets.

    Approach (fully deterministic):
      1. Multiple fixed-seed random barycentric starts plus a jittered-grid start.
      2. Direct greedy polish on the EXACT hard min-area objective: repeatedly
         try perturbations of each point (8 directions), accept any move that
         strictly increases the true minimum area, with geometric step decay.
      3. Return the best configuration found across all starts.
    Robust: no nonlinear solver dependency; always returns a valid (11,2) array.
    """
    n = 11
    triples = np.array(list(combinations(range(n), 3)))
    t0 = triples[:, 0]
    t1 = triples[:, 1]
    t2 = triples[:, 2]
    rng = np.random.default_rng(20240517)

    def min_area(P):
        ax, ay = P[t0, 0], P[t0, 1]
        bx, by = P[t1, 0], P[t1, 1]
        cx, cy = P[t2, 0], P[t2, 1]
        ar = 0.5 * np.abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay))
        return ar.min()

    def bary_to_xy(b):
        return b @ VERTS

    def clip_inside(P):
        # project points back into the triangle via barycentric clipping
        x, y = P[:, 0], P[:, 1]
        l3 = np.clip(y / (SQRT3 / 2.0), 0.0, 1.0)
        l2 = np.clip(x - 0.5 * l3, 0.0, 1.0)
        l1 = np.clip(1.0 - l2 - l3, 0.0, 1.0)
        s = l1 + l2 + l3
        s[s == 0] = 1.0
        b = np.stack([l1 / s, l2 / s, l3 / s], axis=1)
        return bary_to_xy(b)

    def repulsion(P, iters=300, dt=0.015):
        """Deterministic repulsion relaxation: push points apart to spread
        the configuration into a good basin before exact greedy polish."""
        for t in range(iters):
            f = np.zeros_like(P)
            for i in range(n):
                d = P - P[i]
                r2 = (d ** 2).sum(axis=1) + 1e-9
                f[i] = (d / r2[:, None]).sum(axis=0) * 0.5
            P = clip_inside(P + dt * f)
        return P

    def softmin_grad(P, iters=250, beta=200.0, lr=0.002):
        """Soft-min gradient ascent on sum_k w_k * a_k with w_k = softmax(-beta*a_k).
        Exact analytic gradients; anneal beta up so the objective approaches the
        true hard minimum. Projected back into the triangle each step."""
        for it in range(iters):
            b = min(1.0 + 3.0 * it / iters, 4.0) * beta
            ax, ay = P[t0, 0], P[t0, 1]
            bx, by = P[t1, 0], P[t1, 1]
            cx, cy = P[t2, 0], P[t2, 1]
            s = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)
            w = np.exp(-b * (0.5 * np.abs(s) - 0.5 * np.abs(s).min()))
            w = w / w.sum()
            sg = np.sign(s)
            # gradient wrt signed area a = 0.5*s, then chain through |s|
            ga = 0.5 * sg * w  # per-triple weight on d(|s|)/2... effectively dA/ds
            G = np.zeros_like(P)
            np.add.at(G, t0, np.stack([-ga * (by - cy), -ga * (cx - bx)], axis=1))
            np.add.at(G, t1, np.stack([-ga * (cy - ay), -ga * (ax - cx)], axis=1))
            np.add.at(G, t2, np.stack([-ga * (ay - by), -ga * (bx - ax)], axis=1))
            P = clip_inside(P + lr * G / (np.abs(G).max() + 1e-12))
        return P

    def polish(P, iters=400):
        P = clip_inside(P.copy())
        cur = min_area(P)
        step = 0.02
        dirs = np.array([[1, 0], [-1, 0], [0, 1], [0, -1],
                         [1, 1], [1, -1], [-1, 1], [-1, -1]]) / np.sqrt(2)
        for it in range(iters):
            improved = False
            for i in range(n):
                best_v, best_d = cur, None
                for d in dirs * step:
                    Q = P.copy()
                    Q[i] += d
                    Q = clip_inside(Q)
                    v = min_area(Q)
                    if v > best_v + 1e-15:
                        best_v, best_d = v, Q
                if best_d is not None:
                    P, cur, improved = best_d, best_v, True
            if not improved:
                step *= 0.5
                if step < 1e-6:
                    break
        return P, cur

    starts = []
    # jittered grid-ish start
    g = []
    for i in range(4):
        for j in range(4 - i):
            l1 = i / 4.0
            l2 = j / 4.0
            l3 = 1.0 - l1 - l2
            if l3 >= -1e-12:
                g.append([l1, l2, max(l3, 0.0)])
    if len(g) >= n:
        g = np.array(g[:n])
        g = g / g.sum(axis=1, keepdims=True)
        starts.append(g)
    # finer triangular lattice starts at several scales/offsets
    for m in (3, 4, 5):
        for off in (0.0, 0.5 / m):
            pts = []
            for i in range(m + 1):
                for j in range(m + 1 - i):
                    l1 = (i + off) / m
                    l2 = (j + off) / m
                    l3 = 1.0 - l1 - l2
                    if l3 >= -1e-9:
                        pts.append([max(l1, 0.0), max(l2, 0.0), max(l3, 0.0)])
            if len(pts) >= n:
                b = np.array(pts[:n])
                b = b / b.sum(axis=1, keepdims=True)
                starts.append(b)
    # edge-concentrated starts (points near the three edges)
    for k in range(3):
        b = np.zeros((n, 3))
        b[:, k] = 0.05
        b[:, (k + 1) % 3] = np.linspace(0.05, 0.9, n)
        b[:, (k + 2) % 3] = 1.0 - b[:, k] - b[:, (k + 1) % 3]
        starts.append(np.clip(b, 0.0, None))
    # random starts
    for _ in range(8):
        raw = rng.random((n, 3)) + 0.1
        starts.append(raw / raw.sum(axis=1, keepdims=True))

    best_P, best_val = None, -1.0
    for b0 in starts:
        P0 = bary_to_xy(b0)
        # three-phase: repulsion-spread, soft-min gradient, exact greedy polish
        P, val = polish(softmin_grad(repulsion(P0)))
        if val > best_val:
            best_val, best_P = val, P
        # also polish the raw start (repulsion can overshoot)
        P, val = polish(P0)
        if val > best_val:
            best_val, best_P = val, P

    # final safety check
    if best_P is None or not np.all(np.isfinite(best_P)):
        best_P = np.array([[1/3 + 0.01*i, 0.2 + 0.02*(i % 3)] for i in range(n)])
        best_P = clip_inside(best_P)
    return best_P


# EVOLVE-BLOCK-END
