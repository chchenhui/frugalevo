# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations


def heilbronn_convex13() -> np.ndarray:
    """
    Deterministic multi-restart Metropolis simulated-annealing search for 13
    points inside a unit-area triangle, maximizing the minimum area over all
    C(13,3)=286 triples, followed by a multi-scale greedy polish.

    Approach:
      - Region: triangle (0,0),(1,0),(0,2) (area = 1); objective = min |cross|
        over all triples (twice area; scale-invariant).
      - Points projected back into triangle via barycentric clamping.
      - 3-fold symmetric and random inits; geometric cooling schedule.
      - Fixed seeds => fully reproducible; runs in ~1-2 minutes.
    """
    n = 13

    # Unit-area triangle vertices
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.0, 2.0])

    triples = np.array(list(combinations(range(n), 3)))  # (286, 3)

    def project_into_triangle(pts):
        """Clamp points back inside triangle ABC via barycentric coordinates."""
        v0 = B - A
        v1 = C - A
        v2 = pts - A
        d00 = v0 @ v0
        d01 = v0 @ v1
        d11 = v1 @ v1
        d20 = v2 @ v0
        d21 = v2 @ v1
        denom = d00 * d11 - d01 * d01
        b1 = (d11 * d20 - d01 * d21) / denom
        b2 = (d00 * d21 - d01 * d20) / denom
        b1 = np.clip(b1, 0.0, 1.0)
        b2 = np.clip(b2, 0.0, 1.0)
        s = b1 + b2
        over = s > 1.0
        b1[over] /= s[over]
        b2[over] /= s[over]
        return A + np.outer(b1, v0) + np.outer(b2, v1)

    def min_tri_area(pts):
        """Minimum (doubled) triangle area over all triples, vectorized."""
        p = pts[triples]
        a, b, c = p[:, 0], p[:, 1], p[:, 2]
        cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - \
                (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        return np.abs(cross).min()

    def symmetric_init(seed):
        """Centroid + 4 base points each with 120-degree rotational orbit."""
        r = np.random.default_rng(seed)
        G = (A + B + C) / 3.0
        base = []
        for _ in range(4):
            u, v = r.random(), r.random()
            if u + v > 0.9:
                u, v = 0.9 * u / (u + v), 0.9 * v / (u + v)
            base.append(A + u * (B - A) + v * (C - A))
        pts = [G.copy()]
        cs, sn = np.cos(2 * np.pi / 3), np.sin(2 * np.pi / 3)
        for P in base:
            Q = P.copy()
            for _ in range(3):
                pts.append(Q.copy())
                d = Q - G
                Q = G + np.array([cs * d[0] - sn * d[1], sn * d[0] + cs * d[1]])
        return project_into_triangle(np.array(pts[:n]))

    def random_init(seed):
        r = np.random.default_rng(seed)
        pts = []
        while len(pts) < n:
            u, v = r.random(), r.random()
            if u + v > 1.0:
                u, v = 1.0 - u, 1.0 - v
            pts.append(A + u * (B - A) + v * (C - A))
        return np.array(pts)

    def anneal(pts, seed, n_iter=60000):
        """Metropolis simulated annealing with geometric cooling."""
        r = np.random.default_rng(seed)
        pts = pts.copy()
        cur_val = min_tri_area(pts)
        best_pts, best_val = pts.copy(), cur_val
        T0, T1 = 0.02, 1e-5
        for it in range(n_iter):
            T = T0 * (T1 / T0) ** (it / n_iter)
            cand = pts.copy()
            m = int(r.integers(1, 4))
            idx = r.choice(n, size=m, replace=False)
            cand[idx] += r.normal(0.0, 0.02 + 0.15 * (1 - it / n_iter),
                                  size=(m, 2))
            cand = project_into_triangle(cand)
            val = min_tri_area(cand)
            if val >= cur_val or r.random() < np.exp(min(0.0, (val - cur_val) / max(T, 1e-12))):
                pts = cand
                cur_val = val
                if val > best_val:
                    best_val = val
                    best_pts = cand.copy()
        return best_pts, best_val

    # Multi-restart: symmetric and random seeds, keep the best.
    global_best_pts, global_best_val = None, -1.0
    seeds = [42, 7, 123, 2024, 99, 313, 555, 777, 11, 88, 404, 909]
    for si, sd in enumerate(seeds):
        init = symmetric_init(sd) if si % 2 == 0 else random_init(sd)
        bp, bv = anneal(init, sd + 1000)
        if bv > global_best_val:
            global_best_val = bv
            global_best_pts = bp

    best_pts, best_val = global_best_pts, global_best_val

    # --- Analytic gradient polish on the critical (minimal) triangle ---
    # The area of the minimal triangle is smooth in its 3 vertices; move them
    # uphill along the exact area gradient with backtracking line search.
    def tri_crosses(pts):
        p = pts[triples]
        a, b, c = p[:, 0], p[:, 1], p[:, 2]
        return (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - \
               (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])

    def gradient_polish(pts, max_iter=5000):
        pts = pts.copy()
        best = np.abs(tri_crosses(pts)).min()
        for _ in range(max_iter):
            cr = np.abs(tri_crosses(pts))
            i0, i1, i2 = triples[int(cr.argmin())]
            a, b, c = pts[i0], pts[i1], pts[i2]
            s = np.sign((b[0] - a[0]) * (c[1] - a[1]) -
                        (b[1] - a[1]) * (c[0] - a[0]))
            if s == 0:
                s = 1.0
            grads = {i0: s * np.array([b[1] - c[1], c[0] - b[0]]),
                     i1: s * np.array([c[1] - a[1], a[0] - c[0]]),
                     i2: s * np.array([a[1] - b[1], b[0] - a[0]])}
            gn = max(np.linalg.norm(g) for g in grads.values())
            if gn < 1e-14:
                break
            step, improved = 0.02, False
            while step > 1e-9:
                cand = pts.copy()
                for i, g in grads.items():
                    cand[i] = cand[i] + step * g / gn
                cand = project_into_triangle(cand)
                v = np.abs(tri_crosses(cand)).min()
                if v > best + 1e-14:
                    pts, best, improved = cand, v, True
                    break
                step *= 0.5
            if not improved:
                break
        return pts, best

    best_pts, best_val = gradient_polish(best_pts)

    # Multi-scale greedy polish on single-point moves.
    improved = True
    while improved:
        improved = False
        for i in range(n):
            for step in (0.006, 0.002, 0.0006, 0.0002):
                for dx, dy in ((step, 0), (-step, 0), (0, step), (0, -step),
                               (step, step), (-step, -step), (step, -step), (-step, step)):
                    cand = best_pts.copy()
                    cand[i] += np.array([dx, dy])
                    cand = project_into_triangle(cand)
                    val = min_tri_area(cand)
                    if val > best_val + 1e-12:
                        best_val = val
                        best_pts = cand
                        improved = True

    return best_pts


# EVOLVE-BLOCK-END
