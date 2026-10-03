# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations


def heilbronn_convex13() -> np.ndarray:
    """
    Deterministic simulated-annealing search for 13 points inside a unit-area
    triangle maximizing the minimum area over all C(13,3)=286 triangles.

    Approach:
      - Region: triangle with vertices (0,0), (1,0), (0,2) (area = 1).
      - Objective: min over all 286 triples of |cross product| (twice the area;
        scaling is irrelevant since we normalize by hull area).
      - Constraint: points projected back into the triangle via barycentric
        clamping after each perturbation.
      - Vectorized evaluation of all 286 triangles per candidate move.
      - Fixed seed => fully reproducible; ~a few seconds runtime.
    """
    n = 13
    rng = np.random.default_rng(seed=42)

    # Unit-area triangle vertices
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.0, 2.0])

    # Precompute all triangle index triples
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
        # If b1+b2 > 1, scale down toward vertex A
        s = b1 + b2
        over = s > 1.0
        b1[over] /= s[over]
        b2[over] /= s[over]
        return A + np.outer(b1, v0) + np.outer(b2, v1)

    def tri_areas(pts):
        """All 286 (doubled) triangle areas, vectorized."""
        p = pts[triples]              # (286, 3, 2)
        a = p[:, 0]
        b = p[:, 1]
        c = p[:, 2]
        cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - \
                (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        return np.abs(cross)

    def min_tri_area(pts):
        """Minimum (doubled) triangle area over all triples."""
        return tri_areas(pts).min()

    # --- Multi-restart annealing with symmetric + random seeds ---
    # 3-fold symmetric initializations are strong for the Heilbronn problem,
    # since optimal configurations often exhibit triangular symmetry.
    def symmetric_init(seed):
        r = np.random.default_rng(seed)
        # 12 points in 3 orbits of 4 under 120-degree rotation about centroid,
        # plus the centroid itself.
        G = (A + B + C) / 3.0
        base = []
        for _ in range(4):
            u, v = r.random(), r.random()
            if u + v > 0.9:
                u, v = 0.9 * u / (u + v), 0.9 * v / (u + v)
            P = A + u * (B - A) + v * (C - A)
            base.append(P)
        pts = [G.copy()]
        for P in base:
            Q = P.copy()
            for _ in range(3):
                pts.append(Q.copy())
                # rotate 120 degrees about centroid
                d = Q - G
                c, s = np.cos(2 * np.pi / 3), np.sin(2 * np.pi / 3)
                Q = G + np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])
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

    # Precompute, for each point, which triples contain it -> incremental updates
    trip_of_pt = [np.where((triples == i).any(axis=1))[0] for i in range(n)]

    def anneal(pts, seed, n_iter=150000):
        """Metropolis simulated annealing on the hard minimum triangle area
        with INCREMENTAL evaluation: only triangles containing a moved point
        are recomputed; the rest are reused from a cached area array. This
        makes each step ~5-10x cheaper, allowing many more iterations and
        better exploration within the time budget."""
        r = np.random.default_rng(seed)
        pts = pts.copy()
        areas = tri_areas(pts)
        cur_val = areas.min()
        best_pts = pts.copy()
        best_val = cur_val
        T0, T1 = 0.02, 1e-5
        for it in range(n_iter):
            T = T0 * (T1 / T0) ** (it / n_iter)
            m = int(r.integers(1, 4))
            idx = r.choice(n, size=m, replace=False)
            old = pts[idx].copy()
            pts[idx] += r.normal(0.0, 0.02 + 0.15 * (1 - it / n_iter),
                                 size=(m, 2))
            pts[idx] = project_into_triangle(pts[idx])
            affected = np.unique(np.concatenate([trip_of_pt[i] for i in idx]))
            new_areas = areas.copy()
            p = pts[triples[affected]]
            a, b, c = p[:, 0], p[:, 1], p[:, 2]
            new_areas[affected] = np.abs(
                (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) -
                (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
            val = new_areas.min()
            if val >= cur_val or r.random() < np.exp(min(0.0, (val - cur_val) / max(T, 1e-12))):
                areas = new_areas
                cur_val = val
                if val > best_val:
                    best_val = val
                    best_pts = pts.copy()
            else:
                pts[idx] = old
        return best_pts, best_val

    # Run many restarts (symmetric and random seeds), keep the best.
    global_best_pts, global_best_val = None, -1.0
    seeds = [42, 7, 123, 2024, 99, 3, 55, 777, 314, 88, 12, 500]
    for si, sd in enumerate(seeds):
        init = symmetric_init(sd) if si % 2 == 0 else random_init(sd)
        bp, bv = anneal(init, sd + 1000)
        if bv > global_best_val:
            global_best_val = bv
            global_best_pts = bp

    best_pts = global_best_pts
    best_val = global_best_val

    # --- Analytic gradient polish on the critical (minimal) triangle ---
    # Move the 3 vertices of the currently-smallest triangle uphill along the
    # exact area gradient with backtracking line search; the hard min is
    # piecewise smooth in its active triangle, so this converges fast.
    def gradient_polish(pts, max_iter=5000):
        pts = pts.copy()
        best = tri_areas(pts).min()
        for _ in range(max_iter):
            cr = tri_areas(pts)
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
                v = tri_areas(cand).min()
                if v > best + 1e-14:
                    pts, best, improved = cand, v, True
                    break
                step *= 0.5
            if not improved:
                break
        return pts, best

    # --- Alternate gradient polish and greedy coordinate polish ---
    # Each polisher gets stuck in a different local mode of the piecewise
    # min-area landscape; alternating them until neither improves extracts
    # more area than either alone.
    for _ in range(6):
        gp_pts, gp_val = gradient_polish(best_pts)
        if gp_val > best_val:
            best_pts, best_val = gp_pts, gp_val
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
        # Re-run gradient polish on the greedy-refined configuration
        gp_pts, gp_val = gradient_polish(best_pts)
        if gp_val <= best_val + 1e-14:
            break
        best_pts, best_val = gp_pts, gp_val

    return best_pts


# EVOLVE-BLOCK-END
