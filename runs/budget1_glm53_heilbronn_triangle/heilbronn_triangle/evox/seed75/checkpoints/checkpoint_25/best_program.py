# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct 11 points inside/on the equilateral triangle (0,0),(1,0),(0.5,h)
    maximizing the minimum area over all C(11,3)=165 triplets.

    Approach (deterministic, vectorized):
    1. Diverse non-degenerate seeds: boundary-heavy (staggered so no three
       points are collinear on an edge), perturbed triangular lattice, and
       fixed-seed random barycentric restarts.
    2. Greedy local search with annealed step size. Moving point i only
       changes the 45 triangles containing i; the min over the other 120
       triangles is precomputed once per point, and all candidates are scored
       simultaneously via NumPy broadcasting (~50x faster than full recompute).
    3. Points in the currently-worst triangle get extra candidate samples.
    4. Best configuration across all seeds is returned, with kick restarts
       from the best solution to escape local optima.
    """
    n = 11
    h = np.sqrt(3.0) / 2.0
    s = np.sqrt(3.0)
    rng = np.random.RandomState(20240711)
    tri = np.array(list(combinations(range(n), 3)))

    def clamp(p):
        x, y = float(p[0]), float(p[1])
        y = min(max(y, 0.0), h)
        x = min(max(x, y / s), 1.0 - y / s)
        return np.array([x, y])

    def areas(P):
        a = P[tri[:, 0]]
        b = P[tri[:, 1]]
        c = P[tri[:, 2]]
        cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        return 0.5 * np.abs(cross)

    def min_area(P):
        return float(np.min(areas(P)))

    def rand_bary(rng):
        u, v = rng.rand(), rng.rand()
        if u + v > 1.0:
            u, v = 1.0 - u, 1.0 - v
        return clamp(np.array([u + 0.5 * v, v * h]))

    # --- diverse, non-degenerate seed set ---
    seeds = []
    # boundary-heavy: vertices + staggered points along each edge
    bnd = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.5, h])]
    for t in (0.2, 0.5, 0.8):
        bnd.append(np.array([t, 0.0]))
    for t in (0.35, 0.7):
        bnd.append(np.array([t / 2.0, t * h]))       # left edge
        bnd.append(np.array([1.0 - t / 2.0, t * h])) # right edge
    while len(bnd) < n:
        bnd.append(clamp(np.array([0.5, h / 3.0]) + 0.05 * rng.randn(2)))
    seeds.append(np.array(bnd[:n]))
    # lattice: triangular grid rows 1+2+3+4 = 10 + centroid (perturbed to
    # avoid collinear triplets of the perfect lattice)
    lat = []
    for row in range(4):
        y = h * row / 3.0
        for k in range(row + 1):
            lat.append(clamp(np.array([k / 3.0 + 0.5 * y / h, y])
                             + 0.004 * rng.randn(2)))
    lat.append(np.array([0.5, h / 3.0]))
    seeds.append(np.array(lat))
    # fixed-seed random restarts
    for _ in range(5):
        seeds.append(np.array([rand_bary(rng) for _ in range(n)]))

    # triangles containing / not containing each point (incremental eval)
    has_i = [np.where(np.any(tri == i, axis=1))[0] for i in range(n)]
    not_has_i = [np.where(~np.any(tri == i, axis=1))[0] for i in range(n)]

    def optimize(pts, rounds=140, cand_per=400):
        """Greedy local search with vectorized candidate scoring: moving
        point i only changes the 45 triangles containing i; the min over
        the other 120 is precomputed once per point."""
        pts = pts.copy()
        cur = min_area(pts)
        scale = 0.06
        for _ in range(rounds):
            improved = False
            ar = areas(pts)
            worst = tri[int(np.argmin(ar))]
            for i in range(n):
                fixed_min = float(np.min(ar[not_has_i[i]]))
                idx = has_i[i]
                ta, tb, tc = tri[idx, 0], tri[idx, 1], tri[idx, 2]
                fa, fb, fc = pts[ta], pts[tb], pts[tc]
                eq_a = (ta == i)[:, None]
                eq_b = (tb == i)[:, None]
                eq_c = (tc == i)[:, None]
                K = cand_per + (100 if i in worst else 0)
                # random perturbations + axis nudges, all clamped
                cands = pts[i] + rng.randn(K, 2) * scale
                for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    cands = np.vstack([cands, pts[i] + np.array(d, float) * scale])
                # edge-projection candidates: optimal Heilbronn configs are
                # typically boundary-heavy, so actively sample near/on edges
                Ke = 60
                ec = pts[i] + rng.randn(Ke, 2) * (scale * 3.0)
                # project onto base edge y=0
                base = np.stack([np.clip(ec[:, 0], 0.0, 1.0),
                                 np.zeros(Ke)], axis=1)
                # project onto left edge x = y/sqrt(3)
                ly = np.clip(ec[:, 1], 0.0, h)
                left = np.stack([ly / s, ly], axis=1)
                # project onto right edge x = 1 - y/sqrt(3)
                right = np.stack([1.0 - ly / s, ly], axis=1)
                # slight insets so points can also sit just inside the edges
                inset = 0.01 * rng.rand(Ke, 1)
                base_i = np.stack([np.clip(ec[:, 0], 0.0, 1.0),
                                   inset[:, 0]], axis=1)
                cands = np.vstack([cands, base, left, right, base_i])
                ys = np.clip(cands[:, 1], 0.0, h)
                xs = np.clip(cands[:, 0], ys / s, 1.0 - ys / s)
                cands = np.stack([xs, ys], axis=1)
                # broadcast: (K,45,2) arrays with candidate substituted for i
                A = np.where(eq_a[None], cands[:, None, :], fa[None])
                B = np.where(eq_b[None], cands[:, None, :], fb[None])
                C = np.where(eq_c[None], cands[:, None, :], fc[None])
                cross = (B[:, :, 0] - A[:, :, 0]) * (C[:, :, 1] - A[:, :, 1]) \
                    - (B[:, :, 1] - A[:, :, 1]) * (C[:, :, 0] - A[:, :, 0])
                mins = 0.5 * np.min(np.abs(cross), axis=1)
                mins = np.minimum(mins, fixed_min)
                k = int(np.argmax(mins))
                if mins[k] > cur - 1e-12 and not np.allclose(cands[k], pts[i]):
                    # accept equal-or-better moves (plateau traversal)
                    if mins[k] > cur + 1e-15:
                        improved = True
                    pts[i] = cands[k]
                    cur = max(cur, float(mins[k]))
                    ar = areas(pts)
                    worst = tri[int(np.argmin(ar))]
            scale *= 0.93
            if not improved and scale < 1e-4:
                break
        return pts, cur

    best_pts, best_val = None, -1.0
    try:
        for sd in seeds:
            p, v = optimize(sd)
            if v > best_val:
                best_val, best_pts = v, p
            if best_val > 0.0365:
                break
        # kick restarts from the best configuration to escape local optima:
        # varied kick magnitudes, and always re-polish from the incumbent
        for kick_scale in (0.005, 0.01, 0.02, 0.04, 0.08):
            kick = np.array([clamp(best_pts[i] + rng.randn(2) * kick_scale)
                             for i in range(n)])
            p, v = optimize(kick)
            if v > best_val:
                best_val, best_pts = v, p
            if best_val > 0.0365:
                break
        # final fine polish
        p, v = optimize(best_pts, rounds=250, cand_per=600)
        if v > best_val:
            best_val, best_pts = v, p
    except Exception:
        pass

    if best_pts is None:  # graceful fallback
        best_pts = np.array([rand_bary(rng) for _ in range(n)])
    return best_pts


# EVOLVE-BLOCK-END
