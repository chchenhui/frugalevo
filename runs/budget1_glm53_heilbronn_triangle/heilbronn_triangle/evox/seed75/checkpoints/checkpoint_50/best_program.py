# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct 11 points inside/on the equilateral triangle (0,0),(1,0),(0.5,h)
    maximizing the minimum triangle area over all C(11,3)=165 triplets.

    Approach (deterministic):
    1. Generate diverse seeds: boundary-heavy, triangular-lattice, and several
       fixed-seed random barycentric configurations.
    2. For each seed, run greedy local search: for every point, sample random
       perturbations (annealed step size) plus coordinate nudges, accept the
       candidate that maximizes the vectorized min-area; also perturb points
       belonging to the currently-worst triangle to escape local optima.
    3. Keep the best configuration across all seeds.
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

    # --- diverse seed set ---
    seeds = []
    # boundary-heavy: vertices + points along each edge
    bnd = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.5, h])]
    for t in np.linspace(0.2, 0.8, 3):
        bnd.append(np.array([t, 0.0]))
        bnd.append(np.array([t / 2.0, t * h]))
        bnd.append(np.array([1.0 - t / 2.0, t * h]))
    while len(bnd) < n:
        bnd.append(clamp(np.array([0.5, h / 3.0]) + 0.05 * rng.randn(2)))
    seeds.append(np.array(bnd[:n]))
    # lattice: triangular grid rows 1+2+3+4 = 10 points + centroid
    lat = []
    for row in range(4):
        y = h * row / 3.0
        for k in range(row + 1):
            lat.append(clamp(np.array([k / 3.0 + 0.5 * y / h, y])))
    lat.append(np.array([0.5, h / 3.0]))
    seeds.append(np.array(lat))
    # random restarts (fixed seed -> deterministic)
    for _ in range(5):
        seeds.append(np.array([rand_bary(rng) for _ in range(n)]))

    # triangles containing / not containing each point (incremental eval)
    has_i = [np.where(np.any(tri == i, axis=1))[0] for i in range(n)]
    not_has_i = [np.where(~np.any(tri == i, axis=1))[0] for i in range(n)]

    def optimize(pts, rounds=140, cand_per=220):
        """Greedy local search with annealed steps; each point's candidate
        batch is scored simultaneously via broadcasting over the 45
        triangles containing that point (min over the other 120 is fixed),
        making evaluation ~100x faster than full recomputation."""
        pts = pts.copy()
        cur = min_area(pts)
        scale = 0.06
        for _ in range(rounds):
            improved = False
            ar = areas(pts)
            worst = tri[int(np.argmin(ar))]
            for i in range(n):
                fixed_min = float(np.min(ar[not_has_i[i]])) if len(not_has_i[i]) else np.inf
                idx = has_i[i]
                ta, tb, tc = tri[idx, 0], tri[idx, 1], tri[idx, 2]
                oa, ob, oc = pts[ta], pts[tb], pts[tc]
                m = cand_per + (80 if i in worst else 0)
                steps = rng.randn(m, 2) * scale
                fine = rng.rand(m, 1) < 0.4          # multi-scale sampling
                steps = steps * np.where(fine, 0.25, 1.0)
                cands = pts[i] + steps
                cy = np.clip(cands[:, 1], 0.0, h)
                cx = np.clip(cands[:, 0], cy / s, 1.0 - cy / s)
                cands = np.stack([cx, cy], axis=1)
                A = np.where((ta == i)[None, :, None], cands[:, None, :], oa[None, :, :])
                B = np.where((tb == i)[None, :, None], cands[:, None, :], ob[None, :, :])
                C = np.where((tc == i)[None, :, None], cands[:, None, :], oc[None, :, :])
                cross = ((B[:, :, 0] - A[:, :, 0]) * (C[:, :, 1] - A[:, :, 1])
                         - (B[:, :, 1] - A[:, :, 1]) * (C[:, :, 0] - A[:, :, 0]))
                vals = 0.5 * np.abs(cross).min(axis=1)
                np.minimum(vals, fixed_min, out=vals)
                j = int(np.argmax(vals))
                if vals[j] > cur + 1e-15:
                    pts[i] = cands[j]
                    cur = float(vals[j])
                    ar = areas(pts)
                    worst = tri[int(np.argmin(ar))]
                    improved = True
            scale *= 0.95
            if not improved and scale < 1e-4:
                break
        return pts, cur

    def polish(pts, rounds=60, cand_per=300):
        """Fine-grained refinement with tiny annealed steps around the best
        configuration found (same vectorized incremental evaluation)."""
        pts = pts.copy()
        cur = min_area(pts)
        scale = 0.004
        for _ in range(rounds):
            improved = False
            ar = areas(pts)
            for i in range(n):
                fixed_min = float(np.min(ar[not_has_i[i]])) if len(not_has_i[i]) else np.inf
                idx = has_i[i]
                ta, tb, tc = tri[idx, 0], tri[idx, 1], tri[idx, 2]
                oa, ob, oc = pts[ta], pts[tb], pts[tc]
                cands = pts[i] + scale * rng.randn(cand_per, 2)
                cy = np.clip(cands[:, 1], 0.0, h)
                cx = np.clip(cands[:, 0], cy / s, 1.0 - cy / s)
                cands = np.stack([cx, cy], axis=1)
                A = np.where((ta == i)[None, :, None], cands[:, None, :], oa[None, :, :])
                B = np.where((tb == i)[None, :, None], cands[:, None, :], ob[None, :, :])
                C = np.where((tc == i)[None, :, None], cands[:, None, :], oc[None, :, :])
                cross = ((B[:, :, 0] - A[:, :, 0]) * (C[:, :, 1] - A[:, :, 1])
                         - (B[:, :, 1] - A[:, :, 1]) * (C[:, :, 0] - A[:, :, 0]))
                vals = 0.5 * np.abs(cross).min(axis=1)
                np.minimum(vals, fixed_min, out=vals)
                j = int(np.argmax(vals))
                if vals[j] > cur + 1e-15:
                    pts[i] = cands[j]
                    cur = float(vals[j])
                    ar = areas(pts)
                    improved = True
            scale *= 0.9
            if not improved and scale < 1e-6:
                break
        return pts, cur

    import time
    t0 = time.time()
    best_pts, best_val = None, -1.0
    for sd in seeds:
        try:
            p, v = optimize(sd)
        except Exception:
            continue
        if v > best_val:
            best_val, best_pts = v, p
        if best_val > 0.0365 or time.time() - t0 > 150.0:
            break

    # kick restarts: perturb the best configuration and re-optimize to
    # escape local optima; key phase for reaching high min-area configs
    if best_pts is not None:
        try:
            while time.time() - t0 < 300.0:
                kick = np.array([clamp(best_pts[i] + rng.randn(2) * 0.02)
                                 for i in range(n)])
                p, v = optimize(kick)
                if v > best_val:
                    best_val, best_pts = v, p
                if best_val > 0.0365:
                    break
        except Exception:
            pass
    # final fine polish on the best configuration (time-guarded)
    if best_pts is not None:
        try:
            if time.time() - t0 <= 340.0:
                p, v = polish(best_pts, rounds=120, cand_per=500)
                if v > best_val:
                    best_val, best_pts = v, p
        except Exception:
            pass

    if best_pts is None:  # graceful fallback
        best_pts = np.array([rand_bary(rng) for _ in range(n)])
    return best_pts


# EVOLVE-BLOCK-END
