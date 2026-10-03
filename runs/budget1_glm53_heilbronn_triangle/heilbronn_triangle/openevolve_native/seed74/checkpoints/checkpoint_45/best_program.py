# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct 11 points in the unit equilateral triangle maximizing the
    minimum area over all C(11,3)=165 point triplets.

    Approach:
    1. Deterministic seed: 3 triangle vertices + 8 interior points placed on
       a jittered triangular lattice (fixed RNG seed).
    2. Seeded hill-climbing: try small Gaussian perturbations of each point;
       accept only if the minimum normalized triangle area increases.
       Points are projected back into the triangle via barycentric clamping.
    3. Step size decays over iterations for fine convergence.

    Returns:
        points: np.ndarray of shape (11,2).
    """
    n = 11
    rng = np.random.default_rng(2024)
    H = np.sqrt(3) / 2.0
    S3 = np.sqrt(3)

    # All triple indices, precomputed
    triples = np.array(list(combinations(range(n), 3)))
    # Fast incremental evaluation: moving point i only changes the ~45
    # triples containing i. Candidate min-area =
    #   min(min over triples NOT containing i, min over affected triples).
    mask_of = np.array([[i in t for t in triples] for i in range(n)])
    trip_of = [np.nonzero(mask_of[i])[0] for i in range(n)]

    def areas_idx(pts, idxs):
        """Vectorized areas of only the triples indexed by idxs."""
        t = triples[idxs]
        a = pts[t[:, 0]]
        b = pts[t[:, 1]]
        c = pts[t[:, 2]]
        return 0.5 * np.abs(
            (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
            - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        )

    def project(p):
        """Clamp point into the equilateral triangle (barycentric-style)."""
        y = min(max(p[1], 0.0), H)
        xmin = y / S3          # left edge:  x >= y/sqrt(3)
        xmax = 1.0 - y / S3    # right edge: x <= 1 - y/sqrt(3)
        if xmin > xmax:         # degenerate near apex
            x = 0.5
        else:
            x = min(max(p[0], xmin), xmax)
        return np.array([x, y])

    def areas_all(pts):
        """Vectorized areas of all triples."""
        a = pts[triples[:, 0]]
        b = pts[triples[:, 1]]
        c = pts[triples[:, 2]]
        return 0.5 * np.abs(
            (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
            - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        )

    def random_config(seed):
        """Random interior configuration via barycentric sampling."""
        r = np.random.default_rng(seed)
        pts = r.dirichlet(np.ones(3), size=n)
        out = np.zeros((n, 2))
        out[:, 0] = pts[:, 1] + 0.5 * pts[:, 2]
        out[:, 1] = H * pts[:, 2]
        return out

    def local_optimize(pts, rng, iters=1500, anneal_frac=0.35):
        """Simulated-annealing hill-climb on the minimum triangle area.

        Early iterations accept mildly worse moves (prob exp(-d/T), T decaying
        geometrically) to escape local optima; the greedy phase then RESTARTS
        from the best-ever snapshot, so the returned points always achieve
        the reported best value (fixes the degraded-return bug). Only points
        in the worst triples are perturbed, and candidate moves are evaluated
        incrementally: only the ~45 triples containing the moved point are
        recomputed; the rest of the min comes from a cached base-area vector.
        """
        base = areas_all(pts)
        best = base.min()
        best_pts = pts.copy()
        cur = best
        step = 0.03
        T0 = 0.002
        n_anneal = int(iters * anneal_frac)
        for it in range(iters):
            if it == n_anneal and n_anneal > 0:
                # Greedy phase restarts from the best configuration found.
                pts = best_pts.copy()
                base = areas_all(pts)
                cur = best
                step = 0.01
            T = T0 * (0.97 ** it) if it < n_anneal else 0.0
            crit = np.argsort(base)[:6]
            crit_pts = np.unique(triples[crit].ravel())
            if it % 7 == 0:
                crit_pts = np.arange(n)
            improved = False
            for i in crit_pts:
                rest = base[~mask_of[i]].min()
                for _ in range(10):
                    cand = pts.copy()
                    cand[i] = project(pts[i] + rng.normal(0, step, 2))
                    a = min(rest, areas_idx(cand, trip_of[i]).min())
                    if a > cur + 1e-15:
                        cur = a
                        pts = cand
                        base = areas_all(pts)
                        improved = True
                        if a > best + 1e-15:
                            best = a
                            best_pts = pts.copy()
                    elif T > 0.0 and a > cur - 5 * T and rng.random() < np.exp((a - cur) / T):
                        cur = a
                        pts = cand
                        base = areas_all(pts)
            if not improved and T == 0.0:
                step *= 0.6
                if step < 1e-6:
                    break
            if it % 300 == 299:
                step *= 0.8
        return best_pts, best

    def pair_polish(pts, rng, iters=400):
        """Jointly perturb pairs of critical points (greedy, fine steps).

        Single-point moves stall when the critical triple needs two of its
        points to move together; pair moves unlock such configurations.
        Pairs are evaluated incrementally: only the triples touching either
        moved point are recomputed; the rest comes from a cached vector.
        """
        base = areas_all(pts)
        best = base.min()
        step = 0.01
        pair_idx = {}
        for i in range(n):
            for j in range(i + 1, n):
                m = mask_of[i] | mask_of[j]
                pair_idx[(i, j)] = (np.nonzero(~m)[0], np.nonzero(m)[0])
        for it in range(iters):
            crit = np.unique(triples[np.argsort(base)[:8]].ravel())
            pairs = [(i, j) for i in crit for j in crit if i < j]
            if it % 9 == 0:
                pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            improved = False
            for (i, j) in pairs:
                ridx, aidx = pair_idx[(i, j)]
                rest = base[ridx].min()
                for _ in range(4):
                    cand = pts.copy()
                    cand[i] = project(pts[i] + rng.normal(0, step, 2))
                    cand[j] = project(pts[j] + rng.normal(0, step, 2))
                    a = min(rest, areas_idx(cand, aidx).min())
                    if a > best + 1e-15:
                        best = a
                        pts = cand
                        base = areas_all(pts)
                        improved = True
            if not improved:
                step *= 0.6
                if step < 1e-7:
                    break
        return pts, best

    # --- Multi-start: lattice seed + random seeds ---
    candidates = []
    pts0 = np.zeros((n, 2))
    pts0[0] = [0.0, 0.0]
    pts0[1] = [1.0, 0.0]
    pts0[2] = [0.5, H]
    interior = []
    rows = [(0.2, 2), (0.42, 3), (0.65, 3)]
    for y, cnt in rows:
        for k in range(cnt):
            x = (k + 0.5) / cnt
            interior.append([x * (1.0 - y / H) + 0.5 * (y / H), y])
    interior = np.array(interior[:8])
    interior += rng.normal(0, 0.01, interior.shape)
    for i in range(8):
        pts0[3 + i] = project(interior[i])
    candidates.append(pts0)

    # Ring-based structured seeds: 3 vertices + jittered hexagon + 2 inner
    # points. Good coverage of the triangle and no three points collinear
    # after jitter, giving a different optimization basin than random starts.
    def ring_config(seed, R):
        r = np.random.default_rng(seed)
        cx, cy = 0.5, H / 3.0
        pts = [[0.0, 0.0], [1.0, 0.0], [0.5, H]]
        for k in range(6):
            ang = np.pi / 6.0 + k * np.pi / 3.0
            pts.append([cx + R * np.cos(ang), cy + R * np.sin(ang)])
        for ang in (np.pi / 2.0, -np.pi / 2.0):
            pts.append([cx + 0.5 * R * np.cos(ang),
                        cy + 0.5 * R * np.sin(ang)])
        pts = np.array(pts, dtype=float)
        pts[3:] += r.normal(0, 0.01, pts[3:].shape)
        return np.array([project(p) for p in pts])

    for R in (0.24, 0.30, 0.36):
        candidates.append(ring_config(700 + int(100 * R), R))

    for s in range(28):
        candidates.append(random_config(1000 + s))

    best_pts = None
    best_val = -1.0
    for idx, cand in enumerate(candidates):
        r = np.random.default_rng(777 + idx)
        p, v = local_optimize(cand.copy(), r)
        if v > best_val:
            best_val = v
            best_pts = p

    r = np.random.default_rng(999)
    best_pts, best_val = local_optimize(best_pts, r, iters=800)
    r2 = np.random.default_rng(555)
    best_pts, best_val = pair_polish(best_pts, r2, iters=400)
    r3 = np.random.default_rng(321)
    best_pts, best_val = local_optimize(best_pts, r3, iters=400, anneal_frac=0.0)
    r4 = np.random.default_rng(444)
    best_pts, best_val = pair_polish(best_pts, r4, iters=400)
    r5 = np.random.default_rng(222)
    best_pts, best_val = local_optimize(best_pts, r5, iters=300, anneal_frac=0.0)

    return best_pts


# EVOLVE-BLOCK-END
