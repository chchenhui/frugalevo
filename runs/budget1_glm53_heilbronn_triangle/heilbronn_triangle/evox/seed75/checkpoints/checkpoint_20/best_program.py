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

    def optimize(pts, rounds=90, cand_per=50):
        pts = pts.copy()
        cur = min_area(pts)
        scale = 0.06
        for _ in range(rounds):
            improved = False
            # points participating in the worst triangle get extra attention
            ar = areas(pts)
            worst = tri[int(np.argmin(ar))]
            focus = list(range(n))
            for i in focus:
                best_local = cur
                best_p = pts[i].copy()
                n_cand = cand_per + (20 if i in worst else 0)
                cands = [clamp(pts[i] + rng.randn(2) * scale) for _ in range(n_cand)]
                for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    cands.append(clamp(pts[i] + np.array(d, float) * scale))
                for cand in cands:
                    old = pts[i].copy()
                    pts[i] = cand
                    a = min_area(pts)
                    pts[i] = old
                    if a > best_local + 1e-15:
                        best_local = a
                        best_p = cand
                if best_local > cur:
                    pts[i] = best_p
                    cur = best_local
                    improved = True
            scale *= 0.94
            if not improved and scale < 1e-4:
                break
        return pts, cur

    best_pts, best_val = None, -1.0
    for sd in seeds:
        p, v = optimize(sd)
        if v > best_val:
            best_val, best_pts = v, p
        if best_val > 0.0365:
            break

    if best_pts is None:  # graceful fallback
        best_pts = np.array([rand_bary(rng) for _ in range(n)])
    return best_pts


# EVOLVE-BLOCK-END
