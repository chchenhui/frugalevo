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

    def random_config(seed, alpha=1.0):
        """Barycentric sampling; alpha<1 biases points toward the boundary
        (vertices/edges), matching known Heilbronn structure."""
        r = np.random.default_rng(seed)
        pts = r.dirichlet(np.full(3, alpha), size=n)
        # barycentric -> cartesian
        out = np.zeros((n, 2))
        out[:, 0] = pts[:, 1] + 0.5 * pts[:, 2]
        out[:, 1] = H * pts[:, 2]
        return out

    def local_optimize(pts, rng, iters=800, anneal_frac=0.3):
        """Simulated-annealing hill-climb on the minimum triangle area.

        Phase A (annealing, first anneal_frac of iterations): perturb points
        from the worst triples; moves that decrease the min area are accepted
        with probability exp(-delta/T) so the search can escape local optima.
        Temperature T decays geometrically to zero, seamlessly turning into
        Phase B: pure greedy hill-climbing with step decay when stuck.
        """
        best = areas_all(pts).min()
        cur = best
        step = 0.03
        T0 = 0.002
        n_anneal = int(iters * anneal_frac)
        for it in range(iters):
            T = T0 * (0.97 ** it) if it < n_anneal else 0.0
            ar = areas_all(pts)
            # indices of the 6 worst triples -> critical points
            crit = np.argsort(ar)[:6]
            crit_pts = np.unique(triples[crit].ravel())
            # occasionally allow any point to move (escape mechanism)
            if it % 7 == 0:
                crit_pts = np.arange(n)
            improved = False
            for i in crit_pts:
                for _ in range(8):
                    cand = pts.copy()
                    cand[i] = project(pts[i] + rng.normal(0, step, 2))
                    a = areas_all(cand).min()
                    if a > cur + 1e-15:
                        cur = a
                        pts = cand
                        improved = True
                        if a > best:
                            best = a
                    elif T > 0.0 and a > cur - 5 * T and rng.random() < np.exp((a - cur) / T):
                        # annealing: accept mildly worse moves early on
                        cur = a
                        pts = cand
            if not improved and T == 0.0:
                step *= 0.6
                if step < 1e-7:
                    break
            if it % 300 == 299:
                step *= 0.8
        return pts, best

    # --- Multi-start: lattice seed + several random seeds ---
    candidates = []

    # Lattice-based seed (original construction)
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

    # Random multi-start seeds: uniform interior + boundary-biased
    # (deterministic seed sequence)
    for s in range(12):
        candidates.append(random_config(1000 + s))
    for s in range(8):
        candidates.append(random_config(5000 + s, alpha=0.35))
    for s in range(6):
        candidates.append(random_config(9000 + s, alpha=0.15))

    best_pts = None
    best_val = -1.0
    for idx, cand in enumerate(candidates):
        r = np.random.default_rng(777 + idx)
        p, v = local_optimize(cand.copy(), r)
        if v > best_val:
            best_val = v
            best_pts = p

    # Final polish: pair-move hill-climb. Single-point moves can be blocked
    # when the critical triple needs two points to move together; here we
    # jointly perturb pairs of points appearing in the worst triples.
    def pair_polish(pts, rng, iters=400):
        best = areas_all(pts).min()
        step = 0.01
        for it in range(iters):
            ar = areas_all(pts)
            crit = np.unique(triples[np.argsort(ar)[:8]].ravel())
            improved = False
            pairs = [(i, j) for i in crit for j in crit if i < j]
            if it % 9 == 0:
                pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            for (i, j) in pairs:
                for _ in range(4):
                    cand = pts.copy()
                    cand[i] = project(pts[i] + rng.normal(0, step, 2))
                    cand[j] = project(pts[j] + rng.normal(0, step, 2))
                    a = areas_all(cand).min()
                    if a > best + 1e-15:
                        best = a
                        pts = cand
                        improved = True
            if not improved:
                step *= 0.6
                if step < 1e-7:
                    break
        return pts, best

    # Final polish on the winner with fine steps: pair-move hill-climb.
    # Single-point moves can be blocked when the critical triple needs two
    # points to move together; here we jointly perturb pairs of points
    # appearing in the worst triples.
    r = np.random.default_rng(999)
    best_pts, best_val = pair_polish(best_pts, r)
    # one more fine greedy pass after pair moves
    r3 = np.random.default_rng(321)
    best_pts, best_val = local_optimize(best_pts, r3, iters=250, anneal_frac=0.0)

    # Triple-move polish: the binding (minimum-area) triangle often needs
    # ALL THREE of its vertices to move in a coordinated way; single/pair
    # moves are blocked. Jointly perturb the worst triple's points with
    # small correlated steps, and also try snapping each of them onto the
    # triangle edges (optimal small-n configs are boundary-heavy).
    def triple_polish(pts, rng, iters=300):
        """Joint 3-point moves on the worst triple + edge snaps.

        For the worst-area triangle (i,j,k), try moving all three points
        simultaneously by small Gaussians (which lets the triangle 'rotate'
        or 'translate' as a rigid set), plus variants where one or two of
        the points snap to their nearest edge. Accept only improvements.
        """
        edges = [(np.array([0.0, 0.0]), np.array([1.0, 0.0])),
                 (np.array([1.0, 0.0]), np.array([0.5, H])),
                 (np.array([0.5, H]), np.array([0.0, 0.0]))]

        def snap(p):
            """Nearest point on the triangle boundary (edges clamped)."""
            best_p, best_d = None, np.inf
            for e0, e1 in edges:
                d = e1 - e0
                t = np.dot(p - e0, d) / np.dot(d, d)
                t = min(max(t, 0.0), 1.0)
                q = e0 + t * d
                dd = np.dot(p - q, p - q)
                if dd < best_d:
                    best_d, best_p = dd, q
            return best_p

        best = areas_all(pts).min()
        step = 0.008
        for it in range(iters):
            ar = areas_all(pts)
            worst = np.argsort(ar)[:4]
            improved = False
            for w in worst:
                i, j, k = triples[w]
                for _ in range(10):
                    cand = pts.copy()
                    cand[i] = project(pts[i] + rng.normal(0, step, 2))
                    cand[j] = project(pts[j] + rng.normal(0, step, 2))
                    cand[k] = project(pts[k] + rng.normal(0, step, 2))
                    # variants: snap subsets of the triple to the boundary
                    if rng.random() < 0.5:
                        cand[i] = snap(cand[i])
                    if rng.random() < 0.3:
                        cand[k] = snap(cand[k])
                    a = areas_all(cand).min()
                    if a > best + 1e-15:
                        best = a
                        pts = cand
                        improved = True
            if not improved:
                step *= 0.65
                if step < 1e-8:
                    break
        return pts, best

    r4 = np.random.default_rng(4242)
    best_pts, best_val = triple_polish(best_pts, r4)
    # alternate pair / single passes after triple moves unlock new configs
    r5 = np.random.default_rng(555)
    p2, v2 = pair_polish(best_pts, r5)
    if v2 > best_val:
        best_pts, best_val = p2, v2
    r6 = np.random.default_rng(6789)
    best_pts, best_val = triple_polish(best_pts, r6, iters=150)

    return best_pts


# EVOLVE-BLOCK-END
