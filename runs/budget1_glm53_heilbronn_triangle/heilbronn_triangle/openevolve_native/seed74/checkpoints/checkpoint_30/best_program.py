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

    def min_area(pts):
        """Minimum triangle area over all triples, normalized by triangle area."""
        a = pts[triples[:, 0]]
        b = pts[triples[:, 1]]
        c = pts[triples[:, 2]]
        areas = 0.5 * np.abs(
            (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
            - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        )
        return areas.min()

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
        out = np.zeros((n, 2))
        out[:, 0] = pts[:, 1] + 0.5 * pts[:, 2]
        out[:, 1] = H * pts[:, 2]
        return out

    def local_optimize(pts, rng, iters=600):
        """Softmin gradient ascent on the minimum triangle area with annealed
        temperature and adaptive step, followed by an exact hill-climb polish
        that includes snapping critical points onto the triangle edges.

        The smooth surrogate  -tau*log(sum_k exp(-|s_k|/tau))  has analytic
        gradients w.r.t. every point; ascending it directly pushes apart the
        vertices of the smallest-area triangles in the optimal direction.
        """
        best_pts = pts.copy()
        best = areas_all(pts).min()
        lr = 0.02
        tau = 0.01
        for it in range(iters):
            a = pts[triples[:, 0]]
            b = pts[triples[:, 1]]
            c = pts[triples[:, 2]]
            s = 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                       - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
            ab = np.abs(s)
            # softmax weights concentrated on the worst triples
            w = np.exp(-(ab - ab.min()) / tau)
            w /= w.sum()
            sg = np.sign(s) * w
            g = np.zeros_like(pts)
            # analytic gradients of signed area w.r.t. each triple vertex
            np.add.at(g, triples[:, 0],
                      0.5 * sg[:, None] * np.stack(
                          [b[:, 1] - c[:, 1], c[:, 0] - b[:, 0]], axis=1))
            np.add.at(g, triples[:, 1],
                      0.5 * sg[:, None] * np.stack(
                          [c[:, 1] - a[:, 1], a[:, 0] - c[:, 0]], axis=1))
            np.add.at(g, triples[:, 2],
                      0.5 * sg[:, None] * np.stack(
                          [a[:, 1] - b[:, 1], b[:, 0] - a[:, 0]], axis=1))
            gn = np.linalg.norm(g, axis=1, keepdims=True)
            gn[gn == 0] = 1.0
            new_pts = pts + lr * g / gn
            for i in range(n):
                new_pts[i] = project(new_pts[i])
            cur = areas_all(pts).min()
            new = areas_all(new_pts).min()
            if new >= cur - 1e-12:
                pts = new_pts
                if new > best:
                    best = new
                    best_pts = pts.copy()
                lr *= 1.02
            else:
                lr *= 0.5
            tau = max(tau * 0.995, 1e-4)
            if lr < 1e-9:
                break
        # exact hill-climb polish: targeted perturbations plus edge snaps.
        # Optimal small-n Heilbronn configs place many points on the
        # boundary, so for each critical point we also try its orthogonal
        # projection onto each triangle edge (clamped to the segment).
        edges = [(np.array([0.0, 0.0]), np.array([1.0, 0.0])),
                 (np.array([1.0, 0.0]), np.array([0.5, H])),
                 (np.array([0.5, H]), np.array([0.0, 0.0]))]
        step = 0.005
        cur = areas_all(best_pts).min()
        for it in range(400):
            ar = areas_all(best_pts)
            crit = np.argsort(ar)[:5]
            crit_pts = np.unique(triples[crit].ravel())
            improved = False
            for i in crit_pts:
                cands = [project(best_pts[i] + rng.normal(0, step, 2))
                         for _ in range(8)]
                for e0, e1 in edges:
                    d = e1 - e0
                    t = np.dot(best_pts[i] - e0, d) / np.dot(d, d)
                    t = min(max(t, 0.0), 1.0)
                    cands.append(e0 + t * d)
                for cp in cands:
                    cand = best_pts.copy()
                    cand[i] = cp
                    v = areas_all(cand).min()
                    if v > cur + 1e-15:
                        cur = v
                        best_pts = cand
                        improved = True
            if not improved:
                step *= 0.6
                if step < 1e-8:
                    break
        return best_pts, cur

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
    for s in range(10):
        candidates.append(random_config(1000 + s))
    for s in range(6):
        candidates.append(random_config(5000 + s, alpha=0.35))

    # Structured boundary-heavy seeds: optimal small-n Heilbronn configs
    # place most points on the triangle edges. Distribute points along the
    # three edges at staggered fractions (plus the 3 vertices), with a few
    # variants of interior fill and edge jitter.
    edges = [(np.array([0.0, 0.0]), np.array([1.0, 0.0])),
             (np.array([1.0, 0.0]), np.array([0.5, H])),
             (np.array([0.5, H]), np.array([0.0, 0.0]))]
    for variant in range(8):
        r = np.random.default_rng(88 + variant)
        pts = np.zeros((n, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2] = [0.5, H]
        k = 3
        fracs = [[1/4, 3/4], [1/4, 1/2, 3/4], [1/3, 2/3]]
        per_edge = fracs[variant % 3]
        for e0, e1 in edges:
            for f in per_edge:
                if k < n:
                    jf = f + r.normal(0, 0.02)
                    jf = min(max(jf, 0.05), 0.95)
                    pts[k] = e0 + jf * (e1 - e0)
                    k += 1
        while k < n:
            pts[k] = project(np.array([0.3 + 0.4 * r.random(),
                                       0.2 + 0.4 * r.random()]))
            k += 1
        candidates.append(pts)

    # Symmetric seed: 3 vertices + edge midpoints + centroid ring.
    # Known good Heilbronn-style layouts tend to be symmetric; this seed
    # reliably converges to a high local optimum.
    pts_sym = np.zeros((n, 2))
    pts_sym[0] = [0.0, 0.0]
    pts_sym[1] = [1.0, 0.0]
    pts_sym[2] = [0.5, H]
    mids = [0.5 * (np.array([0.0, 0.0]) + np.array([1.0, 0.0])),
            0.5 * (np.array([1.0, 0.0]) + np.array([0.5, H])),
            0.5 * (np.array([0.5, H]) + np.array([0.0, 0.0]))]
    for i, m in enumerate(mids):
        pts_sym[3 + i] = m
    cx, cy = 0.5, H / 3.0
    for i in range(5):
        ang = 2 * np.pi * i / 5 + 0.3
        rad = 0.16 + 0.05 * (i % 2)
        pts_sym[6 + i] = project(np.array([cx + rad * np.cos(ang),
                                           cy + rad * np.sin(ang)]))
    candidates.append(pts_sym)

    best_pts = None
    best_val = -1.0
    for idx, cand in enumerate(candidates):
        r = np.random.default_rng(777 + idx)
        p, v = local_optimize(cand.copy(), r)
        if v > best_val:
            best_val = v
            best_pts = p

    # Final polish on the winner with fine steps (single longer pass
    # replaces two redundant passes, saving eval_time at equal quality)
    r = np.random.default_rng(999)
    best_pts, best_val = local_optimize(best_pts, r, iters=1000)

    return best_pts


# EVOLVE-BLOCK-END
