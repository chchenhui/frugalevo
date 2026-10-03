# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct 11 points in/on the unit equilateral triangle (0,0),(1,0),(0.5,sqrt(3)/2)
    maximizing the minimum area over all C(11,3)=165 point-triangles (Heilbronn problem).

    Approach (fully deterministic):
      1. Multi-start seeds: k=3 triangular lattice + extra point, perimeter spread,
         seeded random interior points, and vertex-pinned random starts.
      2. Projected gradient ascent on a soft-min (log-sum-exp) surrogate of the
         minimum triangle area, with analytic area gradients and beta annealed
         from smooth (20) to sharp (8000). Points are re-projected into the
         container via barycentric clipping after every step.
      3. Hill-climbing polish on the exact minimum area (single-point moves,
         fixed RNG seed, decaying step scale).
      4. Fallback to a lattice arrangement if optimization fails.

    Returns:
        points: np.ndarray of shape (11,2) with the x,y coordinates of the points.
    """
    n = 11
    h = np.sqrt(3) / 2.0
    V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, h]])
    tri = np.array(list(combinations(range(n), 3)), dtype=np.int64)  # (165,3)

    def project(pts):
        # Barycentric coordinates w.r.t. V, clipped to nonnegative, renormalized.
        cc = pts[:, 1] / h
        bb = pts[:, 0] - 0.5 * cc
        aa = 1.0 - bb - cc
        abc = np.clip(np.stack([aa, bb, cc], axis=1), 0.0, None)
        abc = abc / np.maximum(abc.sum(axis=1, keepdims=True), 1e-12)
        return abc @ V

    def min_area(pts):
        p = pts[tri]
        ax, ay = p[:, 0, 0], p[:, 0, 1]
        bx, by = p[:, 1, 0], p[:, 1, 1]
        cx, cy = p[:, 2, 0], p[:, 2, 1]
        s = 0.5 * ((bx - ax) * (cy - ay) - (cx - ax) * (by - ay))
        return float(np.abs(s).min())

    def softmin_ascent(pts, iters=3500, lr0=0.01):
        pts = project(np.array(pts, dtype=float))
        for it in range(iters):
            f = it / (iters - 1)
            beta = 20.0 * (400.0 ** f)
            lr = lr0 * (1.0 - 0.97 * f) + 2e-4
            p = pts[tri]
            a, b, c = p[:, 0], p[:, 1], p[:, 2]
            s = 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                      - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
            A = np.abs(s)
            e = np.exp(-beta * (A - A.min()))
            w = e / e.sum()
            sg = w[:, None] * np.sign(s)[:, None]
            # Analytic gradients of signed area w.r.t. each triangle vertex.
            ga = 0.5 * np.stack([b[:, 1] - c[:, 1], c[:, 0] - b[:, 0]], axis=1)
            gb = 0.5 * np.stack([c[:, 1] - a[:, 1], a[:, 0] - c[:, 0]], axis=1)
            gc = 0.5 * np.stack([a[:, 1] - b[:, 1], b[:, 0] - a[:, 0]], axis=1)
            g = np.zeros_like(pts)
            np.add.at(g, tri[:, 0], sg * ga)
            np.add.at(g, tri[:, 1], sg * gb)
            np.add.at(g, tri[:, 2], sg * gc)
            pts = project(pts + lr * g)
        return pts

    def polish(pts, iters=60000, seed=1234567, T0=2e-4):
        """Simulated-annealing polish: Metropolis acceptance of slightly worse
        single-point moves lets the search escape shallow local optima, while a
        geometrically decaying step scale and temperature converge to a local
        optimum. Returns best configuration ever seen."""
        rng = np.random.default_rng(seed)
        cur = project(np.array(pts, dtype=float))
        curA = min_area(cur)
        best, bestA = cur.copy(), curA
        for t in range(iters):
            f = t / iters
            scale = 0.03 * (0.005 ** f)
            T = T0 * (1e-3 ** f) + 1e-12
            cand = cur.copy()
            idx = int(rng.integers(n))
            cand[idx] = cand[idx] + rng.normal(size=2) * scale
            cand = project(cand)
            mA = min_area(cand)
            d = mA - curA
            if d > -1e-12 or rng.random() < np.exp(d / T):
                cur, curA = cand, mA
                if mA > bestA:
                    best, bestA = cand.copy(), mA
        return best, bestA

    # ---------- deterministic multi-start seeds ----------
    rng = np.random.default_rng(20240517)
    starts = []

    # (a) k=3 triangular lattice (10 pts, min area = container/9) + one extra point.
    lat = []
    for i in range(4):
        for j in range(4 - i):
            k_ = 3 - i - j
            lat.append([(j + 0.5 * k_) / 3.0, k_ * h / 3.0])
    lat = np.array(lat)
    for extra in [np.array([0.5, h / 3.0]), np.array([0.25, h / 2.0]),
                  np.array([0.75, h / 2.0]), np.array([0.5, 2.0 * h / 3.0])]:
        starts.append(np.vstack([lat, extra[None, :]]))

    # (b) 11 points spread evenly along the perimeter.
    per = []
    for i in range(11):
        t = 3.0 * i / 11.0
        side = int(t)
        u = t - side
        if side == 0:
            per.append([u, 0.0])
        elif side == 1:
            per.append([1.0 - 0.5 * u, h * u])
        else:
            per.append([0.5 * (1.0 - u), h * (1.0 - u)])
    starts.append(np.array(per))

    # (c) random interior points (uniform in barycentric coords).
    for _ in range(10):
        abc = rng.random((n, 3))
        abc /= abc.sum(axis=1, keepdims=True)
        starts.append(abc @ V)

    # (d) random starts with the three container vertices pinned.
    for _ in range(6):
        abc = rng.random((n, 3))
        abc /= abc.sum(axis=1, keepdims=True)
        pts = abc @ V
        pts[0], pts[1], pts[2] = V[0], V[1], V[2]
        starts.append(pts)

    # (e) boundary-heavy seeds: 3 vertices + 2 points per side at varied
    # symmetric fractions + 2 interior points. Optimal Heilbronn layouts for
    # small n concentrate points on the perimeter; these seeds give the
    # ascent access to basins random interior starts never reach.
    for frac in (0.25, 0.3, 1.0 / 3.0, 0.4, 0.2):
        bnd = [V[0].copy(), V[1].copy(), V[2].copy()]
        for (p_, q_) in [(V[0], V[1]), (V[1], V[2]), (V[2], V[0])]:
            for t in (frac, 1.0 - frac):
                bnd.append(p_ + t * (q_ - p_))
        bnd.append([0.5, h / 3.0])
        bnd.append([0.25, h / 2.0])
        starts.append(np.array(bnd[:n]))
        # variant: interior points at alternate positions
        bnd2 = bnd[:9] + [[0.75, h / 2.0], [0.5, 2.0 * h / 3.0]]
        starts.append(np.array(bnd2[:n]))

    # ---------- optimize ----------
    cands = []
    for s0 in starts:
        try:
            q = softmin_ascent(s0)
            mA = min_area(q)
            if np.isfinite(mA):
                cands.append((mA, q))
        except Exception:
            continue
    cands.sort(key=lambda z: -z[0])
    # Keep more candidates for polishing (was 4); boundary seeds often need
    # polish to reveal their potential since softmin ascent alone under-rates
    # near-degenerate layouts.

    best_pts = None
    best_A = -1.0
    for A0, q in cands[:8]:
        try:
            pb, pA = polish(q, iters=40000)
            if pA > best_A:
                best_A, best_pts = pA, pb
        except Exception:
            continue

    # Targeted micro-polish: perturb only points on the minimal triangle.
    def targeted(pts, iters=6000, seed=987654321):
        rng = np.random.default_rng(seed)
        cur = project(np.array(pts, dtype=float))
        curA = min_area(cur)
        best, bestA = cur.copy(), curA
        for t in range(iters):
            p = cur[tri]
            s = 0.5 * ((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
                      - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1]))
            k = int(np.argmin(np.abs(s)))
            idxs = tri[k]
            scale = 0.01 * (0.02 ** (t / iters))
            cand = cur.copy()
            j = idxs[int(rng.integers(3))]
            cand[j] = cand[j] + rng.normal(size=2) * scale
            cand = project(cand)
            mA = min_area(cand)
            if mA > curA - 1e-12:
                cur, curA = cand, mA
                if mA > bestA:
                    best, bestA = cand.copy(), mA
        return best, bestA

    if best_pts is not None:
        try:
            tb, tA = targeted(best_pts, iters=15000)
            if tA > best_A:
                best_pts, best_A = tb, tA
        except Exception:
            pass
        # Annealing restarts from the incumbent with varied seeds/temperatures:
        # different random trajectories explore different basins. Budget is
        # large (runtime was ~16s of a 360s limit), so use longer chains.
        for sd, tp in [(11, 3e-4), (22, 6e-4), (33, 1.5e-4), (44, 1e-3),
                       (55, 5e-4), (66, 2e-4)]:
            try:
                rb, rA = polish(best_pts, iters=120000, seed=sd, T0=tp)
                if rA > best_A:
                    best_pts, best_A = rb, rA
            except Exception:
                continue
        # Final greedy targeted polish to squeeze the last bit out.
        try:
            tb, tA = targeted(best_pts, iters=10000)
            if tA > best_A:
                best_pts, best_A = tb, tA
        except Exception:
            pass

    if best_pts is None or not np.isfinite(best_A) or best_A <= 0.0:
        # Fallback: non-degenerate lattice-based arrangement.
        best_pts = np.vstack([lat, np.array([[0.5, h / 3.0]])])

    return np.asarray(best_pts, dtype=float)


# EVOLVE-BLOCK-END
