# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """
    Construct a specific arrangement of 26 circles in a unit square
    that attempts to maximize the sum of their radii.

    Returns:
        Tuple of (centers, radii, sum_of_radii)
        centers: np.array of shape (26, 2) with (x, y) coordinates
        radii: np.array of shape (26) with radius of each circle
        sum_of_radii: Sum of all radii
    """
    # Initialize arrays for 26 circles
    n = 26
    centers = np.zeros((n, 2))

    """Approach: asymmetric graded-radius seeds. The best known n=26
    packings are not four-fold symmetric: they feature graded corner
    radii, unequally spaced edge chains, and/or one dominant circle.
    We generate a small deterministic set of symmetry-broken seeds:
    (a) four corner circles with graded radii and edge chains spaced
    proportionally between them; (b) a dominant circle at (0.2, 0.2)
    with a 5x5 grid of small circles in the remaining area; (c) the
    two-scale mixed seed with a deterministic shear x += 0.03*y. Each
    seed is polished with exactly one bounded SLSQP run; radii are
    always recomputed via compute_max_radii, so validity holds. The
    incumbent hex-lattice result is kept as fallback."""

    def _hex_seed():
        counts = [6, 5, 6, 5, 4]
        dx = 0.16
        dy = 0.2 * np.sqrt(3) / 2
        c = np.zeros((26, 2))
        k = 0
        for row, cnt in enumerate(counts):
            y = 0.1 + row * dy
            x0 = 0.5 - (cnt - 1) * dx / 2
            for col in range(cnt):
                c[k] = [x0 + dx * col, y]
                k += 1
        return c

    def _graded_seed(rcs, re):
        """4 corner circles with graded radii rcs (BL, BR, TR, TL),
        3 edge circles per side (nominal radius re) spaced proportionally
        to the adjacent corner radii, 10 interior circles staggered."""
        c = np.zeros((26, 2))
        k = 0
        corners = [(rcs[0], rcs[0]), (1 - rcs[1], rcs[1]),
                   (1 - rcs[2], 1 - rcs[2]), (rcs[3], 1 - rcs[3])]
        for cx, cy in corners:
            c[k] = [cx, cy]
            k += 1
        # proportional interpolation weights along each edge
        tb = rcs[0] / max(rcs[0] + rcs[1], 1e-9)
        tt = rcs[3] / max(rcs[3] + rcs[2], 1e-9)
        tl = rcs[0] / max(rcs[0] + rcs[3], 1e-9)
        tr = rcs[1] / max(rcs[1] + rcs[2], 1e-9)

        def frac(t):
            return np.array([t * tb, 0.5, (1 - t) + t * tb][:3]) * 0 + \
                np.array([t * tb, 0.5, (1 - t * (1 - tb))])

        # bottom / top edges (x from left corner to right corner)
        for a in range(3):
            f = np.linspace(0, 1, 5)[1:4][a] * 0  # placeholder
            ts = [tb, 0.5 + (tb - 0.5) * 0.0, 1 - (1 - tb)]
            x = rcs[0] + (1 - rcs[0] - rcs[1]) * np.array(
                [tb * 0.5, 0.5, tb + (1 - tb) * 0.5][a])
            c[k] = [x, re]
            k += 1
        for a in range(3):
            x = rcs[3] + (1 - rcs[3] - rcs[2]) * np.array(
                [tt * 0.5, 0.5, tt + (1 - tt) * 0.5][a])
            c[k] = [x, 1 - re]
            k += 1
        for a in range(3):
            y = rcs[0] + (1 - rcs[0] - rcs[3]) * np.array(
                [tl * 0.5, 0.5, tl + (1 - tl) * 0.5][a])
            c[k] = [re, y]
            k += 1
        for a in range(3):
            y = rcs[1] + (1 - rcs[1] - rcs[2]) * np.array(
                [tr * 0.5, 0.5, tr + (1 - tr) * 0.5][a])
            c[k] = [1 - re, y]
            k += 1
        # interior: 10 circles, staggered rows 4/3/3 centered at 0.5
        p = 0.17
        rows = [(4, 0.5 - p), (3, 0.5), (3, 0.5 + p)]
        for cnt, y in rows:
            x0 = 0.5 - (cnt - 1) * p / 2
            for col in range(cnt):
                c[k] = [x0 + p * col, y]
                k += 1
        return c

    def _dominant_seed():
        """One dominant circle at (0.2, 0.2) plus a 5x5 grid of small
        circles (spacing 0.185) shifted to the remaining area."""
        c = np.zeros((26, 2))
        c[0] = [0.2, 0.2]
        k = 1
        for row in range(5):
            for col in range(5):
                c[k] = [0.115 + 0.185 * col, 0.115 + 0.185 * row]
                k += 1
        return c

    def _sheared_seed():
        """Mixed two-scale seed with deterministic shear x += 0.03*y."""
        base = _graded_seed((0.16, 0.16, 0.16, 0.16), 0.10)
        base[:, 0] = base[:, 0] + 0.03 * base[:, 1]
        return np.clip(base, 1e-6, 1 - 1e-6)

    # Incumbent hex result as fallback
    best_c = _hex_seed()
    best_r = compute_max_radii(best_c)
    best_c, best_r = _refine(best_c, best_r)
    best_sum = np.sum(best_r)

    # Multistart: base seeds plus deterministic jitters drawn from an
    # archive of the best distinct refined configurations (multiple basins),
    # each polished with the analytic-Jacobian SLSQP.
    def _transposed_hex_seed():
        """Hex lattice transposed (row counts along a column instead of a
        row): a distinct basin from the standard hex seed that samples the
        orthogonal orientation of the staggered rows."""
        counts = [5, 6, 5, 6, 4]
        dx = 0.16
        dy = 0.2 * np.sqrt(3) / 2
        c = np.zeros((26, 2))
        k = 0
        for row, cnt in enumerate(counts):
            y = 0.9 - row * dy
            x0 = 0.5 - (cnt - 1) * dx / 2 + 0.04  # offset rows
            for col in range(cnt):
                c[k] = [x0 + dx * col, y]
                k += 1
        return c

    def _dominant_corner_seed(cx, cy):
        """Dominant circle at a chosen corner plus a 5x5 grid of small
        circles filling the remaining rectangular area."""
        c = np.zeros((26, 2))
        c[0] = [cx, cy]
        k = 1
        x0 = 0.115 if cx < 0.5 else 1 - (0.115 + 0.185 * 4)
        y0 = 0.115 if cy < 0.5 else 1 - (0.115 + 0.185 * 4)
        for row in range(5):
            for col in range(5):
                c[k] = [x0 + 0.185 * col, y0 + 0.185 * row]
                k += 1
        return c

    def _row_tangency_seed(counts):
        """Exact-tangency row builder. Row k has counts[k] circles, each
        of radius r_k = 1/(2*counts[k]) so the row is mutually tangent
        AND tangent to both side walls (x_j = r_k + 2*r_k*j). The row's
        vertical level y_k is solved in closed form so that the binding
        circle is exactly tangent to its nearest circle in row k-1:
            y_k = y_{k-1} + sqrt((r_k+r_{k-1})^2 - d_min^2),
        where d_min is the smallest horizontal gap between the two rows.
        Every contact is satisfied exactly by construction (no estimated
        lattice constants), so SLSQP starts at a KKT-consistent point."""
        c = np.zeros((26, 2))
        k = 0
        r_prev = None
        y_prev = None
        for m in counts:
            r = 1.0 / (2.0 * m)
            xs = r + 2.0 * r * np.arange(m)
            if r_prev is None:
                y = r
            else:
                d_min = np.min(np.abs(xs[:, None] - x_prev[None, :]))
                disc = (r + r_prev) ** 2 - d_min ** 2
                dy = np.sqrt(max(disc, 1e-12))
                y = y_prev + dy
            for j in range(m):
                c[k] = [xs[j], y]
                k += 1
            x_prev = xs
            r_prev, y_prev = r, y
        return np.clip(c, 1e-6, 1 - 1e-6)

    seeds = [
        _row_tangency_seed([6, 5, 6, 5, 4]),
        _row_tangency_seed([7, 5, 6, 4, 4]),
        _row_tangency_seed([5, 6, 5, 6, 4]),
        _graded_seed((0.19, 0.15, 0.13, 0.11), 0.09),
        _graded_seed((0.11, 0.13, 0.15, 0.19), 0.10),
        _graded_seed((0.17, 0.12, 0.17, 0.12), 0.095),
        _graded_seed((0.20, 0.16, 0.12, 0.10), 0.105),
        _graded_seed((0.14, 0.14, 0.14, 0.14), 0.11),
        _dominant_seed(),
        _dominant_corner_seed(0.8, 0.2),
        _dominant_corner_seed(0.5, 0.5),
        _sheared_seed(),
        _transposed_hex_seed(),
    ]
    rng = np.random.RandomState(12345)
    jitter_magnitudes = [0.004, 0.01, 0.015, 0.02, 0.035, 0.05]
    archive = [(best_sum, best_c.copy())]

    def _update_archive(s, c):
        """Insert a refined config into the archive of distinct basins."""
        for i, (si, ci) in enumerate(archive):
            if np.max(np.abs(ci - c)) < 1e-4:  # same basin
                if s > si:
                    archive[i] = (s, c.copy())
                return
        archive.append((s, c.copy()))
        archive.sort(key=lambda t: -t[0])
        del archive[4:]  # keep top 4 distinct basins

    _update_archive(best_sum, best_c)
    for round_idx in range(90):
        if round_idx < len(seeds):
            c0 = np.asarray(seeds[round_idx], dtype=float)
        else:
            k = (round_idx - len(seeds)) % len(archive)
            base = archive[k][1]
            mag = jitter_magnitudes[(round_idx // len(archive)) % 4]
            c0 = base + rng.uniform(-mag, mag, size=base.shape)
        c0 = np.clip(np.asarray(c0, dtype=float), 1e-6, 1 - 1e-6)
        r0 = compute_max_radii(c0)
        if np.sum(r0) <= 1e-6:
            continue
        c1, r1 = _refine(c0, r0)
        s1 = np.sum(r1)
        _update_archive(s1, c1)
        if s1 > best_sum + 1e-12:
            best_sum = s1
            best_c, best_r = c1.copy(), r1.copy()

    # Final phase: weighted-objective degeneracy sweep. At the incumbent's
    # optimum the sum-radii objective is flat along a degenerate KKT face;
    # re-solving with alternating weight emphases (corner-vs-interior
    # position, small-vs-large radius rank) moves the optimizer to
    # different vertices of that face, each re-polished with the uniform
    # objective. The incumbent is kept unless a vertex strictly improves
    # the unweighted sum.
    best_c, best_r = _refine(best_c, best_r)
    best_sum = np.sum(best_r)

    def _corner_weight(c_arr):
        """Emphasize circles near corners (low min(x, 1-x)+min(y, 1-y))."""
        d = np.minimum(c_arr[:, 0], 1 - c_arr[:, 0]) + \
            np.minimum(c_arr[:, 1], 1 - c_arr[:, 1])
        w = 1.0 + np.exp(-d / 0.25)
        return w / np.sum(w) * 26

    def _edge_weight(c_arr):
        """Emphasize circles near the square boundary."""
        d = np.minimum(np.minimum(c_arr[:, 0], 1 - c_arr[:, 0]),
                       np.minimum(c_arr[:, 1], 1 - c_arr[:, 1]))
        w = 1.0 + np.exp(-d / 0.08)
        return w / np.sum(w) * 26

    def _small_rank_weight(r_arr):
        """Double the weight of the smaller half of the radii."""
        order = np.argsort(r_arr)
        w = np.ones(26)
        w[order[:13]] = 2.0
        return w / np.sum(w) * 26

    def _large_rank_weight(r_arr):
        """Double the weight of the larger half of the radii."""
        order = np.argsort(r_arr)
        w = np.ones(26)
        w[order[13:]] = 2.0
        return w / np.sum(w) * 26

    weight_generators = [
        _corner_weight, _small_rank_weight,
        _edge_weight, _large_rank_weight,
        _corner_weight, _small_rank_weight,
        _edge_weight, _large_rank_weight,
    ]
    for wg in weight_generators:
        cw = best_c.copy()
        rw = best_r.copy()
        try:
            wc, wr = _refine(cw, rw, weights=wg(cw))
        except Exception:
            continue
        # one uniform re-polish from the weighted vertex
        wc, wr = _refine(np.asarray(wc, dtype=float),
                         np.asarray(wr, dtype=float))
        ws = np.sum(wr)
        if ws > best_sum + 1e-12:
            best_sum = ws
            best_c, best_r = np.asarray(wc, dtype=float).copy(), wr.copy()

    # Final residual polish: one more uniform SLSQP pass on the overall
    # winner (the weighted sweep may end on a vertex slightly below the
    # uniform optimum); accept only strict improvements.
    best_c, best_r = _refine(np.asarray(best_c, dtype=float),
                             np.asarray(best_r, dtype=float))
    best_sum = np.sum(best_r)

    # Safety shrink: pull every radius marginally inside the constraint
    # set. compute_max_radii's pairwise loop already converges to exact
    # feasibility (r_i + r_j <= dist, r <= wall), so only an epsilon-scale
    # guard is needed; shrinking by 5e-7 per circle surrendered ~1.3e-5 of
    # achievable sum for no validity benefit.
    best_r = np.maximum(best_r - 1e-9, 1e-9)
    centers, radii = best_c, best_r
    sum_radii = np.sum(radii)

    return centers, radii, sum_radii


def compute_max_radii(centers):
    """
    Compute the maximum possible radii for each circle position
    such that they don't overlap and stay within the unit square.

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates

    Returns:
        np.array of shape (n) with radius of each circle
    """
    n = centers.shape[0]
    radii = np.ones(n)

    # Limit by distance to square borders
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(x, y, 1 - x, 1 - y)

    # Iteratively enforce pairwise non-overlap until convergence so that
    # shrinkage from one pair propagates correctly to all other pairs
    for _ in range(200):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                if radii[i] + radii[j] > dist:
                    # Shrink the larger radius first (keeps small circles
                    # small and preserves the intended size hierarchy)
                    excess = radii[i] + radii[j] - dist
                    if radii[i] >= radii[j]:
                        radii[i] = max(radii[i] - excess, 1e-12)
                    else:
                        radii[j] = max(radii[j] - excess, 1e-12)
                    changed = True
        if not changed:
            break

    return radii


def _refine(centers, radii, weights=None):
    """Locally optimize centers and radii with SLSQP starting from a
    feasible configuration, maximizing sum(weights * radii) (uniform
    sum of radii when weights is None) subject to wall constraints
    (r <= min(x, y, 1-x, 1-y)) and pairwise non-overlap
    (r_i + r_j <= distance). The optimizer's radii are discarded; radii
    are recomputed from the refined centers via compute_max_radii so the
    returned configuration is always valid. Falls back to the input if
    scipy is unavailable or no strict improvement is found."""
    try:
        from scipy.optimize import minimize
    except ImportError:
        return centers, radii

    """Analytic-Jacobian SLSQP polish. Variables z = (x0,y0,...,r0,r1,...).
    Wall constraints (4 per circle) are linear; pair constraints use the
    exact 2-term gradient of the Euclidean distance. Analytic jac removes
    the finite-difference noise floor of the previous version."""
    n = len(centers)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    if weights is None:
        w = np.ones(n)
    else:
        w = np.asarray(weights, dtype=float)
        w = w / np.sum(w) * n  # normalize so total emphasis is comparable

    def obj(z):
        return -np.dot(w, z[2 * n:])

    def obj_jac(z):
        g = np.zeros_like(z)
        g[2 * n:] = -w
        return g

    cons = []
    for i in range(n):
        def wfun(z, i=i):
            return np.array([z[2 * i], z[2 * i + 1],
                             1 - z[2 * i], 1 - z[2 * i + 1]]) - z[2 * n + i]

        def wjac(z, i=i):
            g = np.zeros((4, z.size))
            g[0, 2 * i] = 1.0
            g[1, 2 * i + 1] = 1.0
            g[2, 2 * i] = -1.0
            g[3, 2 * i + 1] = -1.0
            g[:, 2 * n + i] = -1.0
            return g
        cons.append({"type": "ineq", "fun": wfun, "jac": wjac})

    for (i, j) in pairs:
        def pfun(z, i=i, j=j):
            return (np.sqrt(np.sum((z[2 * i:2 * i + 2] - z[2 * j:2 * j + 2]) ** 2))
                    - z[2 * n + i] - z[2 * n + j])

        def pjac(z, i=i, j=j):
            d = z[2 * i:2 * i + 2] - z[2 * j:2 * j + 2]
            dist = np.sqrt(np.sum(d ** 2)) + 1e-300
            g = np.zeros(z.size)
            g[2 * i:2 * i + 2] = d / dist
            g[2 * j:2 * j + 2] = -d / dist
            g[2 * n + i] -= 1.0
            g[2 * n + j] -= 1.0
            return g
        cons.append({"type": "ineq", "fun": pfun, "jac": pjac})

    z0 = np.concatenate([centers.ravel(), radii])
    try:
        res = minimize(obj, z0, method="SLSQP", jac=obj_jac,
                       constraints=cons,
                       options={"maxiter": 400, "ftol": 1e-12})
    except Exception:
        return centers, radii

    c2 = np.asarray(res.x[:2 * n], dtype=float).reshape(n, 2).copy()
    if np.all(c2 > 1e-9) and np.all(c2 < 1 - 1e-9):
        r2 = compute_max_radii(c2)
        if np.sum(r2) > np.sum(radii) + 1e-12:
            return c2, r2
    return centers, radii


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii


def visualize(centers, radii):
    """
    Visualize the circle packing

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radius of each circle
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw unit square
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True)

    # Draw circles
    for i, (center, radius) in enumerate(zip(centers, radii)):
        circle = Circle(center, radius, alpha=0.5)
        ax.add_patch(circle)
        ax.text(center[0], center[1], str(i), ha="center", va="center")

    plt.title(f"Circle Packing (n={len(centers)}, sum={sum(radii):.6f})")
    plt.show()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii}")
    # AlphaEvolve improved this to 2.635

    # Uncomment to visualize:
    visualize(centers, radii)
