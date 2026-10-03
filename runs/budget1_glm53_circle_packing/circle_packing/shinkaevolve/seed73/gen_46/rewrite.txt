# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles."""
import numpy as np


def _row_layout(counts, r):
    """Hex-row layout: rows with counts, spacing dx=2r, dy=sqrt(3)r."""
    sqrt3 = np.sqrt(3.0)
    dx, dy = 2.0 * r, sqrt3 * r
    pts = []
    for row, c in enumerate(counts):
        y = r + row * dy
        span = 2.0 * r + (c - 1) * dx
        start = (1.0 - span) / 2.0 + r
        for k in range(c):
            pts.append([start + k * dx, y])
    return np.array(pts)


def _row_layout_walls(counts, r):
    """Row layout pinned to the left wall (row 1 circle touches x=0)."""
    sqrt3 = np.sqrt(3.0)
    dx, dy = 2.0 * r, sqrt3 * r
    pts = []
    for row, c in enumerate(counts):
        y = r + row * dy
        for k in range(c):
            pts.append([r + k * dx, y])
    return np.array(pts)


def construct_packing():
    sqrt3 = np.sqrt(3.0)
    r0 = min(0.1, 1.0 / (2.0 + 5.0 * sqrt3))

    count_sets = [[5, 4, 5, 4, 5, 3], [5, 4, 5, 4, 4, 4],
                  [4, 5, 4, 5, 4, 4], [3, 5, 4, 5, 4, 5],
                  [5, 5, 4, 5, 4, 3], [4, 4, 5, 4, 5, 4],
                  [5, 4, 5, 4, 5], [6, 5, 4, 5, 6]]
    stretch_factors = [1.0, 1.03, 1.06]

    seeds = []
    for counts in count_sets:
        for sf in stretch_factors:
            r = r0 * sf
            seeds.append(_row_layout(counts, r))
            seeds.append(_row_layout_walls(counts, r))

    # Deterministic jittered variants for extra diversity
    rng = np.random.default_rng(12345)
    candidates = list(seeds)
    for base in seeds:
        for _ in range(2):
            cand = base + rng.normal(0.0, 0.015, size=base.shape)
            candidates.append(np.clip(cand, 0.005, 0.995))

    results = []
    for cand in candidates:
        centers, radii, s = refine_layout(cand, r0)
        results.append((s, centers, radii))
    results.sort(key=lambda t: -t[0])

    # Multi-start SLSQP polish: refine the top-k distinct basins and
    # keep the overall best. Different count sets converge to
    # genuinely different local optima; polishing several of them
    # greatly increases the chance of finding the near-global one.
    best = (results[0][1], results[0][2], results[0][0])
    seen = []
    polished = 0
    for s0, c0, r0v in results:
        if polished >= 5:
            break
        # skip basins too similar to one already polished
        if any(np.linalg.norm(c0 - cc) < 0.05 for cc in seen):
            continue
        seen.append(c0)
        polished += 1
        out = slsqp_refine(c0, max_iters=500)
        if out is not None and out[0].shape[0] == 26 and out[2] > best[2]:
            best = out
        # one extra restart pass from the polished endpoint
        out2 = slsqp_refine(out[0], max_iters=500) if out is not None else None
        if out2 is not None and out2[0].shape[0] == 26 and out2[2] > best[2]:
            best = out2

    centers, radii, sum_radii = best
    return centers, radii, sum_radii


def grow_radii(centers, iters=400):
    """Exact Gauss-Seidel growth: r_i = min(wall_i, min_j (d_ij - r_j))."""
    n = centers.shape[0]
    b = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                   np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    d = np.sqrt(((centers[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(d, np.inf)
    radii = np.zeros(n)
    for _ in range(iters):
        # Vectorized sweep: candidate radius for every circle at once.
        cand = np.minimum(b, np.min(d - radii[None, :], axis=1))
        cand = np.maximum(cand, 1e-9)
        grew_mask = cand > radii + 1e-12
        if not grew_mask.any():
            break
        # Gauss-Seidel flavor: apply updates circle-by-circle so that
        # later circles in the sweep see the already-updated radii.
        for i in np.flatnonzero(grew_mask):
            ci = min(b[i], float(np.min(d[i, :] - radii)))
            ci = max(ci, 1e-9)
            if ci > radii[i] + 1e-12:
                radii[i] = ci
    return radii


def refine_layout(centers, r0, iters=60):
    """Nudge centers toward free space, keeping sum-improving configs."""
    centers = centers.copy()
    radii = grow_radii(centers)
    best_centers = centers.copy()
    best_radii = radii.copy()
    best_sum = float(np.sum(radii))

    for t in range(iters):
        step = 0.02 * (1.0 - t / iters)
        d = np.sqrt(((centers[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))
        np.fill_diagonal(d, np.inf)
        new_centers = centers.copy()
        for i in range(centers.shape[0]):
            j = int(np.argmin(d[i]))
            vec = centers[i] - centers[j]
            norm = np.linalg.norm(vec)
            if norm > 1e-9:
                new_centers[i] += step * vec / norm
            wx = min(centers[i, 0], 1.0 - centers[i, 0])
            wy = min(centers[i, 1], 1.0 - centers[i, 1])
            if wx < wy:
                towall = -centers[i, 0] if centers[i, 0] < 0.5 else 1.0 - centers[i, 0]
                new_centers[i, 0] += 0.5 * step * towall
            else:
                towall = -centers[i, 1] if centers[i, 1] < 0.5 else 1.0 - centers[i, 1]
                new_centers[i, 1] += 0.5 * step * towall
        new_centers = np.clip(new_centers, 0.005, 0.995)
        new_radii = grow_radii(new_centers)
        new_sum = float(np.sum(new_radii))
        if new_sum > best_sum:
            best_sum = new_sum
            best_centers = new_centers.copy()
            best_radii = new_radii.copy()
            centers = new_centers
        elif t < iters // 2:
            centers = new_centers
    return best_centers, best_radii, best_sum


def slsqp_refine(centers, max_iters=500):
    """
    Continuous refinement of (centers, radii) with SLSQP.
    Variables: [x_i, y_i, r_i]. Maximize sum(r) subject to
    r_i + r_j <= |c_i - c_j| and r_i <= wall distances.
    The result is repaired via exact growth so it is always valid.
    Returns (centers, radii, sum) or None.
    """
    try:
        from scipy.optimize import minimize
    except Exception:
        return None
    n = centers.shape[0]
    # Feasible starting radii: exact growth shrunk slightly
    r = grow_radii(centers) * 0.98
    d = np.sqrt(((centers[:, None] - centers[None, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    wall = np.minimum(np.minimum(centers[:, 0], 1 - centers[:, 0]),
                      np.minimum(centers[:, 1], 1 - centers[:, 1]))
    r = np.minimum(np.maximum(r, 1e-9), np.maximum(wall * 0.98, 1e-9))
    for i in range(n):
        r[i] = max(min(r[i], float(np.min(d[i] - r)) - 1e-9), 1e-9)

    x0 = np.concatenate([centers.ravel(), r])
    m2 = 2 * n

    def unpack(z):
        return z[:m2].reshape(n, 2), z[m2:]

    ii, jj = np.triu_indices(n, k=1)

    def obj(z):
        return -float(np.sum(z[m2:]))

    def obj_grad(z):
        g = np.zeros_like(z)
        g[m2:] = -1.0
        return g

    def pair_fun(z):
        c, rr = unpack(z)
        dvec = np.sqrt(((c[ii] - c[jj]) ** 2).sum(axis=1))
        return dvec - rr[ii] - rr[jj]

    def pair_jac(z):
        c, rr = unpack(z)
        dvec = np.sqrt(((c[ii] - c[jj]) ** 2).sum(axis=1))
        dvec = np.maximum(dvec, 1e-12)
        J = np.zeros((ii.size, x0.size))
        diff = (c[ii] - c[jj]) / dvec[:, None]
        rows = np.arange(ii.size)
        J[rows, 2 * ii] = diff[:, 0]
        J[rows, 2 * jj] = -diff[:, 0]
        J[rows, 2 * ii + 1] = diff[:, 1]
        J[rows, 2 * jj + 1] = -diff[:, 1]
        J[rows, m2 + ii] = -1.0
        J[rows, m2 + jj] = -1.0
        return J

    def wall_fun(z):
        c, rr = unpack(z)
        return np.concatenate([c[:, 0] - rr, 1 - c[:, 0] - rr,
                               c[:, 1] - rr, 1 - c[:, 1] - rr])

    def wall_jac(z):
        J = np.zeros((4 * n, x0.size))
        for i in range(n):
            J[i, 2 * i] = 1.0
            J[i, m2 + i] = -1.0
            J[n + i, 2 * i] = -1.0
            J[n + i, m2 + i] = -1.0
            J[2 * n + i, 2 * i + 1] = 1.0
            J[2 * n + i, m2 + i] = -1.0
            J[3 * n + i, 2 * i + 1] = -1.0
            J[3 * n + i, m2 + i] = -1.0
        return J

    cons = [{"type": "ineq", "fun": pair_fun, "jac": pair_jac},
            {"type": "ineq", "fun": wall_fun, "jac": wall_jac}]
    bounds = [(0.001, 0.999)] * m2 + [(1e-6, 0.5)] * n

    try:
        res = minimize(obj, x0, jac=obj_grad, bounds=bounds, constraints=cons,
                       method="SLSQP",
                       options={"maxiter": max_iters, "ftol": 1e-14})
        c, _ = unpack(res.x)
    except Exception:
        return None
    c = np.clip(c, 0.001, 0.999)
    rr = grow_radii(c)
    return c, rr, float(np.sum(rr))


def compute_max_radii(centers):
    n = centers.shape[0]
    radii = np.array([min(c[0], c[1], 1 - c[0], 1 - c[1]) for c in centers])
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            if radii[i] + radii[j] > dist:
                s = dist / (radii[i] + radii[j])
                radii[i] *= s
                radii[j] *= s
    return radii


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