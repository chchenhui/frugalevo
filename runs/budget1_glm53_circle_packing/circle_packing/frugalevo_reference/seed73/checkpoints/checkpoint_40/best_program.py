# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def _greedy_forward_build(grid=24):
    """Forward greedy constructor: place 26 circles one at a time.

    At each step, every grid point (plus wall-hugging variants) is scored by
    the total LP radius sum of the partial packing after insertion there
    (compute_max_radii), not by the new circle's radius alone. This
    sum-maximizing criterion naturally yields large boundary circles and
    smaller interior fillers -- a size-differentiated topology a single hex
    seed cannot reach. Deterministic, strictly feasible at every step.
    """
    centers = np.zeros((0, 2), dtype=np.float64)
    ts = np.linspace(0.0, 1.0, grid)
    for _ in range(26):
        best_sum, best_p = -1.0, np.array([0.5, 0.5])
        for gy in ts:
            for gx in ts:
                p = np.array([gx, gy], dtype=np.float64)
                cand = np.vstack([centers, p[None, :]])
                r = compute_max_radii(cand)
                if not np.all(r > 0):
                    continue
                s = float(np.sum(r))
                if s > best_sum:
                    best_sum, best_p = s, p
        centers = np.vstack([centers, best_p[None, :]])
    return centers


def _hex_lattice_start():
    """Hexagonal-lattice starting configuration (rows 6,5,6,5,4 = 26).

    Strictly feasible: centers spaced 1/6 apart horizontally with
    sqrt(3)/12 vertical offset, so LP radii are strictly positive.
    """
    n = 26
    centers = np.zeros((n, 2), dtype=np.float64)
    rows = [6, 5, 6, 5, 4]
    dx = 1.0 / 6.0
    dy = np.sqrt(3.0) / 12.0
    idx = 0
    for k, cnt in enumerate(rows):
        y = 0.5 + (k - 2) * dy
        x0 = 0.5 - (cnt - 1) * dx / 2.0
        for j in range(cnt):
            centers[idx, 0] = x0 + j * dx
            centers[idx, 1] = y
            idx += 1
    return centers


def compute_max_radii(centers):
    """LP radii: r_i = min(wall distance, half min pairwise distance)."""
    n = centers.shape[0]
    radii = np.empty(n, dtype=np.float64)
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(x, y, 1.0 - x, 1.0 - y)
    for i in range(n):
        for j in range(i + 1, n):
            d = 0.5 * np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if d < radii[i]:
                radii[i] = d
            if d < radii[j]:
                radii[j] = d
    return radii


def _refine_slsqp(centers, radii, maxiter=300, time_limit=60.0):
    """Joint SLSQP refinement over (cx, cy, r) in R^78.

    Minimizes -sum(r) subject to squared-form non-overlap constraints
    (ci-cj)^2 - (ri+rj)^2 >= 0 (325 pairs) and wall constraints
    c - r >= 0, 1 - c - r >= 0 per axis, all with analytic Jacobians.
    Returns (centers, radii) or None on failure/timeout.
    """
    import time as _time
    from scipy.optimize import minimize

    t0 = _time.time()
    n = 26
    idx = np.triu_indices(n, k=1)
    npairs = len(idx[0])
    i_idx, j_idx = idx

    def unpack(x):
        return x[0:2 * n].reshape(n, 2), x[2 * n:]

    def obj(x):
        return -np.sum(x[2 * n:])

    def obj_jac(x):
        g = np.zeros(3 * n)
        g[2 * n:] = -1.0
        return g

    def cons_f(x):
        c, r = unpack(x)
        d = c[i_idx] - c[j_idx]
        sq = d[:, 0] ** 2 + d[:, 1] ** 2
        s = r[i_idx] + r[j_idx]
        walls = np.concatenate([c[:, 0] - r, c[:, 1] - r,
                                1.0 - c[:, 0] - r, 1.0 - c[:, 1] - r])
        return np.concatenate([sq - s ** 2, walls])

    def cons_jac(x):
        c, r = unpack(x)
        J = np.zeros((npairs + 4 * n, 3 * n))
        dx = c[i_idx, 0] - c[j_idx, 0]
        dy = c[i_idx, 1] - c[j_idx, 1]
        s = r[i_idx] + r[j_idx]
        for k in range(npairs):
            i, j = i_idx[k], j_idx[k]
            J[k, 2 * i] = 2 * dx[k]
            J[k, 2 * i + 1] = 2 * dy[k]
            J[k, 2 * j] = -2 * dx[k]
            J[k, 2 * j + 1] = -2 * dy[k]
            J[k, 2 * n + i] = -2 * s[k]
            J[k, 2 * n + j] = -2 * s[k]
        off = npairs
        for a in range(n):
            J[off + 0 * n + a, 2 * a] = 1.0
            J[off + 0 * n + a, 2 * n + a] = -1.0
            J[off + 1 * n + a, 2 * a + 1] = 1.0
            J[off + 1 * n + a, 2 * n + a] = -1.0
            J[off + 2 * n + a, 2 * a] = -1.0
            J[off + 2 * n + a, 2 * n + a] = -1.0
            J[off + 3 * n + a, 2 * a + 1] = -1.0
            J[off + 3 * n + a, 2 * n + a] = -1.0
        return J

    x0 = np.concatenate([centers.reshape(-1), radii])
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 1.0)] * n
    try:
        res = minimize(obj, x0, jac=obj_jac, bounds=bounds,
                       constraints=[{"type": "ineq",
                                     "fun": cons_f, "jac": cons_jac}],
                       method="SLSQP", options={"maxiter": maxiter, "ftol": 1e-12})
        if _time.time() - t0 > time_limit:
            return None
        rc, rr = unpack(res.x)
        if not np.all(np.isfinite(rr)) or np.any(rr <= 0):
            return None
        return rc.copy(), rr.copy()
    except Exception:
        return None


def _largest_empty_point(others, grid=20):
    """Largest-empty-circle center in the unit square avoiding 25 disks.

    Grid scan (grid x grid) on wall-distance objective, then local
    Nelder-Mead polish of f(p) = min(wall dist, dist to nearest center).
    Returns the 2D point maximizing the LP radius a new circle could have.
    """
    from scipy.optimize import minimize

    def gap(p):
        wx = min(p[0], p[1], 1.0 - p[0], 1.0 - p[1])
        if len(others) == 0:
            return wx
        d = np.sqrt(((others - p) ** 2).sum(axis=1))
        return min(wx, float(d.min()))

    best_p, best_v = np.array([0.5, 0.5]), -1.0
    ts = np.linspace(1e-6, 1.0 - 1e-6, grid)
    for gy in ts:
        for gx in ts:
            v = gap((gx, gy))
            if v > best_v:
                best_v, best_p = v, np.array([gx, gy])
    res = minimize(lambda p: -gap(p), best_p, method="Nelder-Mead",
                   options={"maxiter": 200, "xatol": 1e-9, "fatol": 1e-12})
    if -res.fun > best_v:
        best_p = res.x
    return np.clip(best_p, 1e-9, 1.0 - 1e-9)


def construct_packing():
    """Hex-lattice LP start + joint SLSQP center/radius refinement.

    Returns the better (by sum of radii) of the LP solution and the
    SLSQP-refined solution; radii shrunk by 2e-7 for tolerance safety.
    """
    n = 26
    # Incumbent fallback: hex lattice, LP radii.
    hex_centers = _hex_lattice_start()
    hex_radii = compute_max_radii(hex_centers)
    best_sum = float(np.sum(hex_radii))
    centers, radii = hex_centers, hex_radii

    # New mechanism: forward greedy build -- each circle placed at the grid
    # position maximizing the total LP sum of the partial packing. One
    # SLSQP polish of the greedy result (single start, bounded iterations).
    greedy = _greedy_forward_build(grid=24)
    gr = compute_max_radii(greedy)
    if np.all(gr > 0):
        g_result = _refine_slsqp(greedy, gr, maxiter=500, time_limit=60.0)
        if g_result is not None:
            gc, grr = g_result
            s = float(np.sum(grr))
            if s > best_sum:
                best_sum = s
                centers, radii = gc, grr
    # LP radii at the greedy centers can also beat the incumbent directly.
    g_lp = compute_max_radii(greedy)
    if np.all(g_lp > 0) and float(np.sum(g_lp)) > best_sum:
        best_sum = float(np.sum(g_lp))
        centers, radii = greedy, g_lp

    result = _refine_slsqp(centers, radii)
    if result is not None:
        rc, rr = result
        ref_sum = float(np.sum(rr))
        if ref_sum > best_sum:
            centers, radii = rc, rr
            best_sum = ref_sum

    # Drop-reinsert sweep in ascending-radius order: small circles are the
    # ones typically trapped between larger neighbors, so relocating them
    # first (to the largest empty region, which is usually a corner gap)
    # maximizes the chance each accepted swap improves the sum. For each
    # circle i: delete it, run one short SLSQP on the remaining 25, then
    # try two reinsertion points (largest-empty point plus, if it differs,
    # the best of the four corners), running one SLSQP on all 26 per
    # candidate. Accept the best strictly-improving swap only. Bounded:
    # <= 26 trials x (5 s + 2 x 6 s) with a hard 90 s wall-clock guard.
    import time as _time
    t_start = _time.time()
    order = np.argsort(radii)
    for i in order:
        if _time.time() - t_start > 90.0:
            break
        keep = np.ones(n, dtype=bool)
        keep[i] = False
        others = centers[keep].copy()
        out25 = _refine_slsqp(others, compute_max_radii(others),
                              maxiter=120, time_limit=4.0)
        if out25 is not None:
            others = out25[0]
        spots = [_largest_empty_point(others)]
        corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        d = np.sqrt(((corners[:, None, :] - others[None, :, :]) ** 2).sum(axis=2))
        corner_gap = np.minimum(np.minimum(corners, 1.0 - corners).min(axis=1),
                                d.min(axis=1))
        spots.append(corners[int(np.argmax(corner_gap))])
        best_local = -1.0
        best_pack = None
        for p_new in spots:
            cand = np.vstack([others, np.asarray(p_new, dtype=np.float64)[None, :]])
            cand_radii = compute_max_radii(cand)
            if not np.all(cand_radii > 0):
                continue
            out26 = _refine_slsqp(cand, cand_radii, maxiter=250, time_limit=6.0)
            cc, cr = out26 if out26 is not None else (cand, cand_radii)
            s = float(np.sum(cr))
            if s > best_local and np.all(cr > 0):
                best_local, best_pack = s, (cc, cr)
        if best_pack is not None and best_local > best_sum + 1e-12:
            best_sum, centers, radii = best_local, best_pack[0], best_pack[1]

    # Free check: greedy LP radii at the refined centers can exceed the
    # SLSQP radii sum; keep whichever is larger (both are feasible).
    lp_radii = compute_max_radii(centers)
    if float(np.sum(lp_radii)) > best_sum:
        radii = lp_radii

    # Shrink slightly and clamp so evaluator tolerance is never consumed
    radii = np.maximum(radii - 2e-7, 0.0)
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        centers[i, 0] = min(max(x, r + 1e-12), 1.0 - r - 1e-12)
        centers[i, 1] = min(max(y, r + 1e-12), 1.0 - r - 1e-12)

    sum_radii = float(np.sum(radii))
    return centers, radii, sum_radii


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
