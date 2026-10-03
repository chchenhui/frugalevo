# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def _lattice_shift_sweep():
    """Diagonal-lattice-shift sweep over the hex core.

    Translates the centered hex lattice by (dx, dy) over a 7x7 grid of
    offsets in [-0.03, 0.03], computes LP radii (compute_max_radii) for
    each shifted seed, and returns the list of (sum, centers) pairs sorted
    by LP sum descending. Shifting changes which circles contact which
    walls, giving the SLSQP polish a different active set than the
    symmetric centered seed. All shifts keep centers >= 0.05 from walls,
    so seeds remain strictly feasible.
    """
    base = _hex_lattice_start()
    offsets = [-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03]
    scored = []
    for dx in offsets:
        for dy in offsets:
            shifted = base + np.array([dx, dy], dtype=np.float64)
            r = compute_max_radii(shifted)
            if np.all(r > 0):
                scored.append((float(np.sum(r)), shifted))
    scored.sort(key=lambda t: -t[0])
    return scored


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


def _kkt_growth_loop(centers, radii, rounds=4):
    """Iterative contact-network linear solve (contact-network-linear-solve).

    Unlike a single-shot KKT step (which is a no-op at a saturated
    incumbent), this loop repeats the active-set saturation so each round
    detects the *new* contact network of the moved point:

    Per round:
      1. Detect near-active constraints: pair contacts with
         |slack| = |(r_i+r_j) - d_ij| < 3e-6, wall contacts with
         |wall_d - r_i| < 3e-6.
      2. Build one weighted least-squares system over all 78 unknowns
         (dcx, dcy, dr): active rows close the residual slack exactly
         (weight 1.0); uniform growth rows dr_i = g push every radius up
         through the network's free directions (weight 0.05, g decaying
         0.004 -> 0.001 across rounds).
      3. Take the largest damped step t in {1, 0.7, 0.5, 0.3, 0.2, 0.1}
         whose moved centers keep all true LP radii strictly positive and
         strictly increase the LP sum; accept and continue to the next
         round from the new point.
    Returns the best (centers, LP radii) found, or the inputs unchanged.
    Bounded: rounds x 1 lstsq (<= 104 rows x 78 cols) + 6 LP checks each.
    """
    n = 26
    c = centers.copy()
    r = radii.copy()
    best_c, best_r = c.copy(), r.copy()
    best_sum = float(np.sum(r))
    growth = (0.004, 0.003, 0.002, 0.001)
    for rnd in range(min(rounds, len(growth))):
        rows, rhs, wts = [], [], []
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt(np.sum((c[i] - c[j]) ** 2))
                slack = (r[i] + r[j]) - d
                if abs(slack) < 3e-6:
                    u = (c[i] - c[j]) / max(d, 1e-15)
                    row = np.zeros(3 * n)
                    row[2 * i:2 * i + 2] = u
                    row[2 * j:2 * j + 2] = -u
                    row[2 * n + i] = -1.0
                    row[2 * n + j] = -1.0
                    rows.append(row)
                    rhs.append(slack)
                    wts.append(1.0)
        for i in range(n):
            wall_d = min(c[i, 0], c[i, 1], 1.0 - c[i, 0], 1.0 - c[i, 1])
            if abs(wall_d - r[i]) < 3e-6:
                if c[i, 0] == wall_d:
                    normal, const = np.array([1.0, 0.0]), 0.0
                elif 1.0 - c[i, 0] == wall_d:
                    normal, const = np.array([-1.0, 0.0]), 1.0
                elif c[i, 1] == wall_d:
                    normal, const = np.array([0.0, 1.0]), 0.0
                else:
                    normal, const = np.array([0.0, -1.0]), 1.0
                row = np.zeros(3 * n)
                row[2 * i] = normal[0]
                row[2 * i + 1] = normal[1]
                row[2 * n + i] = -1.0
                rows.append(row)
                rhs.append(r[i] - float(normal @ c[i]))
                wts.append(1.0)
        if not rows:
            break
        g = growth[rnd]
        for i in range(n):
            row = np.zeros(3 * n)
            row[2 * n + i] = 1.0
            rows.append(row)
            rhs.append(g)
            wts.append(0.05)
        W = np.array(wts, dtype=np.float64)[:, None]
        A = np.array(rows, dtype=np.float64) * W
        b = np.array(rhs, dtype=np.float64) * W[:, 0]
        dx, *_ = np.linalg.lstsq(A, b, rcond=None)
        x0 = np.concatenate([c.reshape(-1), r])
        moved = False
        for t in (1.0, 0.7, 0.5, 0.3, 0.2, 0.1):
            xc = x0 + t * dx
            cc = xc[0:2 * n].reshape(n, 2).copy()
            rr = xc[2 * n:].copy()
            if np.any(rr <= 0) or np.any(cc <= 0) or np.any(cc >= 1.0):
                continue
            lp = compute_max_radii(cc)
            if np.all(lp > 0) and float(np.sum(lp)) > float(np.sum(r)) + 1e-12:
                c, r = cc, lp
                moved = True
                break
        if not moved:
            break
        cur = float(np.sum(r))
        if cur > best_sum:
            best_sum = cur
            best_c, best_r = c.copy(), r.copy()
    return best_c, best_r


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

    # Diagonal-lattice-shift mechanism: sweep 49 translations of the hex
    # core (LP-sum evaluation only), accept the best LP seed, then run a
    # fine local LP sweep (5x5, +/-0.005/0.0025) around the best coarse
    # shift to sharpen the lattice-to-wall offset. Polish the top 4 seeds
    # of the combined ranking with bounded SLSQP; accept only improvements.
    sweep = _lattice_shift_sweep()
    for shift_sum, shifted in sweep:
        if shift_sum > best_sum:
            best_sum = shift_sum
            centers, radii = shifted.copy(), compute_max_radii(shifted)
    # Fine sweep around the best coarse shift (LP evaluation only, cheap).
    bdx, bdy = centers[0] - _hex_lattice_start()[0]
    fine = []
    for ddx in (-0.005, -0.0025, 0.0, 0.0025, 0.005):
        for ddy in (-0.005, -0.0025, 0.0, 0.0025, 0.005):
            cand = _hex_lattice_start() + np.array(
                [bdx + ddx, bdy + ddy], dtype=np.float64)
            r = compute_max_radii(cand)
            if np.all(r > 0):
                fine.append((float(np.sum(r)), cand))
    fine.sort(key=lambda t: -t[0])
    ranked = sorted(sweep + fine, key=lambda t: -t[0])
    polished = []
    for _, shifted in ranked[:4]:
        sr = compute_max_radii(shifted)
        res = _refine_slsqp(shifted, sr, maxiter=400, time_limit=15.0)
        if res is not None:
            polished.append(res)
    for rc, rr in polished:
        s = float(np.sum(rr))
        if s > best_sum:
            best_sum = s
            centers, radii = rc, rr

    # Contact-network-linear-solve, applied at two points:
    # (a) at the best sweep seed, before the drop-reinsert loop, so the
    # downstream SLSQP/drop-reinsert machinery starts from a different
    # basin than the plain LP seed; (b) replacing the final unbounded
    # polish, followed by one short safety SLSQP (maxiter=100).
    gc, gr = _kkt_growth_loop(centers, radii)
    gs = float(np.sum(gr))
    if gs > best_sum:
        best_sum, centers, radii = gs, gc, gr
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

    # Final stage: iterative KKT growth on the incumbent's active contact
    # network (each round re-detects the active set of the moved point, so
    # the loop makes real progress where a single-shot solve is a no-op),
    # then one short safety SLSQP polish (maxiter=100, 10 s cap).
    gc2, gr2 = _kkt_growth_loop(centers, radii)
    gs2 = float(np.sum(gr2))
    if gs2 > best_sum:
        best_sum, centers, radii = gs2, gc2, gr2
    safety = _refine_slsqp(centers, radii, maxiter=100, time_limit=10.0)
    if safety is not None:
        rc, rr = safety
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
