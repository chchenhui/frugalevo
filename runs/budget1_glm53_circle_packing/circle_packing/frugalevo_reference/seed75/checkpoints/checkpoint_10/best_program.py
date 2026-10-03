# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def _pair_dist(centers):
    """Pairwise distance matrix of an (n,2) array of centers."""
    diff = centers[:, None, :] - centers[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))


def _init_radii(centers):
    """Feasible starting radii: wall distance capped by half nearest-neighbor gap."""
    n = centers.shape[0]
    x, y = centers[:, 0], centers[:, 1]
    r = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
    dist = _pair_dist(centers)
    np.fill_diagonal(dist, np.inf)
    r = np.minimum(r, dist.min(axis=1) / 2.0)
    return r * 0.999


def _constraint_vals(centers, radii):
    """Concatenated inequality constraints (all >= 0): wall clearances and pairwise gaps."""
    x, y = centers[:, 0], centers[:, 1]
    walls = np.concatenate([x - radii, y - radii, (1 - x) - radii, (1 - y) - radii])
    dist = _pair_dist(centers)
    iu = np.triu_indices(centers.shape[0], 1)
    gaps = dist[iu] - radii[iu[0]] - radii[iu[1]]
    return np.concatenate([walls, gaps])


def _expand_symmetric(p):
    """Expand 6 shape parameters into 25 D4-symmetric centers.

    p = (a, b, c, d, g, h):
      center singleton (0.5, 0.5);
      two general orbits of 8 from (a, b) and (c, d);
      one diagonal orbit of 4 from (g, g);
      one edge orbit of 4 from (0.5, h).
    Exact D4 symmetry cannot tile 26 circles (orbit sizes are 1, 4, 8),
    so the skeleton has 25; the 26th is added in the symmetry-breaking stage.
    """
    a, b, c, d, g, h = p
    pts = [(0.5, 0.5)]
    for (u, v) in ((a, b), (c, d)):
        pts += [(u, v), (1 - u, v), (u, 1 - v), (1 - u, 1 - v),
                (v, u), (1 - v, u), (v, 1 - u), (1 - v, 1 - u)]
    pts += [(g, g), (1 - g, g), (g, 1 - g), (1 - g, 1 - g)]
    pts += [(0.5, h), (0.5, 1 - h), (h, 0.5), (1 - h, 0.5)]
    return np.array(pts, dtype=float)


def construct_packing():
    """
    D4-symmetric two-stage construction for n=26 circles.

    Stage 1 (symmetric): SLSQP over 6 shape parameters + 25 radii,
    maximizing total radius of the 25-circle D4 skeleton (corner/edge
    orbits replace border-limited ring circles; <=300 iterations).
    Stage 2 (symmetry-breaking): all 52 positions + 26 radii free; a
    26th circle is seeded in the largest gap and SLSQP refines the
    full asymmetric packing (<=200 iterations).
    Stage 3 (radius polish): with centers frozen, SLSQP maximizes the
    radius sum over 26 radii from the feasible shrink solution — the
    proportional pairwise shrink is suboptimal, and this recovers the
    exact maximal radius vector at the achieved centers.
    Feasibility is guaranteed by a final pairwise-shrink safety clip.
    """
    from scipy.optimize import minimize, dual_annealing, linprog

    # --- Stage 1 (bilevel): global shape search with exact LP radii ---
    def _expand26(z):
        """Expand 12 shape parameters into exactly 26 centers.

        z = (a, b, c, d, g, h, e1x, e1y, e2x, e2y, sx, sy):
        D4 skeleton orbits plus asymmetric offsets (e1, e2) on the two
        8-orbits; one orbit-8 point is split into two half-offset points
        (sx, sy) to reach 26 circles.
        """
        a, b, c, d, g, h, e1x, e1y, e2x, e2y, sx, sy = z
        pts = [(0.5, 0.5)]
        for (u, v) in ((a + e1x, b + e1y), (c + e2x, d + e2y)):
            pts += [(u, v), (1 - u, v), (u, 1 - v), (1 - u, 1 - v),
                    (v, u), (1 - v, u), (v, 1 - u), (1 - v, 1 - u)]
        pts += [(g, g), (1 - g, g), (g, 1 - g), (1 - g, 1 - g)]
        pts += [(0.5, h), (0.5, 1 - h), (h, 0.5), (1 - h, 0.5)]
        ux, uy = pts.pop(1)
        pts.append((ux + sx, uy + sy))
        pts.append((ux - sx, uy - sy))
        return np.array(pts, dtype=float)

    def _lp_radii(cen):
        """Exact maximal radius sum at fixed centers via HiGHS LP."""
        n = cen.shape[0]
        iu = np.triu_indices(n, 1)
        m = 4 * n + iu[0].size
        A = np.zeros((m, n))
        b = np.zeros(m)
        for k in range(n):
            A[k, k] = -1.0
            b[k] = -cen[k, 0]
            A[n + k, k] = -1.0
            b[n + k] = -(1.0 - cen[k, 0])
            A[2 * n + k, k] = -1.0
            b[2 * n + k] = -cen[k, 1]
            A[3 * n + k, k] = -1.0
            b[3 * n + k] = -(1.0 - cen[k, 1])
        r0 = 4 * n
        for k in range(iu[0].size):
            i, j = iu[0][k], iu[1][k]
            A[r0 + k, i] = -1.0
            A[r0 + k, j] = -1.0
            b[r0 + k] = -float(np.sqrt(((cen[i] - cen[j]) ** 2).sum()))
        res = linprog(-np.ones(n), A_ub=A, b_ub=b,
                      bounds=[(0.0, 0.5)] * n, method="highs")
        if res.success:
            rr = np.asarray(res.x, dtype=float)
            return rr, float(rr.sum())
        return None, -1.0

    bounds1 = [(0.02, 0.48)] * 6 + [(-0.06, 0.06)] * 4 + [(0.0, 0.06)] * 2

    def outer_obj(p):
        """Outer objective: negative exact LP radius sum of the skeleton."""
        cen = _expand26(p)
        if cen.min() < 1e-4 or cen.max() > 1.0 - 1e-4:
            return 10.0
        _, s = _lp_radii(cen)
        return -s if np.isfinite(s) else 10.0

    res_da = dual_annealing(outer_obj, bounds1, maxiter=80, seed=0)
    radii = None
    if np.isfinite(res_da.fun):
        centers26 = np.clip(_expand26(res_da.x), 1e-6, 1.0 - 1e-6)
        radii, _ = _lp_radii(centers26)
    if radii is None:
        # Deterministic fallback skeleton (feasible under LP)
        p_def = np.array([0.35, 0.20, 0.28, 0.38, 0.11, 0.10,
                          0.0, 0.0, 0.0, 0.0, 0.01, 0.01])
        centers26 = _expand26(p_def)
        radii, _ = _lp_radii(centers26)
        if radii is None:
            radii = _init_radii(centers26)
    radii = np.clip(np.asarray(radii, dtype=float) * 0.999, 1e-6, 0.5)
    r26 = radii.copy()
    # Deterministic symmetry-breaking perturbation: the bilevel seed may sit
    # point is a stationary point of stage 2, so SLSQP starting there stays
    # symmetric and cannot reach the asymmetric optimum. Nudging each center
    # along a fixed low-amplitude pattern leaves the symmetric basin intact
    # while giving the optimizer an asymmetric direction to descend.
    _pert = np.array([0.004, -0.003, 0.005, -0.002, 0.003, 0.004,
                      -0.004, 0.002, -0.005, 0.003], dtype=float)
    for k in range(26):
        centers26[k, 0] = min(max(centers26[k, 0] + _pert[(2 * k) % 10], 1e-4), 1 - 1e-4)
        centers26[k, 1] = min(max(centers26[k, 1] + _pert[(2 * k + 1) % 10], 1e-4), 1 - 1e-4)
    z2 = np.concatenate([centers26.flatten(), r26])

    def obj2(z):
        return -np.sum(z[52:])

    def cons2(z):
        return _constraint_vals(z[:52].reshape(26, 2), z[52:])

    res2 = minimize(obj2, z2, method="SLSQP", constraints=[{"type": "ineq", "fun": cons2}],
                    bounds=[(0.0, 1.0)] * 52 + [(0.0, 0.5)] * 26,
                    options={"maxiter": 200, "ftol": 1e-9})
    cand_c = res2.x[:52].reshape(26, 2)
    cand_r = res2.x[52:]
    if np.isfinite(res2.fun) and cand_c.min() >= -1e-9 and cand_c.max() <= 1 + 1e-9:
        centers = np.clip(cand_c, 1e-6, 1 - 1e-6)
        radii = np.clip(cand_r, 1e-6, 0.5)

    # Guaranteed-feasible radii for the final configuration (pairwise shrink only)
    radii = compute_max_radii(centers, start=radii)

    # --- Radius polish: exact maximal radii at fixed centers ---
    # compute_max_radii shrinks both circles of a tight pair proportionally,
    # which is suboptimal when one circle has slack via other contacts. With
    # centers frozen, maximizing the radius sum is a 26-variable feasibility
    # problem; SLSQP from the feasible shrink solution strictly improves it.
    def _rcons(r):
        return _constraint_vals(centers, r)

    def _robj(r):
        return -np.sum(r)

    resr = minimize(_robj, radii, method="SLSQP",
                    constraints=[{"type": "ineq", "fun": _rcons}],
                    bounds=[(0.0, 0.5)] * centers.shape[0],
                    options={"maxiter": 300, "ftol": 1e-12})
    if np.isfinite(resr.fun):
        r_pol = np.clip(resr.x, 1e-6, 0.5)
        # Accept only if it does not violate feasibility (safety margin)
        if _constraint_vals(centers, r_pol).min() > -1e-9 and np.sum(r_pol) > np.sum(radii):
            radii = r_pol
            # One final guaranteed-feasible shrink pass as safety net
            radii = compute_max_radii(centers, start=radii)

    # --- Stage 4: relocate smallest circle into the largest void (basin escape) ---
    # A converged SLSQP optimum is locked into one contact topology; small
    # perturbations only explore within its basin. Relocating the smallest
    # circle from its position into the exact largest empty circle of the
    # remaining 25 centers changes the topology itself, giving SLSQP access
    # to a different (potentially better) basin. The result is accepted only
    # if it strictly improves and is made guaranteed-feasible.
    def _refine_full(c0, r0):
        """Joint SLSQP (centers+radii) then frozen-center radius polish."""
        z = np.concatenate([c0.flatten(), r0])

        res = minimize(lambda zz: -np.sum(zz[52:]), z, method="SLSQP",
                       constraints=[{"type": "ineq",
                                     "fun": lambda zz: _constraint_vals(
                                         zz[:52].reshape(26, 2), zz[52:])}],
                       bounds=[(0.0, 1.0)] * 52 + [(0.0, 0.5)] * 26,
                       options={"maxiter": 200, "ftol": 1e-10})
        cc = res.x[:52].reshape(26, 2)
        if not (np.isfinite(res.fun) and cc.min() >= -1e-9 and cc.max() <= 1 + 1e-9):
            return None
        cc = np.clip(cc, 1e-6, 1 - 1e-6)
        rr = compute_max_radii(cc, start=np.clip(res.x[52:], 1e-6, 0.5))
        resp = minimize(lambda r: -np.sum(r), rr, method="SLSQP",
                        constraints=[{"type": "ineq",
                                      "fun": lambda r: _constraint_vals(cc, r)}],
                        bounds=[(0.0, 0.5)] * 26,
                        options={"maxiter": 300, "ftol": 1e-12})
        if np.isfinite(resp.fun):
            rpl = np.clip(resp.x, 1e-6, 0.5)
            if _constraint_vals(cc, rpl).min() > -1e-9 and rpl.sum() > rr.sum():
                rr = compute_max_radii(cc, start=rpl)
        return cc, rr

    k = int(np.argmin(radii))
    rest = np.delete(centers, k, axis=0)
    gx = np.linspace(0.01, 0.99, 70)
    dxg = gx[:, None] - rest[None, :, 0]
    dyg = gx[:, None] - rest[None, :, 1]
    dmin = np.sqrt(dxg ** 2 + dyg ** 2).min(axis=1)      # (70, 70)
    wall = np.minimum(np.minimum(gx[:, None], 1 - gx[:, None]),
                      np.minimum(gx[None, :], 1 - gx[None, :]))
    gap = np.minimum(dmin, wall)
    iy, ix = np.unravel_index(np.argmax(gap), gap.shape)
    best_pt = (gx[ix], gx[iy])
    best_gap = float(gap[iy, ix])

    c_new = np.vstack([rest, np.array(best_pt, dtype=float)])
    r_new = np.append(np.delete(radii, k), max(best_gap * 0.6, 1e-4)).astype(float)
    # Break symmetry of the relocation seed deterministically (same pattern
    # family as stage 2, different amplitude for a different basin).
    _pert2 = np.array([0.002, -0.0025, 0.003, -0.0015, 0.0025, 0.002,
                       -0.002, 0.0015, -0.003, 0.0025], dtype=float)
    for i2 in range(26):
        c_new[i2, 0] = min(max(c_new[i2, 0] + _pert2[(2 * i2) % 10], 1e-4), 1 - 1e-4)
        c_new[i2, 1] = min(max(c_new[i2, 1] + _pert2[(2 * i2 + 1) % 10], 1e-4), 1 - 1e-4)
    cand = _refine_full(np.ascontiguousarray(c_new), r_new)
    if cand is not None and cand[1].sum() > radii.sum():
        centers, radii = cand

    return centers, radii, float(np.sum(radii))


def compute_max_radii(centers, start=None):
    """
    Compute feasible radii for the given centers by shrinking a starting
    radius vector (default: all ones) under wall limits and pairwise
    proportional scaling until no pair overlaps. Only ever decreases
    radii, so the result is guaranteed valid.

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        start: optional np.array of shape (n) with initial radii

    Returns:
        np.array of shape (n) with radius of each circle
    """
    n = centers.shape[0]
    radii = np.ones(n) if start is None else np.array(start, dtype=float, copy=True)

    # First, limit by distance to square borders
    for i in range(n):
        x, y = centers[i]
        # Distance to borders
        radii[i] = min(radii[i], x, y, 1 - x, 1 - y)

    # Then, limit by distance to other circles
    # Each pair of circles with centers at distance d can have
    # sum of radii at most d to avoid overlap
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))

            # If current radii would cause overlap
            if radii[i] + radii[j] > dist:
                # Scale both radii proportionally
                scale = dist / (radii[i] + radii[j])
                radii[i] *= scale
                radii[j] *= scale

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
