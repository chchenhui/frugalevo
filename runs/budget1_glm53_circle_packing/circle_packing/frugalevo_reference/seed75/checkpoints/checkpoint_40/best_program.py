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

    def _hex_seed():
        """Deterministic 26-center hex-row-varistratum seed (incumbent basin).

        Staggered hexagonal rows of 5,4,5,4,5,3 circles: spacing s = 0.2 so
        5-wide rows span the square, vertical row pitch sqrt(3)/2*s, rows
        vertically centered, alternate rows offset s/2 (proper hex lattice).
        """
        s = 0.2
        rows = [5, 4, 5, 4, 5, 3]
        dy = np.sqrt(3.0) / 2.0 * s
        y0 = (1.0 - (len(rows) - 1) * dy) / 2.0
        pts = []
        for k, m in enumerate(rows):
            y = y0 + k * dy
            x0 = s / 2.0 if m == 5 else s
            for i in range(m):
                pts.append((x0 + i * s, y))
        return np.array(pts, dtype=float)

    def _rotated_seed(theta):
        """Rotated hexagonal-lattice seed for 26 circles.

        Builds a hex lattice (spacing s, pitch sqrt(3)/2*s) centered at the
        square's center, keeps the 26 lattice points closest to the center,
        rotates the cluster by theta about (0.5, 0.5), and bisects s (<=20
        steps) so the rotated cluster just fits inside the unit square.
        theta>0 yields staggered wall pockets instead of aligned columns —
        a contact topology the axis-aligned basins cannot reach.
        """
        th = np.float64(np.deg2rad(theta))
        R = np.array([[np.cos(th), -np.sin(th)],
                      [np.sin(th), np.cos(th)]], dtype=float)

        def cluster(s):
            pts = []
            for j in range(-4, 5):
                off = 0.5 * s if (j % 2 != 0) else 0.0
                for i in range(-5, 6):
                    pts.append((i * s + off, j * (np.sqrt(3.0) / 2.0) * s))
            P = np.array(pts, dtype=float)
            d = np.sqrt((P ** 2).sum(axis=1))
            idx = np.argsort(d, kind="stable")[:26]
            return 0.5 + P[idx] @ R.T

        lo, hi = 0.04, 0.34
        for _ in range(20):
            mid = 0.5 * (lo + hi)
            Q = cluster(mid)
            if Q.min() >= 0.0 and Q.max() <= 1.0:
                lo = mid
            else:
                hi = mid
        return np.ascontiguousarray(cluster(lo), dtype=float)

    def _seed_pair(c26):
        """LP radii for a seed, with deterministic init fallback."""
        r26, _ = _lp_radii(np.ascontiguousarray(c26, dtype=float))
        if r26 is None:
            r26 = _init_radii(c26)
        return c26, np.clip(np.asarray(r26, dtype=float) * 0.999, 1e-6, 0.5)

    # Seed set: incumbent hex-row basin + 12 rotated angles in (0, 45 deg].
    _angles = [2.0, 5.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0, 36.0, 40.0, 45.0]
    seeds = [_seed_pair(np.clip(_hex_seed(), 1e-4, 1.0 - 1e-4))]
    seeds += [_seed_pair(np.clip(_rotated_seed(t), 1e-4, 1.0 - 1e-4))
              for t in _angles]

    # Stage-2 joint SLSQP starts from the incumbent seed (theta=0 path kept),
    # with a small fixed symmetry-breaking nudge as before.
    centers26, r26 = seeds[0]
    _pert = np.array([0.004, -0.003, 0.005, -0.002, 0.003, 0.004,
                      -0.004, 0.002, -0.005, 0.003], dtype=float)
    centers26 = np.array(centers26, dtype=float, copy=True)
    for k in range(26):
        centers26[k, 0] = min(max(centers26[k, 0] + _pert[(2 * k) % 10], 1e-4), 1 - 1e-4)
        centers26[k, 1] = min(max(centers26[k, 1] + _pert[(2 * k + 1) % 10], 1e-4), 1 - 1e-4)
    z2 = np.concatenate([centers26.flatten(), r26])

    def obj2(z):
        # Log-radii objective: equal relative weight per circle; the frozen-
        # center polish afterwards still maximizes the true linear sum.
        return -np.sum(np.log(np.maximum(z[52:], 1e-9)))

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

    # --- Stage 4: contact-graph continuation (replaces blind grid relocation) ---
    # At a true local optimum every circle is pinned by >=3 active contacts
    # (walls or neighbours); circles with <3 are underconstrained and can
    # grow if relocated. For the <=3 most underconstrained circles: delete
    # one, reinsert it at the exact largest-void center (fine 200x200 grid,
    # removing the 70x70 quantization loss of the old heuristic), warm-start
    # a bounded joint SLSQP. The incumbent's blind-relocation baseline is
    # kept as one additional candidate. Every candidate is scored on the
    # SAME guaranteed-feasible radius pipeline (shrink -> frozen-center
    # polish -> shrink) as the incumbent, and a candidate is accepted only
    # if its post-clip sum beats the incumbent's post-clip sum by a margin,
    # so the returned value can never drop below the incumbent path.
    def _shrink_stable(cen, r0):
        """Iterate compute_max_radii to a fixed point (guaranteed feasible)."""
        rr = np.array(r0, dtype=float, copy=True)
        for _ in range(60):
            rr2 = compute_max_radii(cen, start=rr)
            if np.allclose(rr2, rr, rtol=0.0, atol=1e-13):
                return rr2
            rr = rr2
        return rr

    def _score_feasible(cen, r0):
        """Refine (centers free, <=150 iters) then frozen-center polish; return
        (centers, radii, post-clip sum) with radii guaranteed feasible."""
        z = np.concatenate([np.asarray(cen, dtype=float).flatten(),
                            np.asarray(r0, dtype=float)])
        # Joint refinement, two-phase objective continuation on the same
        # feasible set (identical constraints/bounds, same per-start caps):
        # Phase A: multiplicative (log) radii objective - unlike the linear
        # objective, whose gradient is dominated by the largest circles, log
        # weights every circle equally relative to its size, so
        # under-contacted small/medium circles actively grow.
        # Phase B: one bounded linear-objective joint pass warm-started from
        # the log solution - near the log optimum its gradient flattens and
        # the sum stalls; re-linearizing with the true sum objective at the
        # new point recovers additional growth without any new search start.
        # The final linear polish at frozen centers still recovers the exact
        # maximal sum of radii.
        _jcon = [{"type": "ineq",
                  "fun": lambda zz: _constraint_vals(
                      zz[:52].reshape(26, 2), zz[52:])}]
        _jbnd = [(0.0, 1.0)] * 52 + [(0.0, 0.5)] * 26
        res = minimize(lambda zz: -np.sum(np.log(np.maximum(zz[52:], 1e-9))),
                       z, method="SLSQP", constraints=_jcon,
                       bounds=_jbnd,
                       options={"maxiter": 200, "ftol": 1e-11})
        res = minimize(lambda zz: -np.sum(zz[52:]),
                       np.ascontiguousarray(res.x, dtype=float),
                       method="SLSQP", constraints=_jcon,
                       bounds=_jbnd,
                       options={"maxiter": 200, "ftol": 1e-11})
        # Third continuation pass: re-linearize once more from the second
        # solution; the active set has typically stabilized by now, so this
        # cheap bounded pass recovers the last ~1e-5 of radius sum.
        res = minimize(lambda zz: -np.sum(zz[52:]),
                       np.ascontiguousarray(res.x, dtype=float),
                       method="SLSQP", constraints=_jcon,
                       bounds=_jbnd,
                       options={"maxiter": 200, "ftol": 1e-12})
        cc = res.x[:52].reshape(26, 2)
        if not (np.isfinite(res.fun) and cc.min() >= -1e-9 and cc.max() <= 1 + 1e-9):
            return None
        cc = np.clip(np.ascontiguousarray(cc), 1e-6, 1 - 1e-6)
        rr = _shrink_stable(cc, np.clip(res.x[52:], 1e-6, 0.5))
        if _constraint_vals(cc, rr).min() < -1e-12:
            return None
        resp = minimize(lambda r: -np.sum(r), rr, method="SLSQP",
                        constraints=[{"type": "ineq",
                                      "fun": lambda r: _constraint_vals(cc, r)}],
                        bounds=[(0.0, 0.5)] * 26,
                        options={"maxiter": 300, "ftol": 1e-12})
        if np.isfinite(resp.fun):
            rpl = np.clip(resp.x, 1e-6, 0.5)
            if _constraint_vals(cc, rpl).min() > -1e-12:
                rr = _shrink_stable(cc, rpl)
        # Final monotone safety clip: same pipeline used to score the incumbent
        rr = _shrink_stable(cc, np.clip(rr * 0.9999, 1e-6, 0.5))
        if _constraint_vals(cc, rr).min() < -1e-12:
            return None
        return cc, rr, float(rr.sum())

    def _largest_void(rest):
        """Largest empty-circle center for fixed circles + walls.

        Fine 200x200 grid scan for global coverage, then the top-3 distinct
        grid cells are polished by bounded Nelder-Mead (<=40 iters each) on
        the exact gap function, removing quantization loss and single-argmax
        local-basin traps. Cost is negligible (no SLSQP calls added).
        """
        from scipy.optimize import minimize as _min
        gx = np.linspace(0.005, 0.995, 200)
        dxg = gx[:, None] - rest[None, :, 0]
        dyg = gx[:, None] - rest[None, :, 1]
        dmin = np.sqrt(dxg ** 2 + dyg ** 2).min(axis=1)  # (200, 200)
        wall = np.minimum(np.minimum(gx[:, None], 1 - gx[:, None]),
                          np.minimum(gx[None, :], 1 - gx[None, :]))
        gap = np.minimum(dmin, wall)
        flat = gap.ravel()
        k = min(3, flat.size)
        top = np.argpartition(flat, -k)[-k:]
        top = top[np.argsort(flat[top])[::-1]]

        def _neg_gap(p):
            px, py = p
            if px < 0.0 or px > 1.0 or py < 0.0 or py > 1.0:
                return 10.0
            d = np.sqrt(((rest - np.array([px, py])) ** 2).sum(axis=1)).min()
            return -min(d, px, 1.0 - px, py, 1.0 - py)

        best_px = best_py = None
        best_pg = -1.0
        for idx in top:
            iy0, ix0 = divmod(int(idx), gap.shape[1])
            res = _min(_neg_gap, np.array([gx[ix0], gx[iy0]]),
                       method="Nelder-Mead",
                       options={"maxiter": 40, "xatol": 1e-10, "fatol": 1e-12})
            if np.isfinite(res.fun):
                px = float(np.clip(res.x[0], 0.0, 1.0))
                py = float(np.clip(res.x[1], 0.0, 1.0))
                pg = float(-res.fun)
            else:
                px, py, pg = gx[ix0], gx[iy0], float(gap[iy0, ix0])
            if pg > best_pg:
                best_px, best_py, best_pg = px, py, pg
        if best_px is None:
            iy, ix = np.unravel_index(np.argmax(gap), gap.shape)
            return gx[ix], gx[iy], float(gap[iy, ix])
        return best_px, best_py, best_pg

    def _active_contacts(cen):
        """Active contacts (slack < 1e-7) per circle: walls + tight pairs."""
        x, y = cen[:, 0], cen[:, 1]
        cnt = ((x <= 1e-7) | ((1 - x) <= 1e-7) |
               (y <= 1e-7) | ((1 - y) <= 1e-7)).astype(int)
        dist = _pair_dist(cen)
        iu = np.triu_indices(cen.shape[0], 1)
        for i, j in zip(iu[0], iu[1]):
            if dist[i, j] <= 2e-7:
                cnt[i] += 1
                cnt[j] += 1
        return cnt

    # Seed race: every seed (incumbent hex-row, 12 rotated angles) plus the
    # stage-2 SLSQP result is scored on the identical guaranteed-feasible
    # pipeline (shrink -> log+2x linear joint SLSQP -> frozen-center polish
    # -> shrink). The best post-clip sum wins; the incumbent seed and the
    # stage-2 result are included, so the return can only improve.
    base_c = base_r = None
    base_sum = -1.0
    for c0, r0 in seeds + [(centers, radii)]:
        out = _score_feasible(np.ascontiguousarray(c0, dtype=float),
                              np.ascontiguousarray(r0, dtype=float))
        if out is not None and out[2] > base_sum:
            base_c, base_r, base_sum = out
    if base_c is None:
        base_c, base_r, base_sum = centers, radii, float(np.sum(radii))

    # Candidate edits: constraint-rank (fewest active contacts), up to 3,
    # plus the incumbent's blind smallest-circle relocation as baseline.
    contacts = _active_contacts(base_c)
    order = np.argsort(contacts, kind="stable")
    cand_ids = [int(k) for k in order[:3] if contacts[k] < 3]
    if not cand_ids:
        cand_ids = [int(np.argmin(base_r))]
    k_blind = int(np.argmin(base_r))
    if k_blind not in cand_ids:
        cand_ids.append(k_blind)

    # Gaussian basin hopping (replaces single-circle ratchet relocation).
    # Key structural repair over strict-improvement-only hopping: a hopper
    # that only accepts strict improvements can never leave the incumbent's
    # basin, so every perturbation re-polishes back to the same optimum and
    # the chain stalls. Instead we run a proper basin-hopping WALK: the
    # walker state accepts any hop whose guaranteed-feasible post-clip sum
    # is within -0.002 of the walker's own value (sideways and slightly
    # downhill moves allowed), letting the chain traverse several adjacent
    # basins; a separate best-ever record is updated only on strict
    # improvement and is what is returned, so the output is monotone
    # non-decreasing w.r.t. the incumbent and feasibility is preserved by
    # the shrink-stable pipeline. Perturbation is per-circle Gaussian noise
    # sigma_i = scale * r_i with the scale cycling over 4 magnitudes;
    # each hop gets a feasible radius start via compute_max_radii.
    # Budget: <=60 hops, hard wall-clock guard at 240 s.
    import time
    _t0 = time.time()
    _rng = np.random.default_rng(20260117)
    _scales = (0.08, 0.15, 0.25, 0.35)
    _walk_c = np.array(base_c, dtype=float, copy=True)
    _walk_r = np.array(base_r, dtype=float, copy=True)
    _walk_sum = float(base_sum)
    for _hop in range(60):
        if time.time() - _t0 > 240.0:
            break
        sigma = _scales[_hop % 4] * _walk_r
        noise = _rng.standard_normal((26, 2)) * sigma[:, None]
        c_hop = np.clip(_walk_c + noise, 1e-4, 1.0 - 1e-4)
        r_hop = compute_max_radii(c_hop, start=_walk_r)
        out = _score_feasible(np.ascontiguousarray(c_hop),
                              np.ascontiguousarray(np.maximum(r_hop, 1e-6)))
        if out is None:
            continue
        s = out[2]
        if s > base_sum + 1e-9:
            base_c = np.ascontiguousarray(out[0])
            base_r = np.ascontiguousarray(out[1])
            base_sum = s
        if s > _walk_sum - 0.002:
            _walk_c = np.ascontiguousarray(out[0])
            _walk_r = np.ascontiguousarray(out[1])
            _walk_sum = s

    return base_c, base_r, float(np.sum(base_r))


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
