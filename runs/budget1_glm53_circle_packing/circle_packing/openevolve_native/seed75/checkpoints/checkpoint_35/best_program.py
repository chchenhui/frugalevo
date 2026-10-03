# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles: hex seed + LP polishing."""
import numpy as np
from math import sqrt

try:
    from scipy.optimize import linprog
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


def _solve_radii(centers):
    """Optimal radii for fixed centers via LP (or safe iterative fallback).

    Maximizes sum(r) subject to r_i + r_j <= dist(i, j) for all pairs and
    r_i <= distance of center i to each wall. Returns (radii, sum).
    """
    n = len(centers)
    ub = np.minimum.reduce([centers[:, 0], 1.0 - centers[:, 0],
                            centers[:, 1], 1.0 - centers[:, 1]])
    if _HAVE_SCIPY:
        from scipy.sparse import coo_matrix
        # Vectorized pairwise-distance constraint matrix (sparse) for speed.
        diff = centers[:, None, :] - centers[None, :, :]
        dmat = np.sqrt((diff ** 2).sum(-1))
        iu, ju = np.triu_indices(n, 1)
        m = len(iu)
        A = coo_matrix(
            (np.ones(2 * m),
             (np.concatenate([np.arange(m), np.arange(m)]),
              np.concatenate([iu, ju]))),
            shape=(m, n)).tocsr()
        res = linprog(-np.ones(n), A_ub=A, b_ub=dmat[iu, ju],
                      bounds=[(1e-12, float(u)) for u in ub], method="highs")
        if res.success:
            radii = np.maximum(res.x, 1e-12)
            return radii, float(np.sum(radii))
    # Fallback: iterate r_i = min(ub_i, min_j (d_ij - r_j)) until feasible.
    radii = ub.copy()
    dmat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dmat[i, j] = sqrt((centers[i, 0] - centers[j, 0]) ** 2 +
                                  (centers[i, 1] - centers[j, 1]) ** 2)
    for _ in range(60):
        newr = ub.copy()
        for i in range(n):
            m = np.min(dmat[i] - radii)
            newr[i] = min(ub[i], m)
        if np.all(newr <= radii + 1e-15):
            radii = np.maximum(newr, 1e-12)
            break
        radii = np.maximum(newr, 1e-12)
    return radii, float(np.sum(radii))


def _optimize(centers):
    """Deterministic coordinate-descent on positions, scored by the radii LP.

    For a ladder of step sizes, repeatedly try shifting each circle by
    +/- step along each axis; accept a move only if the optimal (LP) sum of
    radii at the new position strictly improves. Always feasible because
    the LP enforces wall and pairwise constraints exactly.
    """
    n = len(centers)
    best_c = centers.copy()
    best_r, best_s = _solve_radii(best_c)
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1)]
    for step in (0.02, 0.006, 0.002, 0.0006, 0.0002, 0.00006, 0.00002):
        improved = True
        while improved:
            improved = False
            for i in range(n):
                # Candidate moves: 8 axis/diagonal steps plus a push
                # directly away from the nearest neighbour, which helps
                # separate touching pairs that block radius growth.
                moves = [(dx * step, dy * step) for dx, dy in dirs]
                dvec = np.sqrt(((best_c - best_c[i]) ** 2).sum(1))
                dvec[i] = np.inf
                j = int(np.argmin(dvec))
                vec = best_c[i] - best_c[j]
                nrm = float(np.hypot(vec[0], vec[1]))
                if nrm > 1e-12:
                    moves.append((vec[0] / nrm * step, vec[1] / nrm * step))
                # Push away from the nearest wall: raises this circle's
                # radius upper bound without directly squeezing neighbours.
                wx = min(best_c[i, 0], 1.0 - best_c[i, 0])
                wy = min(best_c[i, 1], 1.0 - best_c[i, 1])
                if wx < wy:
                    sgn = 1.0 if best_c[i, 0] < 0.5 else -1.0
                    moves.append((sgn * step, 0.0))
                else:
                    sgn = 1.0 if best_c[i, 1] < 0.5 else -1.0
                    moves.append((0.0, sgn * step))
                for mx, my in moves:
                    cand = best_c.copy()
                    vx = cand[i, 0] + mx
                    vy = cand[i, 1] + my
                    if vx < 1e-6 or vx > 1.0 - 1e-6:
                        continue
                    if vy < 1e-6 or vy > 1.0 - 1e-6:
                        continue
                    cand[i, 0] = vx
                    cand[i, 1] = vy
                    r, sc = _solve_radii(cand)
                    if sc > best_s + 1e-9:
                        best_s, best_c, best_r = sc, cand, r
                        improved = True
    return best_c, best_r


def construct_packing():
    """
    Construct a packing of 26 circles in a unit square.

    Approach: hexagonal two-tier seed, then LP-based polishing.
    - Seed: bottom tier of 4 staggered rows (5, 4, 5, 4 = 18 circles) at
      r = 0.1 (width-limited), plus a top tier of 2 staggered rows of 4
      smaller circles at r2 ~ 0.083 solved from the height constraint.
    - Polish: with centers fixed, the maximal radii are the solution of a
      linear program (maximize sum r_i s.t. r_i + r_j <= d_ij, wall bounds).
      A deterministic coordinate-descent then nudges each center by
      shrinking steps, keeping a move only when the LP sum improves.
      This exploits slack near the walls/corners of the staggered rows
      that the rigid seed layout cannot use.

    Returns:
        Tuple of (centers, radii, sum_of_radii)
    """
    n = 26
    seeds = []

    # Seed A: 5-4-5-4 hex rows (r=0.1) + two staggered 4-rows on top.
    r = 0.1
    dy = sqrt(3.0) * r
    y4 = r + 3.0 * dy
    # Solve y4 + sqrt(r2^2 + 0.2*r2) + (1+sqrt(3))*r2 = 1 for r2.
    A = (1.0 + sqrt(3.0)) ** 2 - 1.0
    B = -2.0 * (1.0 - y4) * (1.0 + sqrt(3.0)) - 0.2
    C = (1.0 - y4) ** 2
    disc = B * B - 4.0 * A * C
    r2 = min((-B - sqrt(disc)) / (2.0 * A), 0.083)
    c = []
    for row, m in enumerate([5, 4, 5, 4]):
        y = r + row * dy
        xs = (r + 2.0 * r * np.arange(m)) if m == 5 else \
             (2.0 * r + 2.0 * r * np.arange(m))
        c += [[float(x), y] for x in xs]
    y1 = y4 + sqrt(r2 * r2 + 0.2 * r2)
    y2 = y1 + sqrt(3.0) * r2
    c += [[x, y1] for x in [0.1, 0.3, 0.5, 0.7]]
    c += [[x, y2] for x in [0.2, 0.4, 0.6, 0.8]]
    seeds.append(np.array(c))

    # Seed B: 6-5-6-5 hex rows (r=1/12) + a wide top row of 4 (r=0.125).
    r = 1.0 / 12.0
    dy = sqrt(3.0) * r
    c = []
    for row, m in enumerate([6, 5, 6, 5]):
        y = r + row * dy
        xs = (r + 2.0 * r * np.arange(m)) if m == 6 else \
             (2.0 * r + 2.0 * r * np.arange(m))
        c += [[float(x), y] for x in xs]
    c += [[0.125 + 0.25 * k, 0.875] for k in range(4)]
    seeds.append(np.array(c))

    # Seed C: 5-4-5-4-5 hex rows (r=0.1) + 3 small circles near the top.
    r = 0.1
    dy = sqrt(3.0) * r
    c = []
    for row, m in enumerate([5, 4, 5, 4, 5]):
        y = r + row * dy
        xs = (r + 2.0 * r * np.arange(m)) if m == 5 else \
             (2.0 * r + 2.0 * r * np.arange(m))
        c += [[float(x), y] for x in xs]
    ytop = r + 4.0 * dy
    c += [[x, ytop + 0.09] for x in [0.2, 0.5, 0.8]]
    seeds.append(np.array(c))

    # Seed J: corner circles (r=0.09) + staggered hex rows in between.
    # Corner circles exploit the double-wall slack the pure hex seeds
    # cannot use; the LP polisher grows them against both walls.
    rc = 0.09
    c = [[rc, rc], [1.0 - rc, rc], [rc, 1.0 - rc], [1.0 - rc, 1.0 - rc]]
    rh = 0.095
    dyh = sqrt(3.0) * rh
    for row, m in enumerate([5, 4, 5, 4]):
        y = 0.18 + row * dyh
        if y > 0.9:
            break
        xs = (0.1 + 0.2 * np.arange(m)) if m == 5 else \
             (0.2 + 0.2 * np.arange(m))
        c += [[float(x), y] for x in xs]
    c = c[:26]
    if len(c) == 26:
        seeds.append(np.array(c))

    # Seed K: mixed radii - large corner circles + staggered interior.
    # Corner circles exploit double-wall slack (corner distance is
    # r*sqrt(2) from the effective corner point), which pure hex seeds
    # waste; the LP polisher then grows them against both walls.
    rc = 0.105
    c = [[rc, rc], [1.0 - rc, rc], [rc, 1.0 - rc], [1.0 - rc, 1.0 - rc]]
    # Interior: staggered rows between the corner circles.
    rh = 0.092
    dyh = sqrt(3.0) * rh
    for row, m in enumerate([5, 4, 5, 4, 4]):
        y = 0.19 + row * dyh
        if y > 0.86:
            break
        xs = (0.1 + 0.2 * np.arange(m)) if m == 5 else \
             (0.2 + 0.2 * np.arange(m))
        c += [[float(x), y] for x in xs]
    c = c[:26]
    if len(c) == 26:
        seeds.append(np.array(c))

    # Seed L: fine uniform 9x3-ish grid (3 rows of 9 staggered), giving
    # a completely different (non-hex) basin for the polisher.
    r = 1.0 / 18.0
    dy = sqrt(3.0) * r
    c = []
    for row, m in enumerate([9, 8, 9]):
        y = r + row * dy
        xs = (r + 2.0 * r * np.arange(m)) if m == 9 else \
             (2.0 * r + 2.0 * r * np.arange(m))
        c += [[float(x), y] for x in xs]
    # Top region: 26 - 26 = 0 spare, so squeeze remaining rows tighter.
    r = 1.0 / 13.0
    dy = sqrt(3.0) * r
    c = []
    for row, m in enumerate([7, 6, 7, 6]):
        y = r + row * dy
        xs = (r + 2.0 * r * np.arange(m)) if m == 7 else \
             (2.0 * r + 2.0 * r * np.arange(m))
        c += [[float(x), y] for x in xs]
    c += [[0.1 + 0.2 * k, 1.0 - 0.08] for k in range(4)]
    if len(c) == 26:
        seeds.append(np.array(c))

    # Seeds D-I: deterministic jittered variants of seed A (the strongest
    # seed). Alternate circles are shifted left/right (and slightly in y)
    # so the LP-based polisher starts in several nearby basins and can
    # escape the symmetric local optimum of the rigid hex layout.
    base = seeds[0]
    signs = np.where(np.arange(len(base)) % 2 == 0, 1.0, -1.0)
    for sx in (0.008, -0.008, 0.02, -0.02):
        for sy in (0.0, 0.01):
            v = base.copy()
            v[:, 0] += sx * signs
            v[:, 1] += sy * signs
            v[:, 0] = np.clip(v[:, 0], 0.02, 0.98)
            v[:, 1] = np.clip(v[:, 1], 0.02, 0.98)
            seeds.append(v)

    # Seeds M-P: row-alternating jitter of seed A. Whole rows shift
    # together, which breaks row-level symmetry differently from the
    # per-circle jitter above and opens additional basins.
    row_signs = np.array([1.0 if (i // 5) % 2 == 0 else -1.0
                          for i in range(len(base))])
    for sx in (0.012, -0.012):
        for sy in (0.006, -0.006):
            v = base.copy()
            v[:, 0] += sx * row_signs
            v[:, 1] += sy * row_signs
            v[:, 0] = np.clip(v[:, 0], 0.02, 0.98)
            v[:, 1] = np.clip(v[:, 1], 0.02, 0.98)
            seeds.append(v)

    # Pre-screen seeds cheaply: run only the coarse polish steps on each
    # seed, then fully polish only the top few survivors. This keeps the
    # added seeds from blowing the time budget.
    def _partial_optimize(centers, steps):
        n = len(centers)
        best_c = centers.copy()
        best_r, best_s = _solve_radii(best_c)
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1),
                (1, 1), (1, -1), (-1, 1), (-1, -1)]
        for step in steps:
            improved = True
            while improved:
                improved = False
                for i in range(n):
                    moves = [(dx * step, dy * step) for dx, dy in dirs]
                    dvec = np.sqrt(((best_c - best_c[i]) ** 2).sum(1))
                    dvec[i] = np.inf
                    j = int(np.argmin(dvec))
                    vec = best_c[i] - best_c[j]
                    nrm = float(np.hypot(vec[0], vec[1]))
                    if nrm > 1e-12:
                        moves.append((vec[0] / nrm * step,
                                      vec[1] / nrm * step))
                    wx = min(best_c[i, 0], 1.0 - best_c[i, 0])
                    wy = min(best_c[i, 1], 1.0 - best_c[i, 1])
                    if wx < wy:
                        sgn = 1.0 if best_c[i, 0] < 0.5 else -1.0
                        moves.append((sgn * step, 0.0))
                    else:
                        sgn = 1.0 if best_c[i, 1] < 0.5 else -1.0
                        moves.append((0.0, sgn * step))
                    for mx, my in moves:
                        cand = best_c.copy()
                        vx = cand[i, 0] + mx
                        vy = cand[i, 1] + my
                        if vx < 1e-6 or vx > 1.0 - 1e-6:
                            continue
                        if vy < 1e-6 or vy > 1.0 - 1e-6:
                            continue
                        cand[i, 0] = vx
                        cand[i, 1] = vy
                        r, sc = _solve_radii(cand)
                        if sc > best_s + 1e-9:
                            best_s, best_c, best_r = sc, cand, r
                            improved = True
        return best_c, best_r

    coarse = (0.02, 0.006)
    fine = (0.002, 0.0006, 0.0002, 0.00006)
    scored = []
    for s in seeds:
        cc, rr = _partial_optimize(s, coarse)
        scored.append((float(np.sum(rr)), s))
    scored.sort(key=lambda t: -t[0])
    survivors = [s for _, s in scored[:6]]

    best_c = best_r = best_s = None
    for s in survivors:
        cc, rr = _optimize(s)
        ss = float(np.sum(rr))
        if best_s is None or ss > best_s:
            best_c, best_r, best_s = cc, rr, ss
    return best_c, best_r, best_s


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
