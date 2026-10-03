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
        A_rows = []
        b = []
        for i in range(n):
            for j in range(i + 1, n):
                d = sqrt((centers[i, 0] - centers[j, 0]) ** 2 +
                         (centers[i, 1] - centers[j, 1]) ** 2)
                row = np.zeros(n)
                row[i] = 1.0
                row[j] = 1.0
                A_rows.append(row)
                b.append(d)
        res = linprog(-np.ones(n), A_ub=np.array(A_rows), b_ub=np.array(b),
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
    for step in (0.02, 0.006, 0.002, 0.0006, 0.0002, 0.00006):
        improved = True
        while improved:
            improved = False
            for i in range(n):
                for dx, dy in dirs:
                    cand = best_c.copy()
                    vx = cand[i, 0] + dx * step
                    vy = cand[i, 1] + dy * step
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

    best_c = best_r = best_s = None
    for s in seeds:
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
