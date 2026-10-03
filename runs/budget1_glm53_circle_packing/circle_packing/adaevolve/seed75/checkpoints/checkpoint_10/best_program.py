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
    """Hexagonal staggered rows (5,4,5,4,5) + 3 gap fillers. Sweep over
    lattice spacing, refine EVERY candidate layout (a worse initial LP
    sum can still lead to a better local optimum), keep the best."""
    best = None
    for r0 in (0.100, 0.105, 0.110, 0.115):
        centers, radii, s = refine(hex_layout(r0))
        if best is None or s > best[2]:
            best = (centers, radii, s)
    return best


def hex_layout(r0):
    """26 centers: 5 staggered hex rows (5,4,5,4,5) with horizontal
    spacing 2*r0, vertical spacing sqrt(3)*r0, plus 3 gap fillers."""
    t = 2.0 * r0
    v = np.sqrt(3.0) * r0
    pts = []
    for row, m in enumerate([5, 4, 5, 4, 5]):
        y = 0.5 + (row - 2) * v
        for i in range(m):
            pts.append([0.5 + (i - (m - 1) / 2.0) * t, y])
    pts.append([0.5 - 2.0 * t - 0.01, 0.5 - 1.5 * v])
    pts.append([0.5 + 2.0 * t + 0.01, 0.5 - 1.5 * v])
    pts.append([0.5, 0.5 + 2.0 * v + 0.09])
    return np.array(pts)


def refine(centers, rounds=4):
    """Greedy per-circle displacement search: try 8-direction moves at
    decreasing step sizes down to 0.001; accept moves that increase the
    LP sum. Finer terminal steps polish the final arrangement."""
    centers = centers.copy()
    radii = compute_max_radii(centers)
    s = float(np.sum(radii))
    for step in (0.02, 0.008, 0.003, 0.001):
        for _ in range(rounds):
            improved = False
            for i in range(centers.shape[0]):
                best_move = None
                for dx, dy in ((step, 0), (-step, 0), (0, step), (0, -step),
                               (step, step), (step, -step),
                               (-step, step), (-step, -step)):
                    old = centers[i].copy()
                    centers[i] = old + [dx, dy]
                    if (centers[i] < 0.02).any() or (centers[i] > 0.98).any():
                        centers[i] = old
                        continue
                    r2 = compute_max_radii(centers)
                    s2 = float(np.sum(r2))
                    if s2 > s + 1e-9:
                        s, best_move = s2, (centers[i].copy(), r2)
                    centers[i] = old
                if best_move is not None:
                    centers[i], radii = best_move
                    improved = True
            if not improved:
                break
    # Pair-move pass: shift adjacent circle pairs together to escape
    # local optima where two circles need to move in the same direction.
    for step in (0.008, 0.003):
        for _ in range(2):
            improved = False
            n = centers.shape[0]
            for i in range(n):
                for j in range(i + 1, n):
                    if np.hypot(*(centers[i] - centers[j])) > 0.35:
                        continue
                    for dx, dy in ((step, 0), (-step, 0), (0, step),
                                   (0, -step)):
                        oi = centers[i].copy()
                        oj = centers[j].copy()
                        centers[i] = oi + [dx, dy]
                        centers[j] = oj + [dx, dy]
                        if ((centers[i] < 0.02).any() or
                                (centers[i] > 0.98).any() or
                                (centers[j] < 0.02).any() or
                                (centers[j] > 0.98).any()):
                            centers[i], centers[j] = oi, oj
                            continue
                        r2 = compute_max_radii(centers)
                        s2 = float(np.sum(r2))
                        if s2 > s + 1e-9:
                            s = s2
                            radii = r2
                            improved = True
                        else:
                            centers[i], centers[j] = oi, oj
            if not improved:
                break
    return centers, radii, s


def compute_max_radii(centers):
    """Maximize sum of radii subject to r_i + r_j <= dist(i,j) and
    r_i <= border distance, via scipy linprog; greedy fallback."""
    n = centers.shape[0]
    x, y = centers[:, 0], centers[:, 1]
    border = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
    ii, jj = np.triu_indices(n, 1)
    d = np.sqrt(((centers[ii] - centers[jj]) ** 2).sum(axis=1))
    try:
        from scipy.optimize import linprog
        A = np.zeros((len(ii), n))
        A[np.arange(len(ii)), ii] = 1.0
        A[np.arange(len(ii)), jj] = 1.0
        res = linprog(c=-np.ones(n), A_ub=A, b_ub=d,
                      bounds=list(zip(np.zeros(n), border)), method="highs")
        if res.success:
            return np.maximum(res.x, 0.0)
    except Exception:
        pass
    radii = border.copy()
    for _ in range(500):
        over = radii[ii] + radii[jj] > d
        if not over.any():
            break
        for a, b, dist in zip(ii[over], jj[over], d[over]):
            tot = radii[a] + radii[b]
            if tot > dist and tot > 0:
                scale = dist / tot
                radii[a] *= scale
                radii[b] *= scale
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
