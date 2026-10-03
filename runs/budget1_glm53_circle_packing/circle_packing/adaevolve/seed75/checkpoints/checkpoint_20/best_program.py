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
    """Two hexagonal staggered-row layout families swept over lattice
    spacing and global shifts; the top candidates by LP-optimal sum are
    then polished by greedy per-circle coordinate descent, and the best
    refined result is returned."""
    cands = []
    for r0 in np.arange(0.095, 0.135, 0.005):
        for dx in (-0.01, 0.0, 0.01):
            for dy in (-0.01, 0.0, 0.01):
                for layout in (layout_A, layout_B, layout_C, layout_D):
                    centers = np.clip(layout(r0, dx, dy), 0.001, 0.999)
                    radii = compute_max_radii(centers)
                    cands.append((float(np.sum(radii)), centers))
    cands.sort(key=lambda c: -c[0])
    best = None
    # Refine the best 4 candidates; a weaker start can still reach a
    # stronger local optimum after refinement.
    for _, centers in cands[:4]:
        centers, radii, s = refine(centers)
        if best is None or s > best[2]:
            best = (centers, radii, s)
    # Perturbation restarts (mini basin-hopping): jitter the best
    # arrangement and re-refine, escaping local optima that the greedy
    # and pair-move passes cannot leave. Varying jitter scales explore
    # both nearby and farther basins.
    rng = np.random.default_rng(12345)
    centers, radii, s = best
    for sigma in (0.006, 0.012, 0.020, 0.010, 0.008, 0.015, 0.006, 0.012):
        pert = np.clip(centers + rng.normal(0.0, sigma, centers.shape),
                       0.02, 0.98)
        c2, r2, s2 = refine(pert)
        if s2 > s:
            centers, radii, s = c2, r2, s2
    # Final simultaneous polish: Powell derivative-free optimization of
    # all 52 coordinates on the LP-sum objective. Coordinated multi-
    # circle moves escape optima that single/pair greedy moves cannot.
    centers, radii, s = powell_polish(centers, s)
    return (centers, radii, s)


def powell_polish(centers, s, budget=4000):
    """Polish all circle positions simultaneously with Powell's method
    on the LP-optimal sum-of-radii objective (each evaluation is one
    sparse LP solve, ~ms). Bounded to keep centers inside the square."""
    try:
        from scipy.optimize import minimize
    except Exception:
        return centers, compute_max_radii(centers), float(
            np.sum(compute_max_radii(centers)))
    n = centers.shape[0]
    evals = [0]

    def obj(flat):
        evals[0] += 1
        c = np.clip(flat.reshape(n, 2), 0.001, 0.999)
        return -float(np.sum(compute_max_radii(c)))

    try:
        res = minimize(obj, centers.ravel(), method="Powell",
                       options={"maxfev": budget, "xtol": 1e-5,
                                "ftol": 1e-9})
        c = np.clip(res.x.reshape(n, 2), 0.001, 0.999)
        r = compute_max_radii(c)
        s2 = float(np.sum(r))
        if s2 > s:
            return c, r, s2
    except Exception:
        pass
    return centers, compute_max_radii(centers), float(
        np.sum(compute_max_radii(centers)))


def refine(centers, rounds=4):
    """Greedy per-circle displacement search (8 directions, decreasing
    step sizes down to 0.001), then a pair-move pass that shifts close
    circle pairs together to escape local optima where two circles need
    to move in the same direction."""
    centers = centers.copy()
    radii = compute_max_radii(centers)
    s = float(np.sum(radii))
    for step in (0.02, 0.008, 0.003, 0.001, 0.0005, 0.0002):
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
    # Pair-move pass: shift close circle pairs together to escape local
    # optima where two circles need to move in the same direction.
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


def hex_rows(counts, r0, dx, dy):
    """Staggered hex rows: horizontal spacing 2*r0, vertical sqrt(3)*r0."""
    t = 2.0 * r0
    s = np.sqrt(3.0) * r0
    pts = []
    for row, m in enumerate(counts):
        y = 0.5 + (row - (len(counts) - 1) / 2.0) * s + dy
        for i in range(m):
            x = 0.5 + (i - (m - 1) / 2.0) * t + dx
            pts.append([x, y])
    return pts


def layout_A(r0, dx, dy):
    """Rows 5,4,5,4,5 (23 circles) + 3 gap fillers near row ends/edges."""
    t = 2.0 * r0
    s = np.sqrt(3.0) * r0
    pts = hex_rows([5, 4, 5, 4, 5], r0, dx, dy)
    pts.append([0.5 - 2.0 * t - 0.02 + dx, 0.5 - 1.5 * s + dy])
    pts.append([0.5 + 2.0 * t + 0.02 + dx, 0.5 - 1.5 * s + dy])
    pts.append([0.5 + dx, 0.5 + 2.0 * s + 0.08 + dy])
    return np.array(pts)


def layout_B(r0, dx, dy):
    """Rows 4,5,4,5,4 (22 circles) + 4 corner circles + 2 mid-edge fillers.
    Corner circles can grow large (bounded by two walls each)."""
    t = 2.0 * r0
    s = np.sqrt(3.0) * r0
    pts = hex_rows([4, 5, 4, 5, 4], r0, dx, dy)
    c = 0.5 * t  # corner circles offset from each corner
    pts.append([c, c])
    pts.append([1.0 - c, c])
    pts.append([c, 1.0 - c])
    pts.append([1.0 - c, 1.0 - c])
    # mid-edge fillers on left and right at middle row height
    pts.append([0.5 - 1.5 * t - 0.02 + dx, 0.5 + dy])
    pts.append([0.5 + 1.5 * t + 0.02 + dx, 0.5 + dy])
    return np.array(pts)


def layout_C(r0, dx, dy):
    """Layout C: 5 rows 4,5,4,5,4 offset toward bottom + 2 top-corner
    circles + 2 side fillers. Uses vertical asymmetric placement since
    optimum packing in a square is not vertically symmetric."""
    t = 2.0 * r0
    s = np.sqrt(3.0) * r0
    pts = []
    for row, m in enumerate([4, 5, 4, 5, 4]):
        y = 0.5 + (row - (len([4, 5, 4, 5, 4]) - 1) / 2.0) * s + dy
        for i in range(m):
            x = 0.5 + (i - (m - 1) / 2.0) * t + dx
            pts.append([x, y])
    # Top-left and top-right circles near top edge (can be large,
    # bounded by two walls each)
    pts.append([0.12 + dx, 0.12 + dy])
    pts.append([0.88 + dx, 0.12 + dy])
    # Mid-edge fillers on left/right at row 1 height
    pts.append([0.5 - 2.0 * t - 0.02 + dx, 0.5 - s + dy])
    pts.append([0.5 + 2.0 * t + 0.02 + dx, 0.5 - s + dy])
    return np.array(pts)


def layout_D(r0, dx, dy):
    """Boundary-ring layout: 4 corner + 4 mid-edge wall-bounded circles
    (8 circles that can grow large since walls bound them) surrounding
    inner hex rows 4,5,4,5 (18 circles), for 26 total."""
    t = 2.0 * r0
    s = np.sqrt(3.0) * r0
    c = 0.11
    pts = []
    # Corner circles: each bounded by two walls -> radius up to c.
    for cx in (c, 1.0 - c):
        for cy in (c, 1.0 - c):
            pts.append([cx, cy])
    # Mid-edge circles: one per edge, near edge centers.
    pts.append([0.5 + dx, c])
    pts.append([0.5 + dx, 1.0 - c])
    pts.append([c, 0.5 + dy])
    pts.append([1.0 - c, 0.5 + dy])
    # Inner hex cluster of 18 circles in 4 staggered rows.
    for row, m in enumerate([4, 5, 4, 5]):
        y = 0.5 + (row - 1.5) * s + dy
        for i in range(m):
            x = 0.5 + (i - (m - 1) / 2.0) * t + dx
            pts.append([x, y])
    return np.array(pts)


def compute_max_radii(centers):
    """Maximize sum of radii subject to r_i + r_j <= dist(i,j) and
    r_i <= border distance, via scipy linprog; converged greedy fallback.
    Speed: pairs with dist >= border_i + border_j can never bind (since
    r_i <= border_i always), so they are pruned; a sparse constraint
    matrix is used. This roughly triples LP throughput, enabling more
    refinement passes within the time budget."""
    n = centers.shape[0]
    x, y = centers[:, 0], centers[:, 1]
    border = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
    ii, jj = np.triu_indices(n, 1)
    d = np.sqrt(((centers[ii] - centers[jj]) ** 2).sum(axis=1))
    keep = d < border[ii] + border[jj] - 1e-12
    ik, jk, dk = ii[keep], jj[keep], d[keep]
    m = len(ik)
    try:
        from scipy.optimize import linprog
        from scipy.sparse import csr_matrix
        rows = np.concatenate([np.arange(m), np.arange(m)])
        cols = np.concatenate([ik, jk])
        A = csr_matrix((np.ones(2 * m), (rows, cols)), shape=(m, n))
        res = linprog(c=-np.ones(n), A_ub=A, b_ub=dk,
                      bounds=list(zip(np.zeros(n), border)), method="highs")
        if res.success:
            return np.maximum(res.x, 0.0)
    except Exception:
        pass
    radii = border.copy()
    for _ in range(500):
        viol = False
        over = radii[ik] + radii[jk] > dk
        if over.any():
            viol = True
            for a, b, dist in zip(ik[over], jk[over], dk[over]):
                tot = radii[a] + radii[b]
                if tot > dist and tot > 0:
                    scale = dist / tot
                    radii[a] *= scale
                    radii[b] *= scale
        if not viol:
            break
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
