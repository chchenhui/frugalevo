# EVOLVE-BLOCK-START
"""Multi-seed constructor for packing 26 circles in a unit square.

Seeds: hex rows, corner-anchored stagger, wide staggered rows.
Refinement: constraint-repulsion on centers (decaying steps).
Radii: Gauss-Seidel water-fill + exact scipy LP when available.
"""
import numpy as np

N = 26


def build_seeds():
    seeds = []

    # --- Seed A: hex rows 4,5,4,5,4,4 with fixed-point spacing ---
    s = 0.2
    dx = s / 2.0
    r = 0.095
    for _ in range(100):
        dy = (1.0 - 2.0 * r) / 5.0
        r = min(0.1, np.sqrt(dy * dy + dx * dx) / 2.0)
    row_xs = [
        [0.2, 0.4, 0.6, 0.8],
        [0.1, 0.3, 0.5, 0.7, 0.9],
        [0.2, 0.4, 0.6, 0.8],
        [0.1, 0.3, 0.5, 0.7, 0.9],
        [0.2, 0.4, 0.6, 0.8],
        [0.1, 0.3, 0.5, 0.7],
    ]
    c = []
    for k, xs in enumerate(row_xs):
        y = r + k * dy
        for x in xs:
            c.append([x, y])
    seeds.append(np.array(c))

    # --- Seed B: corner anchors + staggered interior rows 4,5,4,5,4 ---
    c = []
    corner = 0.085
    for (cx, cy) in [(corner, corner), (1 - corner, corner),
                     (corner, 1 - corner), (1 - corner, 1 - corner)]:
        c.append([cx, cy])
    rows = [4, 5, 4, 5, 4]
    y0, y1 = 0.17, 0.83
    dy = (y1 - y0) / (len(rows) - 1)
    for ri, cnt in enumerate(rows):
        y = y0 + ri * dy
        sp = 0.72 / (cnt - 1) if cnt > 1 else 0.0
        for j in range(cnt):
            c.append([0.14 + j * sp, y])
    seeds.append(np.array(c))

    # --- Seed C: wide staggered rows 5,4,5,4,5,3 ---
    c = []
    row_counts = [5, 4, 5, 4, 5, 3]
    sp = 0.2
    y0 = 0.1
    gap = (1.0 - 2 * y0) / 5.0
    for row, count in enumerate(row_counts):
        y = y0 + row * gap
        for j in range(count):
            c.append([0.5 + (j - (count - 1) / 2.0) * sp, y])
    seeds.append(np.array(c))

    # --- Seed D: corner anchors + shifted 4x5 interior grid ---
    c = []
    for (cx, cy) in [(0.07, 0.07), (0.93, 0.07), (0.07, 0.93), (0.93, 0.93)]:
        c.append([cx, cy])
    for i in range(4):
        for j in range(5):
            x = 0.16 + i * 0.17 + 0.04 * (j % 2)
            y = 0.16 + j * 0.17
            c.append([x, y])
    seeds.append(np.array(c[:N]))

    return seeds


def compute_max_radii(centers):
    """Gauss-Seidel water-fill + shrink-only repair (always feasible)."""
    n = centers.shape[0]
    D = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
    np.fill_diagonal(D, np.inf)
    wall = np.minimum.reduce([
        centers[:, 0], centers[:, 1],
        1.0 - centers[:, 0], 1.0 - centers[:, 1]
    ])
    r = np.zeros(n)
    for _ in range(300):
        prev = r.copy()
        for i in range(n):
            cap = np.min(D[i] - r)
            r[i] = max(0.0, min(wall[i], cap))
        if np.max(np.abs(prev - r)) < 1e-14:
            break
    r = np.minimum(r, wall)
    iu = np.triu_indices(n, 1)
    for _ in range(20):
        ssum = r[iu[0]] + r[iu[1]]
        bad = ssum > D[iu]
        if not np.any(bad):
            break
        scale = np.where(bad, D[iu] / np.maximum(ssum, 1e-15), 1.0)
        r[iu[0]] *= scale
        r[iu[1]] *= scale
        r = np.minimum(r, wall)
    return r


def solve_radii_lp(centers, eps=1e-9):
    """Exact LP: maximize sum(r_i) s.t. r_i+r_j <= d_ij, wall caps."""
    try:
        from scipy.optimize import linprog
    except ImportError:
        return None
    n = centers.shape[0]
    D = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
    wall = np.minimum.reduce([
        centers[:, 0], centers[:, 1],
        1.0 - centers[:, 0], 1.0 - centers[:, 1]
    ])
    rows, rhs = [], []
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            rows.append(row)
            rhs.append(D[i, j] - 2.0 * eps)
        row = np.zeros(n)
        row[i] = 1.0
        rows.append(row)
        rhs.append(max(wall[i] - eps, 0.0))
    try:
        res = linprog(-np.ones(n), A_ub=np.vstack(rows),
                      b_ub=np.array(rhs),
                      bounds=[(0.0, None)] * n, method="highs")
    except Exception:
        return None
    if not res.success:
        return None
    return np.maximum(res.x, 0.0)


def refine(centers, iters=300):
    """Deterministic constraint-repulsion with decaying step."""
    centers = centers.copy()
    n = centers.shape[0]
    radii = compute_max_radii(centers)
    for it in range(iters):
        step = 0.02 * (1.0 - it / iters) + 0.0005
        for i in range(n):
            xi, yi = centers[i]
            ri = radii[i]
            fx = fy = 0.0
            for j in range(n):
                if j == i:
                    continue
                ddx = xi - centers[j, 0]
                ddy = yi - centers[j, 1]
                dist = np.sqrt(ddx * ddx + ddy * ddy)
                ov = ri + radii[j] - dist
                if ov > 0.0 and dist > 1e-12:
                    fx += ov * ddx / dist
                    fy += ov * ddy / dist
            if xi - ri < 0.0:
                fx += (ri - xi)
            if xi + ri > 1.0:
                fx -= (xi + ri - 1.0)
            if yi - ri < 0.0:
                fy += (ri - yi)
            if yi + ri > 1.0:
                fy -= (yi + ri - 1.0)
            norm = np.sqrt(fx * fx + fy * fy)
            if norm > 1e-12:
                centers[i, 0] = min(max(xi + step * fx / norm, 1e-9), 1 - 1e-9)
                centers[i, 1] = min(max(yi + step * fy / norm, 1e-9), 1 - 1e-9)
        if it % 15 == 14:
            radii = compute_max_radii(centers)
    return centers


def _fallback():
    """Simple valid hex-row packing, guaranteed feasible."""
    dx = 0.1
    r = 0.095
    for _ in range(100):
        dy = (1.0 - 2.0 * r) / 5.0
        r = min(0.1, np.sqrt(dy * dy + dx * dx) / 2.0)
    row_xs = [[0.2, 0.4, 0.6, 0.8], [0.1, 0.3, 0.5, 0.7, 0.9],
              [0.2, 0.4, 0.6, 0.8], [0.1, 0.3, 0.5, 0.7, 0.9],
              [0.2, 0.4, 0.6, 0.8], [0.1, 0.3, 0.5, 0.7]]
    c = []
    for k, xs in enumerate(row_xs):
        y = r + k * dy
        for x in xs:
            c.append([x, y])
    centers = np.array(c)
    radii = compute_max_radii(centers)
    return centers, radii


def construct_packing():
    """
    Construct a specific arrangement of 26 circles in a unit square
    that attempts to maximize the sum of their radii.

    Returns:
        Tuple of (centers, radii, sum_of_radii)
    """
    try:
        best = None
        for centers0 in build_seeds():
            centers = refine(centers0, iters=300)
            radii = compute_max_radii(centers)
            s = float(np.sum(radii))
            if best is None or s > best[2]:
                best = (centers.copy(), radii.copy(), s)

        centers, radii, sum_radii = best

        # Exact LP radii for the winning layout, if scipy is available.
        lp = solve_radii_lp(centers)
        if lp is not None:
            radii = lp

        # Shrink-only safety repair, then tiny safety shrink.
        n = centers.shape[0]
        D = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
        wall = np.minimum.reduce([
            centers[:, 0], centers[:, 1],
            1.0 - centers[:, 0], 1.0 - centers[:, 1]
        ])
        radii = np.minimum(np.maximum(radii, 0.0), wall)
        for _ in range(20):
            iu = np.triu_indices(n, 1)
            ssum = radii[iu[0]] + radii[iu[1]]
            bad = ssum > D[iu]
            if not np.any(bad):
                break
            scale = np.where(bad, D[iu] / np.maximum(ssum, 1e-15), 1.0)
            radii[iu[0]] *= scale
            radii[iu[1]] *= scale
            radii = np.minimum(radii, wall)
        radii = radii * (1.0 - 1e-6)
        sum_radii = float(np.sum(radii))
        return centers, radii, sum_radii
    except Exception:
        centers, radii = _fallback()
        radii = radii * (1.0 - 1e-6)
        return centers, radii, float(np.sum(radii))
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