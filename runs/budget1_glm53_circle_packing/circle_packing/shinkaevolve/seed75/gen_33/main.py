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
    # Initialize arrays for 26 circles
    n = 26
    centers = np.zeros((n, 2))

    # Corner-anchored seed: 4 large circles pinned in the corners
    # (corners have three-sided wall slack, so the radii solver can
    # grow them well beyond the interior uniform cap), plus 22
    # interior circles in staggered rows.
    rc = 0.12  # corner-circle distance from each wall
    centers[0] = [rc, rc]
    centers[1] = [1.0 - rc, rc]
    centers[2] = [rc, 1.0 - rc]
    centers[3] = [1.0 - rc, 1.0 - rc]

    # Interior: 22 circles in 5 staggered rows (bottom to top)
    # of 5, 4, 5, 4, 4, spaced between the corner circles.
    row_counts = [5, 4, 5, 4, 4]
    s = 0.16
    y_lo = rc
    y_hi = 1.0 - rc
    gap = (y_hi - y_lo) / (len(row_counts) - 1)
    idx = 4
    for row, count in enumerate(row_counts):
        y = y_lo + row * gap
        for j in range(count):
            x = 0.5 + (j - (count - 1) / 2.0) * s
            centers[idx] = [x, y]
            idx += 1

    # Deterministic refinement: grow per-circle radii, then nudge each
    # center away from its currently active constraints (touching
    # neighbors and walls), and repeat with a decreasing step size.
    radii = compute_max_radii(centers)
    n_iter = 400
    for it in range(n_iter):
        step = 0.02 * (1.0 - it / n_iter) + 0.0005
        for i in range(n):
            xi, yi = centers[i]
            ri = radii[i]
            fx = 0.0
            fy = 0.0
            for j in range(n):
                if j == i:
                    continue
                dx = xi - centers[j, 0]
                dy = yi - centers[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                overlap = ri + radii[j] - dist
                if overlap > 0.0 and dist > 1e-12:
                    fx += overlap * dx / dist
                    fy += overlap * dy / dist
            # Wall pushes
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
                centers[i, 0] = min(max(xi + step * fx / norm, 1e-9), 1.0 - 1e-9)
                centers[i, 1] = min(max(yi + step * fy / norm, 1e-9), 1.0 - 1e-9)
        if it % 20 == 19:
            radii = compute_max_radii(centers)

    # Final per-circle radii: use the exact LP (optimal unequal radii
    # at these fixed centers) when available; fall back to the fast
    # water-filling heuristic otherwise.
    result = solve_radii_lp(centers)
    if result is not None and np.sum(result) > np.sum(compute_max_radii(centers)):
        radii = result
    else:
        radii = compute_max_radii(centers)

    # Tiny safety shrink so the packing is strictly valid
    radii = radii * (1.0 - 1e-6)

    # Calculate the sum of radii
    sum_radii = float(np.sum(radii))

    return centers, radii, sum_radii


def solve_radii_lp(centers, eps=1e-9):
    """
    Solve the linear program: maximize sum(r_i) subject to
      r_i + r_j <= dist(i, j)  for all pairs
      0 <= r_i <= min(x_i, y_i, 1-x_i, 1-y_i)
    Returns optimal radii array, or None if the LP fails.
    """
    try:
        from scipy.optimize import linprog
    except ImportError:
        return None

    n = centers.shape[0]
    c = -np.ones(n)

    rows = []
    rhs = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            rows.append(row)
            rhs.append(d - 2.0 * eps)

    for i in range(n):
        x, y = centers[i]
        for lim in (x - eps, y - eps, 1.0 - x - eps, 1.0 - y - eps):
            row = np.zeros(n)
            row[i] = 1.0
            rows.append(row)
            rhs.append(max(lim, 0.0))

    A_ub = np.vstack(rows)
    b_ub = np.array(rhs)

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0.0, None)] * n,
                  method="highs")
    if not res.success:
        return None
    return np.maximum(res.x, 0.0)


def compute_max_radii(centers):
    """
    Compute per-circle radii via Gauss-Seidel "water filling": each
    circle grows up to the largest radius allowed by its wall distances
    and its neighbors' current radii. Iterated to (near) convergence,
    then a monotone repair pass (which only ever shrinks radii)
    guarantees the final radii are feasible: inside the square and
    pairwise non-overlapping.

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates

    Returns:
        np.array of shape (n) with radius of each circle
    """
    n = centers.shape[0]

    # Precompute pairwise distances and wall-distance caps
    D = np.zeros((n, n))
    wall = np.zeros(n)
    for i in range(n):
        x, y = centers[i]
        wall[i] = min(x, y, 1.0 - x, 1.0 - y)
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            D[i, j] = d
            D[j, i] = d

    # Gauss-Seidel growth
    r = np.zeros(n)
    for _ in range(300):
        changed = False
        for i in range(n):
            cap = wall[i]
            for j in range(n):
                if j != i and D[i, j] - r[j] < cap:
                    cap = D[i, j] - r[j]
            if cap < 0.0:
                cap = 0.0
            if abs(cap - r[i]) > 1e-12:
                changed = True
            r[i] = cap
        if not changed:
            break

    # Monotone feasibility repair: clamp to walls, then shrink pairs.
    # Both operations only reduce radii, so the result is guaranteed valid.
    r = np.minimum(r, wall)
    for i in range(n):
        for j in range(i + 1, n):
            if r[i] + r[j] > D[i, j]:
                scale = D[i, j] / (r[i] + r[j])
                r[i] *= scale
                r[j] *= scale
    r = np.minimum(r, wall)

    return r


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