# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """
    Hexagonal-lattice seed (rows 5,4,5,4,5,3 = 26 circles, uniform radius
    r = 1/(2+5*sqrt(3)) ~ 0.0938, sum ~ 2.439) refined by hill-climbing:
    for fixed centers, optimal radii solve the LP
        max sum(r) s.t. r_i + r_j <= d_ij, 0 <= r_i <= wall cap_i,
    which is always feasible. Perturb one center, re-solve the LP, keep the
    move only if the objective strictly improves. Result is always valid
    and never worse than the seed. Only nearby pairs (d < 0.4) enter the LP.

    Returns:
        Tuple of (centers, radii, sum_of_radii)
    """
    n = 26
    centers = np.zeros((n, 2))

    r = 1.0 / (2.0 + 5.0 * np.sqrt(3.0))
    dy = np.sqrt(3.0) * r  # vertical spacing between adjacent rows

    row_widths = [5, 4, 5, 4, 5, 3]
    assert sum(row_widths) == n

    idx = 0
    for row, w in enumerate(row_widths):
        y = r + row * dy
        if row % 2 == 0:
            c = w - 1  # even offset parity (w is odd here)
        else:
            c = w - 1 if (w - 1) % 2 == 1 else w  # force odd parity
        for k in range(w):
            x = 0.5 + (2 * k - c) * r
            centers[idx] = [x, y]
            idx += 1

    fallback = (np.full(n, r), float(n * r))
    try:
        from scipy.optimize import linprog
    except Exception:
        return centers, fallback[0], fallback[1]

    def solve_radii(C):
        """LP for optimal radii given centers (r_i = 0 is always feasible)."""
        diff = C[:, None, :] - C[None, :, :]
        D = np.sqrt((diff * diff).sum(axis=2))
        iu, ju = np.triu_indices(n, 1)
        d = D[iu, ju]
        keep = d < 0.4  # far pairs cannot bind (wall caps are small)
        iu, ju, d = iu[keep], ju[keep], d[keep]
        m = len(d)
        A = np.zeros((m, n))
        A[np.arange(m), iu] = 1.0
        A[np.arange(m), ju] = 1.0
        caps = np.minimum(np.minimum(C[:, 0], C[:, 1]),
                          np.minimum(1.0 - C[:, 0], 1.0 - C[:, 1]))
        res = linprog(c=-np.ones(n), A_ub=A, b_ub=d,
                      bounds=list(zip(np.zeros(n), caps)),
                      method="highs")
        if not res.success:
            return None, -np.inf
        return res.x, -res.fun

    cur_rad, cur_sum = solve_radii(centers)
    if cur_rad is None:
        return centers, fallback[0], fallback[1]
    if cur_sum < fallback[1]:
        cur_rad, cur_sum = fallback[0], fallback[1]

    import time
    rng = np.random.default_rng(12345)
    deadline = time.time() + 25.0
    scale = 0.04
    it = 0
    while time.time() < deadline:
        it += 1
        if it % 200 == 0:
            scale = max(scale * 0.85, 0.003)
        i = rng.integers(n)
        C2 = centers.copy()
        C2[i] += rng.normal(0.0, scale, 2)
        C2[i] = np.clip(C2[i], 0.02, 0.98)
        rad2, s2 = solve_radii(C2)
        if rad2 is not None and s2 > cur_sum + 1e-9:
            centers = C2
            cur_sum = s2
            cur_rad = rad2

    return centers, cur_rad, float(cur_sum)


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
