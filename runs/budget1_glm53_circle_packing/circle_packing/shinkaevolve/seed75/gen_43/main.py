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
    # Hexagonal-lattice layout: 5 staggered rows with counts 6,5,6,5,6.
    # Horizontal spacing 0.2, vertical spacing 0.1*sqrt(3) ~ 0.1732.
    # This lattice supports 28 circles of radius 0.1; we drop 2 adjacent
    # circles from the middle row (n=26) to leave a growth hole.
    n = 26
    centers_list = []
    y0 = 0.1
    dy = 0.1 * np.sqrt(3.0)
    counts = [6, 5, 6, 5, 6]
    skip = {(2, 2), (2, 3)}  # remove two adjacent circles in the middle row

    def make_seed(dx, anchor):
        """Build seed with per-row-pair stagger offset dx and corner anchoring."""
        pts = []
        for row, cnt in enumerate(counts):
            y = y0 + row * dy
            for k in range(cnt):
                if (row, k) in skip:
                    continue
                if cnt == 6:
                    x = 0.2 * k
                else:
                    x = dx + 0.2 * k
                pts.append([x, y])
        cs = np.array(pts)
        # Pull the circle nearest each corner diagonally INTO the corner:
        # corner circles enjoy three-sided wall slack, so the LP grants
        # them large radii while interior circles stay small.
        corners = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
        for cx, cy in corners:
            d = np.sum((cs - np.array([cx, cy])) ** 2, axis=1)
            i = int(np.argmin(d))
            cs[i] = [
                anchor if cx == 0.0 else 1.0 - anchor,
                anchor if cy == 0.0 else 1.0 - anchor,
            ]
        return cs

    def cheap_refine(cs, n_iters=120, seed=999):
        """Short LP-based hill-climb + greedy growth pass; returns best config."""
        radii = solve_radii_lp(cs)
        if radii is None:
            radii = compute_max_radii(cs)
        b_sum = np.sum(radii)
        b_c = cs.copy()
        b_r = radii.copy()
        rng = np.random.default_rng(seed)
        for it in range(n_iters):
            frac = 1.0 - it / max(n_iters, 1)
            sigma = 0.006 + 0.03 * frac
            cand = b_c.copy()
            for i in range(n):
                x, y = cand[i]
                r = b_r[i]
                tol = 1e-7
                if r >= x - tol and r >= y - tol:
                    cand[i] -= [0.5 * sigma, 0.5 * sigma]
                elif r >= (1.0 - x) - tol and r >= y - tol:
                    cand[i, 0] += 0.5 * sigma
                    cand[i, 1] -= 0.5 * sigma
                elif r >= x - tol and r >= (1.0 - y) - tol:
                    cand[i, 0] -= 0.5 * sigma
                    cand[i, 1] += 0.5 * sigma
                elif r >= (1.0 - x) - tol and r >= (1.0 - y) - tol:
                    cand[i] += [0.5 * sigma, 0.5 * sigma]
                else:
                    if r >= x - tol:
                        cand[i, 0] += 0.5 * sigma
                    if r >= (1.0 - x) - tol:
                        cand[i, 0] -= 0.5 * sigma
                    if r >= y - tol:
                        cand[i, 1] += 0.5 * sigma
                    if r >= (1.0 - y) - tol:
                        cand[i, 1] -= 0.5 * sigma
            jitter = sigma * (0.3 + 0.7 * (1.0 - b_r / (b_r.max() + 1e-12)))
            cand += rng.normal(0.0, 1.0, cand.shape) * jitter[:, None]
            cand = np.clip(cand, 0.001, 0.999)
            result = solve_radii_lp(cand)
            if result is None:
                continue
            if np.sum(result) > b_sum:
                b_sum = np.sum(result)
                b_c = cand.copy()
                b_r = result.copy()
        # quick greedy shift sweep
        step = 0.01
        improved = True
        while improved and step > 2e-3:
            improved = False
            for i in range(n):
                for axis in (0, 1):
                    for direction in (+1.0, -1.0):
                        cand = b_c.copy()
                        cand[i, axis] = np.clip(
                            cand[i, axis] + direction * step, 0.001, 0.999
                        )
                        if cand[i, axis] == b_c[i, axis]:
                            continue
                        result = solve_radii_lp(cand)
                        if result is None:
                            continue
                        if np.sum(result) > b_sum + 1e-12:
                            b_sum = np.sum(result)
                            b_c = cand.copy()
                            b_r = result.copy()
                            improved = True
            if not improved:
                step *= 0.5
        return b_c, b_r, b_sum

    # Triage over stagger offsets and anchor distances, pick the best seed.
    candidates = []
    for dx in (0.08, 0.09, 0.10, 0.11, 0.12):
        for anchor in (0.06, 0.07, 0.08):
            cs = make_seed(dx, anchor)
            bc, br, bs = cheap_refine(cs)
            candidates.append((bs, bc, br))
    candidates.sort(key=lambda t: -t[0])
    best_sum, best_centers, best_radii = candidates[0]

    # ---- Hill-climb over center positions, re-solving the LP each step ----
    rng = np.random.default_rng(12345)
    n_iters = 300
    for it in range(n_iters):
        frac = 1.0 - it / n_iters
        sigma = 0.008 + 0.04 * frac

        cand = best_centers.copy()

        # Directed move, corner-aware:
        # - Circles touching TWO perpendicular walls have three-sided
        #   slack, so the LP grants them larger radii if they slide
        #   diagonally INTO the corner (only the diagonal neighbor binds).
        # - Circles touching a single wall are pushed away from it to
        #   free the binding wall constraint.
        for i in range(n):
            x, y = cand[i]
            r = best_radii[i]
            tol = 1e-7
            left = r >= x - tol
            right = r >= 1.0 - x - tol
            bot = r >= y - tol
            top = r >= 1.0 - y - tol
            if left and bot:
                cand[i] -= [0.5 * sigma, 0.5 * sigma]
            elif left and top:
                cand[i, 0] -= 0.5 * sigma
                cand[i, 1] += 0.5 * sigma
            elif right and bot:
                cand[i, 0] += 0.5 * sigma
                cand[i, 1] -= 0.5 * sigma
            elif right and top:
                cand[i] += [0.5 * sigma, 0.5 * sigma]
            else:
                if left:
                    cand[i, 0] += 0.5 * sigma
                if right:
                    cand[i, 0] -= 0.5 * sigma
                if bot:
                    cand[i, 1] += 0.5 * sigma
                if top:
                    cand[i, 1] -= 0.5 * sigma

        # Random jitter, larger for small circles (more freedom of movement)
        jitter = sigma * (0.3 + 0.7 * (1.0 - best_radii / (best_radii.max() + 1e-12)))
        cand += rng.normal(0.0, 1.0, cand.shape) * jitter[:, None]

        cand = np.clip(cand, 0.001, 0.999)

        result = solve_radii_lp(cand)
        if result is None:
            continue
        if np.sum(result) > best_sum:
            best_sum = np.sum(result)
            best_centers = cand.copy()
            best_radii = result.copy()

    # ---- Final greedy "growth pass": single-circle coordinate tweaks ----
    # Try small shifts for each circle, one coordinate at a time, and keep
    # any shift that strictly increases the LP-optimal sum. Shrink the step
    # when no improvement is found, until the step becomes negligible.
    step = 0.01
    improved = True
    while improved and step > 1e-4:
        improved = False
        for i in range(n):
            for axis in (0, 1):
                for direction in (+1.0, -1.0):
                    cand = best_centers.copy()
                    cand[i, axis] = np.clip(
                        cand[i, axis] + direction * step, 0.001, 0.999
                    )
                    if cand[i, axis] == best_centers[i, axis]:
                        continue
                    result = solve_radii_lp(cand)
                    if result is None:
                        continue
                    if np.sum(result) > best_sum + 1e-12:
                        best_sum = np.sum(result)
                        best_centers = cand.copy()
                        best_radii = result.copy()
                        improved = True
        if not improved:
            step *= 0.5

    centers = best_centers
    radii = best_radii * (1.0 - 1e-6)  # tiny safety shrink
    sum_radii = np.sum(radii)

    return centers, radii, sum_radii


def solve_radii_lp(centers, eps=1e-9):
    """
    Solve the linear program: maximize sum(r_i) subject to
      r_i + r_j <= dist(i, j)  for all pairs
      0 <= r_i <= min(x_i, y_i, 1-x_i, 1-y_i)
    Returns optimal radii array, or None if scipy is unavailable or LP fails.
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
    Compute the maximum possible radii for each circle position
    such that they don't overlap and stay within the unit square.

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates

    Returns:
        np.array of shape (n) with radius of each circle
    """
    n = centers.shape[0]

    # Distance from each center to the nearest square border
    border = np.minimum(centers.min(axis=1), (1.0 - centers).min(axis=1))

    # For fixed centers, maximizing sum(r_i) subject to
    #   r_i <= border_i  and  r_i + r_j <= dist_ij
    # is a linear program -> solve it exactly when scipy is available.
    try:
        from scipy.optimize import linprog

        A_ub = []
        b_ub = []
        for i in range(n):
            row = np.zeros(n)
            row[i] = 1.0
            A_ub.append(row)
            b_ub.append(border[i])
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                row = np.zeros(n)
                row[i] = 1.0
                row[j] = 1.0
                A_ub.append(row)
                b_ub.append(d)
        res = linprog(
            c=-np.ones(n),
            A_ub=np.array(A_ub),
            b_ub=np.array(b_ub),
            bounds=[(0.0, None)] * n,
            method="highs",
        )
        if res.success:
            return res.x
    except Exception:
        pass

    # Fallback: multi-pass proportional scaling (converges to a valid,
    # reasonably tight set of radii if scipy is unavailable).
    radii = border.copy()
    for _ in range(200):
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                s = radii[i] + radii[j]
                if s > dist and s > 0:
                    scale = dist / s
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