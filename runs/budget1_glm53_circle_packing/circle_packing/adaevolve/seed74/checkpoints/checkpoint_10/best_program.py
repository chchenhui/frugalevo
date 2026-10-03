# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """
    Sweep multiple layout families (staggered hexagonal rows and a 5x5
    grid seed with an inserted 26th circle), solve the exact radius-
    maximization LP for each candidate, keep the best, then hill-climb
    the circle centers under a wall-clock budget.

    Returns:
        Tuple of (centers, radii, sum_of_radii)
        centers: np.array of shape (26, 2) with (x, y) coordinates
        radii: np.array of shape (26) with radius of each circle
        sum_of_radii: Sum of all radii
    """
    best = None

    def consider(centers):
        nonlocal best
        if centers is None:
            return
        radii = compute_max_radii(centers)
        s = float(np.sum(radii))
        if best is None or s > best[2]:
            best = (centers, radii, s)

    # Family 1: staggered hexagonal-row layouts over many row patterns,
    # base radii, and stagger offsets. Rows are spread over full height.
    row_patterns = [
        [5, 4, 5, 4, 5, 3],
        [4, 5, 4, 5, 4, 4],
        [5, 5, 4, 5, 4, 3],
        [5, 4, 5, 4, 4, 4],
        [4, 4, 5, 4, 5, 4],
        [5, 4, 4, 5, 4, 4],
        [3, 5, 4, 5, 4, 5],
        [4, 4, 4, 5, 4, 5],
        [4, 5, 5, 4, 5, 3],
        [5, 4, 4, 5, 5, 3],
        [4, 4, 5, 5, 4, 4],
        [4, 5, 4, 4, 5, 4],
        [5, 3, 5, 4, 5, 4],
        [6, 4, 5, 4, 5, 2],
    ]
    for counts in row_patterns:
        for r0 in np.linspace(0.075, 0.125, 26):
            for stagger in (0.0, 0.5, 1.0):
                consider(build_layout(counts, r0, stagger))

    # Family 2: 5x5 grid at pitch 0.2 (each circle can have radius 0.1,
    # sum 2.5) plus a 26th circle inserted into an interior grid gap
    # (its four neighbours are at distance sqrt(2)/10, so it adds ~0.041).
    for gap in [(0.2, 0.2), (0.4, 0.2), (0.2, 0.4), (0.4, 0.4), (0.5, 0.5)]:
        consider(build_grid_layout(gap))

    centers, radii, sum_radii = best
    centers, radii, sum_radii = refine(centers, radii, sum_radii, budget=200.0)
    return centers, radii, sum_radii


def refine(centers, radii, sum_radii, budget=200.0):
    """Hill-climb on circle centers: repeatedly pick a circle, try small
    random perturbations, re-solve the radius LP, keep improvements.
    Halve the step when a full pass yields no gain; stop under budget."""
    import time
    t0 = time.time()
    rng = np.random.default_rng(0)
    centers = centers.copy()
    n = centers.shape[0]
    step = 0.02
    while time.time() - t0 < budget:
        improved = False
        for i in rng.permutation(n):
            if time.time() - t0 >= budget:
                break
            base = centers[i].copy()
            for _ in range(3):
                dx, dy = rng.uniform(-step, step, 2)
                centers[i] = base + [dx, dy]
                if not (0.01 <= centers[i, 0] <= 0.99 and
                        0.01 <= centers[i, 1] <= 0.99):
                    continue
                r2 = compute_max_radii(centers)
                s2 = float(np.sum(r2))
                if s2 > sum_radii + 1e-9:
                    radii, sum_radii = r2, s2
                    improved = True
                    break
            else:
                centers[i] = base
        if not improved:
            step *= 0.5
            if step < 1e-4:
                break
    return centers, radii, sum_radii


def build_layout(counts, r0, stagger):
    """Staggered hex rows: horizontal pitch 2*r0, rows spread evenly over
    the full unit height, alternate rows offset by stagger*r0."""
    n = sum(counts)
    if n != 26:
        return None
    nrow = len(counts)
    dy = (1.0 - 2.0 * r0) / (nrow - 1)
    centers = np.zeros((n, 2))
    idx = 0
    for row, count in enumerate(counts):
        y = r0 + row * dy
        offset = stagger * r0 if row % 2 == 1 else 0.0
        start = 0.5 - (count - 1) * r0 + offset
        for k in range(count):
            centers[idx] = [start + 2.0 * r0 * k, y]
            idx += 1
    if (centers < 1e-9).any() or (centers > 1 - 1e-9).any():
        return None
    return centers


def build_grid_layout(gap):
    """5x5 grid of circles at pitch 0.2 (each can carry radius 0.1) plus
    one extra circle inserted at the given interior gap position."""
    centers = [[0.1 + 0.2 * i, 0.1 + 0.2 * j]
               for j in range(5) for i in range(5)]
    centers.append(list(gap))
    return np.array(centers)


def compute_max_radii(centers):
    """
    Exact radius maximization via linear programming: for fixed centers,
    maximize sum(r_i) subject to r_i + r_j <= d_ij and r_i <= wall_i.

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates

    Returns:
        np.array of shape (n) with radius of each circle
    """
    from scipy.optimize import linprog

    n = centers.shape[0]
    d = np.sqrt(((centers[:, None, :] - centers[None, :, :]) ** 2).sum(-1))
    wall = np.minimum.reduce([
        centers[:, 0], centers[:, 1], 1.0 - centers[:, 0], 1.0 - centers[:, 1]
    ])

    A, b = [], []
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A.append(row)
            b.append(d[i, j])
        row = np.zeros(n)
        row[i] = 1.0
        A.append(row)
        b.append(wall[i])

    res = linprog(-np.ones(n), A_ub=np.array(A), b_ub=np.array(b),
                  bounds=[(0.0, None)] * n, method="highs")
    if res.success:
        return res.x
    # fallback: iterative proportional shrinking
    radii = wall.copy()
    for _ in range(200):
        viol = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                s = radii[i] + radii[j]
                if s > d[i, j]:
                    sc = d[i, j] / s
                    radii[i] *= sc
                    radii[j] *= sc
                    viol = max(viol, s - d[i, j])
        if viol < 1e-12:
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
