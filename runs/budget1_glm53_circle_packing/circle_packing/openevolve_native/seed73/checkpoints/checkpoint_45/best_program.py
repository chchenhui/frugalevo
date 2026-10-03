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
    best = None

    # Hex-lattice row splits summing to 26, used as hill-climb seeds.
    row_sets = [
        [5, 4, 5, 4, 5, 3],
        [4, 5, 4, 5, 4, 4],
        [3, 5, 4, 5, 4, 5],
        [5, 5, 4, 5, 4, 3],
        [4, 4, 5, 4, 5, 4],
        [5, 4, 4, 5, 4, 4],
    ]
    base_r0 = 1.0 / (2.0 + 5.0 * np.sqrt(3.0))

    seeds = []
    for row_counts in row_sets:
        for scale in (0.85, 1.0, 1.15):
            r0 = base_r0 * scale
            dy = np.sqrt(3.0) * r0
            dx = 2.0 * r0
            centers = np.zeros((n, 2))
            idx = 0
            for row, count in enumerate(row_counts):
                y = r0 + row * dy
                # short rows shifted by half a spacing (hex offset)
                x0 = r0 if count % 2 == 1 else 2.0 * r0
                for j in range(count):
                    centers[idx] = [x0 + j * dx, y]
                    idx += 1
            seeds.append(centers)

    # Corner-anchored family: 4 large circles near the corners (which pure
    # row lattices waste) plus a 22-circle interior hex lattice.
    for c0 in (0.13, 0.16, 0.19):
        for scale in (0.85, 1.0):
            centers = np.zeros((n, 2))
            centers[0] = [c0, c0]
            centers[1] = [1 - c0, c0]
            centers[2] = [c0, 1 - c0]
            centers[3] = [1 - c0, 1 - c0]
            lo, hi = c0, 1.0 - c0
            rows = [5, 4, 5, 4, 4]
            r0 = (hi - lo) / (2.0 + 4.0 * np.sqrt(3.0)) * scale
            dy = np.sqrt(3.0) * r0
            dx = 2.0 * r0
            idx = 4
            ok = True
            for row, count in enumerate(rows):
                y = lo + r0 + row * dy
                if y > hi - r0:
                    ok = False
                    break
                width = (count - 1) * dx
                x0 = lo + ((hi - lo) - width) / 2.0
                for j in range(count):
                    centers[idx] = [x0 + j * dx, y]
                    idx += 1
            if ok and idx == n:
                seeds.append(centers)

    def evaluate(c):
        radii = compute_max_radii(c)
        return float(np.sum(radii)), c, radii

    # Score all seeds; keep the top ones for hill-climbing.
    scored = sorted((evaluate(c) for c in seeds), key=lambda t: -t[0])
    best = scored[0]

    # Stochastic hill-climb on centers: alternate all-point jitter with
    # single-circle targeted moves (which explore the contact graph far
    # more efficiently). Amplitude shrinks while improving and collapses
    # when stuck (exploitation focus).
    rng = np.random.RandomState(1234)
    for s0, c0, _ in scored[:4]:
        cur_s, cur_c = s0, c0.copy()
        amp = 0.05
        while amp > 5e-4:
            hit = False
            for _ in range(10):
                # all-point jitter
                cand = cur_c + rng.uniform(-amp, amp, cur_c.shape)
                cand[:, 0] = np.clip(cand[:, 0], 0.01, 0.99)
                cand[:, 1] = np.clip(cand[:, 1], 0.01, 0.99)
                s, cc, rr = evaluate(cand)
                if s > cur_s:
                    cur_s, cur_c = s, cc
                    hit = True
                if s > best[0]:
                    best = (s, cc, rr)
                # single-circle targeted moves
                for _ in range(6):
                    i = rng.randint(n)
                    cand = cur_c.copy()
                    cand[i] += rng.uniform(-2 * amp, 2 * amp, 2)
                    cand[:, 0] = np.clip(cand[:, 0], 0.01, 0.99)
                    cand[:, 1] = np.clip(cand[:, 1], 0.01, 0.99)
                    s, cc, rr = evaluate(cand)
                    if s > cur_s:
                        cur_s, cur_c = s, cc
                        hit = True
                    if s > best[0]:
                        best = (s, cc, rr)
            amp = amp * (0.93 if hit else 0.6)

    sum_radii, centers, radii = best
    return centers, radii, sum_radii


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
    dist = np.sqrt(np.sum((centers[:, None, :] - centers[None, :, :]) ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)

    # distance to nearest wall caps every radius
    wall = np.minimum.reduce([centers[:, 0], centers[:, 1],
                             1.0 - centers[:, 0], 1.0 - centers[:, 1]])

    def grow_sweeps(order):
        radii = np.full(n, 1e-6)
        for _ in range(50):
            changed = False
            for i in order:
                limit = min(wall[i], np.min(dist[i] - radii))
                if limit > radii[i] + 1e-12:
                    radii[i] = limit
                    changed = True
            if not changed:
                break
        return radii

    def fix_pass(radii):
        for i in range(n):
            for j in range(i + 1, n):
                s = radii[i] + radii[j]
                d = dist[i, j]
                if s > d:
                    scale = d / s
                    radii[i] *= scale
                    radii[j] *= scale
        return radii

    orders = [
        list(range(n)),
        list(range(n - 1, -1, -1)),
        list(np.argsort(wall)),
        list(np.argsort(-wall)),
    ]

    best_radii = None
    best_sum = -1.0
    for order in orders:
        r = grow_sweeps(order)
        r = fix_pass(r)
        s = np.sum(r)
        if s > best_sum:
            best_sum = s
            best_radii = r
    return best_radii


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
