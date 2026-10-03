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

    def hex_layout(rows, scale=1.0, wide_ends=False):
        """Build a true triangular-lattice layout.

        Row spacing g = s*sqrt(3)/2, column spacing s, margin r = s/2, so
        neighboring rows form equilateral triangles (optimal local geometry).
        s is limited by both the square width (max row count) and height.
        """
        R = len(rows)
        k = np.sqrt(3.0) / 2.0
        s_h = 1.0 / ((R - 1) * k + 1.0)   # height: (R-1)*g + 2r <= 1
        s_w = 1.0 / max(rows)             # width: count*s <= 1
        s = min(s_h, s_w) * scale
        r0 = s / 2.0
        g = s * k
        pts = []
        for ri, count in enumerate(rows):
            y = r0 + ri * g
            if count < 3 and wide_ends:
                # Sparser end rows: spread circles a bit wider for the LP
                sp = min(1.6 * s, (1.0 - s) / max(count - 1, 1))
            else:
                sp = s
            for c in range(count):
                x = 0.5 + (c - (count - 1) / 2.0) * sp
                x = min(max(x, 0.005), 0.995)
                pts.append([x, y])
        return np.array(pts)

    def lp_radii(centers):
        """Exact optimal radii at fixed centers via LP:
        maximize sum(r) s.t. r_i + r_j <= d_ij, r_i <= border distance."""
        radii = compute_max_radii(centers)
        try:
            from scipy.optimize import linprog

            A_ub = []
            b_ub = []
            for i in range(n):
                for j in range(i + 1, n):
                    d = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                    if d < 1e-9:
                        continue
                    row = np.zeros(n)
                    row[i] = 1.0
                    row[j] = 1.0
                    A_ub.append(row)
                    b_ub.append(d)
            for i in range(n):
                x, y = centers[i]
                row = np.zeros(n)
                row[i] = 1.0
                A_ub.append(row)
                b_ub.append(min(x, y, 1 - x, 1 - y))
            res = linprog(c=-np.ones(n), A_ub=np.array(A_ub),
                          b_ub=np.array(b_ub), bounds=[(0, None)] * n,
                          method="highs")
            if res.success and np.sum(res.x) > np.sum(radii):
                radii = res.x
        except Exception:
            pass
        return radii

    # Candidate staggered row patterns totaling 26 circles
    candidates = [
        ([5, 4, 5, 4, 5, 3], False),
        ([4, 5, 4, 5, 4, 4], False),
        ([5, 4, 5, 4, 5, 3], True),   # widened sparse end row
        ([4, 5, 4, 5, 4, 4], True),
    ]

    best = None
    for rows, wide in candidates:
        for scale in (0.98, 1.0):
            centers_c = hex_layout(rows, scale=scale, wide_ends=wide)
            radii_c = lp_radii(centers_c)
            total = np.sum(radii_c)
            if best is None or total > best[2]:
                best = (centers_c, radii_c, total)

    centers, radii, sum_radii = best

    # Slack-directed hill-climb: perturb circles preferentially along
    # their direction of maximum available space, with a small random
    # component. Moves are pre-screened with a cheap slack proxy before
    # paying the LP re-solve cost, so the time budget goes further.
    def slack_dirs_and_scores(pts, rads):
        """For each circle, direction of max slack (away from nearest
        blocker: wall or neighbor) and a slack score (how roomy it is)."""
        dirs = np.zeros((n, 2))
        scores = np.zeros(n)
        for i in range(n):
            x, y = pts[i]
            wall_d = np.array([x, y, 1 - x, 1 - y])  # L,R,B,T distances
            wall_dirs = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
            # nearest neighbor direction (pointing away from neighbor)
            nd = pts - pts[i]
            dists = np.linalg.norm(nd, axis=1)
            dists[i] = np.inf
            j = int(np.argmin(dists))
            gap_n = dists[j] - rads[i] - rads[j]
            wi = int(np.argmin(wall_d))
            gap_w = wall_d[wi] - rads[i]
            if gap_w < gap_n:
                dirs[i] = wall_dirs[wi]
                scores[i] = gap_w
            else:
                dirs[i] = nd[j] / (dists[j] + 1e-12)
                scores[i] = gap_n
        return dirs, scores

    try:
        import time
        rng = np.random.default_rng(0)
        deadline = time.time() + 1.2
        step = 0.02
        while time.time() < deadline:
            dirs, scores = slack_dirs_and_scores(centers, radii)
            # Bias selection toward tight circles (they gain most from a move)
            w = np.maximum(-scores, 0.01)
            w = w / w.sum()
            idx = int(rng.choice(n, p=w))
            trial = centers.copy()
            # 70% of the step along the slack direction, 30% random
            trial[idx] += (0.7 * dirs[idx] + 0.3 * rng.normal(size=2)) * step
            trial = np.clip(trial, 0.005, 0.995)
            if np.allclose(trial[idx], centers[idx]):
                continue
            radii_t = lp_radii(trial)
            total_t = np.sum(radii_t)
            if total_t > sum_radii + 1e-9:
                centers, radii, sum_radii = trial, radii_t, total_t
                step = min(step * 1.15, 0.05)
            else:
                if rng.random() < 0.7:
                    step = max(step * 0.92, 0.002)
                else:
                    step = min(step * 1.4, 0.05)
    except Exception:
        pass

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
    radii = np.ones(n)

    # First, limit by distance to square borders
    for i in range(n):
        x, y = centers[i]
        # Distance to borders
        radii[i] = min(x, y, 1 - x, 1 - y)

    # Then, limit by distance to other circles
    # Each pair of circles with centers at distance d can have
    # sum of radii at most d to avoid overlap
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))

            # If current radii would cause overlap
            if radii[i] + radii[j] > dist:
                # Scale both radii proportionally
                scale = dist / (radii[i] + radii[j])
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