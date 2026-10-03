# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """
    Construct a specific arrangement of 26 circles in a unit square
    that attempts to maximize the sum of their radii.

    Returns:
        Tuple of (centers, radii, sum_of_radii)
    """
    n = 26

    def hex_layout(rows, scale=1.0, wide_ends=False):
        R = len(rows)
        k = np.sqrt(3.0) / 2.0
        s_h = 1.0 / ((R - 1) * k + 1.0)
        s_w = 1.0 / max(rows)
        s = min(s_h, s_w) * scale
        r0 = s / 2.0
        g = s * k
        pts = []
        for ri, count in enumerate(rows):
            y = r0 + ri * g
            if count < 3 and wide_ends:
                sp = min(1.6 * s, (1.0 - s) / max(count - 1, 1))
            else:
                sp = s
            for c in range(count):
                x = 0.5 + (c - (count - 1) / 2.0) * sp
                x = min(max(x, 0.005), 0.995)
                pts.append([x, y])
        return np.array(pts)

    def lp_radii(centers):
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

    candidates = [
        ([5, 4, 5, 4, 5, 3], False),
        ([4, 5, 4, 5, 4, 4], False),
        ([4, 5, 4, 5, 4, 4], True),
        ([3, 4, 5, 4, 5, 3, 2], False),
        ([3, 4, 5, 4, 5, 3, 2], True),
        ([2, 5, 4, 4, 4, 5, 2], False),
        ([6, 5, 4, 5, 6], False),
        ([6, 4, 5, 4, 5, 2], False),
        ([4, 4, 5, 4, 5, 4], False),
        ([3, 4, 5, 4, 5, 5], False),
        ([5, 5, 4, 5, 4, 3], False),
        ([2, 4, 5, 6, 5, 4], False),
    ]

    scored = []
    for rows, wide in candidates:
        for scale in (0.96, 1.0, 1.02):
            centers_c = hex_layout(rows, scale=scale, wide_ends=wide)
            radii_c = lp_radii(centers_c)
            scored.append((np.sum(radii_c), centers_c, radii_c))
    scored.sort(key=lambda t: -t[0])

    centers, radii, sum_radii = scored[0][1], scored[0][2], scored[0][0]

    # Slack-directed hill climb within a time budget.
    def slack_dirs_and_scores(pts, rads):
        dirs = np.zeros((n, 2))
        scores = np.zeros(n)
        for i in range(n):
            x, y = pts[i]
            wall_d = np.array([x, y, 1 - x, 1 - y])
            wall_dirs = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
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
        overall_deadline = time.time() + 1.8
        n_starts = min(4, len(scored))
        best = (centers, radii, sum_radii)
        for start in range(n_starts):
            c, r, tot = scored[start][1], scored[start][2], scored[start][0]
            rng_s = np.random.default_rng(1000 + start)
            start_time = time.time()
            budget = (overall_deadline - time.time()) / max(n_starts - start, 1)
            step = 0.02
            while time.time() < min(start_time + budget, overall_deadline):
                dirs, scores = slack_dirs_and_scores(c, r)
                w = np.maximum(-scores, 0.01)
                w = w / w.sum()
                idx = int(rng_s.choice(n, p=w))
                trial = c.copy()
                trial[idx] += (0.7 * dirs[idx] + 0.3 * rng_s.normal(size=2)) * step
                trial = np.clip(trial, 0.005, 0.995)
                if np.allclose(trial[idx], c[idx]):
                    continue
                radii_t = lp_radii(trial)
                total_t = np.sum(radii_t)
                if total_t > tot + 1e-9:
                    c, r, tot = trial, radii_t, total_t
                    step = min(step * 1.15, 0.05)
                else:
                    if rng_s.random() < 0.7:
                        step = max(step * 0.92, 0.002)
                    else:
                        step = min(step * 1.4, 0.05)
                if time.time() > overall_deadline:
                    break
            if tot > best[2]:
                best = (c, r, tot)
        centers, radii, sum_radii = best
    except Exception:
        pass

    return centers, radii, sum_radii


def compute_max_radii(centers):
    """
    Compute the maximum possible radii for each circle position
    such that they don't overlap and stay within the unit square.
    """
    n = centers.shape[0]
    radii = np.ones(n)
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(x, y, 1 - x, 1 - y)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if radii[i] + radii[j] > dist:
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