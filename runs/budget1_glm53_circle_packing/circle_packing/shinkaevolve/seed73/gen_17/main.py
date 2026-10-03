# EVOLVE-BLOCK-START
"""Force-relaxed hexagonal constructor for packing n=26 circles in a unit square."""
import numpy as np


def construct_packing():
    n = 26
    centers = _initial_centers(n)
    centers = _relax(centers, iters=600)
    radii = _max_radii(centers)
    sum_radii = float(np.sum(radii))
    return centers, radii, sum_radii


def _initial_centers(n):
    """Staggered rows 6,5,6,5,4 as a starting lattice."""
    counts = [6, 5, 6, 5, 4]
    r0 = 1.0 / (2 + 4 * np.sqrt(3))
    dy = np.sqrt(3) * r0
    pts = []
    y = r0
    for k in counts:
        dx = (1.0 - 2 * r0) / (k - 1)
        x0 = 0.5 - dx * (k - 1) / 2.0
        for i in range(k):
            pts.append([x0 + i * dx, y])
        y += dy
    return np.array(pts)


def _relax(centers, iters=600):
    """Push overlapping circles apart and away from walls."""
    c = centers.copy()
    n = c.shape[0]
    for it in range(iters):
        # current radii: each circle takes max allowed by neighbors+walls
        r = _quick_radii(c)
        # forces
        force = np.zeros_like(c)
        for i in range(n):
            for j in range(i + 1, n):
                d_vec = c[i] - c[j]
                d = np.sqrt(d_vec @ d_vec) + 1e-12
                overlap = r[i] + r[j] - d
                if overlap > 0:
                    push = 0.5 * overlap * d_vec / d
                    force[i] += push
                    force[j] -= push
        # wall forces
        for i in range(n):
            x, y = c[i]
            if x - r[i] < 0:
                force[i][0] += (r[i] - x)
            if 1 - x - r[i] < 0:
                force[i][0] -= (x + r[i] - 1)
            if y - r[i] < 0:
                force[i][1] += (r[i] - y)
            if 1 - y - r[i] < 0:
                force[i][1] -= (y + r[i] - 1)
        step = 0.35 * (1.0 - it / iters)  # annealed step
        c = np.clip(c + step * force, 0.001, 0.999)
        if it % 50 == 0 and np.abs(force).max() < 1e-6:
            break
    return c


def _quick_radii(c):
    """Fast greedy radii: wall limits, then pairwise shrink."""
    n = c.shape[0]
    r = np.minimum.reduce([c[:, 0], c[:, 1], 1 - c[:, 0], 1 - c[:, 1]])
    for _ in range(20):
        worst = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt(((c[i] - c[j]) ** 2).sum())
                s = r[i] + r[j]
                if s > d and s > 0:
                    sc = d / s
                    r[i] *= sc
                    r[j] *= sc
                    worst = max(worst, 1 - sc)
        if worst < 1e-10:
            break
    return r


def _max_radii(c):
    """Conservative, provably-valid radii with small safety margin."""
    n = c.shape[0]
    dist = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
    r = _quick_radii(c)
    # hard feasibility pass with margin
    eps = 1e-9
    for _ in range(30):
        ok = True
        for i in range(n):
            # wall constraint
            r[i] = min(r[i], c[i, 0], c[i, 1], 1 - c[i, 0], 1 - c[i, 1])
            for j in range(n):
                if i != j:
                    r[i] = min(r[i], dist[i, j] - r[j] - eps)
            if r[i] <= eps:
                r[i] = eps
                ok = False
        if ok:
            break
    r = np.maximum(r, 1e-9)
    # final guarantee: if anything still overlaps, shrink everything slightly
    for i in range(n):
        for j in range(i + 1, n):
            s = r[i] + r[j]
            if s > dist[i, j]:
                sc = dist[i, j] / s
                r[i] *= sc
                r[j] *= sc
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
