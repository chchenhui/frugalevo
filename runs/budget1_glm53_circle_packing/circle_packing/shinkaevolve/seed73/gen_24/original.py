# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def _hex_rows(r):
    """Return list of (count, y, offset) row specs for the hex core."""
    dy = np.sqrt(3.0) * r
    rows = []
    for k, cnt in enumerate([5, 4, 5, 4, 5]):
        off = r if cnt == 5 else 2 * r
        rows.append((cnt, r + k * dy, off))
    return rows


def _place_core(r):
    centers = []
    for cnt, y, off in _hex_rows(r):
        for j in range(cnt):
            centers.append((off + 2 * r * j, y))
    return centers


def construct_packing():
    n = 26
    # Hex core: 23 circles in 5 rows (5,4,5,4,5)
    r0 = 1.0 / (2.0 + 5.0 * np.sqrt(3.0))
    pts = _place_core(r0)
    # Corner fillers: small circles tucked into each corner gap
    c = 0.10
    pts += [(c, c), (1 - c, c), (c, 1 - c), (1 - c, 1 - c)]
    # 26 total? core 5+4+5+4+5=23 + 4 = 27 -> drop one core circle
    pts = pts[:26]
    centers = np.array(pts)

    # Vectorized greedy radius expansion
    d = np.sqrt(((centers[:, None] - centers[None, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    radii = np.full(26, 0.4 * r0)
    for _ in range(200):
        wall = np.minimum(centers.min(axis=1), 1.0 - centers.max(axis=1))
        nb = (d - radii[None, :]).min(axis=1)
        newr = np.minimum(wall, nb)
        newr = np.maximum(newr, radii * 0.999)  # monotone shrink-safe
        radii = 0.5 * (radii + newr)
    radii = np.minimum(radii, np.minimum(centers.min(axis=1), 1.0 - centers.max(axis=1)))
    radii = np.maximum(radii, 1e-9)

    # Final exact pass: ensure strict validity
    for _ in range(300):
        ok = True
        for i in range(26):
            wall = min(centers[i, 0], centers[i, 1], 1 - centers[i, 0], 1 - centers[i, 1])
            lim = wall
            for j in range(26):
                if j != i:
                    lim = min(lim, d[i, j] - radii[j])
            lim = max(lim, 1e-9)
            if lim < radii[i] - 1e-12:
                radii[i] = lim
                ok = False
            elif lim > radii[i] + 1e-9:
                radii[i] = min(lim, radii[i] + 0.05 * r0)
                ok = False
        if ok:
            break

    sum_radii = float(np.sum(radii))
    return centers, radii, sum_radii


def compute_max_radii(centers):
    n = centers.shape[0]
    radii = np.array([min(c[0], c[1], 1 - c[0], 1 - c[1]) for c in centers])
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            if radii[i] + radii[j] > dist:
                s = dist / (radii[i] + radii[j])
                radii[i] *= s
                radii[j] *= s
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