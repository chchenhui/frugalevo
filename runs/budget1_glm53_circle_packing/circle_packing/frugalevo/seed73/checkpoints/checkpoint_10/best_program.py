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

    """Approach: mixed-radius two-scale seed. Four large circles sit in
    the corners (radius limited only by two walls), three medium circles
    run along each edge between the corner circles, and ten small circles
    fill the interior on a staggered lattice. This realizes the varied-
    radius contact topology of the best known n=26 packings, which a
    uniform lattice seed cannot reach by local moves. A small sweep over
    corner radius and edge radius generates seed variants; each is
    polished with one bounded SLSQP run (radii always recomputed via
    compute_max_radii, so validity holds). The best strictly-feasible
    variant is returned; the incumbent hex-lattice result is the
    fallback if no variant improves on it."""

    def _hex_seed():
        counts = [6, 5, 6, 5, 4]
        dx = 0.16
        dy = 0.2 * np.sqrt(3) / 2
        c = np.zeros((26, 2))
        k = 0
        for row, cnt in enumerate(counts):
            y = 0.1 + row * dy
            x0 = 0.5 - (cnt - 1) * dx / 2
            for col in range(cnt):
                c[k] = [x0 + dx * col, y]
                k += 1
        return c

    def _mixed_seed(rc, re):
        """4 corner circles (radius rc), 3 edge circles per side
        (nominal radius re), 10 interior circles on a staggered lattice."""
        c = np.zeros((26, 2))
        k = 0
        # corners
        for cx, cy in [(rc, rc), (1 - rc, rc), (rc, 1 - rc), (1 - rc, 1 - rc)]:
            c[k] = [cx, cy]
            k += 1
        # edge chains: 3 circles per edge at nominal radius re, tangent
        # to the wall, evenly spaced between the corner circles
        t = np.array([0.25, 0.5, 0.75])
        for a in range(3):  # bottom edge
            c[k] = [rc + (1 - 2 * rc) * t[a], re]
            k += 1
        for a in range(3):  # top edge
            c[k] = [rc + (1 - 2 * rc) * t[a], 1 - re]
            k += 1
        for a in range(3):  # left edge
            c[k] = [re, rc + (1 - 2 * rc) * t[a]]
            k += 1
        for a in range(3):  # right edge
            c[k] = [1 - re, rc + (1 - 2 * rc) * t[a]]
            k += 1
        # interior: 10 circles, staggered rows 4/3/3 centered at 0.5
        p = 0.17
        rows = [(4, 0.5 - p), (3, 0.5), (3, 0.5 + p)]
        for cnt, y in rows:
            x0 = 0.5 - (cnt - 1) * p / 2
            for col in range(cnt):
                c[k] = [x0 + p * col, y]
                k += 1
        return c

    # Incumbent hex result as fallback
    best_c = _hex_seed()
    best_r = compute_max_radii(best_c)
    best_c, best_r = _refine(best_c, best_r)
    best_sum = np.sum(best_r)

    # Sweep mixed-radius seeds; one bounded SLSQP polish per variant
    for rc in (0.13, 0.145, 0.16):
        for re in (0.09, 0.10, 0.11):
            c0 = _mixed_seed(rc, re)
            r0 = compute_max_radii(c0)
            if np.sum(r0) <= 1e-6:
                continue
            c1, r1 = _refine(c0, r0)
            s1 = np.sum(r1)
            if s1 > best_sum + 1e-9:
                best_sum = s1
                best_c, best_r = c1, r1

    centers, radii = best_c, best_r
    sum_radii = np.sum(radii)

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

    # Limit by distance to square borders
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(x, y, 1 - x, 1 - y)

    # Iteratively enforce pairwise non-overlap until convergence so that
    # shrinkage from one pair propagates correctly to all other pairs
    for _ in range(200):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                if radii[i] + radii[j] > dist:
                    # Shrink the larger radius first (keeps small circles
                    # small and preserves the intended size hierarchy)
                    excess = radii[i] + radii[j] - dist
                    if radii[i] >= radii[j]:
                        radii[i] = max(radii[i] - excess, 1e-12)
                    else:
                        radii[j] = max(radii[j] - excess, 1e-12)
                    changed = True
        if not changed:
            break

    return radii


def _refine(centers, radii):
    """Locally optimize centers and radii with SLSQP starting from a
    feasible configuration, maximizing the sum of radii subject to wall
    constraints (r <= min(x, y, 1-x, 1-y)) and pairwise non-overlap
    (r_i + r_j <= distance). The optimizer's radii are discarded; radii
    are recomputed from the refined centers via compute_max_radii so the
    returned configuration is always valid. Falls back to the input if
    scipy is unavailable or no strict improvement is found."""
    try:
        from scipy.optimize import minimize
    except ImportError:
        return centers, radii

    n = len(centers)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    def obj(z):
        return -np.sum(z[2 * n:])

    cons = []
    for i in range(n):
        cons.append({"type": "ineq",
                     "fun": lambda z, i=i: (np.array([z[2 * i], z[2 * i + 1],
                                                      1 - z[2 * i], 1 - z[2 * i + 1]])
                                            - z[2 * n + i])})
    for (i, j) in pairs:
        cons.append({"type": "ineq",
                     "fun": lambda z, i=i, j=j:
                     np.sqrt(np.sum((z[2 * i:2 * i + 2] - z[2 * j:2 * j + 2]) ** 2))
                     - z[2 * n + i] - z[2 * n + j]})

    z0 = np.concatenate([centers.ravel(), radii])
    try:
        res = minimize(obj, z0, method="SLSQP", constraints=cons,
                       options={"maxiter": 300, "ftol": 1e-10})
    except Exception:
        return centers, radii

    c2 = np.asarray(res.x[:2 * n]).reshape(n, 2)
    if np.all(c2 > 1e-9) and np.all(c2 < 1 - 1e-9):
        r2 = compute_max_radii(c2)
        if np.sum(r2) > np.sum(radii) + 1e-9:
            return c2, r2
    return centers, radii


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
