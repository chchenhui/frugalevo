# EVOLVE-BLOCK-START
"""Hexagonal staggered-row constructor for packing n=26 circles in a unit square."""
import numpy as np


def construct_packing():
    """
    Enumerate hexagonal lattice candidates, refine each with
    nudge-and-grow relaxation, then SLSQP-polish the best layout
    jointly over centers and radii.

    Returns:
        centers, radii, sum_radii
    """
    n = 26
    sqrt3 = np.sqrt(3.0)
    r0 = min(1.0 / 10.0, 1.0 / (2.0 + 5.0 * sqrt3))

    count_sets = [[5, 4, 5, 4, 5, 3], [5, 4, 5, 4, 4, 4],
                  [4, 5, 4, 5, 4, 4], [3, 5, 4, 5, 4, 5],
                  [6, 5, 6, 5, 4], [5, 6, 5, 6, 4]]
    stretch_factors = [1.0, 1.03, 1.06]

    candidates = []
    for counts in count_sets:
        for sf in stretch_factors:
            r = r0 * sf
            dx = 2.0 * r
            dy = sqrt3 * r
            pts = []
            for row, c in enumerate(counts):
                y = r + row * dy
                span = 2.0 * r + (c - 1) * dx
                start = (1.0 - span) / 2.0 + r
                for k in range(c):
                    pts.append([start + k * dx, y])
            pts = np.clip(np.array(pts), 0.005, 0.995)
            candidates.append(pts)

    best = None
    for cand in candidates:
        centers, radii, s = refine_layout(cand, r0)
        if best is None or s > best[2]:
            best = (centers, radii, s)

    centers, radii, sum_radii = best

    # SLSQP polish: jointly optimize centers and radii to maximize
    # sum(r) subject to pairwise and wall constraints.
    try:
        polished = slsqp_polish(centers, radii)
        if polished is not None and polished[2] > sum_radii:
            centers, radii, sum_radii = polished
    except Exception:
        pass

    return centers, radii, sum_radii


def slsqp_polish(centers, radii):
    """Joint SLSQP refinement of centers and radii with repair fallback."""
    from scipy.optimize import minimize
    n = centers.shape[0]
    pairs = np.array(np.triu_indices(n, k=1)).T

    def unpack(z):
        return z[:2 * n].reshape(n, 2), z[2 * n:]

    def neg_obj(z):
        return -z[2 * n:].sum()

    def neg_obj_grad(z):
        g = np.zeros_like(z)
        g[2 * n:] = -1.0
        return g

    def cons(z):
        c, r = unpack(z)
        d = np.sqrt(((c[:, None] - c[None, :]) ** 2).sum(-1))
        out = [d[i, j] - r[i] - r[j] for (i, j) in pairs]
        out.extend((c[:, 0] - r).tolist())
        out.extend((1.0 - c[:, 0] - r).tolist())
        out.extend((c[:, 1] - r).tolist())
        out.extend((1.0 - c[:, 1] - r).tolist())
        return np.array(out)

    def repair(z):
        c, r = unpack(z)
        r = np.minimum.reduce([c[:, 0], c[:, 1], 1 - c[:, 0], 1 - c[:, 1]])
        for _ in range(200):
            changed = False
            for (i, j) in pairs:
                d = np.sqrt(((c[i] - c[j]) ** 2).sum())
                s = r[i] + r[j]
                if s > d and s > 0:
                    sc = d / s
                    r[i] *= sc
                    r[j] *= sc
                    changed = True
            if not changed:
                break
        r = np.maximum(r, 1e-9)
        z2 = z.copy()
        z2[:2 * n] = c.ravel()
        z2[2 * n:] = r
        return z2

    z0 = np.concatenate([centers.ravel(), radii])
    best_z, best_s = z0, float(radii.sum())
    # Restart passes: after each SLSQP run, repair to feasibility and
    # restart from that repaired point with tighter tolerance. This
    # lets SLSQP make fine tangency adjustments (circles sliding into
    # full contact) unreachable from the crude lattice seed.
    schedule = [(150, 1e-10), (300, 1e-12), (500, 1e-14)]
    z_start = z0
    for maxiter, ftol in schedule:
        res = minimize(neg_obj, z_start, jac=neg_obj_grad, method="SLSQP",
                       constraints=[{"type": "ineq", "fun": cons}],
                       options={"maxiter": maxiter, "ftol": ftol})
        z = repair(res.x)
        s = float(z[2 * n:].sum())
        if s > best_s:
            best_z, best_s = z, s
        # restart from the repaired feasible point (best if repair lost)
        z_start = z if s > 0 else best_z
    # One final restart from the best repaired point, tightest settings
    res = minimize(neg_obj, best_z, jac=neg_obj_grad, method="SLSQP",
                   constraints=[{"type": "ineq", "fun": cons}],
                   options={"maxiter": 500, "ftol": 1e-14})
    z = repair(res.x)
    s = float(z[2 * n:].sum())
    if s > best_s:
        best_z, best_s = z, s
    c, r = unpack(best_z)
    return c, r, best_s


def refine_layout(centers, r0, iters=80):
    """
    Deterministically nudge centers toward free space and re-grow radii.
    Each circle is pushed away from its closest neighbor and toward the
    nearest walls; only configurations with a higher total radius are kept.
    """
    centers = centers.copy()
    radii = grow_radii(centers, r0)
    best_centers = centers.copy()
    best_radii = radii.copy()
    best_sum = float(np.sum(radii))

    for t in range(iters):
        step = 0.02 * (1.0 - t / iters)
        d = np.sqrt(((centers[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))
        np.fill_diagonal(d, np.inf)
        new_centers = centers.copy()
        for i in range(centers.shape[0]):
            j = np.argmin(d[i])
            vec = centers[i] - centers[j]
            norm = np.linalg.norm(vec)
            if norm > 1e-9:
                new_centers[i] += step * vec / norm
            walls = np.array([[0.0, centers[i, 1]], [1.0, centers[i, 1]],
                              [centers[i, 0], 0.0], [centers[i, 0], 1.0]])
            w = np.argmin(np.linalg.norm(walls - centers[i], axis=1))
            towall = walls[w] - centers[i]
            new_centers[i] += 0.5 * step * towall
        new_centers = np.clip(new_centers, 0.005, 0.995)
        new_radii = grow_radii(new_centers, r0)
        new_sum = float(np.sum(new_radii))
        if new_sum > best_sum:
            best_sum = new_sum
            best_centers = new_centers.copy()
            best_radii = new_radii.copy()
            centers = new_centers
        elif t < iters // 2:
            centers = new_centers
    return best_centers, best_radii, best_sum


def grow_radii(centers, r_init, iters=1000):
    """
    Grow radii from a small feasible start: each circle's radius can
    grow up to min(wall distance, distance to others minus their radii).
    """
    n = centers.shape[0]
    b = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                   np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    d = np.sqrt(((centers[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(d, np.inf)

    radii = np.full(n, 0.25 * r_init)
    for _ in range(iters):
        grew = False
        for i in range(n):
            cand = min(b[i], np.min(d[i, :] - radii))
            if cand > radii[i] + 1e-12:
                radii[i] = cand
                grew = True
        if not grew:
            break
    return radii


def compute_max_radii(centers):
    """
    Compute radii limited by walls and pairwise distances,
    resolving overlaps by proportional pairwise scaling until valid.
    """
    n = centers.shape[0]
    x = centers[:, 0]
    y = centers[:, 1]
    radii = np.minimum.reduce([x, y, 1 - x, 1 - y])

    # Precompute pairwise distances
    dist = np.sqrt(((centers[:, None, :] - centers[None, :, :]) ** 2).sum(-1))

    # Iteratively fix overlaps
    for _ in range(50):
        worst = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                d = dist[i, j]
                s = radii[i] + radii[j]
                if s > d and s > 0:
                    scale = d / s
                    radii[i] *= scale
                    radii[j] *= scale
                    worst = max(worst, 1 - scale)
        if worst < 1e-12:
            break

    # Final safety: hard clip against walls and neighbors
    for i in range(n):
        for j in range(n):
            if i != j:
                allowed = dist[i, j] - radii[j]
                if radii[i] > allowed:
                    radii[i] = max(allowed, 1e-9)
    radii = np.minimum(radii, np.minimum.reduce([x, y, 1 - x, 1 - y]))
    radii = np.maximum(radii, 1e-9)

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