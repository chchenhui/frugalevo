# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def _largest_empty_center(centers):
    """Return the point in the unit square maximizing the distance to the
    nearest existing circle center or wall (greedy Apollonian insertion
    point), found by a coarse grid scan plus local refinement."""
    xs = np.linspace(0.0, 1.0, 161)
    X, Y = np.meshgrid(xs, xs)
    P = np.stack([X.ravel(), Y.ravel()], axis=1)
    d = np.sqrt(((P - 0.5) ** 2).sum(axis=1)) * 0.0 + 0.5  # wall distance cap
    d = np.minimum(d, 0.5 - np.abs(P[:, 0] - 0.5))
    d = np.minimum(d, 0.5 - np.abs(P[:, 1] - 0.5))
    for c in centers:
        d = np.minimum(d, np.sqrt(((P - c) ** 2).sum(axis=1)))
    k = int(np.argmax(d))
    bx, by = P[k]
    bd = float(d[k])
    step = 1.0 / 160
    for _ in range(40):
        improved = False
        for dx in (-step, 0.0, step):
            for dy in (-step, 0.0, step):
                qx = min(max(bx + dx, 0.0), 1.0)
                qy = min(max(by + dy, 0.0), 1.0)
                qd = min(0.5 - abs(qx - 0.5), 0.5 - abs(qy - 0.5))
                if len(centers):
                    qd = min(qd, np.sqrt(((centers - [qx, qy]) ** 2).sum(axis=1)).min())
                if qd > bd + 1e-12:
                    bx, by, bd = qx, qy, qd
                    improved = True
        if not improved:
            step *= 0.5
            if step < 1e-9:
                break
    return np.array([bx, by]), bd


def _polish(centers, radii, maxiter=200):
    """SLSQP refinement of the 78 variables (52 center coords + 26 radii)
    maximizing the sum of radii subject to wall and pairwise-distance
    constraints. Returns refined centers, radii, or the inputs on failure."""
    try:
        from scipy.optimize import minimize
    except Exception:
        return centers, radii
    n = len(centers)
    x0 = np.concatenate([centers.ravel(), radii])

    def unpack(v):
        return v[: 2 * n].reshape(n, 2), v[2 * n:]

    def obj(v):
        return -np.sum(v[2 * n:])

    def grad(v):
        g = np.zeros_like(v)
        g[2 * n:] = -1.0
        return g

    def cons_wall(v):
        c, r = unpack(v)
        return np.concatenate([c[:, 0] - r, c[:, 1] - r,
                               1 - c[:, 0] - r, 1 - c[:, 1] - r])

    def cons_pair(v):
        c, r = unpack(v)
        out = []
        for i in range(n):
            for j in range(i + 1, n):
                dd = np.sqrt(np.sum((c[i] - c[j]) ** 2))
                out.append(dd - r[i] - r[j])
        return np.array(out)

    res = minimize(obj, x0, jac=grad, method="SLSQP",
                   constraints=[{"type": "ineq", "fun": cons_wall},
                                {"type": "ineq", "fun": cons_pair}],
                   options={"maxiter": maxiter, "ftol": 1e-10})
    c, r = unpack(res.x)
    if not np.all(np.isfinite(c)) or not np.all(np.isfinite(r)):
        return centers, radii
    return c, np.maximum(r, 1e-9)


def construct_packing():
    """
    Apollonian/Soddy-style greedy gap-filling construction: repeatedly
    insert a new circle at the point maximizing distance to walls and
    existing circles (multi-scale radius distribution), then polish with
    one bounded SLSQP pass over centers and radii. Final radii are always
    assigned via compute_max_radii, which guarantees a valid packing.
    """
    n = 26
    centers = np.zeros((n, 2))
    placed = []
    for k in range(n):
        p, _ = _largest_empty_center(np.array(placed))
        centers[k] = p
        placed.append(p)

    radii = compute_max_radii(centers)
    centers, radii = _polish(centers.copy(), radii.copy())

    # Guaranteed-valid final radius assignment (only shrinks if needed).
    centers = np.clip(centers, 1e-9, 1 - 1e-9)
    radii = compute_max_radii(centers)
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
