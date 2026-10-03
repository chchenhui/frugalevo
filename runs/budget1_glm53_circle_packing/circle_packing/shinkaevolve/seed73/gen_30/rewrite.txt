# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np

N = 26


def _grow_radii(pts, iters=60):
    m = pts.shape[0]
    rad = np.zeros(m)
    dm = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    b = np.minimum(np.minimum(pts[:, 0], 1 - pts[:, 0]),
                   np.minimum(pts[:, 1], 1 - pts[:, 1]))
    for _ in range(iters):
        for i in range(m):
            cand = min(b[i], np.min(dm[i] - rad) if m > 1 else b[i])
            rad[i] = max(0.0, cand)
    return rad


def _hex_seed(rnd=None):
    centers = np.zeros((N, 2))
    r0 = 1.0 / (2.0 + 5.0 * np.sqrt(3.0))
    dy = np.sqrt(3.0) * r0
    row_counts = [5, 4, 5, 4, 5, 3]
    idx = 0
    for k, cnt in enumerate(row_counts):
        y = r0 + k * dy
        if cnt == 5:
            xs = [r0 + 2 * r0 * j for j in range(5)]
        elif cnt == 4:
            xs = [2 * r0 + 2 * r0 * j for j in range(4)]
        else:
            xs = [0.5 - 2 * r0 + 2 * r0 * j for j in range(3)]
        for x in xs:
            centers[idx] = [x, y]
            idx += 1
    if rnd is not None:
        centers = centers + rnd.uniform(-0.01, 0.01, centers.shape)
        centers = np.clip(centers, 1e-3, 1 - 1e-3)
    return centers


def _slsqp_optimize(centers, radii):
    from scipy.optimize import minimize
    n = centers.shape[0]

    def pack(z):
        c = z[:2 * n].reshape(n, 2)
        r = z[2 * n:]
        return c, r

    def neg_sum(z):
        return -np.sum(z[2 * n:])

    def neg_sum_grad(z):
        g = np.zeros_like(z)
        g[2 * n:] = -1.0
        return g

    def cons_f(z):
        c, r = pack(z)
        d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
        iu = np.triu_indices(n, 1)
        pair = d[iu] - (r[iu[0]] + r[iu[1]])
        walls = np.concatenate([
            c[:, 0] - r, 1 - c[:, 0] - r,
            c[:, 1] - r, 1 - c[:, 1] - r,
        ])
        return np.concatenate([pair, walls])

    def cons_jac(z):
        c, r = pack(z)
        d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
        J = np.zeros((n * (n - 1) // 2 + 4 * n, z.size))
        row = 0
        for i in range(n):
            for j in range(i + 1, n):
                diff = c[i] - c[j]
                nrm = d[i, j] + 1e-12
                J[row, 2 * i] = diff[0] / nrm
                J[row, 2 * i + 1] = diff[1] / nrm
                J[row, 2 * j] = -diff[0] / nrm
                J[row, 2 * j + 1] = -diff[1] / nrm
                J[row, 2 * n + i] = -1.0
                J[row, 2 * n + j] = -1.0
                row += 1
        for i in range(n):
            # x - r >= 0
            J[row, 2 * i] = 1.0
            J[row, 2 * n + i] = -1.0
            row += 1
            # 1 - x - r >= 0
            J[row, 2 * i] = -1.0
            J[row, 2 * n + i] = -1.0
            row += 1
            # y - r >= 0
            J[row, 2 * i + 1] = 1.0
            J[row, 2 * n + i] = -1.0
            row += 1
            # 1 - y - r >= 0
            J[row, 2 * i + 1] = -1.0
            J[row, 2 * n + i] = -1.0
            row += 1
        return J

    z0 = np.concatenate([centers.ravel(), radii])
    res = minimize(neg_sum, z0, jac=neg_sum_grad,
                   constraints=[{'type': 'ineq', 'fun': cons_f, 'jac': cons_jac}],
                   method='SLSQP', options={'maxiter': 300, 'ftol': 1e-10})
    c, r = pack(res.x)
    r = np.maximum(r, 1e-6)
    # repair feasibility: shrink to guarantee validity
    for _ in range(50):
        d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
        viol = d - (r[:, None] + r[None, :])
        np.fill_diagonal(viol, np.inf)
        bad = viol < 0
        if not bad.any():
            break
        scale = np.minimum(1.0, np.where(bad, viol / (2 * np.minimum(r[:, None], r[None, :]) + 1e-12), 1.0))
        r *= np.min(scale, axis=1)
    r = np.minimum(r, np.minimum(np.minimum(c[:, 0], 1 - c[:, 0]),
                                 np.minimum(c[:, 1], 1 - c[:, 1])))
    return c, r, np.sum(r)


def construct_packing():
    """
    Construct a specific arrangement of 26 circles in a unit square
    that attempts to maximize the sum of their radii.

    Returns:
        Tuple of (centers, radii, sum_of_radii)
    """
    rng = np.random.RandomState(0)
    best = None
    for trial in range(3):
        seed = _hex_seed(rng if trial > 0 else None)
        rad = _grow_radii(seed)
        c, r, s = _slsqp_optimize(seed, rad)
        # extra polish: re-grow radii at the optimized centers
        r2 = _grow_radii(c)
        s2 = np.sum(r2)
        if s2 > s:
            r, s = r2, s2
        if best is None or s > best[2]:
            best = (c, r, s)

    centers, radii, sum_radii = best
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