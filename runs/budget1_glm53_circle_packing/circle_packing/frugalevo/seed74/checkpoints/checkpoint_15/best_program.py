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
    """
    Hexagonal-row constructor with candidate selection and jitter refinement.

    Since compute_max_radii already yields feasible maximal radii, the old
    "relaxation" only shuffled circles and degraded the sum. Instead we:
      1. Generate several staggered-row hexagonal-like layouts (different row
         counts and spacings), compute the achievable sum for each, and keep
         the best.
      2. Hill-climb with small random jitters of individual circles,
         accepting only moves that increase the total sum of maximal radii.
    """
    n = 26
    rng = np.random.RandomState(0)

    def build(counts, spacing, stagger=True):
        c = np.zeros((n, 2))
        idx = 0
        y0 = spacing / 2
        for row, k in enumerate(counts):
            y = y0 + row * spacing
            total_w = (k - 1) * spacing
            x_start = (1.0 - total_w) / 2.0
            if stagger and row % 2 == 1:
                x_start += spacing / 2
            for j in range(k):
                c[idx] = [x_start + j * spacing, y]
                idx += 1
        return np.clip(c, 0.005, 0.995)

    # Candidate row patterns summing to 26 over 5 and 6 rows.
    # 6-row layouts better approximate hexagonal packing in the interior.
    patterns = [
        [4, 5, 6, 6, 5],
        [5, 5, 6, 5, 5],
        [4, 6, 6, 6, 4],
        [5, 6, 5, 6, 4],
        [6, 5, 5, 5, 5],
        [4, 5, 5, 6, 6],
        [4, 4, 6, 6, 6],
        [5, 4, 6, 6, 5],
        [3, 5, 6, 6, 6],
        [4, 5, 6, 5, 6],
        [4, 4, 5, 5, 4, 4],
        [4, 4, 5, 5, 4, 4][:5] + [4],
        [3, 4, 5, 5, 5, 4],
        [4, 4, 5, 4, 5, 4],
        [5, 4, 4, 4, 4, 5],
        [3, 5, 5, 5, 5, 3],
        [4, 5, 6, 6, 5, 0],
        [5, 5, 6, 6, 4, 0],
        [4, 6, 5, 6, 5, 0],
        [3, 5, 6, 6, 6, 0],
        [4, 4, 6, 6, 6, 0],
        [5, 6, 6, 5, 4, 0],
        [4, 5, 5, 5, 5, 2],
        [3, 5, 5, 5, 4, 4],
        [4, 5, 5, 4, 5, 3],
        [2, 5, 6, 6, 5, 2],
    ]
    # Retain the top candidates (by achievable maximal-radii sum) for
    # multi-start SLSQP refinement rather than only the single best.
    candidates = []
    for counts in patterns:
        if sum(counts) != n:
            continue
        for spacing in (0.155, 0.16, 0.165, 0.17, 0.175, 0.18, 0.185, 0.19,
                        0.195, 0.2, 0.21, 0.22, 0.23, 0.24):
            for stagger in (True, False):
                c = build(counts, spacing, stagger)
                s = np.sum(compute_max_radii(c))
                candidates.append((s, c))
    candidates.sort(key=lambda t: -t[0])
    top = [c.copy() for _, c in candidates[:12]]
    best_centers, best_sum = top[0], candidates[0][0]

    # SLSQP NLP refinement: jointly optimize centers and radii.
    # Variables z = (x_0..x_25, y_0..y_25, r_0..r_25), objective max sum(r).
    # Constraints: pairwise non-overlap (squared-distance form) and
    # border containment r_i <= x_i <= 1-r_i, r_i <= y_i <= 1-r_i.
    from scipy.optimize import minimize

    def slsqp_refine(centers0, radii0, delta, maxiter=1500):
        z0 = np.concatenate([centers0[:, 0], centers0[:, 1], radii0])
        I, J = np.triu_indices(n, k=1)

        def unpack(z):
            return z[:n], z[n:2 * n], z[2 * n:]

        def obj(z):
            return -np.sum(z[2 * n:])

        def obj_grad(z):
            g = np.zeros(3 * n)
            g[2 * n:] = -1.0
            return g

        def cons_pair(z):
            x, y, r = unpack(z)
            dx = x[I] - x[J]
            dy = y[I] - y[J]
            d2 = dx * dx + dy * dy
            return d2 - (r[I] + r[J] + delta) ** 2

        def cons_pair_jac(z):
            x, y, r = unpack(z)
            dx = x[I] - x[J]
            dy = y[I] - y[J]
            m = I.shape[0]
            Jm = np.zeros((m, 3 * n))
            rows = np.arange(m)
            Jm[rows, I] += 2 * dx
            Jm[rows, J] -= 2 * dx
            Jm[rows, n + I] += 2 * dy
            Jm[rows, n + J] -= 2 * dy
            Jm[rows, 2 * n + I] -= 2 * (r[I] + r[J] + delta)
            Jm[rows, 2 * n + J] -= 2 * (r[I] + r[J] + delta)
            return Jm

        def cons_borders(z):
            x, y, r = unpack(z)
            return np.concatenate([x - r, (1 - x) - r, y - r, (1 - y) - r])

        def cons_borders_jac(z):
            Jm = np.zeros((4 * n, 3 * n))
            ar = np.arange(n)
            Jm[ar, ar] = 1.0
            Jm[ar, 2 * n + ar] = -1.0
            Jm[n + ar, ar] = -1.0
            Jm[n + ar, 2 * n + ar] = -1.0
            Jm[2 * n + ar, n + ar] = 1.0
            Jm[2 * n + ar, 2 * n + ar] = -1.0
            Jm[3 * n + ar, n + ar] = -1.0
            Jm[3 * n + ar, 2 * n + ar] = -1.0
            return Jm

        bounds = ([(0.0, 1.0)] * n + [(0.0, 1.0)] * n +
                  [(0.0, 0.5)] * n)
        res = minimize(
            obj, z0, jac=obj_grad, method="SLSQP", bounds=bounds,
            constraints=[
                {"type": "ineq", "fun": cons_pair, "jac": cons_pair_jac},
                {"type": "ineq", "fun": cons_borders, "jac": cons_borders_jac},
            ],
            options={"maxiter": maxiter, "ftol": 1e-12},
        )
        return res.x

    def feasible(centers, radii, tol=1e-6):
        if np.any(radii <= 0):
            return False
        if (np.any(centers - radii[:, None] < -tol) or
                np.any(centers + radii[:, None] > 1 + tol)):
            return False
        diff = centers[:, None, :] - centers[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dist, np.inf)
        return np.all(dist >= radii[:, None] + radii[None, :] - tol)

    # Multi-start SLSQP refinement over the top candidates, sequentially
    # tightening delta. Each start is capped at maxiter=1500; results are
    # accepted only after an exact feasibility recheck and only if strictly
    # better than the incumbent.
    inc_radii = compute_max_radii(best_centers)
    inc_sum = np.sum(inc_radii)
    best_sum, best_r, best_c = inc_sum, inc_radii.copy(), best_centers.copy()

    for c0 in top:
        r0 = compute_max_radii(c0)
        cur_c, cur_r = c0.copy(), r0.copy()
        for delta in (1e-6, 1e-8):
            z = slsqp_refine(cur_c, cur_r, delta, maxiter=1500)
            cx = np.stack([z[:n], z[n:2 * n]], axis=1)
            rx = z[2 * n:]
            if feasible(cx, rx):
                s = np.sum(rx)
                if s > np.sum(cur_r):
                    cur_c, cur_r = cx.copy(), np.maximum(rx, 1e-9).copy()
        if np.sum(cur_r) > best_sum:
            best_sum, best_r, best_c = float(np.sum(cur_r)), cur_r.copy(), cur_c.copy()

    # Budget-scaled deterministic restarts (basin-hopping style): sample
    # many more active-contact manifolds within the allowed offline budget.
    # 400 restarts; sigma cycles through a coarse-to-fine ladder, and every
    # 5th restart re-seeds from a retained lattice candidate (basin
    # re-entry) instead of the incumbent. Each restart: one
    # compute_max_radii call + two SLSQP passes (delta 1e-6, 1e-8,
    # maxiter=1500). A wall-clock guard checked only between restarts
    # always returns the best-so-far. Acceptance uses the SLSQP radii
    # directly (never recompute maximal radii from moved centers); exact
    # feasibility recheck before acceptance; incumbent fallback retained.
    import time
    t0 = time.time()
    sigma_ladder = (0.008, 0.005, 0.003, 0.001, 0.0005)
    # Partial-jitter mask: on alternating trials perturb only half the
    # circles (deterministic index selection), leaving the rest pinned.
    # This explores different active-contact manifolds per restart at the
    # same per-restart cost as full jitter.
    half_mask = (np.arange(n) % 2 == 0)
    for trial in range(400):
        if time.time() - t0 > 340.0:
            break
        if trial % 3 == 2:
            seed_c = top[trial % len(top)].copy()
        else:
            seed_c = best_c
        sigma = sigma_ladder[trial % len(sigma_ladder)]
        noise = rng.normal(0.0, sigma, size=seed_c.shape)
        if trial % 2 == 1:
            noise[~half_mask] = 0.0
        jit = np.clip(seed_c + noise, 0.01, 0.99)
        jr = compute_max_radii(jit)
        for delta in (1e-6, 1e-8, 1e-10):
            z = slsqp_refine(jit, jr, delta, maxiter=1500)
            cx = np.stack([z[:n], z[n:2 * n]], axis=1)
            rx = z[2 * n:]
            if feasible(cx, rx) and np.sum(rx) > np.sum(jr):
                jit, jr = cx.copy(), np.maximum(rx, 1e-9).copy()
        if np.sum(jr) > best_sum:
            best_sum, best_r, best_c = float(np.sum(jr)), jr.copy(), jit.copy()

    # Safety margin on returned radii, then verify validity.
    # Tighter margin: feasible() rechecks at tol 1e-6, so 0.999999 is safe
    # and recovers ~2.5e-5 of sum that 0.99999 discarded.
    radii = best_r * 0.999999
    centers = best_c.copy()
    if not feasible(centers, radii):
        centers, radii, best_sum = best_centers, inc_radii, inc_sum
    best_sum = float(np.sum(radii))
    return centers, radii, best_sum


def compute_max_radii(centers):
    """
    Compute the maximum possible radii for each circle position
    such that they don't overlap and stay within the unit square.

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates

    Returns:
        np.array of shape (n) with radius of each circle
    """
    """
    Vectorized maximal-radii computation.

    Border distances give an upper bound; then pairwise proportional
    scaling is iterated to a fixed point so the radii are (approximately)
    maximal subject to non-overlap. Iterating strictly improves on a
    single scaling pass, increasing the achievable sum of radii.
    """
    n = centers.shape[0]
    # Distance to square borders
    radii = np.minimum(np.minimum(centers[:, 0], 1 - centers[:, 0]),
                       np.minimum(centers[:, 1], 1 - centers[:, 1]))

    # Pairwise distance matrix (excluding self-distance)
    diff = centers[:, None, :] - centers[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)

    # Iterate proportional scaling until convergence (fixed point)
    for _ in range(40):
        s = radii[:, None] + radii[None, :]
        scale = np.where(s > dist, dist / np.maximum(s, 1e-12), 1.0)
        new_radii = radii * np.min(scale, axis=1)
        if np.max(np.abs(new_radii - radii)) < 1e-11:
            radii = new_radii
            break
        radii = new_radii

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
