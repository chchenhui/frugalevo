# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def _hex_lattice_start():
    """Hexagonal-lattice starting configuration (rows 6,5,6,5,4 = 26).

    Strictly feasible: centers spaced 1/6 apart horizontally with
    sqrt(3)/12 vertical offset, so LP radii are strictly positive.
    """
    n = 26
    centers = np.zeros((n, 2), dtype=np.float64)
    rows = [6, 5, 6, 5, 4]
    dx = 1.0 / 6.0
    dy = np.sqrt(3.0) / 12.0
    idx = 0
    for k, cnt in enumerate(rows):
        y = 0.5 + (k - 2) * dy
        x0 = 0.5 - (cnt - 1) * dx / 2.0
        for j in range(cnt):
            centers[idx, 0] = x0 + j * dx
            centers[idx, 1] = y
            idx += 1
    return centers


def compute_max_radii(centers):
    """LP radii: r_i = min(wall distance, half min pairwise distance)."""
    n = centers.shape[0]
    radii = np.empty(n, dtype=np.float64)
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(x, y, 1.0 - x, 1.0 - y)
    for i in range(n):
        for j in range(i + 1, n):
            d = 0.5 * np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if d < radii[i]:
                radii[i] = d
            if d < radii[j]:
                radii[j] = d
    return radii


def _refine_slsqp(centers, radii, maxiter=300, time_limit=60.0):
    """Joint SLSQP refinement over (cx, cy, r) in R^78.

    Minimizes -sum(r) subject to squared-form non-overlap constraints
    (ci-cj)^2 - (ri+rj)^2 >= 0 (325 pairs) and wall constraints
    c - r >= 0, 1 - c - r >= 0 per axis, all with analytic Jacobians.
    Returns (centers, radii) or None on failure/timeout.
    """
    import time as _time
    from scipy.optimize import minimize

    t0 = _time.time()
    n = 26
    idx = np.triu_indices(n, k=1)
    npairs = len(idx[0])
    i_idx, j_idx = idx

    def unpack(x):
        return x[0:2 * n].reshape(n, 2), x[2 * n:]

    def obj(x):
        return -np.sum(x[2 * n:])

    def obj_jac(x):
        g = np.zeros(3 * n)
        g[2 * n:] = -1.0
        return g

    def cons_f(x):
        c, r = unpack(x)
        d = c[i_idx] - c[j_idx]
        sq = d[:, 0] ** 2 + d[:, 1] ** 2
        s = r[i_idx] + r[j_idx]
        walls = np.concatenate([c[:, 0] - r, c[:, 1] - r,
                                1.0 - c[:, 0] - r, 1.0 - c[:, 1] - r])
        return np.concatenate([sq - s ** 2, walls])

    def cons_jac(x):
        c, r = unpack(x)
        J = np.zeros((npairs + 4 * n, 3 * n))
        dx = c[i_idx, 0] - c[j_idx, 0]
        dy = c[i_idx, 1] - c[j_idx, 1]
        s = r[i_idx] + r[j_idx]
        for k in range(npairs):
            i, j = i_idx[k], j_idx[k]
            J[k, 2 * i] = 2 * dx[k]
            J[k, 2 * i + 1] = 2 * dy[k]
            J[k, 2 * j] = -2 * dx[k]
            J[k, 2 * j + 1] = -2 * dy[k]
            J[k, 2 * n + i] = -2 * s[k]
            J[k, 2 * n + j] = -2 * s[k]
        off = npairs
        for a in range(n):
            J[off + 0 * n + a, 2 * a] = 1.0
            J[off + 0 * n + a, 2 * n + a] = -1.0
            J[off + 1 * n + a, 2 * a + 1] = 1.0
            J[off + 1 * n + a, 2 * n + a] = -1.0
            J[off + 2 * n + a, 2 * a] = -1.0
            J[off + 2 * n + a, 2 * n + a] = -1.0
            J[off + 3 * n + a, 2 * a + 1] = -1.0
            J[off + 3 * n + a, 2 * n + a] = -1.0
        return J

    x0 = np.concatenate([centers.reshape(-1), radii])
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 1.0)] * n
    try:
        res = minimize(obj, x0, jac=obj_jac, bounds=bounds,
                       constraints=[{"type": "ineq",
                                     "fun": cons_f, "jac": cons_jac}],
                       method="SLSQP", options={"maxiter": maxiter, "ftol": 1e-12})
        if _time.time() - t0 > time_limit:
            return None
        rc, rr = unpack(res.x)
        if not np.all(np.isfinite(rr)) or np.any(rr <= 0):
            return None
        return rc.copy(), rr.copy()
    except Exception:
        return None


def construct_packing():
    """Hex-lattice LP start + joint SLSQP center/radius refinement.

    Returns the better (by sum of radii) of the LP solution and the
    SLSQP-refined solution; radii shrunk by 2e-7 for tolerance safety.
    """
    n = 26
    centers = _hex_lattice_start()
    radii = compute_max_radii(centers)
    base_sum = float(np.sum(radii))

    # Symmetry-breaking perturbation of the hex start: the exact lattice is
    # a degenerate saddle (interior circles touch 6 neighbors, boundary
    # circles touch walls), so SLSQP's QP steps stall there. A small
    # deterministic asymmetric nudge lets boundary circles migrate outward
    # and inflate. Perturbation is tiny enough that LP radii stay strictly
    # positive; still a single SLSQP start.
    pert = centers.copy()
    k = np.arange(n, dtype=np.float64)
    pert[:, 0] += 0.004 * np.sin(1.7 * k + 0.3)
    pert[:, 1] += 0.004 * np.cos(2.3 * k + 1.1)
    best_sum = base_sum
    k = np.arange(n, dtype=np.float64)
    # Two deterministic perturbation starts (small / medium amplitude):
    # the exact hex lattice is a degenerate saddle for SLSQP, and small
    # asymmetric nudges let boundary circles migrate outward and inflate.
    for amp, w1, w2, p1, p2 in ((0.004, 1.7, 2.3, 0.3, 1.1),
                                (0.010, 1.3, 2.9, 0.7, 2.1)):
        pert = centers.copy()
        pert[:, 0] += amp * np.sin(w1 * k + p1)
        pert[:, 1] += amp * np.cos(w2 * k + p2)
        pert_radii = compute_max_radii(pert)
        if not np.all(pert_radii > 0):
            continue
        p_result = _refine_slsqp(pert, pert_radii, maxiter=500, time_limit=45.0)
        if p_result is None:
            continue
        pc, pr = p_result
        s = float(np.sum(pr))
        if s > best_sum:
            best_sum = s
            centers, radii = pc, pr

    result = _refine_slsqp(centers, radii)
    if result is not None:
        rc, rr = result
        ref_sum = float(np.sum(rr))
        if ref_sum > best_sum:
            centers, radii = rc, rr
            best_sum = ref_sum

    # Iterative re-polish: restart SLSQP from the incumbent with radii
    # inflated by 3% (capped by wall distance). The mild deliberate
    # infeasibility pushes circles apart so boundary circles keep growing
    # across successive bounded rounds; accept-only-if-better guards keep
    # the incumbent valid at every step.
    for _ in range(3):
        if best_sum <= base_sum:
            break
        wall = np.minimum.reduce([centers[:, 0], centers[:, 1],
                                  1.0 - centers[:, 0], 1.0 - centers[:, 1]])
        inflated = np.minimum(radii * 1.03, wall)
        out = _refine_slsqp(centers, inflated, maxiter=300, time_limit=40.0)
        if out is None:
            break
        rc, rr = out
        s = float(np.sum(rr))
        if s > best_sum:
            best_sum, centers, radii = s, rc, rr
        else:
            break

    # Free check: greedy LP radii at the refined centers can exceed the
    # SLSQP radii sum; keep whichever is larger (both are feasible).
    lp_radii = compute_max_radii(centers)
    if float(np.sum(lp_radii)) > best_sum:
        radii = lp_radii

    # Shrink slightly and clamp so evaluator tolerance is never consumed
    radii = np.maximum(radii - 2e-7, 0.0)
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        centers[i, 0] = min(max(x, r + 1e-12), 1.0 - r - 1e-12)
        centers[i, 1] = min(max(y, r + 1e-12), 1.0 - r - 1e-12)

    sum_radii = float(np.sum(radii))
    return centers, radii, sum_radii


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
