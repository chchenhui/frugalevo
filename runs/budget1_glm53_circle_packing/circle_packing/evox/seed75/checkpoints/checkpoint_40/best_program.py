# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles.

Layout: hexagonal rows 5-4-5-4-5 of near-equal circles (23) plus 3
extra circles in the slack band at the top. Radii are optimized by
a linear program (max sum r_i s.t. r_i + r_j <= d_ij, r_i inside walls).
"""
import numpy as np


def _hex_layout(r, top_y=0.93):
    """Staggered hex rows 5-4-5-4-5 plus 3 circles in the top slack band."""
    centers = []
    dy = np.sqrt(3.0) * r
    row_counts = [5, 4, 5, 4, 5]
    y0 = r
    for k, cnt in enumerate(row_counts):
        y = y0 + k * dy
        x0 = r + (r if cnt == 4 else 0.0)
        for i in range(cnt):
            centers.append([x0 + 2 * r * i, y])
    for x in (0.2, 0.5, 0.8):
        centers.append([x, top_y])
    return np.array(centers)


def _corner_layout():
    """4 corner circles (wall-supported, can grow large) + rows 4-5-4-5-4."""
    centers = [[0.09, 0.09], [0.91, 0.09], [0.09, 0.91], [0.91, 0.91]]
    r = 0.1
    dy = np.sqrt(3.0) * r
    y0 = 0.1 + dy
    for k, cnt in enumerate([4, 5, 4, 5, 4]):
        y = y0 + k * dy
        x0 = 0.3 if cnt == 4 else 0.2
        for i in range(cnt):
            centers.append([x0 + 2 * r * i, y])
    return np.array(centers)


def construct_packing():
    """Multi-candidate constructor: build several hex/corner layouts,
    refine each with SLSQP, jitter-refine the best many (decaying
    schedule), then Powell-polish the winner. Keep the best valid."""
    candidates = [_hex_layout(r) for r in (0.1, 0.095, 0.105, 0.09, 0.085)]
    candidates.append(_corner_layout())
    # Extra seeds: same hex grid but shifted top-band circles.
    for tx in (0.0, 0.05, -0.05):
        c = _hex_layout(0.1, top_y=0.93)
        c[-3:, 0] = np.clip(c[-3:, 0] + tx, 0.15, 0.85)
        candidates.append(c)
    best_c, best_r = None, None
    for c in candidates:
        radii = solve_radii_lp(c)
        c2, r2 = _refine(c, radii, maxiter=200)
        if best_r is None or np.sum(r2) > np.sum(best_r):
            best_c, best_r = c2, r2
    # Deterministic multi-start: jitter the best layout and re-refine,
    # with a decaying jitter schedule (coarse exploration -> fine tune).
    rng = np.random.default_rng(0)
    for k in range(14):
        scale = 0.010 * (1.0 - k / 14.0) + 0.002
        jitter = rng.normal(0.0, scale, size=best_c.shape)
        c2 = np.clip(best_c + jitter, 0.03, 0.97)
        c2, r2 = _refine(c2, solve_radii_lp(c2), maxiter=200)
        if np.sum(r2) > np.sum(best_r):
            best_c, best_r = c2, r2
    # Final Powell polish: derivative-free fine-tuning of the winner.
    best_c, best_r = _polish(best_c, best_r)
    centers, radii = _make_safe(best_c, best_r)
    return centers, radii, float(np.sum(radii))


def solve_radii_lp(centers):
    """Maximize sum of radii via LP; fall back to greedy if scipy absent."""
    n = centers.shape[0]
    try:
        from scipy.optimize import linprog
        rows, rhs = [], []
        # Wall constraints: r_i <= min(x, y, 1-x, 1-y)
        for i in range(n):
            x, y = centers[i]
            for coeff, b in ((1.0, x), (1.0, y), (1.0, 1 - x), (1.0, 1 - y)):
                row = np.zeros(n)
                row[i] = coeff
                rows.append(row)
                rhs.append(b)
        # Pair constraints: r_i + r_j <= dist
        for i in range(n):
            for j in range(i + 1, n):
                d = np.hypot(*(centers[i] - centers[j]))
                row = np.zeros(n)
                row[i] = 1.0
                row[j] = 1.0
                rows.append(row)
                rhs.append(d)
        res = linprog(c=-np.ones(n), A_ub=np.array(rows), b_ub=np.array(rhs),
                      bounds=[(0, None)] * n, method="highs")
        if res.success:
            return np.maximum(res.x, 0.0)
    except Exception:
        pass
    return _greedy_radii(centers)


def _refine(centers, radii, maxiter=200):
    """Jointly optimize centers and radii (SLSQP) maximizing sum of radii.

    Variables z = [x_0,y_0,...,x_{n-1},y_{n-1}, r_0..r_{n-1}].
    Constraints: r_i+r_j <= dist(i,j) and circles inside the square
    (wall constraints), all with analytic Jacobians for speed.
    Falls back to the input (already valid) if the refinement fails.
    """
    try:
        from scipy.optimize import minimize
        n = centers.shape[0]
        z0 = np.concatenate([centers.ravel(), radii])
        m = 3 * n  # number of variables

        def cons_f(z):
            x = z[0:2 * n:2]
            y = z[1:2 * n:2]
            r = z[2 * n:]
            walls = np.concatenate([x - r, 1 - x - r, y - r, 1 - y - r])
            d = []
            for i in range(n):
                for j in range(i + 1, n):
                    d.append(np.hypot(x[i] - x[j], y[i] - y[j]) - r[i] - r[j])
            return np.concatenate([walls, np.array(d)])

        def cons_j(z):
            x = z[0:2 * n:2]
            y = z[1:2 * n:2]
            r = z[2 * n:]
            J = np.zeros((4 * n, m))
            for i in range(n):
                J[i, 2 * i] = 1.0
                J[i, 2 * n + i] = -1.0
                J[n + i, 2 * i] = -1.0
                J[n + i, 2 * n + i] = -1.0
                J[2 * n + i, 2 * i + 1] = 1.0
                J[2 * n + i, 2 * n + i] = -1.0
                J[3 * n + i, 2 * i + 1] = -1.0
                J[3 * n + i, 2 * n + i] = -1.0
            rows = [J]
            for i in range(n):
                for j in range(i + 1, n):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.hypot(dx, dy) + 1e-12
                    row = np.zeros(m)
                    row[2 * i] = dx / dist
                    row[2 * i + 1] = dy / dist
                    row[2 * j] = -dx / dist
                    row[2 * j + 1] = -dy / dist
                    row[2 * n + i] = -1.0
                    row[2 * n + j] = -1.0
                    rows.append(row)
            return np.vstack(rows)

        bounds = [(0.0, 1.0)] * (2 * n) + [(1e-6, 0.5)] * n
        res = minimize(
            lambda z: -np.sum(z[2 * n:]), z0,
            method="SLSQP",
            bounds=bounds,
            constraints=[{"type": "ineq", "fun": cons_f, "jac": cons_j}],
            options={"maxiter": maxiter, "ftol": 1e-10},
        )
        z = max((z0, res.x), key=lambda t: np.sum(t[2 * n:]))
        if np.sum(z[2 * n:]) > np.sum(radii):
            c = np.column_stack([z[0:2 * n:2], z[1:2 * n:2]])
            return c, np.maximum(z[2 * n:], 1e-9)
    except Exception:
        pass
    return centers, radii


def _polish(centers, radii):
    """Powell polish: derivative-free refinement of centers+radii.

    Uses the same constraint set as _refine via a penalized objective;
    only accepts strictly better and valid results.
    """
    try:
        from scipy.optimize import minimize
        n = centers.shape[0]
        z0 = np.concatenate([centers.ravel(), radii])

        def obj(z):
            r = z[2 * n:]
            pen = 0.0
            x = z[0:2 * n:2]
            y = z[1:2 * n:2]
            pen += np.sum(np.maximum(r - x, 0) ** 2)
            pen += np.sum(np.maximum(r - (1 - x), 0) ** 2)
            pen += np.sum(np.maximum(r - y, 0) ** 2)
            pen += np.sum(np.maximum(r - (1 - y), 0) ** 2)
            for i in range(n):
                dx = x[i] - x[i + 1:]
                dy = y[i] - y[i + 1:]
                viol = r[i] + r[i + 1:] - np.hypot(dx, dy)
                pen += np.sum(np.maximum(viol, 0) ** 2)
            return -np.sum(r) + 1e4 * pen

        res = minimize(obj, z0, method="Powell",
                       options={"maxiter": 4000, "xtol": 1e-8, "ftol": 1e-10})
        if -res.fun > np.sum(radii) + 1e-12:
            r = np.maximum(res.x[2 * n:], 1e-9)
            c = np.column_stack([res.x[0:2 * n:2], res.x[1:2 * n:2]])
            # Accept only if actually valid after safety shrink check.
            c2, r2 = _make_safe(c, r)
            if np.sum(r2) > np.sum(radii):
                return c2, r2
    except Exception:
        pass
    return centers, radii


def _make_safe(centers, radii):
    """Uniformly shrink radii (if needed) so the packing is strictly valid.

    Guarantees r_i + r_j <= dist(i,j) and containment in the unit square
    even if the optimizer left a tiny numerical violation.
    """
    n = centers.shape[0]
    s = 1.0
    for i in range(n):
        x, y = centers[i]
        for lim in (x, y, 1 - x, 1 - y):
            if radii[i] > lim:
                s = min(s, lim / radii[i])
        for j in range(i + 1, n):
            d = np.hypot(*(centers[i] - centers[j]))
            need = radii[i] + radii[j]
            if need > d:
                s = min(s, d / need)
    if s < 1.0:
        radii = radii * s * (1.0 - 1e-12)
    radii = np.minimum(radii, np.array(
        [min(x, y, 1 - x, 1 - y) for x, y in centers]))
    return centers, np.maximum(radii, 0.0)


def _greedy_radii(centers):
    """Fallback: iterative proportional shrinking (always valid)."""
    n = centers.shape[0]
    radii = np.array([min(x, y, 1 - x, 1 - y) for x, y in centers])
    for _ in range(200):
        ok = True
        for i in range(n):
            for j in range(i + 1, n):
                d = np.hypot(*(centers[i] - centers[j]))
                if radii[i] + radii[j] > d + 1e-12:
                    s = d / (radii[i] + radii[j] + 1e-15)
                    radii[i] *= s
                    radii[j] *= s
                    ok = False
        if ok:
            break
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
