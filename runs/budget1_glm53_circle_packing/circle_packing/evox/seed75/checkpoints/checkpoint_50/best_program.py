# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles in the unit square.

Approach (fundamentally different from the penalty-descent version):
  1. Generate staggered hexagonal-row template layouts for 26 circles.
  2. Jointly optimize centers AND radii with scipy SLSQP using EXACT
     analytic nonlinear constraints (pairwise non-overlap + wall
     containment), maximizing sum of radii. No penalty weights, no
     changing neighbor sets -> consistent, feasible convergence.
  3. Re-solve radii exactly with a linear program (linprog "highs")
     at the final centers, which is optimal for fixed centers.
  4. Multi-start over templates; keep the best valid layout.
  5. Guaranteed-valid fallback: greedy proportional shrinking plus a
     final uniform scale-down, so output is always feasible.
"""
import numpy as np

# Row decompositions of 26 circles: staggered hex-style templates.
_TEMPLATES = [
    [5, 4, 5, 4, 5],
    [5, 4, 5, 4, 5, 3],
    [4, 5, 4, 5, 4, 4],
    [6, 5, 4, 5, 6],
    [5, 5, 4, 5, 4, 3],
    [4, 4, 5, 5, 4, 4],
    [6, 4, 6, 4, 6],
    [4, 6, 6, 6, 4],
    [6, 5, 5, 5, 5],
    [5, 6, 5, 5, 5],
    [4, 4, 4, 5, 4, 5],
    [3, 5, 4, 5, 4, 5],
    [5, 4, 4, 5, 4, 4],
    [4, 5, 5, 4, 5, 3],
    [3, 4, 5, 5, 4, 5],
    [4, 3, 5, 4, 5, 5],
    [2, 4, 5, 5, 5, 5],
    [5, 5, 5, 5, 4, 2],
    [4, 4, 4, 4, 5, 5],
    [5, 5, 4, 4, 4, 4],
    [7, 5, 4, 5, 5],
    [5, 5, 4, 5, 7],
    [6, 6, 4, 4, 6],
    [3, 6, 5, 6, 6],
    [6, 5, 6, 5, 4],
]


def _rows_layout(counts):
    """Staggered-row template: rows centered horizontally, vertical spacing
    capped so the stack fits the square; narrower rows offset half a gap to
    approximate hexagonal close packing."""
    counts = list(counts)
    nrow = len(counts)
    maxc = max(counts)
    gap = 1.0 / maxc
    r = gap / 2.0
    dy = min(np.sqrt(3.0) * r, (1.0 - 2 * r) / max(1, nrow - 1))
    y0 = r + max(0.0, (1.0 - 2 * r - (nrow - 1) * dy)) / 2.0
    centers = []
    for k, cnt in enumerate(counts):
        w = (cnt - 1) * gap
        x0 = (1.0 - w) / 2.0 + (gap / 2.0 if cnt < maxc else 0.0)
        y = y0 + k * dy
        for i in range(cnt):
            centers.append([x0 + gap * i, y])
    return np.array(centers)


def _slsqp_refine(centers, radii, maxiter=300):
    """Joint SLSQP refinement of centers and radii with EXACT constraints.

    Variables z = [x_1..x_n, y_1..y_n, r_1..r_n]. Maximize sum(r) subject
    to r_i <= wall distances and r_i + r_j <= dist(i, j). All constraint
    Jacobians are analytic, so SLSQP converges reliably to a feasible
    local optimum. Falls back to the input on any failure.
    """
    try:
        from scipy.optimize import minimize
        n = centers.shape[0]
        z0 = np.concatenate([centers[:, 0], centers[:, 1],
                             np.maximum(radii, 1e-3)])

        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        ix = np.array([p[0] for p in pairs])
        jx = np.array([p[1] for p in pairs])

        def con_walls(z):
            """Wall containment: r_i <= x_i, 1-x_i, y_i, 1-y_i (linear)."""
            x, y, r = z[:n], z[n:2 * n], z[2 * n:]
            return np.concatenate([x - r, 1 - x - r, y - r, 1 - y - r])

        def con_walls_jac(z):
            J = np.zeros((4 * n, 3 * n))
            for i in range(n):
                J[i, i] = 1.0            # x_i - r_i
                J[n + i, i] = -1.0       # 1 - x_i - r_i
                J[2 * n + i, n + i] = 1.0    # y_i - r_i
                J[3 * n + i, n + i] = -1.0   # 1 - y_i - r_i
                J[i, 2 * n + i] = -1.0
                J[n + i, 2 * n + i] = -1.0
                J[2 * n + i, 2 * n + i] = -1.0
                J[3 * n + i, 2 * n + i] = -1.0
            return J

        def con_pairs(z):
            """Pairwise non-overlap: dist_ij - r_i - r_j >= 0."""
            x, y, r = z[:n], z[n:2 * n], z[2 * n:]
            dx = x[ix] - x[jx]
            dy = y[ix] - y[jx]
            return np.hypot(dx, dy) - r[ix] - r[jx]

        def con_pairs_jac(z):
            x, y, r = z[:n], z[n:2 * n], z[2 * n:]
            dx = x[ix] - x[jx]
            dy = y[ix] - y[jx]
            d = np.hypot(dx, dy) + 1e-12
            m = len(ix)
            J = np.zeros((m, 3 * n))
            rows = np.arange(m)
            J[rows, ix] = dx / d
            J[rows, jx] = -dx / d
            J[rows, n + ix] = dy / d
            J[rows, n + jx] = -dy / d
            J[rows, 2 * n + ix] = -1.0
            J[rows, 2 * n + jx] = -1.0
            return J

        def obj(z):
            g = np.zeros(3 * n)
            g[2 * n:] = -1.0
            return -np.sum(z[2 * n:]), g

        cons = [
            {"type": "ineq", "fun": con_walls, "jac": con_walls_jac},
            {"type": "ineq", "fun": con_pairs, "jac": con_pairs_jac},
        ]
        bounds = [(0.01, 0.99)] * n + [(0.01, 0.99)] * n + [(1e-4, 0.5)] * n
        res = minimize(obj, z0, jac=True, method="SLSQP",
                       bounds=bounds, constraints=cons,
                       options={"maxiter": maxiter, "ftol": 1e-12})
        cand = [z0, res.x] if np.isfinite(res.fun) else [z0]
        best = max(cand, key=lambda t: np.sum(t[2 * n:]))
        c = np.column_stack([best[:n], best[n:2 * n]])
        return c, np.maximum(best[2 * n:], 1e-9)
    except Exception:
        return centers, radii


def construct_packing():
    """Multi-start template construction + exact-constraint SLSQP + LP.

    For each staggered-row template: refine with SLSQP (exact analytic
    constraints), re-solve radii with the exact LP, repeat once, and keep
    the best layout. Output is always validated/shrunk to be feasible.
    """
    best_c, best_r, best_s = None, None, -1.0
    for counts in _TEMPLATES:
        c = _rows_layout(counts)
        r = solve_radii_lp(c)
        for _ in range(3):
            c, r = _slsqp_refine(c, r, maxiter=400)
            r = solve_radii_lp(c)
        s = float(np.sum(r))
        if s > best_s:
            best_c, best_r, best_s = c, r, s
    # Jittered restarts around the best layout: small random perturbations
    # of centers, then SLSQP + LP polish; keep any improvement.
    rng = np.random.default_rng(12345)
    for trial in range(40):
        scale = 0.02 * (0.5 ** (trial // 10))
        c2 = best_c + rng.normal(scale=scale, size=best_c.shape)
        c2[:, 0] = np.clip(c2[:, 0], 0.02, 0.98)
        c2[:, 1] = np.clip(c2[:, 1], 0.02, 0.98)
        r2 = solve_radii_lp(c2)
        for _ in range(2):
            c2, r2 = _slsqp_refine(c2, r2, maxiter=400)
            r2 = solve_radii_lp(c2)
        s2 = float(np.sum(r2))
        if s2 > best_s:
            best_c, best_r, best_s = c2, r2, s2
    centers, radii = _make_safe(best_c, best_r)
    return centers, radii, float(np.sum(radii))


def _make_safe(centers, radii):
    """Greedy proportional shrink + uniform scale-down for strict validity."""
    n = centers.shape[0]
    radii = np.array(radii, dtype=float)
    # Iterative pairwise proportional shrinking (always converges here).
    for _ in range(300):
        ok = True
        for i in range(n):
            for j in range(i + 1, n):
                d = np.hypot(*(centers[i] - centers[j]))
                need = radii[i] + radii[j]
                if need > d - 1e-12:
                    s = max(d - 1e-12, 0.0) / (need + 1e-15)
                    radii[i] *= s
                    radii[j] *= s
                    ok = False
        if ok:
            break
    # Uniform scale-down for any residual violation (walls included).
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
        radii = radii * s * (1.0 - 1e-9)
    radii = np.minimum(radii, np.array(
        [min(x, y, 1 - x, 1 - y) for x, y in centers]))
    return centers, np.maximum(radii, 0.0)


def solve_radii_lp(centers):
    """Maximize sum of radii via LP (optimal for fixed centers); greedy
    proportional shrinking as fallback if scipy is unavailable."""
    n = centers.shape[0]
    try:
        from scipy.optimize import linprog
        rows, rhs = [], []
        for i in range(n):
            x, y = centers[i]
            for b in (x, y, 1 - x, 1 - y):
                row = np.zeros(n)
                row[i] = 1.0
                rows.append(row)
                rhs.append(b)
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
