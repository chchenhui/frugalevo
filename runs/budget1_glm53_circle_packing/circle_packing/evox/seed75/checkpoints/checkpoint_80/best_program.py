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


def _hex_layout_var(r):
    """Variant: same 5-4-5-4-5 stagger but rows offset (phase-shifted x)."""
    centers = []
    dy = np.sqrt(3.0) * r
    row_counts = [5, 4, 5, 4, 5]
    y0 = r
    for k, cnt in enumerate(row_counts):
        y = y0 + k * dy
        # phase-shift: rows of 5 start at right side instead of left
        width = 2 * r * (cnt - 1) + (2 * r if cnt == 4 else 0)
        x0 = 1.0 - r - width if cnt == 5 else 0.5 - (cnt - 1) * r
        for i in range(cnt):
            centers.append([x0 + 2 * r * i, y])
    for x in (0.2, 0.5, 0.8):
        centers.append([x, 0.93])
    return np.array(centers)


def _inflate(centers, radii, factor=1.02):
    """Slightly inflate radii (may violate constraints) so the subsequent
    constrained re-refine can escape slack and find larger tangency sets."""
    lims = np.array([min(x, y, 1 - x, 1 - y) for x, y in centers])
    return np.minimum(radii * factor, lims)


def _farthest_points(seed, n=26, g=35):
    """Farthest-point (k-center) layout: start from one random grid point and
    repeatedly add the grid point maximizing min-distance to the chosen set.
    These well-separated configurations are excellent seeds for LP+SLSQP."""
    rng = np.random.default_rng(seed)
    xs = np.linspace(0.02, 0.98, g)
    grid = np.array([[x, y] for x in xs for y in xs])
    chosen = [int(rng.integers(grid.shape[0]))]
    dmin = np.linalg.norm(grid - grid[chosen[0]], axis=1)
    while len(chosen) < n:
        i = int(np.argmax(dmin))
        chosen.append(i)
        dmin = np.minimum(dmin, np.linalg.norm(grid - grid[i], axis=1))
    return grid[chosen].copy()


def _grow_to_contact(centers, radii, passes=40):
    """Greedy grow-to-contact polish (monotone, always valid).

    Repeatedly grow each circle i to the largest radius allowed by the
    walls and its current neighbors: cap_i = min(walls, min_j dist(i,j)
    - r_j). Only circle i grows, so validity is preserved and the sum
    of radii never decreases. Harvests slack left by LP/SLSQP.
    """
    centers = np.asarray(centers, dtype=float)
    r = np.asarray(radii, dtype=float).copy()
    n = centers.shape[0]
    for _ in range(passes):
        improved = False
        for i in range(n):
            x, y = centers[i]
            cap = min(x, y, 1.0 - x, 1.0 - y)
            for j in range(n):
                if j != i:
                    d = np.hypot(*(centers[i] - centers[j])) - r[j]
                    if d < cap:
                        cap = d
            if cap > r[i] + 1e-12:
                r[i] = cap
                improved = True
        if not improved:
            break
    return centers, r


def _grid_jitter(seed, n=26):
    """Jittered regular grid layout (another diverse template family)."""
    rng = np.random.default_rng(seed)
    side = int(np.ceil(np.sqrt(n)))
    pts = []
    for i in range(n):
        r_i, c_i = divmod(i, side)
        pts.append([(c_i + 0.5) / side, (r_i + 0.5) / side])
    return np.clip(np.array(pts) + rng.normal(0, 0.02, (n, 2)), 0.03, 0.97)


def construct_packing():
    """Multi-family constructor: hex/corner templates + farthest-point and
    jittered-grid seeds; each is LP-radiused, SLSQP-refined, then re-LP'd
    (pushing every radius to exact tangency). Annealed jitter multi-start
    and inflate-refine deep polish around the incumbent. Keep the winner."""
    candidates = [_hex_layout(r) for r in (0.1, 0.095, 0.105, 0.09)]
    candidates.append(_hex_layout_var(0.1))
    candidates.append(_hex_layout_var(0.095))
    candidates.append(_corner_layout())
    candidates += [_farthest_points(s) for s in range(10)]
    candidates += [_grid_jitter(s) for s in range(6)]
    best_c, best_r = None, None
    for c in candidates:
        c2, r2 = _refine(c, solve_radii_lp(c), maxiter=200)
        r2 = solve_radii_lp(c2)          # re-LP: radii to exact tangency
        if np.sum(r2) > np.sum(best_r if best_r is not None else r2 * 0):
            best_c, best_r = c2, r2
    # Grow-to-contact on the incumbent before deep search.
    best_c, best_r = _grow_to_contact(best_c, best_r)
    # Annealed deterministic multi-start: jitter the best layout and re-refine.
    # Later trials jitter only a random subset of ~6 circles so good
    # regions are preserved while jammed spots escape.
    rng = np.random.default_rng(0)
    n_circles = best_c.shape[0]
    scales = np.linspace(0.012, 0.0005, 40)
    for trial, s in enumerate(scales):
        c2 = best_c.copy()
        if trial >= 15:
            idx = rng.choice(n_circles, size=6, replace=False)
            c2[idx] += rng.normal(0.0, s, size=(6, 2))
        else:
            c2 += rng.normal(0.0, s, size=c2.shape)
        c2 = np.clip(c2, 0.02, 0.98)
        c2, r2 = _refine(c2, solve_radii_lp(c2), maxiter=200)
        r2 = solve_radii_lp(c2)
        c2, r2 = _grow_to_contact(c2, r2)
        if np.sum(r2) > np.sum(best_r):
            best_c, best_r = c2, r2
    # Deep polish: inflate-then-refine loops around the incumbent.
    for _ in range(8):
        c2, r2 = _refine(best_c, _inflate(best_c, best_r), maxiter=400)
        r2 = solve_radii_lp(c2)
        c2, r2 = _grow_to_contact(c2, r2)
        if np.sum(r2) > np.sum(best_r):
            best_c, best_r = c2, r2
        else:
            break
    best_c, best_r = _grow_to_contact(best_c, best_r)
    best_r = solve_radii_lp(best_c)      # final tangency push
    best_c, best_r = _grow_to_contact(best_c, best_r)
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


def _refine(centers, radii, maxiter=150):
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
            options={"maxiter": maxiter, "ftol": 1e-12},
        )
        z = max((z0, res.x), key=lambda t: np.sum(t[2 * n:]))
        if np.sum(z[2 * n:]) > np.sum(radii):
            c = np.column_stack([z[0:2 * n:2], z[1:2 * n:2]])
            return c, np.maximum(z[2 * n:], 1e-9)
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
