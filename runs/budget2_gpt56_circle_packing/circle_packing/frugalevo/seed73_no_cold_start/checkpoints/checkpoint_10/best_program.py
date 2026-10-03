"""Deterministic multistart constructor-based circle packing for n=26."""
import numpy as np


def compute_max_radii(centers):
    """Conservative feasible radii for a supplied set of centers."""
    n = centers.shape[0]
    radii = np.minimum.reduce(
        (centers[:, 0], centers[:, 1], 1.0 - centers[:, 0], 1.0 - centers[:, 1])
    ).copy()
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(centers[i] - centers[j]))
            s = radii[i] + radii[j]
            if s > d and s > 0.0:
                q = d / s
                radii[i] *= q
                radii[j] *= q
    return np.maximum(radii, 0.0)


def _row_seed(counts, offsets, phase=0.0):
    """Build layered centers with count-dependent spans and alternating stagger."""
    rows = len(counts)
    ys = np.linspace(0.095, 0.905, rows)
    pts = []
    for k, (m, off) in enumerate(zip(counts, offsets)):
        # Denser rows receive a slightly wider span; alternating offsets create
        # distinct inter-row contact graphs without relying on randomness.
        width = 0.79 + 0.018 * (m - 5)
        left = 0.5 - width / 2.0 + off
        right = 0.5 + width / 2.0 + off
        xs = np.linspace(left, right, m)
        stagger = (0.5 * (right - left) / max(m - 1, 1)
                   if k % 2 else 0.0)
        for q, x in enumerate(xs):
            dx = 0.0055 * np.sin(1.71 * (k + 1) * (q + 1) + phase)
            dy = 0.0035 * np.cos(1.13 * (k + 1) * (q + 1) + phase)
            pts.append((np.clip(x + stagger + dx, 0.04, 0.96),
                        np.clip(ys[k] + dy, 0.04, 0.96)))
    return np.asarray(pts, dtype=float)


def _frame_seed(kind):
    """A boundary-frame / two-row-core seed, with exactly 26 centers."""
    pts = []
    # Four groups of four side circles.  The variants deliberately make
    # horizontal, vertical, or asymmetric wall contact graphs.
    if kind == 0:
        a, b, skew = 0.105, 0.895, 0.0
    elif kind == 1:
        a, b, skew = 0.085, 0.915, 0.012
    else:
        a, b, skew = 0.11, 0.89, -0.014

    side = np.linspace(0.18, 0.82, 4)
    for t in side:
        pts.append((t + skew, a))
        pts.append((t - skew, b))
        pts.append((a, t - skew))
        pts.append((b, t + skew))

    # Ten core centers in two staggered rows.
    for y, shift in ((0.405, -0.035 + skew), (0.595, 0.035 - skew)):
        for x in np.linspace(0.19, 0.81, 5):
            pts.append((x + shift, y))

    return np.clip(np.asarray(pts, dtype=float), 0.035, 0.965)


def _radius_lp(centers, fallback):
    """Exact maximum-radius LP for fixed centers."""
    try:
        from scipy.optimize import linprog
        n = len(centers)
        A = []
        b = []

        for i, (x, y) in enumerate(centers):
            row = np.zeros(n)
            row[i] = 1.0
            A.append(row); b.append(x)
            A.append(row.copy()); b.append(y)
            A.append(row.copy()); b.append(1.0 - x)
            A.append(row.copy()); b.append(1.0 - y)

        for i in range(n):
            for j in range(i + 1, n):
                row = np.zeros(n)
                row[i] = 1.0
                row[j] = 1.0
                A.append(row)
                b.append(float(np.linalg.norm(centers[i] - centers[j])))

        ans = linprog(
            -np.ones(n), A_ub=np.asarray(A), b_ub=np.asarray(b),
            bounds=[(0.0, 0.25)] * n, method="highs"
        )
        if ans.success and np.all(np.isfinite(ans.x)):
            return ans.x
    except Exception:
        pass
    return fallback


def _feasible(centers, radii, tol=2e-7):
    if centers.shape != (26, 2) or radii.shape != (26,):
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(radii)):
        return False
    if np.min(radii) < -tol:
        return False
    if np.min(centers[:, 0] - radii) < -tol:
        return False
    if np.min(centers[:, 1] - radii) < -tol:
        return False
    if np.min(1.0 - centers[:, 0] - radii) < -tol:
        return False
    if np.min(1.0 - centers[:, 1] - radii) < -tol:
        return False
    for i in range(26):
        d = centers[i + 1:] - centers[i]
        if len(d) and np.min(np.sum(d * d, axis=1) -
                             (radii[i] + radii[i + 1:]) ** 2) < -tol:
            return False
    return True


def construct_packing():
    n = 26

    # The first seed preserves the parent's useful layered construction;
    # subsequent seeds change row contact topology or use a frame/core topology.
    seeds = [
        _row_seed((5, 5, 6, 5, 5), (0.00, 0.018, 0.00, 0.018, 0.00), 0.0),
        _row_seed((5, 6, 5, 5, 5), (0.00, -0.018, 0.018, -0.010, 0.012), 0.7),
        _row_seed((5, 5, 5, 6, 5), (0.012, -0.010, 0.018, -0.018, 0.00), 1.4),
        _row_seed((4, 5, 4, 5, 4, 4), (0.00, 0.018, -0.010, 0.018, 0.00, 0.012), 2.1),
        _row_seed((4, 5, 5, 4, 5, 3), (0.008, -0.014, 0.012, -0.012, 0.014, 0.0), 2.8),
        _frame_seed(0),
        _frame_seed(1),
        _frame_seed(2),
    ]

    # A guaranteed valid baseline, used whenever SciPy is unavailable or a
    # numerical optimizer fails.
    base_centers = seeds[0]
    base_radii = compute_max_radii(base_centers)
    best_centers = base_centers.copy()
    best_radii = base_radii.copy()
    best_sum = float(np.sum(best_radii))

    try:
        from scipy.optimize import minimize

        ii, jj = np.triu_indices(n, 1)

        def objective(z):
            return -float(np.sum(z[2 * n:]))

        def constraints(z):
            c = z[:2 * n].reshape(n, 2)
            r = z[2 * n:]
            d = c[jj] - c[ii]
            return np.concatenate((
                c[:, 0] - r, c[:, 1] - r,
                1.0 - c[:, 0] - r, 1.0 - c[:, 1] - r,
                np.einsum("ij,ij->i", d, d) - (r[ii] + r[jj]) ** 2
            ))

        # Retain the eight established starts and add two deterministic
        # centroidal-cell layouts containing an interior dislocation.
        portfolio = list(seeds)

        def _cellular_defect_seed(defect):
            """Generate a 5x5 centroidal-cell layout plus one interstitial site."""
            grid = np.linspace(0.12, 0.88, 5)
            sites = []
            for row, y in enumerate(grid):
                shift = 0.038 if row % 2 else 0.0
                for x in grid:
                    sites.append((np.clip(x + shift, 0.035, 0.965), y))
            sites.append(defect)
            sites = np.asarray(sites, dtype=float)
            samples = np.linspace(0.02, 0.98, 41)
            sample_grid = np.asarray(
                [(x, y) for y in samples for x in samples], dtype=float
            )
            corner_sites = (0, 4, 20, 24)
            for _ in range(18):
                delta = sample_grid[:, None, :] - sites[None, :, :]
                labels = np.argmin(np.einsum("gsi,gsi->gs", delta, delta), axis=1)
                updated = sites.copy()
                for k in range(26):
                    assigned = sample_grid[labels == k]
                    if len(assigned):
                        centroid = np.mean(assigned, axis=0)
                        amount = 0.10 if k in corner_sites else 0.35
                        updated[k] += amount * (centroid - updated[k])
                sites = np.clip(updated, 0.035, 0.965)
            return sites

        portfolio.append(_cellular_defect_seed((0.50, 0.43)))
        portfolio.append(_cellular_defect_seed((0.57, 0.52)))

        candidates = []
        for centers0 in portfolio[:10]:
            # All starts are strictly feasible at this radius.
            radii0 = np.full(n, 0.018, dtype=float)
            z0 = np.concatenate((centers0.ravel(), radii0))
            result = minimize(
                objective, z0, method="SLSQP",
                bounds=[(0.001, 0.999)] * (2 * n) + [(0.001, 0.22)] * n,
                constraints={"type": "ineq", "fun": constraints},
                options={"maxiter": 500, "ftol": 3e-10, "disp": False},
            )
            if not result.success:
                continue

            centers = result.x[:2 * n].reshape(n, 2)
            radii = result.x[2 * n:]
            if _feasible(centers, radii, tol=5e-7):
                candidates.append(
                    (float(np.sum(radii)), centers.copy(), radii.copy())
                )

        # Fixed-center LP refinement is reserved for the two strongest
        # nonlinear candidates, maintaining the bounded runtime contract.
        candidates.sort(key=lambda item: item[0], reverse=True)
        for _, centers, radii in candidates[:2]:
            radii = _radius_lp(centers, radii)
            if not _feasible(centers, radii, tol=5e-7):
                continue
            value = float(np.sum(radii))
            if value > best_sum:
                best_centers, best_radii, best_sum = centers, radii, value

    except (ImportError, ValueError, RuntimeError, FloatingPointError):
        pass

    # Certified inward margin against strict external overlap checks.
    best_radii = np.maximum(0.0, best_radii * (1.0 - 3e-7))
    return best_centers, best_radii, float(np.sum(best_radii))


def run_packing():
    """Run the circle packing constructor for n=26."""
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii


def visualize(centers, radii):
    """Visualize a packing."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True)

    for i, (center, radius) in enumerate(zip(centers, radii)):
        ax.add_patch(Circle(center, radius, alpha=0.5))
        ax.text(center[0], center[1], str(i), ha="center", va="center")

    plt.title(f"Circle Packing (n={len(centers)}, sum={sum(radii):.6f})")
    plt.show()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii}")
    visualize(centers, radii)