# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles."""
import numpy as np


def construct_packing():
    n = 26
    best = (None, None, -1.0)

    # ---- Stage 1: enumerate all row compositions summing to n ----
    layouts = []
    for n_rows in (4, 5, 6, 7):
        _gen_rows([], n_rows, n, 3, 7, layouts)

    # Score every composition with a couple of vertical spacings.
    scored = []
    for rows in layouts:
        for vscale in (1.0, 0.94):
            c = _build_layout(rows, vscale=vscale)
            r = _lp_radii(c)
            s = float(np.sum(r))
            scored.append((s, rows, vscale))
            if s > best[2]:
                best = (c, r, s)
    scored.sort(key=lambda t: -t[0])

    # ---- Stage 2: row-shift refinement on top layouts ----
    top = scored[:6]
    for _, rows, vscale in top:
        base = _build_layout(rows, vscale=vscale)
        # shift grid: shift each row independently by small deltas
        deltas = (-0.012, -0.004, 0.0, 0.004, 0.012)
        # Greedy per-row shift: optimize rows one at a time, 2 sweeps
        cur = base.copy()
        cur_r = _lp_radii(cur)
        cur_s = float(np.sum(cur_r))
        boundaries = []
        idx0 = 0
        for cnt in rows:
            boundaries.append((idx0, idx0 + cnt))
            idx0 += cnt
        for _sweep in range(2):
            improved = False
            for (a, b) in boundaries:
                for d in deltas:
                    trial = cur.copy()
                    trial[a:b, 0] += d
                    np.clip(trial[:, 0], 0.01, 0.99, out=trial[:, 0])
                    tr = _lp_radii(trial)
                    ts = float(np.sum(tr))
                    if ts > cur_s + 1e-9:
                        cur, cur_r, cur_s = trial, tr, ts
                        improved = True
            if not improved:
                break
        if cur_s > best[2]:
            best = (cur, cur_r, cur_s)

        # Also try vertical squeeze on the best-shifted layout
        for vy in (0.97, 1.03):
            trial = cur.copy()
            trial[:, 1] = 0.5 + (trial[:, 1] - 0.5) * vy
            tr = _lp_radii(trial)
            ts = float(np.sum(tr))
            if ts > best[2]:
                best = (trial, tr, ts)

    centers, radii, sum_radii = best
    return centers, radii, float(np.sum(radii))


def _gen_rows(prefix, remaining_rows, remaining, lo, hi, out):
    if remaining_rows == 0:
        if remaining == 0:
            out.append(list(prefix))
        return
    for c in range(lo, hi + 1):
        if c <= remaining - lo * (remaining_rows - 1):
            _gen_rows(prefix + [c], remaining_rows - 1,
                      remaining - c, lo, hi, out)


def _build_layout(rows, vscale=1.0, stagger=True):
    """Staggered hex-like centers for a row-count pattern."""
    n_rows = len(rows)
    dy = (1.0 / (n_rows + 1)) * vscale
    y0 = 0.5 - (n_rows - 1) * dy / 2.0
    centers = []
    for r, count in enumerate(rows):
        y = y0 + r * dy
        dx = 1.0 / (count + 1)
        offset = 0.5 * dx if (stagger and r % 2 == 1) else 0.0
        for c in range(count):
            x = (c + 1) * dx + offset
            x = min(max(x, 0.01), 0.99)
            centers.append([x, y])
    return np.array(centers)


def _lp_radii(centers):
    """Maximize sum(r) s.t. r_i + r_j <= d_ij, r_i <= border distance."""
    n = centers.shape[0]
    radii = compute_max_radii(centers)
    try:
        from scipy.optimize import linprog

        d = np.sqrt(((centers[:, None, :] - centers[None, :, :]) ** 2).sum(-1))
        iu, ju = np.triu_indices(n, 1)
        A_ub = np.zeros((len(iu) + n, n))
        b_ub = np.empty(len(iu) + n)
        A_ub[np.arange(len(iu)), iu] = 1.0
        A_ub[np.arange(len(iu)), ju] = 1.0
        b_ub[:len(iu)] = d[iu, ju]
        wall = np.minimum.reduce([centers[:, 0], centers[:, 1],
                                  1 - centers[:, 0], 1 - centers[:, 1]])
        A_ub[len(iu) + np.arange(n), np.arange(n)] = 1.0
        b_ub[len(iu):] = wall
        res = linprog(c=-np.ones(n), A_ub=A_ub, b_ub=b_ub,
                      bounds=[(0, None)] * n, method="highs")
        if res.success:
            radii = res.x
    except Exception:
        pass
    return radii


def compute_max_radii(centers):
    """Iterative shrink fallback: valid, non-overlapping, in-square radii."""
    n = centers.shape[0]
    radii = np.array([min(x, y, 1 - x, 1 - y) for x, y in centers])
    dist = np.sqrt(((centers[:, None, :] - centers[None, :, :]) ** 2).sum(-1))
    for _ in range(200):
        max_change = 0.0
        for i in range(n):
            new_r = min(centers[i, 0], centers[i, 1],
                        1 - centers[i, 0], 1 - centers[i, 1])
            if n > 1:
                new_r = min(new_r, (dist[i] - radii).min())
            new_r = max(new_r, 0.0)
            max_change = max(max_change, radii[i] - new_r)
            radii[i] = new_r
        if max_change < 1e-12:
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