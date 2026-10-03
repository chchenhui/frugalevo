# EVOLVE-BLOCK-START
"""Hexagonal-row constructor with greedy largest-first radius growth."""
import numpy as np

SQ3 = np.sqrt(3.0)


def construct_packing():
    """
    Enumerate staggered hexagonal-row layouts of 26 circles (ordered row
    partitions into 5..8 rows of 2..6 circles), rank by uniform-radius sum,
    then grow radii on the top candidates using greedy largest-first
    assignment plus water-filling, keeping the best valid result.
    """
    """Enumerate hex-row layouts, grow radii (greedy/water-fill), then
    multi-seed annealed local search on centers; keep best valid sum."""
    patterns = []
    for R in range(5, 9):
        _gen_patterns(26, R, 2, 6, [], patterns)
    cands = []
    for counts in patterns:
        centers, r0 = _build_layout(counts)
        if centers is None:
            continue
        for radii in (_scale_safe(centers, np.full(len(centers), r0)),
                      _grow(centers, r0),
                      _greedy_max(centers)):
            s = float(np.sum(radii))
            cands.append((s, centers, radii))
    cands.sort(key=lambda t: -t[0])
    best = cands[0]
    # Multi-seed annealed local search: nudge centers off the rigid
    # lattice to exploit corners/edges; evaluate greedy and water-filled.
    for seed in (12345, 777, 2024):
        rng = np.random.default_rng(seed)
        for _, centers, _ in cands[:6]:
            c, r, s = _anneal(centers, rng, iters=900, step=0.04)
            for rr in (r, _grow(c, float(np.min(r)), iters=200)):
                ss = float(np.sum(rr))
                if ss > best[0]:
                    best = (ss, c, rr)
            if s > best[0]:
                best = (s, c, r)
    return best[1], best[2], best[0]


def _anneal(centers, rng, iters=900, step=0.04):
    """Hill-climb: move one circle per step, accept if greedy radius sum
    improves; step size anneals geometrically. Returns best (c, r, s)."""
    cur = centers.copy()
    cur_r = _greedy_max(cur)
    cur_s = float(np.sum(cur_r))
    bc, br, bs = cur.copy(), cur_r, cur_s
    size, n = step, len(cur)
    for _ in range(iters):
        prop = cur.copy()
        i = rng.integers(n)
        prop[i] = np.clip(prop[i] + rng.normal(0, size, 2), 0.02, 0.98)
        pr = _greedy_max(prop)
        ps = float(np.sum(pr))
        if ps >= cur_s - 1e-9:
            cur, cur_r, cur_s = prop, pr, ps
            if ps > bs:
                bc, br, bs = prop.copy(), pr, ps
        size *= 0.997
    return bc, br, bs


def _gen_patterns(total, rows, lo, hi, cur, out):
    """Recursively generate ordered row counts summing to total."""
    if rows == 0:
        if total == 0:
            out.append(list(cur))
        return
    for m in range(lo, min(hi, total) + 1):
        cur.append(m)
        _gen_patterns(total - m, rows - 1, lo, hi, cur, out)
        cur.pop()


def _build_layout(counts):
    """Place circles in staggered hex rows; return centers and uniform radius."""
    R = len(counts)
    if R < 2 or sum(counts) != 26:
        return None, None
    r = 1.0 / (2.0 + SQ3 * (R - 1))
    for k, m in enumerate(counts):
        if m <= 1:
            continue
        if k % 2 == 0:
            r = min(r, 0.5 / m)
        else:
            r = min(r, 0.5 / (m - 1))
    if r <= 0:
        return None, None
    dy = SQ3 * r
    total = 2 * r + dy * (R - 1)
    y0 = (1.0 - total) / 2.0 + r
    centers = []
    for k, m in enumerate(counts):
        y = y0 + k * dy
        off = r if k % 2 == 1 else 0.0
        x0 = 0.5 - (m - 1) * r + off
        for j in range(m):
            centers.append([x0 + 2 * r * j, y])
    return np.array(centers), r


def _greedy_max(centers):
    """
    Greedy largest-first radius assignment: process circles in order of
    increasing border slack; each radius is set to the maximum feasible
    value given already-assigned neighbors. Yields a maximal radius vector.
    """
    n = len(centers)
    x, y = centers[:, 0], centers[:, 1]
    border = np.minimum(np.minimum(x, y), np.minimum(1 - x, 1 - y))
    d = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    np.fill_diagonal(d, np.inf)
    r = np.zeros(n)
    assigned = np.zeros(n, dtype=bool)
    for i in np.argsort(border):
        ub = border[i]
        if assigned.any():
            ub = min(ub, float(np.min(d[i, assigned] - r[assigned])))
        r[i] = max(ub, 1e-12)
        assigned[i] = True
    return _scale_safe(centers, r)


def _grow(centers, r0, iters=300):
    """Water-filling: each radius expands to its maximal feasible value."""
    n = len(centers)
    x, y = centers[:, 0], centers[:, 1]
    border = np.minimum(np.minimum(x, y), np.minimum(1 - x, 1 - y))
    d = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    np.fill_diagonal(d, np.inf)
    r = np.full(n, r0)
    for _ in range(iters):
        ub = np.minimum(border, np.min(d - r[None, :], axis=1))
        r = _scale_safe(centers, np.maximum(ub, 1e-12))
    return r


def _scale_safe(centers, radii):
    """Uniformly scale radii down so packing is strictly valid."""
    n = len(centers)
    x, y = centers[:, 0], centers[:, 1]
    border = np.minimum(np.minimum(x, y), np.minimum(1 - x, 1 - y))
    s = 1.0
    over = radii > border
    if over.any():
        s = min(s, float(np.min(border[over] / radii[over])))
    d = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    need = radii[:, None] + radii[None, :]
    iu = np.triu_indices(n, 1)
    viol = need[iu] > d[iu]
    if viol.any():
        s = min(s, float(np.min(d[iu][viol] / need[iu][viol])))
    return radii * s * 0.999999


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
