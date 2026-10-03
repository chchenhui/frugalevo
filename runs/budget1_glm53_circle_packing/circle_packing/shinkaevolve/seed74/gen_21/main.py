# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles.

Architecture:
  1. multi-start layouts: staggered hex-style row variants
  2. vectorized exact LP for radii (sparse constraint matrix, triu indices)
  3. hill-climb with fast-radii pre-screen + exact LP evaluation
"""
import numpy as np

try:
    from scipy.optimize import linprog
    from scipy.sparse import csr_matrix
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# ------------------------------------------------------------- layouts ----
def build_layout(counts, ys):
    centers = []
    for k, (cnt, y) in enumerate(zip(counts, ys)):
        x0 = 0.5 / cnt if (k % 2 == 1) else 0.0
        xs = x0 + (np.arange(cnt) + 0.5) / cnt
        for x in xs:
            centers.append([x, y])
    return np.array(centers)


def initial_layouts():
    return [
        build_layout([5, 6, 5, 6, 4], [0.10, 0.30, 0.50, 0.70, 0.90]),
        build_layout([6, 5, 6, 5, 4], [0.10, 0.30, 0.50, 0.70, 0.90]),
        build_layout([4, 6, 6, 6, 4], [0.09, 0.30, 0.50, 0.70, 0.91]),
        build_layout([5, 6, 6, 5, 4], [0.10, 0.30, 0.50, 0.70, 0.90]),
        build_layout([6, 4, 6, 4, 6], [0.09, 0.30, 0.50, 0.70, 0.91]),
    ]


# ------------------------------------------------- fast radius solving ----
def fast_radii(centers, iters=60):
    """Iterative shrinking: r_i = min(walls, min_j (d_ij - r_j)). Monotone, safe."""
    x = centers[:, 0]
    y = centers[:, 1]
    walls = np.minimum(np.minimum(x, y), np.minimum(1 - x, 1 - y))
    d = np.sqrt(((centers[:, None, :] - centers[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    r = walls.copy()
    for _ in range(iters):
        cand = np.min(d - r[None, :], axis=1)
        new_r = np.minimum(walls, np.maximum(cand, 0.0))
        new_r = np.minimum(new_r, r)
        if np.max(r - new_r) < 1e-12:
            r = new_r
            break
        r = new_r
    return r


def pairwise_data(centers):
    """Vectorized pairwise distances and index arrays."""
    n = centers.shape[0]
    i_idx, j_idx = np.triu_indices(n, 1)
    diffs = centers[i_idx] - centers[j_idx]
    dists = np.sqrt((diffs ** 2).sum(axis=1))
    return i_idx, j_idx, dists


def lp_radii(centers):
    """Exact LP (vectorized, sparse): max sum r s.t. r_i+r_j<=d_ij, r_i<=walls."""
    n = centers.shape[0]
    if _HAS_SCIPY:
        i_idx, j_idx, dists = pairwise_data(centers)
        m = len(i_idx)

        # wall constraints: n*4 rows, each with a single unit entry
        walls = np.concatenate([centers[:, 0], centers[:, 1],
                                1.0 - centers[:, 0], 1.0 - centers[:, 1]])
        wall_rows = np.repeat(np.arange(4 * n), 1)
        wall_cols = np.tile(np.arange(n), 4)
        # build via coo-style arrays
        wall_rows = np.concatenate([np.arange(n)] * 4)
        wall_cols = np.tile(np.arange(n), 4)
        wall_data = np.ones(4 * n)

        # pairwise constraints: m rows, entries at (i,1) and (j,1)
        pair_rows = np.concatenate([np.arange(m), np.arange(m)])
        pair_cols = np.concatenate([i_idx, j_idx])
        pair_data = np.ones(2 * m)

        rows = np.concatenate([wall_rows, pair_rows])
        cols = np.concatenate([wall_cols, pair_cols])
        data = np.concatenate([wall_data, pair_data])
        b = np.concatenate([walls, dists])

        A = csr_matrix((data, (rows, cols)), shape=(4 * n + m, n))
        res = linprog(c=-np.ones(n), A_ub=A, b_ub=b,
                      bounds=[(0, None)] * n, method="highs")
        if res.success:
            return np.maximum(res.x, 0.0)
    return fast_radii(centers, iters=200)


def ensure_valid(centers, radii):
    """Hard-clip radii so all constraints hold exactly (numerical safety)."""
    r = np.minimum(radii, np.min(np.minimum(centers, 1.0 - centers), axis=1))
    n = len(r)
    i_idx, j_idx, d = pairwise_data(centers)
    for _ in range(6):
        s = r[i_idx] + r[j_idx]
        bad = s > d
        if not bad.any():
            break
        f = np.where(s > 0, d / np.maximum(s, 1e-18), 1.0)
        # apply shrink factor per pair (last write wins per iteration is
        # still safe because we iterate until no violations remain)
        np.minimum.at(r, i_idx[bad], r[i_idx[bad]] * f[bad])
        np.minimum.at(r, j_idx[bad], r[j_idx[bad]] * f[bad])
    return r


# ------------------------------------------------- hill-climb search ----
def slack_direction(centers, radii, d_full, i):
    """Direction of maximum available slack for circle i (vectorized)."""
    n = len(centers)
    x, y = centers[i]
    ri = radii[i]
    # wall slacks
    wall_slack = np.array([x - ri, 1 - x - ri, y - ri, 1 - y - ri])
    wall_vecs = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=float)
    # neighbor gaps
    diff = centers[i] - centers
    dist = np.linalg.norm(diff, axis=1)
    dist[i] = np.inf
    gaps = dist - (ri + radii)
    ok = (gaps > 0) & (dist < 1e12)
    dirs = np.zeros((n, 2))
    dirs[ok] = diff[ok] / dist[ok, None]

    cands_slack = np.concatenate([wall_slack, gaps[ok]])
    cands_dir = np.concatenate([wall_vecs, dirs[ok]])
    if len(cands_slack) == 0:
        return np.zeros(2)
    avg = cands_dir.sum(axis=0)
    na = np.linalg.norm(avg)
    best_dir = cands_dir[np.argmax(cands_slack)]
    dvec = 0.7 * best_dir + 0.3 * (avg / na if na > 1e-12 else 0.0)
    nn = np.linalg.norm(dvec)
    return dvec / nn if nn > 1e-12 else np.zeros(2)


def local_search(centers, steps=1200, step0=0.02, seed=0):
    """Hill-climb with fast pre-screen + exact LP evaluation."""
    rng = np.random.default_rng(seed)
    n = len(centers)
    cur = centers.copy()
    cur_r = ensure_valid(cur, lp_radii(cur))
    cur_sum = cur_r.sum()
    cur_fast_sum = fast_radii(cur).sum()
    best, best_r, best_sum = cur.copy(), cur_r.copy(), cur_sum
    since_gain = 0
    for t in range(steps):
        step = step0 * (0.3 + 0.7 * (1 - t / steps)) + 1e-4
        cand = cur.copy()
        i = rng.integers(n)
        d_slack = slack_direction(cur, cur_r, None, i)
        rand = rng.normal(size=2)
        rand = rand / max(np.linalg.norm(rand), 1e-12)
        cand[i] = cand[i] + (0.7 * d_slack + 0.3 * rand) * step * rng.uniform(0.5, 1.5)
        cand[i] = np.clip(cand[i], 0.02, 0.98)

        # cheap pre-screen: fast radii must beat current fast sum
        f_fast = fast_radii(cand, iters=40).sum()
        if f_fast < cur_fast_sum - 1e-9:
            since_gain += 1
            if since_gain > 120:
                cur, cur_r, cur_sum = best.copy(), best_r.copy(), best_sum
                cur_fast_sum = fast_radii(cur).sum()
                since_gain = 0
            continue

        r = ensure_valid(cand, lp_radii(cand))
        s = r.sum()
        if s > cur_sum + 1e-9:
            cur, cur_r, cur_sum = cand, r, s
            cur_fast_sum = f_fast
            since_gain = 0
            if s > best_sum:
                best, best_r, best_sum = cand.copy(), r.copy(), s
        else:
            since_gain += 1
            if since_gain > 120:
                cur, cur_r, cur_sum = best.copy(), best_r.copy(), best_sum
                cur_fast_sum = fast_radii(cur).sum()
                since_gain = 0
    return best, best_r


# ------------------------------------------------------------- interface ----
def construct_packing():
    """
    Construct an arrangement of 26 circles in a unit square maximizing the
    sum of radii. Returns (centers, radii, sum_of_radii).
    """
    best_c, best_r, best_sum = None, None, -1.0
    layouts = initial_layouts()[:3]
    for li, layout in enumerate(layouts):
        c, r = local_search(layout, steps=1200, seed=li)
        if r.sum() > best_sum:
            best_c, best_r, best_sum = c, r, r.sum()
    # final exact LP polish on the best centers
    r = ensure_valid(best_c, lp_radii(best_c))
    if r.sum() > best_r.sum():
        best_r = r
    return best_c, best_r, float(best_r.sum())


def compute_max_radii(centers):
    """Backward-compatible helper: radii for given centers."""
    return ensure_valid(centers, lp_radii(centers))

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