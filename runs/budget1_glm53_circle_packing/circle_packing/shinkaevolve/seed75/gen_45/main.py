# EVOLVE-BLOCK-START
"""Multi-start grow-and-relax circle packing for n=26 circles"""
import numpy as np

N = 26


def construct_packing():
    """
    Build 26 circles in the unit square maximizing the sum of radii.
    Deterministic multi-start: several structured seeds, each refined by
    a grow/relax/compact loop; the best final layout is returned.
    """
    # Single deterministic staggered-lattice seed (the layout family that
    # scored best previously): rows of 5,4,5,4,5,3 = 26 circles, spaced
    # wider than the uniform optimum so sparse rows can grow larger radii.
    s = 0.2
    row_counts = [5, 4, 5, 4, 5, 3]
    row_spacings = [s, s, s, s, s, 0.3]
    y0 = 0.1
    gap = (1.0 - 2.0 * y0) / 5.0
    idx = 0
    centers = np.zeros((N, 2))
    for row, count in enumerate(row_counts):
        sp = row_spacings[row]
        y = y0 + row * gap
        for j in range(count):
            centers[idx] = [0.5 + (j - (count - 1) / 2.0) * sp, y]
            idx += 1

    _refine(centers)

    # Greedy local-search on centers: try small shifts per circle, keep
    # any shift that raises the total radius sum after re-solving.
    centers = _local_shift_search(centers)

    # Shift->relax->shift fixed-point: after the first shift pass plateaus,
    # re-run the grow/relax refinement so coordinated multi-circle moves
    # can unlock further tiny shifts, then shift-search again at fine
    # deltas only. Always keep the best-scoring layout.
    best_centers = centers.copy()
    best_sum = float(np.sum(compute_max_radii(best_centers)))
    for _ in range(6):
        _refine(centers)
        centers = _local_shift_search(centers, deltas=[0.002, 0.001, 0.0005])
        cand_sum = float(np.sum(compute_max_radii(centers)))
        if cand_sum > best_sum + 1e-9:
            best_sum = cand_sum
            best_centers = centers.copy()
        else:
            break
    centers = best_centers

    # Exact LP for optimal (unequal) radii at the fixed centers; fall back
    # to the fast heuristic if scipy is unavailable or the LP fails.
    radii = solve_radii_lp(centers)
    if radii is None:
        radii = compute_max_radii(centers)
        radii = _greedy_enlarge(centers, radii)
    radii = radii * (1.0 - 1e-9)  # tiny safety shrink for strict validity
    sum_radii = float(np.sum(radii))
    return centers, radii, sum_radii


def solve_radii_lp(centers, eps=1e-9):
    """
    Exact maximum-sum radii at fixed centers via LP:
      maximize sum(r_i) s.t. r_i + r_j <= dist(i,j),
                            r_i <= min(x_i, y_i, 1-x_i, 1-y_i), r_i >= 0.
    Returns the optimal radii array, or None on failure.
    """
    try:
        from scipy.optimize import linprog
    except Exception:
        return None
    n = centers.shape[0]
    diff = centers[:, None, :] - centers[None, :, :]
    D = np.sqrt((diff ** 2).sum(-1))
    iu = np.triu_indices(n, 1)
    m = len(iu[0])
    A_pairs = np.zeros((m, n))
    A_pairs[np.arange(m), iu[0]] = 1.0
    A_pairs[np.arange(m), iu[1]] = 1.0
    b_pairs = D[iu] - 2.0 * eps
    x, y = centers[:, 0], centers[:, 1]
    wall = np.minimum(np.minimum(x, y), np.minimum(1.0 - x, 1.0 - y))
    A_wall = np.zeros((n, n))
    A_wall[np.arange(n), np.arange(n)] = 1.0
    A_ub = np.vstack([A_pairs, A_wall])
    b_ub = np.concatenate([b_pairs, wall - eps])
    res = linprog(c=-np.ones(n), A_ub=A_ub, b_ub=b_ub,
                  bounds=[(0.0, None)] * n, method="highs")
    if not res.success:
        return None
    r = np.maximum(res.x, 0.0)
    # guarantee feasibility against exact pairwise distances
    r = np.minimum(r, np.maximum(wall, 0.0))
    for _ in range(4):
        s = r[iu[0]] + r[iu[1]]
        bad = s > D[iu] + 1e-12
        if not bad.any():
            break
        idx = np.nonzero(bad)[0]
        i, j = iu[0][idx], iu[1][idx]
        scale = D[iu][idx] / s[idx]
        r[i] *= scale
        r[j] *= scale
    return r


def _local_shift_search(centers, deltas=(0.01, 0.005, 0.002, 0.001, 0.0005)):
    """Greedily accept small center shifts that increase total radii.

    Tries axis-aligned shifts and diagonal (both-axis) shifts, since
    improving directions in hexagonal layouts are often diagonal.
    The radius solver is warm-started from the previous (feasible)
    radii vector, so each candidate evaluation converges in a few
    Gauss-Seidel sweeps instead of a full cold-start solve.
    """
    centers = centers.copy()
    r = compute_max_radii(centers)
    best = float(np.sum(r))
    for delta in deltas:
        improved = True
        while improved:
            improved = False
            for i in range(N):
                # axis-aligned moves
                for axis in (0, 1):
                    for sign in (+1.0, -1.0):
                        old = centers[i, axis]
                        new = min(max(old + sign * delta, 1e-9), 1.0 - 1e-9)
                        if abs(new - old) < 1e-12:
                            continue
                        centers[i, axis] = new
                        cand = compute_max_radii(centers, r_init=r, max_sweeps=25)
                        tot = float(np.sum(cand))
                        if tot > best + 1e-9:
                            best = tot
                            improved = True
                            r = cand
                        else:
                            centers[i, axis] = old
                # diagonal moves
                for sx in (+1.0, -1.0):
                    for sy in (+1.0, -1.0):
                        old0, old1 = centers[i, 0], centers[i, 1]
                        new0 = min(max(old0 + sx * delta, 1e-9), 1.0 - 1e-9)
                        new1 = min(max(old1 + sy * delta, 1e-9), 1.0 - 1e-9)
                        if abs(new0 - old0) < 1e-12 and abs(new1 - old1) < 1e-12:
                            continue
                        centers[i, 0], centers[i, 1] = new0, new1
                        cand = compute_max_radii(centers, r_init=r, max_sweeps=25)
                        tot = float(np.sum(cand))
                        if tot > best + 1e-9:
                            best = tot
                            improved = True
                            r = cand
                        else:
                            centers[i, 0], centers[i, 1] = old0, old1
    # final exact cold solve to remove any warm-start bias
    r = compute_max_radii(centers)
    return centers


def _make_seeds():
    """Deterministic set of structured starting layouts."""
    seeds = []
    # Staggered lattices with varying row-count patterns and spacings
    patterns = [
        [5, 4, 5, 4, 5, 3],
        [4, 5, 4, 5, 4, 4],
        [6, 5, 5, 5, 5],
        [5, 5, 6, 5, 5],
        [7, 6, 6, 7],
        [4, 6, 6, 6, 4],
        [3, 5, 5, 5, 5, 3],
    ]
    for counts in patterns:
        for s_extra in (0.0, 0.01, 0.02):
            nrow = len(counts)
            # vertical span: rows separated by hex spacing where possible
            s = (1.0 + s_extra) / max(max(counts), (1.0 + (nrow - 1) * np.sqrt(3) / 2.0) / 1.0)
            s = min(1.0 / max(counts), 1.0 / (1.0 + (nrow - 1) * np.sqrt(3) / 2.0))
            s *= (1.0 + s_extra * 0.5)
            h = s * np.sqrt(3) / 2.0
            total_h = s + (nrow - 1) * h
            if total_h > 1.0 or s * max(counts) > 1.0:
                h = min(h, (1.0 - s) / (nrow - 1)) if nrow > 1 else 0
            pts = []
            for row, cnt in enumerate(counts):
                y = s / 2.0 + row * h
                off = (s / 2.0) if row % 2 else 0.0
                # center rows horizontally inside leftover margin
                width = (cnt - 1) * s + off
                x0 = max(s / 2.0, (1.0 - width) / 2.0)
                for j in range(cnt):
                    x = x0 + off + j * s
                    if x <= 1.0 - 1e-6 and y <= 1.0 - 1e-6:
                        pts.append((x, min(y, 1.0 - 1e-6)))
            if len(pts) >= N:
                seeds.append(np.array(pts[:N]))
    # Corner-anchored seed: 4 corner circles + inner staggered lattice
    c = np.array([[0.15, 0.15], [0.85, 0.15], [0.15, 0.85], [0.85, 0.85]])
    rest = []
    for row in range(4):
        cnt = 6 if row % 2 == 0 else 5
        y = 0.25 + row * 0.17
        off = 0.0 if row % 2 == 0 else 0.09
        for j in range(cnt):
            x = 0.12 + off + j * 0.155
            rest.append((x, y))
    seed = np.vstack([c, np.array(rest[:N - 4])])
    if len(seed) >= N:
        seeds.append(seed[:N])
    # Random-ish deterministic seed (Halton-like) as diversity
    pts = []
    p1, p2 = 2, 3
    def halton(i, b):
        f, r = 1.0, 0.0
        while i > 0:
            f /= b
            r += f * (i % b)
            i //= b
        return r
    for i in range(1, N + 1):
        pts.append((0.05 + 0.9 * halton(i, p1), 0.05 + 0.9 * halton(i, p2)))
    seeds.append(np.array(pts))
    return seeds


def _refine(centers):
    """Grow-relax-compact loop, returns best sum seen."""
    n = len(centers)
    radii = compute_max_radii(centers)
    best_sum = np.sum(radii)
    best_centers = centers.copy()
    n_iter = 250
    for it in range(n_iter):
        step = 0.03 * (1.0 - it / n_iter) + 0.0002
        for i in range(n):
            xi, yi = centers[i]
            ri = radii[i]
            fx = fy = 0.0
            for j in range(n):
                if j == i:
                    continue
                dx = xi - centers[j, 0]
                dy = yi - centers[j, 1]
                dist = np.sqrt(dx * dx + dy * dy) + 1e-12
                overlap = ri + radii[j] - dist
                if overlap > 0.0:
                    fx += overlap * dx / dist
                    fy += overlap * dy / dist
            if xi - ri < 0.0:
                fx += (ri - xi)
            if xi + ri > 1.0:
                fx -= (xi + ri - 1.0)
            if yi - ri < 0.0:
                fy += (ri - yi)
            if yi + ri > 1.0:
                fy -= (yi + ri - 1.0)
            norm = np.sqrt(fx * fx + fy * fy)
            if norm > 1e-12:
                centers[i, 0] = min(max(xi + step * fx / norm, 1e-9), 1.0 - 1e-9)
                centers[i, 1] = min(max(yi + step * fy / norm, 1e-9), 1.0 - 1e-9)
        if it % 10 == 9:
            radii = compute_max_radii(centers)
            tot = np.sum(radii)
            if tot > best_sum:
                best_sum = tot
                best_centers = centers.copy()
            # occasional compaction toward center
            if it % 50 == 49:
                bbox = centers.max(axis=0) - centers.min(axis=0)
                scale = min((1.0 - 1e-6) / bbox[0] if bbox[0] > 0 else 1.0,
                            (1.0 - 1e-6) / bbox[1] if bbox[1] > 0 else 1.0)
                if scale < 1.0 and scale > 0.5:
                    mid = (centers.max(axis=0) + centers.min(axis=0)) / 2.0
                    centers[:] = mid + (centers - mid) * scale
    centers[:] = best_centers
    return best_sum


def compute_max_radii(centers, r_init=None, max_sweeps=300):
    """
    Per-circle radii via Gauss-Seidel water filling (optionally warm-started
    from r_init), then monotone repair (only shrinks) to guarantee
    feasibility. Vectorized inner loop for speed.
    """
    n = centers.shape[0]
    diff = centers[:, None, :] - centers[None, :, :]
    D = np.sqrt((diff ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)
    x, y = centers[:, 0], centers[:, 1]
    wall = np.minimum(np.minimum(x, y), np.minimum(1.0 - x, 1.0 - y))
    if r_init is not None and len(r_init) == n:
        r = np.minimum(np.maximum(r_init, 0.0), np.maximum(wall, 0.0)).copy()
    else:
        r = np.zeros(n)
    for _ in range(max_sweeps):
        changed = False
        for i in range(n):
            cap = np.min(D[i] - r)
            if wall[i] < cap:
                cap = wall[i]
            cap = max(cap, 0.0)
            if abs(cap - r[i]) > 1e-12:
                changed = True
            r[i] = cap
        if not changed:
            break
    r = np.minimum(r, wall)
    for i in range(n):
        for j in range(i + 1, n):
            if r[i] + r[j] > D[i, j]:
                scale = D[i, j] / (r[i] + r[j])
                r[i] *= scale
                r[j] *= scale
    r = np.minimum(r, wall)
    return r


def _greedy_enlarge(centers, radii):
    """Try enlarging each circle slightly, keep center if feasible."""
    n = len(centers)
    for _ in range(3):
        for i in range(n):
            x, y = centers[i]
            cap = min(x, y, 1.0 - x, 1.0 - y)
            for j in range(n):
                if j == i:
                    continue
                d = float(np.sqrt(np.sum((centers[i] - centers[j]) ** 2)))
                cap = min(cap, d - radii[j])
            if cap > radii[i]:
                radii[i] = max(cap, 0.0)
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