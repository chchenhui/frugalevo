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
    for _ in range(3):
        _refine(centers)
        centers = _local_shift_search(centers, deltas=[0.002, 0.001, 0.0005])
        cand_sum = float(np.sum(compute_max_radii(centers)))
        if cand_sum > best_sum + 1e-9:
            best_sum = cand_sum
            best_centers = centers.copy()
        else:
            break
    centers = best_centers

    radii = compute_max_radii(centers)
    radii = _greedy_enlarge(centers, radii)
    sum_radii = float(np.sum(radii))
    return centers, radii, sum_radii


def _local_shift_search(centers, deltas=(0.01, 0.005, 0.002, 0.001, 0.0005)):
    """Greedily accept small center shifts that increase total radii.

    Tries axis-aligned shifts and diagonal (both-axis) shifts, since
    improving directions in hexagonal layouts are often diagonal.
    """
    centers = centers.copy()
    best = float(np.sum(compute_max_radii(centers)))
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
                        tot = float(np.sum(compute_max_radii(centers)))
                        if tot > best + 1e-9:
                            best = tot
                            improved = True
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
                        tot = float(np.sum(compute_max_radii(centers)))
                        if tot > best + 1e-9:
                            best = tot
                            improved = True
                        else:
                            centers[i, 0], centers[i, 1] = old0, old1
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


def compute_max_radii(centers):
    """
    Per-circle radii via Gauss-Seidel water filling, then monotone
    repair (only shrinks) to guarantee feasibility.
    """
    n = centers.shape[0]
    D = np.zeros((n, n))
    wall = np.zeros(n)
    for i in range(n):
        x, y = centers[i]
        wall[i] = min(x, y, 1.0 - x, 1.0 - y)
        for j in range(i + 1, n):
            d = float(np.sqrt(np.sum((centers[i] - centers[j]) ** 2)))
            D[i, j] = d
            D[j, i] = d
    r = np.zeros(n)
    for _ in range(300):
        changed = False
        for i in range(n):
            cap = wall[i]
            for j in range(n):
                if j != i and D[i, j] - r[j] < cap:
                    cap = D[i, j] - r[j]
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