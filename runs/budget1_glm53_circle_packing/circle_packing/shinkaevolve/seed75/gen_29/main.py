# EVOLVE-BLOCK-START
"""Multi-start grow-and-relax circle packing for n=26 circles.
Optimized with incremental warm-started radius solving for fast
local shift search."""
import numpy as np

N = 26


def construct_packing():
    """
    Build 26 circles in the unit square maximizing the sum of radii.
    Deterministic multi-start: structured seeds refined by a
    grow/relax loop, then incremental local shift search with
    warm-started radius re-solving, then greedy enlargement plus
    a second fine shift round.
    """
    best_sum = -1.0
    best_centers = None
    for seed in _make_seeds():
        centers = seed.copy()
        _refine(centers)
        for _ in range(5):
            centers = _pressure_perturb(centers)
            _refine(centers)
        tot = float(np.sum(compute_max_radii(centers)))
        if tot > best_sum:
            best_sum = tot
            best_centers = centers.copy()

    centers = _local_shift_search(best_centers)
    radii = compute_max_radii(centers)
    radii = _greedy_enlarge(centers, radii)
    # Second, finer shift round after enlargement: greedy enlarge
    # frees tight constraints, exposing new improving moves.
    centers = _local_shift_search(centers, deltas=(0.0005, 0.0002),
                                  max_rounds=2)
    radii = compute_max_radii(centers)
    radii = _greedy_enlarge(centers, radii)
    sum_radii = float(np.sum(radii))
    return centers, radii, sum_radii


class _RadiusSolver:
    """Incremental radius evaluator with cached distance matrix and
    warm-started Gauss-Seidel."""

    def __init__(self, centers):
        self.set_centers(centers)

    def set_centers(self, centers):
        self.c = centers.copy()
        n = len(centers)
        diff = self.c[:, None, :] - self.c[None, :, :]
        self.D = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(self.D, np.inf)
        self.wall = np.minimum.reduce([self.c[:, 0], self.c[:, 1],
                                       1.0 - self.c[:, 0], 1.0 - self.c[:, 1]])
        self.r = np.zeros(n)

    def move(self, i, new_xy):
        """Update cached distance row/col for circle i only."""
        self.c[i] = new_xy
        d = np.sqrt(np.sum((self.c - new_xy) ** 2, axis=1))
        d[i] = np.inf
        self.D[i, :] = d
        self.D[:, i] = d
        x, y = new_xy
        self.wall[i] = min(x, y, 1.0 - x, 1.0 - y)

    def solve(self, passes=25, tol=1e-13):
        """Warm-started Gauss-Seidel; radii converge quickly when
        centers change only slightly."""
        r = self.r
        D = self.D
        wall = self.wall
        n = len(r)
        for _ in range(passes):
            max_change = 0.0
            for i in range(n):
                cap = np.min(D[i] - r)
                if wall[i] < cap:
                    cap = wall[i]
                if cap < 0.0:
                    cap = 0.0
                if abs(cap - r[i]) > max_change:
                    max_change = abs(cap - r[i])
                r[i] = cap
            if max_change < tol:
                break
        return r

    def repair(self):
        """Monotone shrinkage repair to guarantee strict feasibility
        (only shrinks radii, so warm start stays conservative)."""
        r = self.r
        n = len(r)
        D = self.D
        wall = self.wall
        np.minimum(r, wall, out=r)
        changed = True
        guard = 0
        while changed and guard < 10:
            changed = False
            guard += 1
            for i in range(n):
                for j in range(i + 1, n):
                    s = r[i] + r[j]
                    d = D[i, j]
                    if s > d:
                        scale = d / s
                        r[i] *= scale
                        r[j] *= scale
                        changed = True
        np.minimum(r, wall, out=r)
        return r


def _local_shift_search(centers, deltas=(0.01, 0.005, 0.002, 0.001,
                                         0.0005), max_rounds=4):
    """Greedily accept small center shifts that increase total radii,
    using incremental warm-started evaluation (fast)."""
    solver = _RadiusSolver(centers)
    solver.solve(passes=300)
    solver.repair()
    best = float(np.sum(solver.r))
    c = solver.c
    for delta in deltas:
        for _ in range(max_rounds):
            improved = False
            for i in range(N):
                for axis in (0, 1):
                    for sign in (+1.0, -1.0):
                        old = c[i, axis]
                        new = min(max(old + sign * delta, 1e-9), 1.0 - 1e-9)
                        if abs(new - old) < 1e-12:
                            continue
                        old_xy = c[i].copy()
                        solver.move(i, np.array(
                            [new, old_xy[1]] if axis == 0
                            else [old_xy[0], new]))
                        r = solver.solve(passes=25)
                        tot = float(np.sum(r))
                        if tot > best + 1e-10:
                            best = tot
                            improved = True
                        else:
                            solver.move(i, old_xy)
                            # restore previous radii warm start
                            solver.r[:] = r
                            # re-solve from the restored geometry to get
                            # the correct warm start back
                            solver.solve(passes=25)
            if not improved:
                break
    solver.solve(passes=300)
    solver.repair()
    return solver.c


def _pressure_perturb(centers):
    """Move each center away from its tightest constraints."""
    n = len(centers)
    radii = compute_max_radii(centers)
    out = centers.copy()
    for i in range(n):
        x, y = centers[i]
        ri = radii[i]
        worst_gap = 1e9
        fx = fy = 0.0
        for j in range(n):
            if j == i:
                continue
            dx = x - centers[j, 0]
            dy = y - centers[j, 1]
            dist = np.sqrt(dx * dx + dy * dy) + 1e-12
            gap = dist - (ri + radii[j])
            worst_gap = min(worst_gap, gap)
            if gap < 1e-4:
                fx += dx / dist
                fy += dy / dist
        for wall_gap, ddx, ddy in ((x - ri, -1.0, 0.0),
                                   (1.0 - x - ri, 1.0, 0.0),
                                   (y - ri, 0.0, -1.0),
                                   (1.0 - y - ri, 0.0, 1.0)):
            worst_gap = min(worst_gap, wall_gap)
            if wall_gap < 1e-4:
                fx += ddx
                fy += ddy
        norm = np.sqrt(fx * fx + fy * fy)
        if norm > 1e-9:
            step = 0.02 * max(0.0, -worst_gap) + 0.0015
            out[i, 0] = min(max(x + step * fx / norm, 1e-9), 1.0 - 1e-9)
            out[i, 1] = min(max(y + step * fy / norm, 1e-9), 1.0 - 1e-9)
    return out


def _make_seeds():
    """Small deterministic set of well-shaped starting layouts."""
    seeds = []
    # Seed 1: staggered lattice 5,4,5,4,5,3 (best family)
    pts = []
    s = 0.2
    row_counts = [5, 4, 5, 4, 5, 3]
    row_spacings = [s, s, s, s, s, 0.3]
    y0 = 0.1
    gap = (1.0 - 2.0 * y0) / 5.0
    for row, count in enumerate(row_counts):
        sp = row_spacings[row]
        y = y0 + row * gap
        for j in range(count):
            pts.append((0.5 + (j - (count - 1) / 2.0) * sp, y))
    seeds.append(np.array(pts[:N]))
    # Seed 2: corner-anchored with mixed radii
    pts = []
    for cx in (0.12, 0.88):
        for cy in (0.12, 0.88):
            pts.append((cx, cy))
    for x in (0.3, 0.5, 0.7):
        pts.append((x, 0.06))
        pts.append((x, 0.94))
    for y in (0.3, 0.5, 0.7):
        pts.append((0.06, y))
        pts.append((0.94, y))
    for row in range(3):
        cnt = 4 if row % 2 == 0 else 3
        y = 0.30 + row * 0.20
        off = 0.0 if row % 2 == 0 else 0.10
        for j in range(cnt):
            pts.append((0.22 + off + j * 0.20, y))
    seeds.append(np.array(pts[:N]))
    # Seed 3: denser hex lattice 6,5,6,5,4
    pts = []
    s = 1.0 / 6.0
    h = s * np.sqrt(3.0) / 2.0
    row_counts = [6, 5, 6, 5, 4]
    y = 0.08
    for row, cnt in enumerate(row_counts):
        off = (s / 2.0) if row % 2 else 0.0
        x0 = (1.0 - ((cnt - 1) * s + off)) / 2.0 + off
        for j in range(cnt):
            pts.append((x0 + j * s, min(y, 1.0 - 1e-6)))
        y += h
    seeds.append(np.array(pts[:N]))
    # Seed 4: ring-based
    pts = [(0.5, 0.5)]
    for i in range(8):
        a = 2 * np.pi * i / 8
        pts.append((0.5 + 0.22 * np.cos(a), 0.5 + 0.22 * np.sin(a)))
    for i in range(17):
        a = 2 * np.pi * i / 17
        pts.append((0.5 + 0.42 * np.cos(a), 0.5 + 0.42 * np.sin(a)))
    seeds.append(np.clip(np.array(pts[:N]), 0.02, 0.98))
    # Seed 5: alternate staggered lattice 4,5,4,5,4,4
    pts = []
    s = 0.2
    y0 = 0.08
    gap = (1.0 - 2.0 * y0) / 5.0
    row_counts = [4, 5, 4, 5, 4, 4]
    for row, count in enumerate(row_counts):
        y = y0 + row * gap
        off = 0.0 if row % 2 == 0 else s / 2.0
        x0 = 0.5 - ((count - 1) * s + off) / 2.0 + off
        for j in range(count):
            pts.append((x0 + j * s, y))
    seeds.append(np.array(pts[:N]))
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
            if it % 50 == 49:
                bbox = centers.max(axis=0) - centers.min(axis=0)
                scale = min((1.0 - 1e-6) / bbox[0] if bbox[0] > 0 else 1.0,
                            (1.0 - 1e-6) / bbox[1] if bbox[1] > 0 else 1.0)
                if 0.5 < scale < 1.0:
                    mid = (centers.max(axis=0) + centers.min(axis=0)) / 2.0
                    centers[:] = mid + (centers - mid) * scale
    centers[:] = best_centers
    return best_sum


def compute_max_radii(centers):
    """
    Per-circle radii via vectorized Gauss-Seidel water filling, then
    monotone repair (only shrinks) to guarantee feasibility.
    """
    n = centers.shape[0]
    diff = centers[:, None, :] - centers[None, :, :]
    D = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(D, np.inf)
    wall = np.minimum.reduce([centers[:, 0], centers[:, 1],
                              1.0 - centers[:, 0], 1.0 - centers[:, 1]])
    r = np.zeros(n)
    for _ in range(300):
        max_change = 0.0
        for i in range(n):
            cap = np.min(D[i] - r)
            if wall[i] < cap:
                cap = wall[i]
            if cap < 0.0:
                cap = 0.0
            if abs(cap - r[i]) > max_change:
                max_change = abs(cap - r[i])
            r[i] = cap
        if max_change < 1e-12:
            break
    np.minimum(r, wall, out=r)
    for i in range(n):
        for j in range(i + 1, n):
            if r[i] + r[j] > D[i, j]:
                scale = D[i, j] / (r[i] + r[j])
                r[i] *= scale
                r[j] *= scale
    np.minimum(r, wall, out=r)
    return r


def _greedy_enlarge(centers, radii):
    """Try enlarging each circle to its exact constraint, keep if feasible."""
    n = len(centers)
    D = np.sqrt(np.sum((centers[:, None, :] - centers[None, :, :]) ** 2,
                       axis=2))
    np.fill_diagonal(D, np.inf)
    for _ in range(3):
        for i in range(n):
            x, y = centers[i]
            cap = min(x, y, 1.0 - x, 1.0 - y)
            cap = min(cap, np.min(D[i] - radii))
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