# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles.

Approach: generate diverse hexagonal-lattice candidate layouts, run a
cheap SLSQP pass on each, deduplicate results into distinct basins, then
spend the remaining budget on high-budget SLSQP polish restarts for the
top-k distinct basins.
"""
import numpy as np
from scipy.optimize import minimize

SQRT3 = np.sqrt(3.0)
EPS = 1e-9
N = 26


def _lattice_candidates():
    """Generate diverse initial center layouts on clipped hex lattices."""
    cands = []
    row_sets = [
        [5, 4, 5, 4, 5, 3], [5, 4, 5, 4, 4, 4], [4, 5, 4, 5, 4, 4],
        [3, 5, 4, 5, 4, 5], [4, 4, 5, 5, 4, 4], [5, 5, 4, 4, 5, 3],
        [3, 4, 5, 5, 4, 5], [4, 5, 5, 4, 5, 3], [5, 4, 4, 5, 4, 4],
        [5, 4, 5, 5, 4, 3], [4, 5, 4, 5, 5, 3], [3, 4, 5, 4, 5, 5],
    ]
    for counts in row_sets:
        if sum(counts) != N:
            continue
        r = min(1.0 / 10.0, 1.0 / (2.0 + 5.0 * SQRT3))
        for sf in (1.0, 1.05):
            rr = r * sf
            dx = 2.0 * rr
            dy = SQRT3 * rr
            pts = []
            for row, c in enumerate(counts):
                y = rr + row * dy
                span = 2.0 * rr + (c - 1) * dx
                start = (1.0 - span) / 2.0 + rr
                for k in range(c):
                    pts.append([start + k * dx, y])
            pts = np.clip(np.array(pts), 0.005, 0.995)
            cands.append(pts)
    # Deterministic jittered variants for layout diversity
    rng = np.random.default_rng(12345)
    for base in list(cands[:6]):
        for _ in range(2):
            jit = rng.normal(0.0, 0.012, size=base.shape)
            cands.append(np.clip(base + jit, 0.005, 0.995))
    return cands


def _cons_all(z):
    """All constraint values (>= 0 means feasible): walls + pairwise gaps."""
    x = z[0:N]
    y = z[N:2 * N]
    r = z[2 * N:3 * N]
    walls = np.concatenate([
        x - r, (1.0 - x) - r, y - r, (1.0 - y) - r
    ])
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx * dx + dy * dy)
    iu = np.triu_indices(N, k=1)
    gaps = d[iu] - r[iu[0]] - r[iu[1]] - EPS
    return np.concatenate([walls, gaps])


def _slsqp_solve(z0, maxiter=250, ftol=1e-12):
    """Run one SLSQP pass maximizing sum of radii from z0."""
    bounds = [(0.002, 0.998)] * (2 * N) + [(0.001, 0.5)] * N
    cons = [{'type': 'ineq', 'fun': _cons_all}]
    try:
        res = minimize(lambda z: -np.sum(z[2 * N:3 * N]), z0,
                       method='SLSQP', bounds=bounds, constraints=cons,
                       options={'maxiter': maxiter, 'ftol': ftol})
        z = res.x if res.x is not None else z0
    except Exception:
        z = z0
    return z


def _repair(z):
    """Ensure strict feasibility by shrinking radii as needed."""
    x = np.clip(z[0:N], 0.001, 0.999)
    y = np.clip(z[N:2 * N], 0.001, 0.999)
    r = np.clip(z[2 * N:3 * N], 0.001, 0.5)
    wall = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
    r = np.minimum(r, wall)
    d = np.sqrt(((x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2))
    np.fill_diagonal(d, np.inf)
    for _ in range(200):
        gap = d - r[:, None] - r[None, :]
        np.fill_diagonal(gap, np.inf)
        violation = -gap.min(axis=1)
        if violation.max() <= -1e-12:
            break
        for i in np.where(violation > -1e-12)[0]:
            j = np.argmin(gap[i])
            need = d[i, j] - r[j] - 1e-8
            r[i] = min(r[i], max(need, 0.001))
    r = np.minimum(r, wall - 1e-9)
    r = np.maximum(r, 0.0005)
    centers = np.stack([x, y], axis=1)
    return centers, r, float(np.sum(r))


def _refine(centers, r_init):
    """Cheap SLSQP optimization of centers + radii from a lattice seed."""
    d = np.sqrt(((centers[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(d, np.inf)
    wall = np.minimum(np.minimum(centers[:, 0], 1 - centers[:, 0]),
                      np.minimum(centers[:, 1], 1 - centers[:, 1]))
    r = np.full(N, 0.5 * r_init)
    for _ in range(50):
        for i in range(N):
            r[i] = min(wall[i], np.min(d[i, :] - r) - 2 * EPS)
            r[i] = max(r[i], 0.005)
    z0 = np.concatenate([centers[:, 0], centers[:, 1], r])
    z = _slsqp_solve(z0, maxiter=150, ftol=1e-11)
    return _repair(z)


def _polish(centers, radii, rounds=6):
    """High-budget iterative SLSQP restarts from repaired feasible points."""
    best = (centers, radii, float(np.sum(radii)))
    z_cur = np.concatenate([centers[:, 0], centers[:, 1], radii])
    for _ in range(rounds):
        z_next = _slsqp_solve(z_cur, maxiter=600, ftol=1e-14)
        c, r, s = _repair(z_next)
        if s > best[2] + 1e-12:
            best = (c, r, s)
            z_cur = np.concatenate([c[:, 0], c[:, 1], r])
        else:
            break
    return best


def _basin_key(centers, existing, thresh=0.05):
    """Return True if this configuration is a distinct basin from all
    configurations in `existing` (max center-distance > thresh)."""
    for c0 in existing:
        if np.max(np.linalg.norm(centers - c0, axis=1)) < thresh:
            return False
    return True


def construct_packing():
    """
    Construct an arrangement of 26 circles in a unit square maximizing
    the sum of radii: cheap multi-start SLSQP, basin deduplication,
    then high-budget polish on the top-k distinct basins.
    """
    results = []
    for cand in _lattice_candidates():
        try:
            centers, radii, s = _refine(cand, 0.05)
        except Exception:
            continue
        results.append((centers, radii, s))

    results.sort(key=lambda t: t[2], reverse=True)

    # Deduplicate into distinct basins; keep top-5 distinct ones.
    basins = []
    for centers, radii, s in results:
        if _basin_key(centers, [b[0] for b in basins]):
            basins.append((centers, radii, s))
        if len(basins) >= 5:
            break

    best = None
    for centers, radii, s in basins:
        centers, radii, s = _polish(centers, radii, rounds=6)
        if best is None or s > best[2]:
            best = (centers, radii, s)

    centers, radii, sum_radii = best
    return centers, radii, sum_radii


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