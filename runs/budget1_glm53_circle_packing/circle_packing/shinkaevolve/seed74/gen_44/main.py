# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles.

Architecture:
  1. layouts:  many staggered hex-row patterns (each sums to 26 circles)
  2. ranking:  cheap vectorized greedy radii (fixed-point shrinking)
  3. climb:   slack-directed hill-climb where every trial is pre-screened
              with the greedy solver; only promising trials pay the LP cost
  4. polish:  8-direction coordinate descent, LP-verified
  5. safety:  hard validity clipping and full fallbacks without scipy
"""
import numpy as np
import time

try:
    from scipy.optimize import linprog
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

_N = 26

# Row patterns (circle counts per row), all summing to 26.
_PATTERNS = [
    [5, 6, 5, 6, 4],
    [4, 6, 6, 6, 4],
    [6, 5, 4, 5, 6],
    [4, 5, 6, 5, 4, 2],
    [5, 4, 5, 4, 5, 3],
    [2, 4, 5, 5, 5, 4, 1],
    [3, 5, 5, 5, 5, 3],
    [1, 4, 5, 6, 5, 4, 1],
    [6, 4, 6, 4, 6],
    [2, 5, 4, 4, 4, 5, 2],
]


def _dist(centers):
    return np.sqrt(((centers[:, None, :] - centers[None, :, :]) ** 2).sum(-1))


def fast_radii(centers, iters=50):
    """Greedy monotone-shrinking radii (valid, lower bound on LP optimum).

    r_i <- min(wall_i, min_j (d_ij - r_j)), shrunk monotonically from walls.
    Vectorized; converges in a handful of iterations for n=26.
    """
    n = centers.shape[0]
    x = centers[:, 0]
    y = centers[:, 1]
    walls = np.minimum(np.minimum(x, y), np.minimum(1 - x, 1 - y))
    d = _dist(centers)
    np.fill_diagonal(d, np.inf)
    r = walls.copy()
    for _ in range(iters):
        cand = (d - r[None, :]).min(axis=1)
        new_r = np.minimum(walls, np.maximum(cand, 0.0))
        new_r = np.minimum(new_r, r)  # monotone shrink -> always valid
        if np.max(r - new_r) < 1e-13:
            r = new_r
            break
        r = new_r
    return r


def lp_radii(centers):
    """Exact LP: maximize sum(r) s.t. r_i + r_j <= d_ij, r_i <= wall dist."""
    n = centers.shape[0]
    if not _HAS_SCIPY:
        return fast_radii(centers, 200)
    try:
        d = _dist(centers)
        iu, ju = np.triu_indices(n, 1)
        m = len(iu)
        A = np.zeros((m + n, n))
        rows = np.arange(m)
        A[rows, iu] = 1.0
        A[rows, ju] = 1.0
        walls = np.minimum(np.minimum(centers[:, 0], 1 - centers[:, 0]),
                           np.minimum(centers[:, 1], 1 - centers[:, 1]))
        idx = np.arange(n)
        A[m + idx, idx] = 1.0
        b = np.concatenate([d[iu, ju], walls])
        res = linprog(c=-np.ones(n), A_ub=A, b_ub=b,
                      bounds=[(0, None)] * n, method="highs")
        if res.success:
            return np.maximum(res.x, 0.0)
    except Exception:
        pass
    return fast_radii(centers, 200)


def ensure_valid(centers, radii):
    """Hard-clip radii so wall and pairwise constraints hold exactly."""
    x = centers[:, 0]
    y = centers[:, 1]
    r = np.array(radii, dtype=float)
    r = np.minimum(r, np.minimum(np.minimum(x, y), np.minimum(1 - x, 1 - y)))
    r = np.maximum(r, 0.0)
    n = len(r)
    d = _dist(centers)
    iu, ju = np.triu_indices(n, 1)
    for _ in range(6):
        s = r[iu] + r[ju]
        dd = d[iu, ju]
        viol = s > dd + 1e-12
        if not viol.any():
            break
        for i, j in zip(iu[viol], ju[viol]):
            ss = r[i] + r[j]
            if ss > d[i, j] and ss > 0:
                f = d[i, j] / ss
                r[i] *= f
                r[j] *= f
    return r


def _hex_rows(rows, scale=1.0, stagger=0.5):
    """Staggered triangular-lattice rows; spacing limited by width & height."""
    R = len(rows)
    k = np.sqrt(3.0) / 2.0
    s_h = 1.0 / ((R - 1) * k + 1.0)   # height: (R-1)*g + s <= 1
    s_w = 1.0 / max(rows)             # width: count*s <= 1
    s = min(s_h, s_w) * scale
    g = s * k
    pts = []
    for ri, count in enumerate(rows):
        y = 0.5 + (ri - (R - 1) / 2.0) * g
        off = (stagger * s) if (ri % 2 == 1) else 0.0
        for c in range(count):
            x = 0.5 + (c - (count - 1) / 2.0) * s + off
            x = min(max(x, 0.004), 0.996)
            pts.append([x, min(max(y, 0.004), 0.996)])
    return np.array(pts)


def _slack_dir(centers, radii, i):
    """Direction pointing away from circle i's tightest blocker."""
    x, y = centers[i]
    diffs = centers - centers[i]
    d = np.linalg.norm(diffs, axis=1)
    d[i] = np.inf
    j = int(np.argmin(d))
    gap_n = d[j] - radii[i] - radii[j]
    walls = np.array([x, 1.0 - x, y, 1.0 - y])
    wdirs = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]])
    wi = int(np.argmin(walls))
    gap_w = walls[wi] - radii[i]
    if gap_w <= gap_n:
        return wdirs[wi]
    return diffs[j] / d[j]


def _tightness(centers, radii):
    """How tight each circle is (gap to nearest blocker; negative = touching)."""
    d = _dist(centers)
    np.fill_diagonal(d, np.inf)
    nn_gap = (d - radii[None, :]).min(axis=1) - radii
    x = centers[:, 0]
    y = centers[:, 1]
    wgap = np.minimum(np.minimum(x, y), np.minimum(1 - x, 1 - y)) - radii
    return np.minimum(nn_gap, wgap)


def climb(centers, deadline, seed=0):
    """Slack-directed hill-climb with greedy pre-screen before each LP."""
    rng = np.random.default_rng(seed)
    centers = centers.copy()
    radii = ensure_valid(centers, lp_radii(centers))
    cur = float(radii.sum())
    best_c, best_r, best_s = centers.copy(), radii.copy(), cur
    step = 0.02
    fails = 0
    while time.time() < deadline:
        tight = _tightness(centers, radii)
        w = 0.01 + np.maximum(-tight, 0.0)
        w = w / w.sum()
        idx = int(rng.choice(_N, p=w))
        trial = centers.copy()
        sd = _slack_dir(centers, radii, idx)
        rnd = rng.normal(size=2)
        rnd = rnd / max(np.linalg.norm(rnd), 1e-12)
        move = (0.65 * sd + 0.35 * rnd) * step * rng.uniform(0.5, 1.5)
        trial[idx] = np.clip(trial[idx] + move, 0.003, 0.997)
        # ---- greedy pre-screen: skip LP for clearly-bad trials ----
        fr = fast_radii(trial, 40)
        if fr.sum() < cur - 0.02:
            fails += 1
            if rng.random() < 0.7:
                step = max(step * 0.9, 0.002)
            if fails > 80:
                centers = best_c.copy()
                radii = best_r.copy()
                cur = best_s
                step = rng.uniform(0.008, 0.03)
                fails = 0
            continue
        # ---- exact LP evaluation ----
        rt = ensure_valid(trial, lp_radii(trial))
        st = float(rt.sum())
        if st > cur + 1e-9:
            centers, radii, cur = trial, rt, st
            fails = 0
            step = min(step * 1.12, 0.05)
            if st > best_s + 1e-9:
                best_c, best_r, best_s = trial.copy(), rt.copy(), st
        else:
            fails += 1
            if rng.random() < 0.7:
                step = max(step * 0.9, 0.002)
            else:
                step = min(step * 1.3, 0.05)
            if fails > 80:
                centers = best_c.copy()
                radii = best_r.copy()
                cur = best_s
                step = rng.uniform(0.008, 0.03)
                fails = 0
    return best_c, best_r, best_s


_DIRS8 = np.array([[1, 0], [-1, 0], [0, 1], [0, -1],
                  [1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=float) / np.sqrt(2.0)


def polish(centers, radii, cur, deadline):
    """Greedy 8-direction coordinate descent, greedy-pre-screened, LP-verified."""
    centers = centers.copy()
    radii = radii.copy()
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(_N):
            if time.time() > deadline:
                break
            for step in (0.01, 0.004):
                for dd in _DIRS8:
                    if time.time() > deadline:
                        break
                    trial = centers.copy()
                    trial[i] = np.clip(trial[i] + dd * step, 0.003, 0.997)
                    fr = fast_radii(trial, 40)
                    if fr.sum() < cur - 0.02:
                        continue
                    rt = ensure_valid(trial, lp_radii(trial))
                    st = float(rt.sum())
                    if st > cur + 1e-9:
                        centers, radii, cur = trial, rt, st
                        improved = True
    return centers, radii, cur


def construct_packing():
    """
    Construct an arrangement of 26 circles in a unit square maximizing the
    sum of radii. Returns (centers, radii, sum_of_radii).
    """
    t0 = time.time()
    deadline = t0 + 2.6
    try:
        # ---- cheap ranking of many layouts with the greedy solver ----
        layouts = []
        for rows in _PATTERNS:
            for scale in (0.98, 1.0):
                layouts.append(_hex_rows(rows, scale))
        scored = sorted(range(len(layouts)),
                        key=lambda k: -fast_radii(layouts[k], 60).sum())
        top = [layouts[k] for k in scored[:4]]

        # ---- exact LP on the top few layouts ----
        best = None
        for L in top:
            r = ensure_valid(L, lp_radii(L))
            if best is None or r.sum() > best[2]:
                best = (L.copy(), r, float(r.sum()))

        # ---- dedicated climb time slices for the two best layouts ----
        for rank, L in enumerate(top[:2]):
            now = time.time()
            remaining = deadline - now
            if remaining < 0.15:
                break
            share = 0.45 if rank == 0 else 0.6
            c, r, s = climb(L, now + remaining * share, seed=rank)
            if s > best[2]:
                best = (c, r, s)

        # ---- final polish with whatever time remains ----
        now = time.time()
        if deadline - now > 0.1:
            c, r, s = polish(best[0], best[1], best[2], deadline - 0.03)
            if s > best[2]:
                best = (c, r, s)

        centers, radii, tot = best
    except Exception:
        centers = _hex_rows([5, 6, 5, 6, 4], 1.0)
        radii = fast_radii(centers, 200)
        tot = float(radii.sum())

    radii = ensure_valid(centers, radii)
    return centers, radii, float(radii.sum())


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