# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


try:
    from scipy.optimize import linprog
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def build_initial_centers():
    """Staggered hex-style rows: 5,6,5,6,4 = 26 circles."""
    counts = [5, 6, 5, 6, 4]
    ys = [0.10, 0.30, 0.50, 0.70, 0.90]
    centers = []
    for k, (cnt, y) in enumerate(zip(counts, ys)):
        x0 = (0.5 / cnt) if (k % 2 == 1) else 0.0
        xs = x0 + (np.arange(cnt) + 0.5) / cnt
        for x in xs:
            centers.append([x, y])
    return np.array(centers)


def solve_radii_lp(centers):
    """Exact LP: maximize sum(r) s.t. r_i+r_j <= d_ij, r_i <= wall dists."""
    n = centers.shape[0]

    if _HAS_SCIPY:
        ii, jj = np.triu_indices(n, 1)
        d = np.hypot(centers[ii, 0] - centers[jj, 0],
                     centers[ii, 1] - centers[jj, 1])
        A = np.zeros((len(ii) + 4 * n, n))
        b = np.zeros(len(ii) + 4 * n)
        A[np.arange(len(ii)), ii] = 1.0
        A[np.arange(len(ii)), jj] = 1.0
        b[:len(ii)] = d
        wall = np.minimum(centers, 1.0 - centers).ravel()
        for k in range(4):
            rows = np.arange(len(ii) + k * n, len(ii) + (k + 1) * n)
            A[rows, np.arange(n)] = 1.0
            b[rows] = wall.reshape(n, 4)[:, k] if wall.size == 4 * n else wall
        # wall distances per circle (correct construction)
        A = A[:0]
        b = b[:0]
        A1 = np.zeros((len(ii) + 4 * n, n))
        b1 = np.zeros(len(ii) + 4 * n)
        A1[np.arange(len(ii)), ii] = 1.0
        A1[np.arange(len(ii)), jj] = 1.0
        b1[:len(ii)] = d
        wd = np.minimum(centers, 1.0 - centers)  # (n,2)
        wd4 = np.minimum(wd[:, 0], wd[:, 1])
        for k in range(4):
            rows = np.arange(len(ii) + k * n, len(ii) + (k + 1) * n)
            A1[rows, np.arange(n)] = 1.0
            b1[rows] = wd4
        res = linprog(c=-np.ones(n), A_ub=A1, b_ub=b1,
                      bounds=[(0, None)] * n, method="highs")
        if res.success:
            return np.maximum(res.x, 0.0)

    # fallback: greedy proportional scaling
    radii = np.min(np.minimum(centers, 1.0 - centers), axis=1)
    for i in range(n):
        for j in range(i + 1, n):
            dd = np.hypot(*(centers[i] - centers[j]))
            if radii[i] + radii[j] > dd and radii[i] + radii[j] > 0:
                sc = dd / (radii[i] + radii[j])
                radii[i] *= sc
                radii[j] *= sc
    return radii


def ensure_valid(centers, radii):
    """Hard-clip radii so all constraints hold exactly."""
    r = np.minimum(radii, np.min(np.minimum(centers, 1.0 - centers), axis=1))
    n = len(r)
    for _ in range(4):
        ok = True
        for i in range(n):
            for j in range(i + 1, n):
                dd = np.hypot(*(centers[i] - centers[j]))
                ssum = r[i] + r[j]
                if ssum > dd:
                    f = dd / ssum if ssum > 0 else 1.0
                    r[i] *= f
                    r[j] *= f
                    ok = False
        if ok:
            break
    return r


def refine_centers(centers, steps=600, step0=0.03, seed=0):
    """Annealed hill-climb: perturb one center, LP re-solved each step."""
    rng = np.random.default_rng(seed)
    best = centers.copy()
    best_r = ensure_valid(best, solve_radii_lp(best))
    best_sum = best_r.sum()

    cur, cur_sum = best.copy(), best_sum
    for t in range(steps):
        step = step0 * (1.0 - t / steps) + 1e-4
        cand = cur.copy()
        i = rng.integers(len(cand))
        cand[i] = cand[i] + rng.normal(size=2) * step
        cand[i] = np.clip(cand[i], 0.02, 0.98)
        r = ensure_valid(cand, solve_radii_lp(cand))
        s = r.sum()
        # annealing acceptance: allow small worsening moves early
        temp = 1e-3 * (1.0 - t / steps)
        if s > cur_sum - temp:
            cur, cur_sum = cand, s
            if s > best_sum:
                best, best_r, best_sum = cand, r, s
    return best, best_r


def construct_packing():
    """
    Construct an arrangement of 26 circles in a unit square maximizing the
    sum of radii. Returns (centers, radii, sum_of_radii).
    """
    best_centers, best_radii = None, None
    best_sum = -1.0
    for seed in range(3):
        centers0 = build_initial_centers()
        if seed > 0:
            rng = np.random.default_rng(seed)
            centers0 = centers0 + rng.normal(size=centers0.shape) * 0.015
            centers0 = np.clip(centers0, 0.03, 0.97)
        c, r = refine_centers(centers0, steps=500, seed=seed)
        s = r.sum()
        if s > best_sum:
            best_centers, best_radii, best_sum = c, r, s
    radii = ensure_valid(best_centers, best_radii)
    return best_centers, radii, float(radii.sum())


def compute_max_radii(centers):
    """Backward-compatible helper: radii for given centers."""
    return ensure_valid(centers, solve_radii_lp(centers))


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