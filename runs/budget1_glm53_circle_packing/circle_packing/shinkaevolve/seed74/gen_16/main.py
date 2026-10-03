# EVOLVE-BLOCK-START
"""Circle packing for n=26 circles: simulated annealing on circle centers,
maximizing the sum of maximal feasible (water-filled) radii."""
import numpy as np


def construct_packing():
    n = 26
    rng = np.random.default_rng(12345)

    pts = _initial_hex_grid(n)
    best_pts = pts.copy()

    radii = _max_radii(pts)
    cur_score = radii.sum()
    best_score = cur_score

    T0, Tend = 2e-3, 1e-6
    iters = 3500
    reheat_every = 1200
    T = T0
    decay = (Tend / T0) ** (1.0 / (iters / 2))
    no_improve = 0

    for it in range(iters):
        # periodic reheat to escape local optima
        if it > 0 and it % reheat_every == 0:
            T = T0
            pts = best_pts.copy()
            radii = _max_radii(pts)
            cur_score = radii.sum()
            no_improve = 0

        # propose a move: single circle or small global jitter
        cand = pts.copy()
        if rng.random() < 0.75:
            i = rng.integers(n)
            scale = T / T0 * 0.08 + 0.005
            cand[i] += rng.normal(0, scale, 2)
        else:
            cand += rng.normal(0, 0.15 * T / T0 + 0.002, (n, 2))

        # keep centers strictly inside the square
        cand = np.clip(cand, 0.001, 0.999)

        cand_radii = _max_radii(cand)
        cand_score = cand_radii.sum()

        if cand_score >= cur_score or rng.random() < np.exp((cand_score - cur_score) / max(T, 1e-12)):
            pts = cand
            cur_score = cand_score
            if cand_score > best_score:
                best_score = cand_score
                best_pts = pts.copy()
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1

        T *= decay
        if T < Tend:
            T = Tend

    radii = _max_radii(best_pts)
    return best_pts, radii, float(radii.sum())


def _initial_hex_grid(n):
    """Staggered rows 5,6,5,6,4 -> 26 centers, hex-like spacing."""
    rows = [5, 6, 5, 6, 4]
    n_rows = len(rows)
    dy = 1.0 / (n_rows + 1)
    centers = []
    for r, count in enumerate(rows):
        y = (r + 1) * dy
        dx = 1.0 / (count + 1)
        offset = 0.0 if r % 2 == 0 else 0.5 * dx
        for c in range(count):
            x = (c + 1) * dx + offset
            x = min(max(x, 0.02), 0.98)
            centers.append([x, y])
    return np.array(centers)


def _max_radii(centers):
    """Maximal feasible radii: start at wall limits, iteratively scale down
    violating pairs. Vectorized joint scaling for speed."""
    pts = np.asarray(centers, dtype=float)
    n = pts.shape[0]
    x, y = pts[:, 0], pts[:, 1]
    r = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))

    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(d, np.inf)

    for _ in range(60):
        s = r[:, None] + r[None, :]
        ratio = d / np.maximum(s, 1e-15)
        # scale each radius by the worst ratio it participates in
        min_ratio = ratio.min(axis=1)
        min_ratio = np.minimum(min_ratio, 1.0)
        if (min_ratio > 1.0 - 1e-9).all():
            break
        r *= min_ratio
    return r


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
