"""Deterministic constrained constructor for 26 non-overlapping circles."""
import numpy as np


def construct_packing():
    """Optimize six starts from each of two staggered five-layer topologies."""
    n = 26

    # Compare the incumbent five-five-six-five-five arrangement with the
    # four-six-six-six-four arrangement, whose three interior rows provide
    # additional room for unequal radii away from the walls.
    patterns = [
        ((5, 5, 6, 5, 5), (0.095, 0.297, 0.500, 0.703, 0.905)),
        ((4, 6, 6, 6, 4), (0.085, 0.285, 0.500, 0.715, 0.915)),
    ]

    seeds = []
    variations = (
        (1.00, 1.00, 0.000),
        (0.96, 1.00, -0.010),
        (1.04, 1.00, 0.010),
        (1.00, 0.975, 0.000),
        (1.00, 1.025, 0.000),
        (0.98, 1.010, 0.014),
    )

    for row_sizes, ys in patterns:
        p = []
        for iy, (count, y) in enumerate(zip(row_sizes, ys)):
            pitch = 0.158 if count == 6 else (0.205 if count == 4 else 0.175)
            start = 0.5 - 0.5 * pitch * (count - 1)
            if iy & 1:
                start += 0.5 * pitch
            lo = 0.065
            hi = 0.935 - pitch * (count - 1)
            start = min(max(start, lo), hi)
            for k in range(count):
                p.append((start + k * pitch, y))
        seed = np.asarray(p, dtype=float)
        rows = np.repeat(np.arange(5), row_sizes)

        # Exactly six deterministic affine variants are retained per
        # topology, giving a balanced twelve-start optimization portfolio.
        for ys_scale, xs_scale, shear in variations:
            q = seed.copy()
            q[:, 1] = 0.5 + ys_scale * (q[:, 1] - 0.5)
            q[:, 0] = 0.5 + xs_scale * (q[:, 0] - 0.5)
            q[:, 0] += shear * (rows - 2.0)
            seeds.append(q)

    def cons(z):
        c = z[:2*n].reshape(n, 2)
        r = z[2*n:]
        out = [
            c[:, 0] - r, c[:, 1] - r,
            1.0 - c[:, 0] - r, 1.0 - c[:, 1] - r
        ]
        for i in range(n - 1):
            d = c[i+1:] - c[i]
            out.append(np.einsum("ij,ij->i", d, d) -
                       (r[i] + r[i+1:])**2)
        return np.concatenate(out)

    try:
        from scipy.optimize import minimize
        best = None
        for trial, seed in enumerate(seeds):
            r0 = 0.056 + 0.003 * (trial % 4)
            z0 = np.r_[seed.ravel(), np.full(n, r0)]
            res = minimize(
                lambda z: -np.sum(z[2*n:]),
                z0, method="SLSQP",
                bounds=[(0.0, 1.0)]*(2*n) + [(0.002, 0.25)]*n,
                constraints={"type": "ineq", "fun": cons},
                    options={"maxiter": 600, "ftol": 2e-10, "disp": False},
            )
            if res.success and np.all(np.isfinite(res.x)):
                if best is None or res.fun < best.fun:
                    best = res
        if best is None:
            centers = seeds[0]
            radii = np.full(n, 0.055)
        else:
            centers = best.x[:2*n].reshape(n, 2)
            radii = best.x[2*n:].copy()
    except Exception:
        centers = seeds[0]
        radii = np.full(n, 0.055)

    centers = np.clip(centers, 0.0, 1.0)
    radii = np.maximum(radii, 0.001)

    # Enforce a strict final margin against numerical validity tolerances.
    margin = 1.0
    margin = min(margin, float(np.min(centers / radii[:, None])))
    margin = min(margin, float(np.min((1.0 - centers) / radii[:, None])))
    for i in range(n):
        for j in range(i):
            margin = min(
                margin,
                float(np.linalg.norm(centers[i] - centers[j]) /
                      (radii[i] + radii[j]))
            )
    radii *= min(1.0, 0.999999 * margin)
    return centers, radii, float(np.sum(radii))


def compute_max_radii(centers):
    n = centers.shape[0]
    radii = np.array(
        [min(x, y, 1-x, 1-y) for x, y in centers], dtype=float
    )
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if radii[i] + radii[j] > d:
                scale = d / (radii[i] + radii[j])
                radii[i] *= scale
                radii[j] *= scale
    return radii


def run_packing():
    return construct_packing()


def visualize(centers, radii):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True)
    for i, (center, radius) in enumerate(zip(centers, radii)):
        ax.add_patch(Circle(center, radius, alpha=0.5))
        ax.text(center[0], center[1], str(i), ha="center", va="center")
    plt.title(f"Circle Packing (n={len(centers)}, sum={sum(radii):.6f})")
    plt.show()


if __name__ == "__main__":
    c, r, s = run_packing()
    print(f"Sum of radii: {s}")