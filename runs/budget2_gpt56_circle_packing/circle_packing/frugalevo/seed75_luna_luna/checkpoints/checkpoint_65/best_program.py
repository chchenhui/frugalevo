"""Deterministic constrained constructor for 26 non-overlapping circles."""
import numpy as np


def construct_packing():
    n = 26

    # Four distinct five-layer populations.  The last two are asymmetric
    # enough to give SLSQP different contact graphs without using random data.
    patterns = [
        ((5, 5, 6, 5, 5), (0.095, 0.297, 0.500, 0.703, 0.905)),
        ((4, 6, 6, 6, 4), (0.085, 0.285, 0.500, 0.715, 0.915)),
        ((6, 4, 6, 4, 6), (0.090, 0.292, 0.500, 0.708, 0.910)),
        ((5, 6, 4, 6, 5), (0.090, 0.292, 0.500, 0.708, 0.910)),
    ]

    variations = (
        (1.00, 1.00, 0.000),
        (0.97, 1.00, -0.008),
        (1.03, 1.00, 0.008),
        (1.00, 0.975, 0.000),
    )

    seeds = []
    for row_sizes, ys in patterns:
        points = []
        rows = []
        for iy, (count, y) in enumerate(zip(row_sizes, ys)):
            if count == 6:
                pitch = 0.158
            elif count == 5:
                pitch = 0.175
            else:
                pitch = 0.205
            start = 0.5 - 0.5 * pitch * (count - 1)
            if iy & 1:
                start += 0.5 * pitch
            lo = 0.060
            hi = 0.940 - pitch * (count - 1)
            start = min(max(start, lo), hi)
            for k in range(count):
                points.append((start + k * pitch, y))
                rows.append(iy)

        seed = np.asarray(points, dtype=float)
        rows = np.asarray(rows, dtype=float)

        for ys_scale, xs_scale, shear in variations:
            q = seed.copy()
            q[:, 1] = 0.5 + ys_scale * (q[:, 1] - 0.5)
            q[:, 0] = 0.5 + xs_scale * (q[:, 0] - 0.5)
            q[:, 0] += shear * (rows - 2.0)
            seeds.append(q)

    # Add a deterministic clearance-insertion seed.  It is deliberately
    # generated from geometry rather than from another row population.
    for mode in range(2):
        pts = []
        rr = []
        candidates = [(0.08, 0.08), (0.92, 0.08),
                      (0.08, 0.92), (0.92, 0.92)]
        for p in candidates:
            pts.append(p)
            rr.append(0.045)

        while len(pts) < n:
            best_p = None
            best_r = -1.0
            for gx in range(1, 18):
                for gy in range(1, 18):
                    p = np.array((gx / 18.0, gy / 18.0))
                    wall = min(p[0], p[1], 1-p[0], 1-p[1])
                    if pts:
                        d = np.linalg.norm(np.asarray(pts) - p, axis=1)
                        avail = np.min(d - np.asarray(rr))
                    else:
                        avail = wall
                    rad = min(wall, 0.5 * avail)
                    if rad <= 0.010:
                        continue
                    tie = (p[0] + p[1]) if mode == 0 else (p[0] - p[1])
                    score = rad + 1e-5 * tie
                    if score > best_r:
                        best_r = score
                        best_p = p
            if best_p is None:
                break
            pts.append(tuple(best_p))
            rr.append(best_r)
        if len(pts) == n:
            seeds.append(np.asarray(pts, dtype=float))

    def cons(z):
        c = z[:2*n].reshape(n, 2)
        r = z[2*n:]
        out = [
            c[:, 0] - r,
            c[:, 1] - r,
            1.0 - c[:, 0] - r,
            1.0 - c[:, 1] - r
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
            # Keep insertion seeds conservative; row seeds benefit from the
            # incumbent initialization range.
            if trial >= 16:
                r0 = 0.042
            else:
                r0 = 0.056 + 0.002 * (trial % 4)
            z0 = np.r_[seed.ravel(), np.full(n, r0)]
            # Make the initial radii feasible for every seed.
            d = seed[:, None, :] - seed[None, :, :]
            dist = np.sqrt(np.sum(d*d, axis=2) + np.eye(n))
            upper = np.full(n, r0)
            upper = np.minimum(upper, seed[:, 0])
            upper = np.minimum(upper, 1.0-seed[:, 0])
            upper = np.minimum(upper, seed[:, 1])
            upper = np.minimum(upper, 1.0-seed[:, 1])
            for i in range(n):
                for j in range(i):
                    v = 0.46 * dist[i, j]
                    upper[i] = min(upper[i], v)
                    upper[j] = min(upper[j], v)
            z0[2*n:] = np.maximum(0.018, upper)

            res = minimize(
                lambda z: -float(np.sum(z[2*n:])),
                z0, method="SLSQP",
                bounds=[(0.0, 1.0)]*(2*n) + [(0.002, 0.25)]*n,
                constraints={"type": "ineq", "fun": cons},
                options={"maxiter": 550, "ftol": 2e-10, "disp": False},
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

    centers = np.clip(np.asarray(centers, dtype=float), 0.0, 1.0)
    radii = np.maximum(np.asarray(radii, dtype=float), 0.001)

    margin = 1.0
    margin = min(margin, float(np.min(centers / radii[:, None])))
    margin = min(margin, float(np.min((1.0 - centers) / radii[:, None])))
    for i in range(n):
        for j in range(i):
            d = np.linalg.norm(centers[i] - centers[j])
            margin = min(margin, float(d / (radii[i] + radii[j])))
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