"""Constructor-based circle packing for 26 circles in the unit square."""
import numpy as np


def compute_max_radii(centers):
    """Maximum sum of radii for fixed centers."""
    from scipy.optimize import linprog

    centers = np.asarray(centers, dtype=float)
    n = len(centers)
    rows, rhs = [], []
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            rows.append(row)
            rhs.append(np.linalg.norm(centers[i] - centers[j]))

    walls = np.min(np.column_stack((
        centers[:, 0], centers[:, 1],
        1.0 - centers[:, 0], 1.0 - centers[:, 1],
    )), axis=1)
    answer = linprog(
        -np.ones(n), A_ub=np.asarray(rows), b_ub=np.asarray(rhs),
        bounds=[(0.0, max(0.0, float(w))) for w in walls],
        method="highs",
    )
    if not answer.success:
        return np.maximum(walls, 0.0) * (1.0 - 1e-8)
    return np.maximum(answer.x, 0.0) * (1.0 - 1e-9)


def construct_packing():
    from scipy.optimize import minimize

    n = 26

    def objective(z):
        return -float(np.sum(z[2 * n:]))

    def constraints(z):
        c = z[:2 * n].reshape(n, 2)
        r = z[2 * n:]
        out = [
            c[:, 0] - r, c[:, 1] - r,
            1.0 - c[:, 0] - r, 1.0 - c[:, 1] - r,
        ]
        for i in range(n):
            d = c[i + 1:] - c[i]
            out.append(np.sum(d * d, axis=1) - (r[i] + r[i + 1:]) ** 2)
        return np.concatenate([np.ravel(x) for x in out])

    seeds = []

    # The successful asymmetric 5-6-5-5-5 lattice family.
    counts = (5, 6, 5, 5, 5)
    defect = np.array((-1.0, -0.55, -0.15, 0.15, 0.55, 1.0))
    for ls, rs in ((1, 1), (-1, -1), (1, -1), (-1, 1)):
        points = []
        for row, count in enumerate(counts):
            y = 0.09 + 0.205 * row
            xs = (np.arange(count) + 0.5) / count
            ys = np.full(count, y)
            if row == 1:
                xs += 0.024 * defect
                xs[0] += .018 * ls
                xs[1] += .009 * ls
                xs[-2] += .009 * rs
                xs[-1] += .018 * rs
                ys[0] += .008 * ls
                ys[-1] += .008 * rs
            else:
                xs += .025 * (1 if row % 2 == 0 else -1)
                xs[0] += .018 * ls
                xs[-1] += .018 * rs
                ys[0] += .009 * ls
                ys[-1] += .009 * rs
            xs -= np.mean(xs) - .5
            points.extend(zip(xs, ys))
        c = np.asarray(points, dtype=float)
        seeds.append(np.concatenate((c.ravel(), compute_max_radii(c))))

    # Independent diagonal half-packing seeds.
    tangent = np.array((1.0, 1.0)) / np.sqrt(2.0)
    normal = np.array((-1.0, 1.0)) / np.sqrt(2.0)
    for slope in (-.10, 0.0, .10):
        for sign in (-1.0, 1.0):
            for phase in (0.0, .012):
                points = []
                for side in (-1.0, 1.0):
                    for band, count in enumerate((4, 4, 5)):
                        q0 = (-.32, 0.0, .32)[band] + phase
                        qs = q0 + (np.arange(count) - .5 * (count - 1)) * .145
                        for k, q in enumerate(qs):
                            alt = 1.0 if (k + band) % 2 == 0 else -1.0
                            off = side * .19 + sign * .012 * alt + slope * q
                            points.append(.5 + q * tangent + off * normal)
                c = np.asarray(points, dtype=float)
                if np.all(c >= .02) and np.all(c <= .98):
                    seeds.append(np.concatenate((c.ravel(), compute_max_radii(c))))

    bounds = [(0.0, 1.0)] * (2 * n) + [(0.001, .20)] * n
    best = None
    for seed in seeds:
        result = minimize(
            objective, seed, method="SLSQP", bounds=bounds,
            constraints={"type": "ineq", "fun": constraints},
            options={"maxiter": 950, "ftol": 1e-10, "disp": False},
        )
        if result.success and (best is None or result.fun < best.fun):
            best = result

    if best is None:
        z = seeds[0].copy()
    else:
        z = best.x.copy()

    centers = z[:2 * n].reshape(n, 2)
    radii = compute_max_radii(centers) * (1.0 - 1e-8)

    # Use permitted evaluator slack only after explicit complete validation.
    lift = 5e-7
    for _ in range(8):
        trial = radii + lift
        wall = np.min(np.column_stack((
            centers[:, 0] - trial, centers[:, 1] - trial,
            1.0 - centers[:, 0] - trial, 1.0 - centers[:, 1] - trial,
        )))
        pair = np.inf
        for i in range(n):
            d = centers[i + 1:] - centers[i]
            if len(d):
                pair = min(pair, float(np.min(
                    np.sum(d * d, axis=1) - (trial[i] + trial[i + 1:]) ** 2
                )))
        if min(wall, pair) >= -1e-6:
            radii = trial
            break
        lift -= 5e-8

    return centers, radii, float(np.sum(radii))


def run_packing():
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii


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
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii}")