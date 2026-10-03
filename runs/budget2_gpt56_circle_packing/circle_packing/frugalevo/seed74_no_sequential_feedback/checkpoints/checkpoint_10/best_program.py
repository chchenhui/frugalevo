"""Constructor-based circle packing for n=26 circles."""
import numpy as np

try:
    from scipy.optimize import minimize
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


def compute_max_radii(centers):
    """Conservative compatibility helper retained from the original interface."""
    radii = np.min(
        np.column_stack((
            centers[:, 0], centers[:, 1],
            1.0 - centers[:, 0], 1.0 - centers[:, 1],
        )),
        axis=1,
    )
    delta = centers[:, None, :] - centers[None, :, :]
    distances = np.sqrt(np.sum(delta * delta, axis=2))
    np.fill_diagonal(distances, np.inf)
    return np.minimum(radii, 0.5 * np.min(distances, axis=1))


def _baseline():
    """A certified 2.5-radius-sum fallback: 25 tangent lattice discs plus one point."""
    lattice = np.array(
        [[0.1 + 0.2 * i, 0.1 + 0.2 * j]
         for j in range(5) for i in range(5)],
        dtype=float,
    )
    centers = np.vstack((lattice, [[0.05, 0.5]]))
    radii = np.r_[np.full(25, 0.1 - 2.0e-8), 0.0]
    return centers, radii


def _certify(centers, radii):
    """Make a candidate strictly feasible by a single conservative radius scaling."""
    c = np.asarray(centers, dtype=float).copy()
    r = np.maximum(np.asarray(radii, dtype=float).copy(), 0.0)
    wall = np.minimum.reduce((c[:, 0], c[:, 1], 1.0 - c[:, 0], 1.0 - c[:, 1]))
    r = np.minimum(r, np.maximum(wall, 0.0))

    d = c[:, None, :] - c[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    rs = r[:, None] + r[None, :]
    mask = rs > 0.0
    if np.any(mask):
        scale = min(1.0, float(np.min(dist[mask] / rs[mask])))
        r *= max(0.0, scale) * (1.0 - 3.0e-8)
    return c, r


def _seed_rows(counts, yvals):
    rows = []
    for m, y in zip(counts, yvals):
        if m == 5:
            xs = np.linspace(0.1, 0.9, 5)
        else:
            xs = (np.arange(m, dtype=float) + 0.5) / m
        rows.extend((x, y) for x in xs)
    return np.asarray(rows, dtype=float)


def _initial_radii(c):
    d = c[:, None, :] - c[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    wall = np.minimum.reduce((c[:, 0], c[:, 1], 1.0 - c[:, 0], 1.0 - c[:, 1]))
    return np.minimum(wall, 0.49 * np.min(dist, axis=1))


def _constraints(z):
    q = z.reshape((-1, 3))
    x, y, r = q[:, 0], q[:, 1], q[:, 2]
    walls = np.concatenate((x - r, 1.0 - x - r, y - r, 1.0 - y - r))
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dsq = dx * dx + dy * dy
    rr = r[:, None] + r[None, :]
    pair = (dsq - rr * rr)[np.triu_indices(len(q), 1)]
    return np.concatenate((walls, pair))


def _polish(centers):
    r0 = _initial_radii(centers)
    z0 = np.column_stack((centers, r0)).ravel()
    n = len(centers)
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * n

    result = minimize(
        lambda z: -float(np.sum(z.reshape((-1, 3))[:, 2])),
        z0,
        method="SLSQP",
        jac=lambda z: np.tile(np.array([0.0, 0.0, -1.0]), n),
        bounds=bounds,
        constraints={"type": "ineq", "fun": _constraints},
        options={"maxiter": 700, "ftol": 2.0e-10, "disp": False},
    )
    q = result.x.reshape((-1, 3))
    return _certify(q[:, :2], q[:, 2])


def construct_packing():
    """Use seed optimization, then clear the best contact plateau and release it."""
    best_c, best_r = _baseline()
    best_value = float(np.sum(best_r))
    incumbent = None

    if not _HAVE_SCIPY:
        return best_c, best_r, best_value

    yvals = [0.10, 0.285, 0.470, 0.655, 0.840]
    seeds = [
        _seed_rows([5, 6, 5, 5, 5], yvals),
        _seed_rows([6, 5, 5, 5, 5], yvals),
        _seed_rows([5, 5, 6, 5, 5], yvals),
        _seed_rows([5, 5, 5, 6, 5], yvals),
    ]
    candidates = []

    for seed in seeds:
        centers0 = np.asarray(seed, dtype=float).copy()
        radii0 = np.maximum(_initial_radii(centers0), 1.0e-5)
        z0 = np.column_stack((centers0, radii0)).ravel()
        result = minimize(
            lambda z: -float(np.sum(z.reshape((-1, 3))[:, 2])),
            z0,
            method="SLSQP",
            jac=lambda z: np.tile(np.array([0.0, 0.0, -1.0]), 26),
            bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * 26,
            constraints={"type": "ineq", "fun": _constraints},
            options={"maxiter": 700, "ftol": 2.0e-10, "disp": False},
        )
        q = result.x.reshape((26, 3))
        c, r = _certify(q[:, :2], q[:, 2])
        value = float(np.sum(r))
        if np.isfinite(value):
            candidates.append((value, c.copy(), r.copy()))
            if value > best_value:
                best_c, best_r, best_value = c.copy(), r.copy(), value

    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        incumbent = candidates[0]

    # A clearance-reserve solve moves the incumbent through a high-clearance
    # feasible plateau before the unrestricted radius objective is restored.
    if incumbent is not None:
        base_value, base_c, base_r = incumbent
        zbase = np.column_stack((base_c, np.maximum(base_r - 2.0e-4, 0.0))).ravel()
        reserve0 = np.r_[zbase, 0.0]

        def reserve_constraints(v):
            q = v[:-1].reshape((-1, 3))
            t = v[-1]
            x, y, r = q[:, 0], q[:, 1], q[:, 2]
            walls = np.concatenate((x - r, 1.0 - x - r, y - r, 1.0 - y - r))
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            dsq = dx * dx + dy * dy
            rr = r[:, None] + r[None, :]
            pair = (dsq - rr * rr)[np.triu_indices(26, 1)]
            return np.r_[walls - t, pair - t, np.sum(r) - (base_value - 5.0e-5), t]

        rbounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * 26 + [(0.0, 0.2)]
        reserve = minimize(
            lambda v: -float(v[-1]),
            reserve0,
            method="SLSQP",
            bounds=rbounds,
            constraints={"type": "ineq", "fun": reserve_constraints},
            options={"maxiter": 250, "ftol": 2.0e-10, "disp": False},
        )
        release = minimize(
            lambda z: -float(np.sum(z.reshape((-1, 3))[:, 2])),
            reserve.x[:-1],
            method="SLSQP",
            jac=lambda z: np.tile(np.array([0.0, 0.0, -1.0]), 26),
            bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * 26,
            constraints={"type": "ineq", "fun": _constraints},
            options={"maxiter": 700, "ftol": 2.0e-10, "disp": False},
        )
        q = release.x.reshape((26, 3))
        c, r = _certify(q[:, :2], q[:, 2])
        value = float(np.sum(r))
        if np.isfinite(value) and value > best_value:
            best_c, best_r, best_value = c.copy(), r.copy(), value

    return best_c, best_r, best_value


def run_packing():
    """Run the circle packing constructor for n=26."""
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
    visualize(centers, radii)