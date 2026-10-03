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
    """Insert largest-clearance circles, then alternate fixed-center LP growth and SLSQP relaxation."""
    best_c, best_r = _baseline()
    best_value = float(np.sum(best_r))

    if not _HAVE_SCIPY:
        return best_c, best_r, best_value

    from scipy.optimize import linprog

    # Four initially separated corner discs make the subsequent candidates
    # explore the large central and edge gaps instead of a Cartesian lattice.
    centers = np.array(
        [[0.12, 0.12], [0.88, 0.12], [0.12, 0.88], [0.88, 0.88]],
        dtype=float,
    )

    # A fixed grid gives deterministic largest-clearance insertion.  Adding
    # wall and circle clearance explicitly avoids selecting nearly coincident
    # points when the grid happens to miss a narrow gap.
    grid = np.array(
        [(i / 40.0, j / 40.0) for i in range(1, 40) for j in range(1, 40)],
        dtype=float,
    )
    for _ in range(22):
        wall = np.min(
            np.column_stack((
                grid[:, 0], grid[:, 1],
                1.0 - grid[:, 0], 1.0 - grid[:, 1],
            )),
            axis=1,
        )
        delta = grid[:, None, :] - centers[None, :, :]
        clearance = np.minimum(wall, np.min(np.sqrt(np.sum(delta * delta, axis=2)), axis=1))
        k = int(np.argmax(clearance))
        centers = np.vstack((centers, grid[k]))
        grid = np.delete(grid, k, axis=0)

    def allocate(c):
        """Solve the exact fixed-center linear program for all radii."""
        n = len(c)
        wall = np.min(
            np.column_stack((c[:, 0], c[:, 1], 1.0 - c[:, 0], 1.0 - c[:, 1])),
            axis=1,
        )
        d = np.sqrt(np.sum((c[:, None, :] - c[None, :, :]) ** 2, axis=2))
        pairs = np.triu_indices(n, 1)
        A = np.zeros((len(pairs[0]) + n, n), dtype=float)
        b = np.zeros(len(bairs := pairs[0]), dtype=float)
        for row, (i, j) in enumerate(zip(pairs[0], pairs[1])):
            A[row, i] = 1.0
            A[row, j] = 1.0
            b[row] = d[i, j]
        A[len(pairs[0]):] = np.eye(n)
        b = np.r_[b, wall]
        result = linprog(
            -np.ones(n), A_ub=A, b_ub=b,
            bounds=[(0.0, None)] * n, method="highs",
        )
        return result.x if result.success else _initial_radii(c)

    for _ in range(3):
        radii = allocate(centers)
        z0 = np.column_stack((centers, radii)).ravel()
        result = minimize(
            lambda z: -float(np.sum(z.reshape((-1, 3))[:, 2])),
            z0,
            method="SLSQP",
            jac=lambda z: np.tile(np.array([0.0, 0.0, -1.0]), 26),
            bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * 26,
            constraints={"type": "ineq", "fun": _constraints},
            options={"maxiter": 80, "ftol": 1.0e-9, "disp": False},
        )
        q = result.x.reshape((-1, 3))
        centers, radii = _certify(q[:, :2], q[:, 2])
        value = float(np.sum(radii))
        if np.isfinite(value) and value > best_value:
            best_c, best_r, best_value = centers.copy(), radii.copy(), value

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