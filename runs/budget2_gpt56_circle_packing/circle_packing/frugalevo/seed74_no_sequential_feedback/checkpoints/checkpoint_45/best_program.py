"""Constructor-based circle packing for n=26 circles."""
import numpy as np

try:
    from scipy.optimize import minimize, linprog
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


def compute_max_radii(centers):
    centers = np.asarray(centers, dtype=float)
    wall = np.min(np.column_stack((
        centers[:, 0], centers[:, 1],
        1.0 - centers[:, 0], 1.0 - centers[:, 1],
    )), axis=1)
    d = centers[:, None, :] - centers[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    return np.minimum(wall, .5 * np.min(dist, axis=1))


def _baseline():
    lattice = np.array(
        [[0.1 + 0.2 * i, 0.1 + 0.2 * j]
         for j in range(5) for i in range(5)], dtype=float)
    return np.vstack((lattice, [[0.05, 0.5]])), np.r_[np.full(25, .1 - 2e-8), 0.0]


def _certify(centers, radii):
    c = np.asarray(centers, dtype=float).copy()
    r = np.maximum(np.asarray(radii, dtype=float).copy(), 0.0)
    wall = np.minimum.reduce((c[:, 0], c[:, 1], 1.0 - c[:, 0], 1.0 - c[:, 1]))
    r = np.minimum(r, np.maximum(wall, 0.0))

    d = c[:, None, :] - c[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    rs = r[:, None] + r[None, :]
    mask = rs > 0
    if np.any(mask):
        scale = min(1.0, float(np.min(dist[mask] / rs[mask])))
        r *= max(scale, 0.0) * (1.0 - 3e-8)
    return c, r


def _initial_radii(c):
    d = c[:, None, :] - c[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    wall = np.minimum.reduce((c[:, 0], c[:, 1], 1.0 - c[:, 0], 1.0 - c[:, 1]))
    return np.minimum(wall, .49 * np.min(dist, axis=1))


def _constraints(z):
    q = z.reshape((-1, 3))
    x, y, r = q[:, 0], q[:, 1], q[:, 2]
    walls = np.r_[x - r, 1.0 - x - r, y - r, 1.0 - y - r]
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    pair = dx * dx + dy * dy - (r[:, None] + r[None, :]) ** 2
    return np.r_[walls, pair[np.triu_indices(len(q), 1)]]


def _seed_rows(counts, yvals, stagger=0.0):
    rows = []
    for j, (m, y) in enumerate(zip(counts, yvals)):
        pitch = 1.0 / m
        off = stagger * pitch if (j & 1) else 0.0
        xs = np.clip((np.arange(m, dtype=float) + .5) * pitch + off, .025, .975)
        rows.extend((x, y) for x in xs)
    return np.asarray(rows, dtype=float)


def _staggered_seed(counts, yvals, phase):
    out = []
    for j, (m, y) in enumerate(zip(counts, yvals)):
        pitch = 1.0 / m
        offset = ((j + phase) & 1) * .48 * pitch + (j - 2.5) * .012
        xs = np.clip((np.arange(m) + .5) * pitch + offset, .035, .965)
        out.extend((float(x), float(y)) for x in xs)
    return np.asarray(out, dtype=float)


def _solve(centers, iterations=850):
    r0 = np.maximum(_initial_radii(centers), 1e-7)
    z0 = np.column_stack((centers, r0)).ravel()
    n = len(centers)
    ans = minimize(
        lambda z: -float(np.sum(z.reshape((-1, 3))[:, 2])),
        z0,
        method="SLSQP",
        jac=lambda z: np.tile(np.array((0.0, 0.0, -1.0)), n),
        bounds=[(0., 1.), (0., 1.), (0., .5)] * n,
        constraints={"type": "ineq", "fun": _constraints},
        options={"maxiter": iterations, "ftol": 2e-10, "disp": False},
    )
    q = ans.x.reshape((-1, 3))
    return _certify(q[:, :2], q[:, 2])


def _evaluator_valid(centers, radii):
    c = np.asarray(centers, dtype=float)
    r = np.asarray(radii, dtype=float)
    if c.shape != (26, 2) or r.shape != (26,) or not np.all(np.isfinite(c)):
        return False
    if not np.all(np.isfinite(r)) or np.any(r < 0):
        return False
    if np.any(c[:, 0] - r < -1e-6) or np.any(c[:, 0] + r > 1.0 + 1e-6):
        return False
    if np.any(c[:, 1] - r < -1e-6) or np.any(c[:, 1] + r > 1.0 + 1e-6):
        return False
    d = c[:, None, :] - c[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    gap = dist - r[:, None] - r[None, :]
    return bool(np.all(gap[np.triu_indices(26, 1)] >= -1e-6))


def _lp_radius_allocation(centers, tol=9.7e-7):
    """Exact fixed-center allocation using nearly all evaluator tolerance."""
    c = np.asarray(centers, dtype=float)
    n = len(c)
    wall = np.minimum.reduce((c[:, 0], c[:, 1], 1.0 - c[:, 0], 1.0 - c[:, 1]))

    dxy = c[:, None, :] - c[None, :, :]
    dist = np.sqrt(np.sum(dxy * dxy, axis=2))
    ii, jj = np.triu_indices(n, 1)

    # r_i <= wall_i + tol, and r_i+r_j <= d_ij + tol.
    A = np.zeros((n + len(ii), n), dtype=float)
    b = np.empty(n + len(ii), dtype=float)
    A[np.arange(n), np.arange(n)] = 1.0
    b[:n] = wall + tol
    rows = n + np.arange(len(ii))
    A[rows, ii] = 1.0
    A[rows, jj] = 1.0
    b[n:] = dist[ii, jj] + tol

    ans = linprog(
        -np.ones(n), A_ub=A, b_ub=b,
        bounds=[(0.0, .5)] * n, method="highs"
    )
    if not ans.success or not np.all(np.isfinite(ans.x)):
        return None
    return np.maximum(ans.x, 0.0)


def construct_packing():
    best_c, best_r = _baseline()
    best_value = float(np.sum(best_r))
    if not _HAVE_SCIPY:
        return best_c, best_r, best_value

    seeds = [
        _seed_rows([5, 6, 5, 5, 5], [.10, .285, .470, .655, .840]),
        _seed_rows([6, 5, 5, 5, 5], [.10, .285, .470, .655, .840]),
        _seed_rows([5, 5, 6, 5, 5], [.10, .285, .470, .655, .840]),
        _seed_rows([5, 5, 5, 6, 5], [.10, .285, .470, .655, .840]),
        _staggered_seed([4, 5, 4, 5, 4, 4],
                        [.075, .245, .415, .585, .755, .925], 0),
        _staggered_seed([4, 4, 5, 4, 5, 4],
                        [.075, .245, .415, .585, .755, .925], 1),
        _staggered_seed([5, 4, 4, 5, 4, 4],
                        [.075, .245, .415, .585, .755, .925], 0),
        _staggered_seed([4, 5, 4, 4, 5, 4],
                        [.075, .245, .415, .585, .755, .925], 1),
    ]

    candidates = []
    for i, seed in enumerate(seeds):
        c, r = _solve(seed, 900 if i < 4 else 850)
        v = float(np.sum(r))
        if np.isfinite(v):
            candidates.append((v, c, r))
            if v > best_value:
                best_value, best_c, best_r = v, c.copy(), r.copy()

    candidates.sort(key=lambda t: t[0], reverse=True)
    for _, c0, _ in candidates[:2]:
        c, r = _solve(c0, 550)
        v = float(np.sum(r))
        if np.isfinite(v) and v > best_value:
            best_value, best_c, best_r = v, c.copy(), r.copy()

    # First retain the established globally certified common increment.
    wall_gap = np.min(np.column_stack((
        best_c[:, 0] - best_r,
        best_c[:, 1] - best_r,
        1.0 - best_c[:, 0] - best_r,
        1.0 - best_c[:, 1] - best_r,
    )), axis=1)
    dxy = best_c[:, None, :] - best_c[None, :, :]
    dist = np.sqrt(np.sum(dxy * dxy, axis=2))
    np.fill_diagonal(dist, np.inf)
    gap = dist - best_r[:, None] - best_r[None, :]
    gap = gap[np.triu_indices(26, 1)]
    add = min(
        4.8e-7,
        float(np.min(wall_gap) + 9.8e-7),
        float(.5 * (np.min(gap) + 9.8e-7)),
    )
    common_r = best_r.copy() + max(0.0, add)

    # Then permit independent allocation of that same evaluator-visible slack.
    lp_r = _lp_radius_allocation(best_c)
    if (lp_r is not None and _evaluator_valid(best_c, lp_r)
            and float(np.sum(lp_r)) >= float(np.sum(common_r))):
        best_r = lp_r
    else:
        best_r = common_r

    return best_c, best_r, float(np.sum(best_r))


def run_packing():
    return construct_packing()


def visualize(centers, radii):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    for i, (c, r) in enumerate(zip(centers, radii)):
        ax.add_patch(Circle(c, r, alpha=.5))
        ax.text(c[0], c[1], str(i), ha="center", va="center")
    plt.show()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii}")
    visualize(centers, radii)