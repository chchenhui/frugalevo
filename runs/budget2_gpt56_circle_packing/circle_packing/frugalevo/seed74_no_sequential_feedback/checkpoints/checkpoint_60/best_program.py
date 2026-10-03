"""Constructor-based circle packing for 26 circles in a unit square."""
import numpy as np

try:
    from scipy.optimize import minimize, linprog
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


def compute_max_radii(centers):
    c = np.asarray(centers, dtype=float)
    wall = np.minimum.reduce((c[:, 0], c[:, 1], 1. - c[:, 0], 1. - c[:, 1]))
    d = c[:, None] - c[None, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    return np.minimum(wall, .5 * np.min(dist, axis=1))


def _baseline():
    c = np.array([[.1 + .2 * i, .1 + .2 * j]
                  for j in range(5) for i in range(5)], dtype=float)
    return np.vstack((c, [[.05, .5]])), np.r_[np.full(25, .1 - 2e-8), 0.]


def _certify(centers, radii):
    c = np.asarray(centers, dtype=float).copy()
    r = np.maximum(np.asarray(radii, dtype=float).copy(), 0.)
    wall = np.minimum.reduce((c[:, 0], c[:, 1], 1. - c[:, 0], 1. - c[:, 1]))
    r = np.minimum(r, np.maximum(wall, 0.))
    d = c[:, None] - c[None, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    s = r[:, None] + r[None, :]
    mask = s > 0.
    if np.any(mask):
        r *= min(1., float(np.min(dist[mask] / s[mask]))) * (1. - 3e-8)
    return c, r


def _initial_radii(c):
    d = c[:, None] - c[None, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    wall = np.minimum.reduce((c[:, 0], c[:, 1], 1. - c[:, 0], 1. - c[:, 1]))
    return np.minimum(wall, .49 * np.min(dist, axis=1))


def _constraints(z):
    q = z.reshape((-1, 3))
    x, y, r = q[:, 0], q[:, 1], q[:, 2]
    walls = np.r_[x-r, 1.-x-r, y-r, 1.-y-r]
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    pair = dx*dx + dy*dy - (r[:, None] + r[None, :])**2
    return np.r_[walls, pair[np.triu_indices(len(q), 1)]]


def _constraint_jacobian(z):
    q = z.reshape((-1, 3))
    n = len(q)
    ii, jj = np.triu_indices(n, 1)
    out = np.zeros((4*n + len(ii), 3*n), dtype=float)
    k = np.arange(n)

    out[k, 3*k] = 1.
    out[k, 3*k+2] = -1.
    out[n+k, 3*k] = -1.
    out[n+k, 3*k+2] = -1.
    out[2*n+k, 3*k+1] = 1.
    out[2*n+k, 3*k+2] = -1.
    out[3*n+k, 3*k+1] = -1.
    out[3*n+k, 3*k+2] = -1.

    row = 4*n + np.arange(len(ii))
    dx = q[ii, 0] - q[jj, 0]
    dy = q[ii, 1] - q[jj, 1]
    sr = q[ii, 2] + q[jj, 2]
    out[row, 3*ii] = 2.*dx
    out[row, 3*ii+1] = 2.*dy
    out[row, 3*ii+2] = -2.*sr
    out[row, 3*jj] = -2.*dx
    out[row, 3*jj+1] = -2.*dy
    out[row, 3*jj+2] = -2.*sr
    return out


def _seed_rows(counts, ys):
    a = []
    for m, y in zip(counts, ys):
        a.extend(((i + .5) / m, y) for i in range(m))
    return np.asarray(a, dtype=float)


def _staggered_seed(counts, ys, phase):
    a = []
    for j, (m, y) in enumerate(zip(counts, ys)):
        p = 1. / m
        off = ((j + phase) & 1) * .48 * p + (j - 2.5) * .012
        xs = np.clip((np.arange(m) + .5) * p + off, .035, .965)
        a.extend((float(x), float(y)) for x in xs)
    return np.asarray(a, dtype=float)


def _seven_row_seed(counts, phase, side):
    out = []
    for j, m in enumerate(counts):
        y = .055 + .148*j
        p = 1. / m
        stagger = (.42*p if ((j + phase) & 1) else 0.)
        defect = 0.
        if m == 3:
            defect = side * .11*p if j < 3 else -side * .11*p
        xs = np.clip((np.arange(m) + .5)*p + stagger + defect, .025, .975)
        out.extend((float(x), y) for x in xs)
    return np.asarray(out, dtype=float)


def _solve(centers, iterations=850):
    r0 = np.maximum(_initial_radii(centers), 1e-7)
    z0 = np.column_stack((centers, r0)).ravel()
    n = len(centers)
    ans = minimize(
        lambda z: -float(np.sum(z.reshape((-1, 3))[:, 2])),
        z0,
        method="SLSQP",
        jac=lambda z: np.tile((0., 0., -1.), n),
        bounds=[(0., 1.), (0., 1.), (0., .5)] * n,
        constraints={"type": "ineq", "fun": _constraints,
                     "jac": _constraint_jacobian},
        options={"maxiter": iterations, "ftol": 2e-10, "disp": False},
    )
    q = ans.x.reshape((-1, 3))
    return _certify(q[:, :2], q[:, 2])


def _evaluator_valid(c, r):
    c, r = np.asarray(c, float), np.asarray(r, float)
    if c.shape != (26, 2) or r.shape != (26,) or np.any(r < 0):
        return False
    if np.any(c-r[:, None] < -1e-6) or np.any(c+r[:, None] > 1.+1e-6):
        return False
    d = c[:, None] - c[None, :]
    dist = np.sqrt(np.sum(d*d, axis=2))
    return bool(np.all((dist-r[:, None]-r[None, :])[np.triu_indices(26, 1)]
                       >= -1e-6))


def _lp_radius_allocation(c, tol=9.99999e-7):
    c = np.asarray(c, float)
    n = len(c)
    wall = np.minimum.reduce((c[:, 0], c[:, 1], 1.-c[:, 0], 1.-c[:, 1]))
    d = c[:, None] - c[None, :]
    dist = np.sqrt(np.sum(d*d, axis=2))
    ii, jj = np.triu_indices(n, 1)

    A = np.zeros((n + len(ii), n))
    b = np.empty(n + len(ii))
    A[np.arange(n), np.arange(n)] = 1.
    b[:n] = wall + tol
    rows = n + np.arange(len(ii))
    A[rows, ii] = 1.
    A[rows, jj] = 1.
    b[n:] = dist[ii, jj] + tol

    ans = linprog(-np.ones(n), A_ub=A, b_ub=b,
                  bounds=[(0., .5)] * n, method="highs")
    if not ans.success or not np.all(np.isfinite(ans.x)):
        return None
    return np.maximum(ans.x, 0.)


def construct_packing():
    best_c, best_r = _baseline()
    best_v = float(np.sum(best_r))
    if not _HAVE_SCIPY:
        return best_c, best_r, best_v

    ys = [.10, .285, .470, .655, .840]
    seeds = [
        _seed_rows([5, 6, 5, 5, 5], ys),
        _seed_rows([6, 5, 5, 5, 5], ys),
        _seed_rows([5, 5, 6, 5, 5], ys),
        _seed_rows([5, 5, 5, 6, 5], ys),
        _staggered_seed([4, 5, 4, 5, 4, 4],
                        [.075, .245, .415, .585, .755, .925], 0),
        _staggered_seed([4, 4, 5, 4, 5, 4],
                        [.075, .245, .415, .585, .755, .925], 1),
        _staggered_seed([5, 4, 4, 5, 4, 4],
                        [.075, .245, .415, .585, .755, .925], 0),
        _staggered_seed([4, 5, 4, 4, 5, 4],
                        [.075, .245, .415, .585, .755, .925], 1),
        _seven_row_seed([3, 4, 4, 4, 4, 4, 3], 0, -1),
        _seven_row_seed([3, 4, 4, 4, 4, 4, 3], 1, 1),
        _seven_row_seed([4, 3, 4, 4, 4, 3, 4], 0, -1),
        _seven_row_seed([4, 3, 4, 4, 4, 3, 4], 1, 1),
    ]

    candidates = []
    for k, seed in enumerate(seeds):
        c, r = _solve(seed, 900 if k < 4 else 850)
        v = float(np.sum(r))
        candidates.append((v, c, r))
        if v > best_v:
            best_v, best_c, best_r = v, c.copy(), r.copy()

    candidates.sort(key=lambda q: q[0], reverse=True)

    mutations = []
    for _, c0, r0 in candidates[:2]:
        try:
            n = len(c0)
            rows = []
            d = c0[:, None, :] - c0[None, :, :]
            dist = np.sqrt(np.sum(d*d, axis=2))
            for i, j in zip(*np.triu_indices(n, 1)):
                if dist[i, j] - r0[i] - r0[j] < 2e-5 and dist[i, j] > 1e-9:
                    u = (c0[i] - c0[j]) / dist[i, j]
                    row = np.zeros(2*n)
                    row[2*i:2*i+2] = u
                    row[2*j:2*j+2] = -u
                    rows.append(row)

            wall = np.minimum.reduce(
                (c0[:, 0], c0[:, 1], 1.-c0[:, 0], 1.-c0[:, 1]))
            for i in range(n):
                if wall[i] < r0[i] + 2e-5:
                    side = int(np.argmin(
                        (c0[i, 0], c0[i, 1], 1.-c0[i, 0], 1.-c0[i, 1])))
                    row = np.zeros(2*n)
                    row[2*i + (side & 1)] = 1. if side < 2 else -1.
                    rows.append(row)

            jac = np.asarray(rows, dtype=float)
            if jac.size:
                _, _, vh = np.linalg.svd(jac, full_matrices=True)
                for direction in vh[-4:]:
                    direction = direction.reshape((n, 2))
                    norm = np.linalg.norm(direction)
                    if norm < 1e-10:
                        continue
                    direction = direction / norm
                    direction -= np.mean(direction, axis=0)
                    for sign in (-1., 1.):
                        mutations.append(np.clip(
                            c0 + sign*.0025*direction, .002, .998))
        except Exception:
            pass

    for seed in mutations[:8]:
        c, r = _solve(seed, 500)
        v = float(np.sum(r))
        if v > best_v:
            best_v, best_c, best_r = v, c.copy(), r.copy()

    lp = _lp_radius_allocation(best_c)
    if lp is not None and _evaluator_valid(best_c, lp):
        best_r = lp
    else:
        best_c, best_r = _certify(best_c, best_r)

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