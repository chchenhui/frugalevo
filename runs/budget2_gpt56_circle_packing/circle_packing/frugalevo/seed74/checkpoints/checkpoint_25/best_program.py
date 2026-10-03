"""Joint constrained constructor-based circle packing for n=26 circles."""
import numpy as np
from scipy.optimize import minimize, linprog


def compute_max_radii(centers):
    n = centers.shape[0]
    radii = np.min(np.column_stack((
        centers[:, 0], centers[:, 1], 1.0 - centers[:, 0], 1.0 - centers[:, 1]
    )), axis=1)
    for i in range(n):
        d = centers - centers[i]
        ds = np.sqrt(np.sum(d * d, axis=1))
        ds[i] = np.inf
        radii[i] = min(radii[i], 0.5 * np.min(ds))
    return radii


def _incumbent_centers():
    rows = []
    rows.extend([(x, 0.16) for x in ((i + 0.5) / 6.0 for i in range(6))])
    rows.extend([(x, 0.34) for x in (0.10, 0.30, 0.50, 0.70, 0.90)])
    rows.extend([(x, 0.52) for x in (0.08, 0.28, 0.48, 0.68, 0.88)])
    rows.extend([(x, 0.70) for x in (0.10, 0.30, 0.50, 0.70, 0.90)])
    rows.extend([(x, 0.88) for x in (0.08, 0.28, 0.48, 0.68, 0.88)])
    return np.asarray(rows, dtype=float)


def _constraint_values(z, pairs):
    c = z[:52].reshape(26, 2)
    r = z[52:]
    boundary = np.concatenate((
        c[:, 0] - r, c[:, 1] - r,
        1.0 - c[:, 0] - r, 1.0 - c[:, 1] - r
    ))
    i, j = pairs[:, 0], pairs[:, 1]
    d = c[i] - c[j]
    pair = np.sum(d * d, axis=1) - (r[i] + r[j]) ** 2
    return np.concatenate((boundary, pair))


def _constraint_jacobian(z, pairs):
    c = z[:52].reshape(26, 2)
    r = z[52:]
    m = len(pairs)
    jac = np.zeros((104 + m, 78), dtype=float)
    q = np.arange(26)

    jac[q, 2 * q] = 1.0
    jac[q, 52 + q] = -1.0
    jac[26 + q, 2 * q + 1] = 1.0
    jac[26 + q, 52 + q] = -1.0
    jac[52 + q, 2 * q] = -1.0
    jac[52 + q, 52 + q] = -1.0
    jac[78 + q, 2 * q + 1] = -1.0
    jac[78 + q, 52 + q] = -1.0

    for k, (i, j) in enumerate(pairs):
        row = 104 + k
        dx = c[i] - c[j]
        s = r[i] + r[j]
        jac[row, 2 * i:2 * i + 2] = 2.0 * dx
        jac[row, 2 * j:2 * j + 2] = -2.0 * dx
        jac[row, 52 + i] = -2.0 * s
        jac[row, 52 + j] = -2.0 * s
    return jac


def _strictly_valid(centers, radii):
    if centers.shape != (26, 2) or radii.shape != (26,):
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(radii)):
        return False
    if np.min(radii) < 0.0:
        return False
    if np.min(centers - radii[:, None]) < -1e-8:
        return False
    if np.min(1.0 - centers - radii[:, None]) < -1e-8:
        return False
    for i in range(26):
        for j in range(i):
            if np.linalg.norm(centers[i] - centers[j]) < radii[i] + radii[j] - 1e-8:
                return False
    return True


def _optimal_radii_for_centers(centers, margin=5e-9):
    n = 26
    pairs = [(i, j) for i in range(n) for j in range(i)]
    upper = np.minimum.reduce((
        centers[:, 0] - margin, centers[:, 1] - margin,
        1.0 - centers[:, 0] - margin, 1.0 - centers[:, 1] - margin
    ))
    if np.min(upper) < 0.0:
        return None

    a_ub = []
    b_ub = []
    for i in range(n):
        row = np.zeros(n)
        row[i] = 1.0
        a_ub.append(row)
        b_ub.append(upper[i])
    for i, j in pairs:
        d = float(np.linalg.norm(centers[i] - centers[j])) - margin
        if d < 0.0:
            return None
        row = np.zeros(n)
        row[i] = row[j] = 1.0
        a_ub.append(row)
        b_ub.append(d)

    ans = linprog(
        -np.ones(n), A_ub=np.asarray(a_ub), b_ub=np.asarray(b_ub),
        bounds=[(0.0, 0.5)] * n, method="highs"
    )
    return np.maximum(ans.x, 0.0) if ans.success else None


def _sheared_seeds(base):
    """Small deterministic defect-window modes, including the original seed."""
    seeds = [base.copy()]
    row = np.repeat(np.arange(5), (6, 5, 5, 5, 5))
    col = np.concatenate((
        np.arange(6), np.arange(5), np.arange(5), np.arange(5), np.arange(5)
    ))
    for sign in (-1.0, 1.0):
        s = base.copy()
        middle = (row >= 1) & (row <= 3)
        s[middle, 0] += sign * 0.010 * np.sin(0.9 * col[middle] + 0.7 * row[middle])
        s[middle, 1] += sign * 0.007 * np.cos(1.1 * col[middle] - 0.5 * row[middle])
        seeds.append(s)
    s = base.copy()
    middle = (row == 2)
    s[middle, 0] += np.array((-0.012, 0.008, 0.016, -0.008, 0.012))
    s[middle, 1] += np.array((0.006, -0.008, 0.007, -0.008, 0.006))
    seeds.append(s)
    return seeds


def construct_packing():
    base = _incumbent_centers()
    pairs = np.asarray([(i, j) for i in range(26) for j in range(i)], dtype=int)
    constraint = {
        "type": "ineq",
        "fun": lambda z: _constraint_values(z, pairs),
        "jac": lambda z: _constraint_jacobian(z, pairs),
    }
    objjac = np.concatenate((np.zeros(52), -np.ones(26)))
    bounds = [(1e-6, 1.0 - 1e-6)] * 52 + [(1e-8, 0.5)] * 26

    best_c = base.copy()
    best_r = _optimal_radii_for_centers(best_c)
    if best_r is None:
        best_r = compute_max_radii(best_c)
    best_value = float(np.sum(best_r))

    outer = list(range(5)) + list(range(21, 26))
    for seed in _sheared_seeds(base):
        z0 = np.concatenate((seed.ravel(), 0.97 * compute_max_radii(seed)))
        restricted = list(bounds)
        for i in outer:
            restricted[2 * i] = (seed[i, 0], seed[i, 0])
            restricted[2 * i + 1] = (seed[i, 1], seed[i, 1])

        first = minimize(
            lambda z: -np.sum(z[52:]), z0, jac=lambda z: objjac,
            bounds=restricted, constraints=constraint, method="SLSQP",
            options={"maxiter": 360, "ftol": 1e-11, "disp": False}
        )
        final = minimize(
            lambda z: -np.sum(z[52:]), first.x, jac=lambda z: objjac,
            bounds=bounds, constraints=constraint, method="SLSQP",
            options={"maxiter": 160, "ftol": 1e-11, "disp": False}
        )
        c = final.x[:52].reshape(26, 2)
        r = _optimal_radii_for_centers(c)
        if r is not None and _strictly_valid(c, r):
            value = float(np.sum(r))
            if value > best_value:
                best_c, best_r, best_value = c.copy(), r.copy(), value

    return best_c, best_r, float(np.sum(best_r))


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
    visualize(centers, radii)