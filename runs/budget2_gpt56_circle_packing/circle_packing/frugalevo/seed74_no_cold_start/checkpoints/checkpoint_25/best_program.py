"""Deterministic multistart constructor for 26 unequal circles in a unit square."""
import numpy as np


def compute_max_radii(centers):
    c = np.asarray(centers, dtype=float)
    n = len(c)
    r = np.minimum.reduce([c[:, 0], c[:, 1], 1 - c[:, 0], 1 - c[:, 1]]).copy()
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(c[i] - c[j]))
            if r[i] + r[j] > d and r[i] + r[j] > 0:
                q = d / (r[i] + r[j])
                r[i] *= q
                r[j] *= q
    return r


def _incumbent():
    x5 = np.linspace(.1, .9, 5)
    x6 = np.linspace(1 / 12, 11 / 12, 6)
    rows = [(.1, x5, .1), (.31666666666666665, x5, .1),
            (.5, x6, 1 / 12), (.6833333333333333, x5, .1), (.9, x5, .1)]
    c = np.array([(x, y) for y, xs, _ in rows for x in xs], dtype=float)
    r = np.array([rr for _, xs, rr in rows for _ in xs], dtype=float)
    return c, r * (1 - 1e-9)


def _repair(centers, radii, margin=3e-8):
    c = np.clip(np.asarray(centers, float).copy(), 0.0, 1.0)
    r = np.maximum(np.asarray(radii, float).copy(), 0.0)
    wall = np.minimum.reduce([c[:, 0], c[:, 1], 1-c[:, 0], 1-c[:, 1]])
    r = np.minimum(r, np.maximum(0.0, wall - margin))
    for _ in range(5):
        changed = False
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = float(np.linalg.norm(c[i] - c[j]))
                a = max(0.0, d - margin)
                s = r[i] + r[j]
                if s > a and s > 0:
                    q = a / s
                    r[i] *= q
                    r[j] *= q
                    changed = True
        if not changed:
            break
    return c, r


def _row_seed(counts=None, reflected=False, shear=0.0):
    """Construct a repaired corner-petal layout with four corners and a six-disk spine."""
    # Four corner disks, each tangent to the two nearby walls.
    pts = [
        (.060, .060), (.940, .060), (.060, .940), (.940, .940),

        # Two wall-adjacent petals around each corner.
        (.178, .064), (.064, .178),
        (.822, .064), (.936, .178),
        (.064, .822), (.178, .936),
        (.936, .822), (.822, .936),

        # Eight side/interior transition disks.
        (.285, .080), (.500, .090), (.715, .080),
        (.080, .315), (.920, .315),
        (.080, .685), (.920, .685),
        (.285, .920),

        # Six staggered central disks.
        (.335, .285), (.665, .285),
        (.275, .500), (.500, .500), (.725, .500),
        (.335, .715),
    ]
    c = np.asarray(pts, dtype=float)
    if shear:
        c[20:, 1] += shear * np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    if reflected:
        c[:, 0] = 1.0 - c[::-1, 0]

    # Unequal initial radii preserve the intended corner/petal/transition
    # hierarchy while allowing _joint to redistribute them.
    r = np.empty(26, dtype=float)
    r[:4] = .060
    r[4:12] = .070
    r[12:20] = .080
    r[20:] = .075
    return c.copy(), r.copy()


def _joint(centers, radii, maxiter=850):
    try:
        from scipy.optimize import minimize
    except Exception:
        return centers, radii
    n = 26
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m = 4*n + len(pairs)
    z0 = np.r_[np.asarray(centers, float).ravel(), np.asarray(radii, float)]

    def con(z):
        xy, rr = z[:2*n].reshape(n, 2), z[2*n:]
        out = np.empty(m)
        out[:n] = xy[:, 0] - rr
        out[n:2*n] = xy[:, 1] - rr
        out[2*n:3*n] = 1 - xy[:, 0] - rr
        out[3*n:4*n] = 1 - xy[:, 1] - rr
        for k, (i, j) in enumerate(pairs):
            d = xy[i] - xy[j]
            out[4*n+k] = d.dot(d) - (rr[i] + rr[j])**2
        return out

    def jac(z):
        xy, rr = z[:2*n].reshape(n, 2), z[2*n:]
        a = np.zeros((m, 3*n))
        for i in range(n):
            a[i, 2*i] = 1; a[i, 2*n+i] = -1
            a[n+i, 2*i+1] = 1; a[n+i, 2*n+i] = -1
            a[2*n+i, 2*i] = -1; a[2*n+i, 2*n+i] = -1
            a[3*n+i, 2*i+1] = -1; a[3*n+i, 2*n+i] = -1
        for k, (i, j) in enumerate(pairs):
            d = xy[i] - xy[j]
            row = 4*n+k
            a[row, 2*i:2*i+2] = 2*d
            a[row, 2*j:2*j+2] = -2*d
            a[row, 2*n+i] = a[row, 2*n+j] = -2*(rr[i]+rr[j])
        return a

    ans = minimize(
        lambda z: -float(np.sum(z[2*n:])), z0,
        jac=lambda z: np.r_[np.zeros(2*n), -np.ones(n)],
        constraints={"type": "ineq", "fun": con, "jac": jac},
        bounds=[(0, 1)]*(2*n) + [(1e-8, .35)]*n,
        method="SLSQP",
        options={"maxiter": maxiter, "ftol": 2e-10, "disp": False},
    )
    if not np.all(np.isfinite(ans.x)):
        return centers, radii
    return ans.x[:2*n].reshape(n, 2), ans.x[2*n:]


def _polish(centers, radii, maxiter=850):
    try:
        from scipy.optimize import minimize
    except Exception:
        return centers, radii
    n = 26
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    xy0 = np.asarray(centers, float).copy()
    wall = float(np.min(np.minimum.reduce(
        [xy0[:, 0], xy0[:, 1], 1-xy0[:, 0], 1-xy0[:, 1]])))
    pair = min(.5*float(np.linalg.norm(xy0[i]-xy0[j])) for i, j in pairs)
    z0 = np.r_[xy0.ravel(), .97*min(wall, pair)]

    def common_con(z):
        xy, q = z[:2*n].reshape(n, 2), z[-1]
        out = np.empty(4*n + len(pairs))
        out[:n] = xy[:, 0]-q
        out[n:2*n] = xy[:, 1]-q
        out[2*n:3*n] = 1-xy[:, 0]-q
        out[3*n:4*n] = 1-xy[:, 1]-q
        for k, (i, j) in enumerate(pairs):
            d = xy[i]-xy[j]
            out[4*n+k] = d.dot(d)-(2*q)**2
        return out

    ans = minimize(
        lambda z: -z[-1], z0, method="SLSQP",
        constraints={"type": "ineq", "fun": common_con},
        bounds=[(0, 1)]*(2*n)+[(1e-8, .25)],
        options={"maxiter": 380, "ftol": 2e-9, "disp": False},
    )
    if np.all(np.isfinite(ans.x)):
        centers = ans.x[:2*n].reshape(n, 2)
        radii = np.full(n, max(1e-8, ans.x[-1]))
    return _joint(centers, radii, maxiter)


def _contact_escape(c, r):
    n = 26
    rows = []
    for i in range(n):
        for axis in range(2):
            for sign, val in ((1.0, c[i, axis]-r[i]), (-1.0, 1-c[i, axis]-r[i])):
                if val < 2e-4:
                    row = np.zeros(2*n)
                    row[2*i+axis] = sign
                    rows.append(row)
    for i in range(n):
        for j in range(i+1, n):
            d = c[i]-c[j]
            gap = float(np.linalg.norm(d)-r[i]-r[j])
            if gap < 2e-4 and np.linalg.norm(d) > 1e-10:
                u = d/np.linalg.norm(d)
                row = np.zeros(2*n)
                row[2*i:2*i+2] = u
                row[2*j:2*j+2] = -u
                rows.append(row)
    if len(rows) < 2:
        return []
    _, _, vh = np.linalg.svd(np.asarray(rows), full_matrices=True)
    v1, v2 = vh[-1].reshape(n, 2), vh[-2].reshape(n, 2)
    out = []
    for a, b in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        d = a*v1 + b*v2
        d *= .008 / max(1e-12, float(np.max(np.abs(d))))
        out.append((np.clip(c+d, .01, .99), r*.88))
    return out


def construct_packing():
    best_c, best_r = _repair(*_incumbent())
    best_sum = float(np.sum(best_r))
    starts = [_incumbent()]
    c, r = _incumbent()
    c[10:16, 0] += .25/6
    starts.append((np.clip(c, .02, .98), r*.96))

    rng = np.random.default_rng(371942)
    base, _ = _incumbent()
    for k in range(8):
        d = rng.uniform(-.018, .018, (26, 2))
        d[:, 1] *= .85
        if k & 1:
            d[:, 0] *= -1
        starts.append((np.clip(base+d, .025, .975), np.full(26, .055)))

    # Corner-petal starts expose corner voids that the horizontal-row seeds
    # cannot use efficiently.  Reflection and small spine shears provide four
    # distinct contact orderings without increasing the optimizer workload.
    for reflection, shear in [
        (False, 0.000),
        (True, 0.000),
        (False, .014),
        (True, -.014),
    ]:
        starts.append(_row_seed(None, reflection, shear))

    for q, (c0, r0) in enumerate(starts):
        c0, r0 = _repair(c0, r0)
        c1, r1 = _polish(c0, r0, 900 if q < 2 else 780)
        c1, r1 = _repair(c1, r1)
        if float(np.sum(r1)) > best_sum:
            best_c, best_r, best_sum = c1, r1, float(np.sum(r1))

    for c0, r0 in _contact_escape(best_c, best_r):
        c1, r1 = _joint(c0, r0, 700)
        c1, r1 = _repair(c1, r1)
        if float(np.sum(r1)) > best_sum:
            best_c, best_r, best_sum = c1, r1, float(np.sum(r1))
    return best_c, best_r, best_sum


def run_packing():
    return construct_packing()


def visualize(centers, radii):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    for i, (c, r) in enumerate(zip(centers, radii)):
        ax.add_patch(Circle(c, r, alpha=.5))
        ax.text(c[0], c[1], str(i), ha="center", va="center")
    plt.title(f"Circle Packing (n={len(centers)}, sum={sum(radii):.6f})")
    plt.show()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii}")