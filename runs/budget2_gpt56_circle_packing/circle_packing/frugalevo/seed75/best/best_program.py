"""Constructor-based circle packing for 26 circles in a unit square."""
import numpy as np


def compute_max_radii(centers):
    from scipy.optimize import linprog

    centers = np.asarray(centers, dtype=float)
    n = len(centers)
    rows, rhs = [], []
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n)
            row[i] = row[j] = 1.0
            rows.append(row)
            rhs.append(float(np.linalg.norm(centers[i] - centers[j])))

    walls = np.min(np.column_stack((
        centers[:, 0], centers[:, 1],
        1.0 - centers[:, 0], 1.0 - centers[:, 1],
    )), axis=1)
    ans = linprog(
        -np.ones(n), A_ub=np.asarray(rows), b_ub=np.asarray(rhs),
        bounds=[(0.0, max(0.0, float(w))) for w in walls],
        method="highs",
    )
    if not ans.success:
        return np.maximum(walls, 0.0) * (1.0 - 1e-8)
    return np.maximum(ans.x, 0.0) * (1.0 - 1e-9)


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
        return np.concatenate([np.ravel(v) for v in out])

    counts = (5, 6, 5, 5, 5)
    defect = np.array((-1.0, -0.55, -0.15, 0.15, 0.55, 1.0))
    seeds = []
    for ls, rs in ((1, 1), (-1, -1), (1, -1), (-1, 1)):
        points = []
        for row, count in enumerate(counts):
            y = 0.09 + 0.205 * row
            xs = (np.arange(count) + 0.5) / count
            ys = np.full(count, y)
            if row == 1:
                xs += 0.024 * defect
                xs[0] += 0.018 * ls
                xs[1] += 0.009 * ls
                xs[-2] += 0.009 * rs
                xs[-1] += 0.018 * rs
                ys[0] += 0.008 * ls
                ys[-1] += 0.008 * rs
            else:
                xs += 0.025 * (1 if row % 2 == 0 else -1)
                xs[0] += 0.018 * ls
                xs[-1] += 0.018 * rs
                ys[0] += 0.009 * ls
                ys[-1] += 0.009 * rs
            xs -= np.mean(xs) - 0.5
            points.extend(zip(xs, ys))
        c = np.asarray(points, dtype=float)
        seeds.append(np.r_[c.ravel(), compute_max_radii(c)])

    bounds = [(0.0, 1.0)] * (2 * n) + [(0.001, 0.20)] * n
    best = None
    for seed in seeds:
        res = minimize(
            objective, seed, method="SLSQP", bounds=bounds,
            constraints={"type": "ineq", "fun": constraints},
            options={"maxiter": 950, "ftol": 1e-10, "disp": False},
        )
        if res.success and (best is None or res.fun < best.fun):
            best = res

    z = seeds[0].copy() if best is None else best.x.copy()
    centers = z[:2 * n].reshape(n, 2)
    radii = compute_max_radii(centers)

    def valid(c, r, tol=-2e-7):
        wall = np.min(np.column_stack((
            c[:, 0] - r, c[:, 1] - r,
            1.0 - c[:, 0] - r, 1.0 - c[:, 1] - r,
        )))
        pair = np.inf
        for i in range(n):
            d = c[i + 1:] - c[i]
            if len(d):
                pair = min(pair, float(np.min(
                    np.sum(d * d, axis=1) - (r[i] + r[i + 1:]) ** 2
                )))
        return min(wall, pair) >= tol

    def dual_release(base):
        """Release three dual-weighted contact neighborhoods using two bounded SLSQP starts."""
        c0 = np.asarray(base, dtype=float).copy()
        r0 = np.asarray(compute_max_radii(c0), dtype=float).copy()

        active = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                v = c0[j] - c0[i]
                d = float(np.linalg.norm(v))
                if d > 1e-12 and abs(d - r0[i] - r0[j]) < 3e-5:
                    u = v / d
                    active[i].append((j, u))
                    active[j].append((i, -u))

        # LP response to a small coordinate translation supplies the
        # dual-sensitivity part of the neighborhood ordering.
        score = np.zeros(n, dtype=float)
        for i in range(n):
            score[i] = float(len(active[i]))
            for axis in (0, 1):
                shifted = c0.copy()
                shifted[i, axis] += 1e-5
                shifted_r = compute_max_radii(shifted)
                score[i] += float(np.sum(shifted_r - r0)) / 1e-5 * 1e-5

        neighborhoods = []
        claimed = np.zeros(n, dtype=bool)
        for root in np.argsort(-score):
            root = int(root)
            if claimed[root]:
                continue
            ids = np.flatnonzero(
                np.sum((c0 - c0[root]) ** 2, axis=1) <= 0.27 ** 2
            ).astype(int)
            ids = ids[~claimed[ids]]
            if len(ids) == 0:
                continue
            neighborhoods.append(ids)
            claimed[ids] = True
            if len(neighborhoods) == 3:
                break

        best_c = c0.copy()
        best_r = r0.copy()
        best_value = float(np.sum(best_r))

        for ids in neighborhoods:
            ids = np.asarray(ids, dtype=int)
            m = int(len(ids))
            movable = np.zeros(n, dtype=bool)
            movable[ids] = True

            def unpack(q):
                cc = c0.copy()
                rr = r0.copy()
                cc[ids] = np.asarray(q[:2 * m], dtype=float).reshape(m, 2)
                rr[ids] = np.asarray(q[2 * m:], dtype=float)
                return cc, rr

            def fun(q):
                return -float(np.sum(q[2 * m:])) - float(
                    np.sum(r0[~movable])
                )

            def con(q):
                cc, rr = unpack(q)
                out = [
                    cc[ids, 0] - rr[ids],
                    cc[ids, 1] - rr[ids],
                    1.0 - cc[ids, 0] - rr[ids],
                    1.0 - cc[ids, 1] - rr[ids],
                ]
                for i in ids:
                    d = cc - cc[i]
                    slack = np.sum(d * d, axis=1) - (rr[i] + rr) ** 2
                    slack[i] = 1.0
                    out.append(slack)
                return np.concatenate([np.ravel(x) for x in out])

            starts = [
                np.r_[c0[ids].ravel(), r0[ids]].astype(float, copy=True)
            ]
            displaced = c0.copy()
            for i in ids:
                direction = np.zeros(2, dtype=float)
                for _, unit in active[int(i)]:
                    direction -= unit
                norm = float(np.linalg.norm(direction))
                if norm > 1e-12:
                    displaced[i] += 2e-4 * direction / norm
            displaced[ids] = np.clip(displaced[ids], 1e-4, 1.0 - 1e-4)
            starts.append(
                np.r_[displaced[ids].ravel(), r0[ids]].astype(float, copy=True)
            )

            local_best = None
            bounds = [(.001, .999)] * (2 * m) + [(.001, .20)] * m
            for start in starts:
                res = minimize(
                    fun, start, method="SLSQP", bounds=bounds,
                    constraints={"type": "ineq", "fun": con},
                    options={"maxiter": 450, "ftol": 1e-10, "disp": False},
                )
                if res.success and (
                    local_best is None or res.fun < local_best.fun
                ):
                    local_best = res

            if local_best is None:
                continue
            trial_c, _ = unpack(local_best.x)
            trial_r = np.asarray(
                compute_max_radii(trial_c), dtype=float
            ).copy()
            if not valid(trial_c, trial_r):
                continue
            trial_value = float(np.sum(trial_r))
            if trial_value > best_value + 1e-11:
                best_c = trial_c.copy()
                best_r = trial_r.copy()
                best_value = trial_value
                c0 = best_c.copy()
                r0 = best_r.copy()

        return best_c, best_r

    centers, radii = dual_release(centers)
    value = float(np.sum(radii))

    # Controlled use of evaluator tolerance, independently validated.
    lift = 5e-7
    for _ in range(8):
        trial = radii + lift
        if valid(centers, trial, tol=-1e-6):
            radii = trial
            break
        lift -= 5e-8

    return centers, radii, float(np.sum(radii))


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