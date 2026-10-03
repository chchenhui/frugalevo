# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def _largest_empty_center(centers):
    """Return the point in the unit square maximizing the distance to the
    nearest existing circle center or wall (greedy Apollonian insertion
    point), found by a coarse grid scan plus local refinement."""
    xs = np.linspace(0.0, 1.0, 161)
    X, Y = np.meshgrid(xs, xs)
    P = np.stack([X.ravel(), Y.ravel()], axis=1)
    d = np.sqrt(((P - 0.5) ** 2).sum(axis=1)) * 0.0 + 0.5  # wall distance cap
    d = np.minimum(d, 0.5 - np.abs(P[:, 0] - 0.5))
    d = np.minimum(d, 0.5 - np.abs(P[:, 1] - 0.5))
    for c in centers:
        d = np.minimum(d, np.sqrt(((P - c) ** 2).sum(axis=1)))
    k = int(np.argmax(d))
    bx, by = P[k]
    bd = float(d[k])
    step = 1.0 / 160
    for _ in range(40):
        improved = False
        for dx in (-step, 0.0, step):
            for dy in (-step, 0.0, step):
                qx = min(max(bx + dx, 0.0), 1.0)
                qy = min(max(by + dy, 0.0), 1.0)
                qd = min(0.5 - abs(qx - 0.5), 0.5 - abs(qy - 0.5))
                if len(centers):
                    qd = min(qd, np.sqrt(((centers - [qx, qy]) ** 2).sum(axis=1)).min())
                if qd > bd + 1e-12:
                    bx, by, bd = qx, qy, qd
                    improved = True
        if not improved:
            step *= 0.5
            if step < 1e-9:
                break
    return np.array([bx, by]), bd


def _polish(centers, radii, maxiter=200):
    """SLSQP refinement of the 78 variables (52 center coords + 26 radii)
    maximizing the sum of radii subject to wall and pairwise-distance
    constraints. Returns refined centers, radii, or the inputs on failure."""
    try:
        from scipy.optimize import minimize
    except Exception:
        return centers, radii
    n = len(centers)
    x0 = np.concatenate([centers.ravel(), radii])

    def unpack(v):
        return v[: 2 * n].reshape(n, 2), v[2 * n:]

    def obj(v):
        return -np.sum(v[2 * n:])

    def grad(v):
        g = np.zeros_like(v)
        g[2 * n:] = -1.0
        return g

    def cons_wall(v):
        c, r = unpack(v)
        return np.concatenate([c[:, 0] - r, c[:, 1] - r,
                               1 - c[:, 0] - r, 1 - c[:, 1] - r])

    def jac_wall(v):
        c, r = unpack(v)
        J = np.zeros((4 * n, 3 * n))
        for k in range(n):
            J[k, 2 * k] = 1.0
            J[k, 2 * n + k] = -1.0
            J[n + k, 2 * k + 1] = 1.0
            J[n + k, 2 * n + k] = -1.0
            J[2 * n + k, 2 * k] = -1.0
            J[2 * n + k, 2 * n + k] = -1.0
            J[3 * n + k, 2 * k + 1] = -1.0
            J[3 * n + k, 2 * n + k] = -1.0
        return J

    def cons_pair(v):
        c, r = unpack(v)
        out = []
        for i in range(n):
            for j in range(i + 1, n):
                dd = np.sqrt(np.sum((c[i] - c[j]) ** 2))
                out.append(dd - r[i] - r[j])
        return np.array(out)

    def jac_pair(v):
        c, r = unpack(v)
        m = n * (n - 1) // 2
        J = np.zeros((m, 3 * n))
        row = 0
        for i in range(n):
            for j in range(i + 1, n):
                diff = c[i] - c[j]
                dd = max(np.sqrt(np.sum(diff ** 2)), 1e-12)
                J[row, 2 * i] = diff[0] / dd
                J[row, 2 * i + 1] = diff[1] / dd
                J[row, 2 * j] = -diff[0] / dd
                J[row, 2 * j + 1] = -diff[1] / dd
                J[row, 2 * n + i] = -1.0
                J[row, 2 * n + j] = -1.0
                row += 1
        return J

    res = minimize(obj, x0, jac=grad, method="SLSQP",
                   constraints=[{"type": "ineq", "fun": cons_wall,
                                 "jac": jac_wall},
                                {"type": "ineq", "fun": cons_pair,
                                 "jac": jac_pair}],
                   options={"maxiter": maxiter, "ftol": 1e-10})
    c, r = unpack(res.x)
    if not np.all(np.isfinite(c)) or not np.all(np.isfinite(r)):
        return centers, radii
    return c, np.maximum(r, 1e-9)


def _tighten(centers, radii, rounds=12, grow=1.004):
    """Active-set contact-graph tightening via grow-and-reproject:
    detect touching pairs (d_ij ~ r_i+r_j) and wall contacts, then
    repeatedly inflate all radii by a small factor and re-solve the
    equality system (pair distances = r_i+r_j, wall coords = r / 1-r)
    with bounded damped least-squares so centers shift to restore exact
    simultaneous contact. A shrink guard enforces strict feasibility;
    result accepted only if it improves sum(radii)."""
    n = len(radii)
    c0 = np.asarray(centers, dtype=float).copy()
    r0 = np.asarray(radii, dtype=float).copy()
    pairs, walls = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(c0[i] - c0[j]) - r0[i] - r0[j] < 5e-4:
                pairs.append((i, j))
    for i in range(n):
        for a in range(2):
            v = c0[i, a]
            if v - r0[i] < 5e-4:
                walls.append((i, a, 0))
            if 1.0 - v - r0[i] < 5e-4:
                walls.append((i, a, 1))

    def F(x):
        cc = x[: 2 * n].reshape(n, 2)
        rr = x[2 * n:]
        out = []
        for i, j in pairs:
            out.append(np.sqrt(np.sum((cc[i] - cc[j]) ** 2)) - rr[i] - rr[j])
        for i, a, s in walls:
            out.append((cc[i, a] - rr[i]) if s == 0
                       else (1.0 - cc[i, a] - rr[i]))
        return np.asarray(out)

    def guard(cc, rr):
        rr = np.maximum(rr, 1e-9)
        for i in range(n):
            lim = min(cc[i, 0], cc[i, 1], 1 - cc[i, 0], 1 - cc[i, 1])
            if rr[i] > lim:
                rr[i] = max(lim, 1e-9)
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt(np.sum((cc[i] - cc[j]) ** 2))
                if rr[i] + rr[j] > d:
                    sc = d / (rr[i] + rr[j])
                    rr[i] *= sc
                    rr[j] *= sc
        return cc, np.maximum(rr - 1e-9, 1e-9)

    x = np.concatenate([c0.ravel(), r0])
    try:
        from scipy.optimize import least_squares
        for _ in range(rounds):
            x[2 * n:] *= grow  # inflate, then reproject to exact contact
            sol = least_squares(F, x, method="lm", max_nfev=80,
                                xtol=1e-12, ftol=1e-12)
            x = sol.x
    except Exception:
        pass
    cc = np.asarray(x[: 2 * n]).reshape(n, 2)
    rr = np.asarray(x[2 * n:], dtype=float)
    cc, rr = guard(cc, rr)
    if np.all(np.isfinite(cc)) and np.all(np.isfinite(rr)) \
            and np.sum(rr) > np.sum(radii):
        return cc, rr
    return np.asarray(centers, dtype=float), np.asarray(radii, dtype=float)


def _final_guard(centers, radii):
    """Clip centers into the unit square and shrink any radius violating
    a wall or pairwise contact. Only shrinks; returns valid arrays."""
    n = len(radii)
    centers = np.clip(np.asarray(centers, dtype=float), 1e-9, 1 - 1e-9)
    radii = np.maximum(np.asarray(radii, dtype=float), 1e-9)
    for i in range(n):
        lim = min(centers[i, 0], centers[i, 1],
                  1 - centers[i, 0], 1 - centers[i, 1])
        if radii[i] > lim:
            radii[i] = max(lim, 1e-9)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if radii[i] + radii[j] > d:
                sc = d / (radii[i] + radii[j])
                radii[i] *= sc
                radii[j] *= sc
    return centers, radii


def construct_packing():
    """
    Basin-hopping over the incumbent packing: build the greedy Apollonian
    seed, polish and tighten once to obtain the base incumbent, then run
    time-budgeted restarts that perturb the *current best* solution
    (Gaussian jitter on centers, small multiplicative jitter on radii),
    re-polish and re-tighten each perturbed copy, and keep the best valid
    result. Monotone non-decreasing: the incumbent is returned unless a
    restart strictly improves its guarded sum.
    """
    import time
    rng = np.random.default_rng(0)
    n = 26
    centers = np.zeros((n, 2))
    placed = []
    for k in range(n):
        p, _ = _largest_empty_center(np.array(placed))
        centers[k] = p
        placed.append(p)

    radii = compute_max_radii(centers)
    centers, radii = _polish(centers.copy(), radii.copy())
    centers, radii = _tighten(centers, radii)
    centers, radii = _final_guard(centers, radii)
    best_sum = float(np.sum(radii))

    t0 = time.time()
    # Scale effort with the available wall budget: analytic constraint
    # Jacobians make each restart much cheaper, so a longer budget (~300 s)
    # now fits many more restarts inside the 360 s limit.
    budget = 300.0
    for it in range(150):
        if time.time() - t0 > budget:
            break
        # Four-way perturbation ladder of the incumbent: fine jitter
        # (re-solves the same contact topology), medium jitter for nearby
        # basins, occasional larger escape jitter, and a radius-only
        # squeeze-regrow restart that reshuffles the active contact set
        # without moving centers first.
        if it % 10 == 8:
            pc = centers.copy()
            pr = radii * rng.uniform(0.90, 0.99, size=radii.shape)
        else:
            sig = 0.004 if (it % 4) else (0.012 if (it % 8 == 4) else 0.028)
            pc = centers + rng.normal(0.0, sig, size=centers.shape)
            pr = radii * rng.uniform(0.97, 1.03, size=radii.shape)
        pc, pr = _polish(pc.copy(), pr.copy(), maxiter=300)
        pc, pr = _tighten(pc, pr, rounds=18, grow=1.002)
        pc, pr = _final_guard(pc, pr)
        s = float(np.sum(pr))
        if np.all(np.isfinite(pc)) and np.all(np.isfinite(pr)) and s > best_sum:
            best_sum = s
            centers, radii = pc, pr
            # One extra gentle re-tighten on the newly accepted incumbent
            # to squeeze residual slack before the next restart.
            qc, qr = _tighten(centers.copy(), radii.copy(),
                              rounds=10, grow=1.0015)
            qc, qr = _final_guard(qc, qr)
            qs = float(np.sum(qr))
            if np.all(np.isfinite(qc)) and np.all(np.isfinite(qr)) and qs > best_sum:
                best_sum = qs
                centers, radii = qc, qr

    return centers, radii, best_sum


def compute_max_radii(centers):
    """
    Compute the maximum possible radii for each circle position
    such that they don't overlap and stay within the unit square.

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates

    Returns:
        np.array of shape (n) with radius of each circle
    """
    n = centers.shape[0]
    radii = np.ones(n)

    # First, limit by distance to square borders
    for i in range(n):
        x, y = centers[i]
        # Distance to borders
        radii[i] = min(x, y, 1 - x, 1 - y)

    # Then, limit by distance to other circles
    # Each pair of circles with centers at distance d can have
    # sum of radii at most d to avoid overlap
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))

            # If current radii would cause overlap
            if radii[i] + radii[j] > dist:
                # Scale both radii proportionally
                scale = dist / (radii[i] + radii[j])
                radii[i] *= scale
                radii[j] *= scale

    return radii


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii


def visualize(centers, radii):
    """
    Visualize the circle packing

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radius of each circle
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw unit square
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True)

    # Draw circles
    for i, (center, radius) in enumerate(zip(centers, radii)):
        circle = Circle(center, radius, alpha=0.5)
        ax.add_patch(circle)
        ax.text(center[0], center[1], str(i), ha="center", va="center")

    plt.title(f"Circle Packing (n={len(centers)}, sum={sum(radii):.6f})")
    plt.show()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii}")
    # AlphaEvolve improved this to 2.635

    # Uncomment to visualize:
    visualize(centers, radii)
