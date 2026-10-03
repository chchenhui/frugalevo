# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles via joint SLSQP optimization."""
import numpy as np
from scipy.optimize import minimize

_N = 26


def _hex_seed():
    """Deterministic hexagonal seed: 6 staggered rows [4,5,4,5,4,5], drop last circle."""
    row_counts = [4, 5, 4, 5, 4, 5]
    dx = 0.2
    dy = np.sqrt(3) * dx / 2
    y0 = (1 - dy * (len(row_counts) - 1)) / 2
    pts = []
    for r, count in enumerate(row_counts):
        y = y0 + r * dy
        xs = [0.2 + i * dx for i in range(count)] if r % 2 == 0 else [0.1 + i * dx for i in range(count)]
        pts.extend([x, y] for x in xs)
    return np.array(pts[:_N], dtype=float)


def _grid_seed(rng):
    """Jittered 5x5 grid plus one center circle, perturbed."""
    g = np.linspace(0.1, 0.9, 5)
    pts = [[x, y] for y in g for x in g]
    pts.append([0.5, 0.5])
    pts = np.array(pts[:_N], dtype=float)
    pts += rng.uniform(-0.02, 0.02, pts.shape)
    return np.clip(pts, 0.05, 0.95)


def _pack(z):
    """Unpack 78-vector into centers (26,2) and radii (26,)."""
    z = z.reshape(3, _N)
    return z[0].copy(), z[1].copy(), z[2].copy()


def _unpack_to_z(centers, radii):
    z = np.empty(3 * _N)
    z[0:_N] = centers[:, 0]
    z[_N:2 * _N] = centers[:, 1]
    z[2 * _N:3 * _N] = radii
    return z


def _objective(z):
    return -np.sum(z[2 * _N:3 * _N])


def _make_constraints():
    idx = np.triu_indices(_N, k=1)
    cons = [
        {
            "type": "ineq",
            "fun": lambda z, ii=idx[0], jj=idx[1]: (
                (z[ii] - z[jj]) ** 2 + (z[_N + ii] - z[_N + jj]) ** 2
                - (z[2 * _N + ii] + z[2 * _N + jj]) ** 2
            ),
        }
    ]
    for k in range(_N):
        cons.append({"type": "ineq", "fun": lambda z, k=k: z[k] - z[2 * _N + k]})
        cons.append({"type": "ineq", "fun": lambda z, k=k: z[_N + k] - z[2 * _N + k]})
        cons.append({"type": "ineq", "fun": lambda z, k=k: 1 - z[k] - z[2 * _N + k]})
        cons.append({"type": "ineq", "fun": lambda z, k=k: 1 - z[_N + k] - z[2 * _N + k]})
    return cons


def _bounds():
    b = []
    for _ in range(_N):
        b += [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)]
    return b


def _feasible(centers, radii, tol=1e-6):
    """Validate non-overlap and containment within tolerance."""
    if np.any(radii <= 0) or np.any(radii > 0.5 + tol):
        return False
    if np.any(centers - radii[:, None] < -tol) or np.any(centers + radii[:, None] > 1 + tol):
        return False
    d = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    return np.all(d >= radii[:, None] + radii[None, :] - tol)


def _shell_seed():
    """
    Parametric heterogeneous-radius shell layout (mixed-radius escape from the
    equal-radius hex ceiling): one large center circle (radius a), 8 circles of
    radius b on a ring of radius a+b, 4 corner circles of radius c at (c,c),
    (1-c,c), (c,1-c), (1-c,1-c), 4 edge-midpoint circles of radius c at
    (0.5,c),(0.5,1-c),(c,0.5),(1-c,0.5), and 9 circles of radius b on a partial
    outer ring of radius a+3b at angles k*30 deg (k=0..8). The three structural
    parameters (a, b, c) are optimized (<=6 starts, maxiter 200 each on 3
    variables) to minimize pairwise/wall max-violation; greedy-maximal radii are
    then computed for the resulting centers.
    """
    idx = np.triu_indices(_N, k=1)

    def centers_for(a, b, c):
        pts = [[0.5, 0.5]]
        ring1 = a + b
        for k in range(8):
            t = 2 * np.pi * k / 8 + np.pi / 8
            pts.append([0.5 + ring1 * np.cos(t), 0.5 + ring1 * np.sin(t)])
        pts += [[c, c], [1 - c, c], [c, 1 - c], [1 - c, 1 - c]]
        pts += [[0.5, c], [0.5, 1 - c], [c, 0.5], [1 - c, 0.5]]
        ring2 = a + 3 * b
        for k in range(9):
            t = np.pi * k / 6
            pts.append([0.5 + ring2 * np.cos(t), 0.5 + ring2 * np.sin(t)])
        return np.array(pts[:_N], dtype=float)

    def viol(p):
        a, b, c = p
        if a <= 0.02 or b <= 0.02 or c <= 0.02:
            return 10.0
        C = centers_for(a, b, c)
        wall = np.max(np.maximum(0.0, np.concatenate([-C, C - 1.0], axis=1)))
        d = np.linalg.norm(C[idx[0]] - C[idx[1]], axis=1)
        worst_pair = 2 * b + 2 * c - 2 * b  # baseline gap needed: b+b or b+c
        need = np.where(
            np.isin(idx[0], [0]) | np.isin(idx[1], [0]), a + b, 0.0
        )
        need = np.maximum(need, np.where((idx[0] >= 13) | (idx[1] >= 13), b, 0.0))
        need = np.maximum(need, np.where(
            ((idx[0] >= 1) & (idx[0] <= 12)) & ((idx[1] >= 1) & (idx[1] <= 12)), b, 0.0))
        pen = np.sum(np.maximum(0.0, need - d) ** 2)
        return float(wall ** 2 + pen + worst_pair * 0.0)

    best_p, best_v = None, np.inf
    for a0 in (0.14, 0.18):
        for b0 in (0.07, 0.09):
            p0 = np.array([a0, b0, 0.10])
            try:
                res = minimize(viol, p0, method="Nelder-Mead",
                               options={"maxiter": 200, "xatol": 1e-6, "fatol": 1e-10})
                if res.fun < best_v:
                    best_v, best_p = float(res.fun), res.x
            except Exception:
                continue
    if best_p is None:
        best_p = np.array([0.16, 0.08, 0.10])
    a, b, c = np.clip(best_p, 0.03, 0.45)
    C = centers_for(a, b, c)
    C = np.clip(C, 0.02, 0.98)

    # Greedy-maximal radii: wall distance then pairwise shrink.
    r = np.minimum(np.min(C, axis=1), 1 - np.max(C, axis=1))
    r = np.minimum(np.min(np.column_stack([C, 1 - C]), axis=1), 0.5)
    for i in range(_N):
        for j in range(i + 1, _N):
            d = np.linalg.norm(C[i] - C[j])
            if r[i] + r[j] > d and d > 1e-12:
                s = d / (r[i] + r[j])
                r[i] *= s
                r[j] *= s
    return C, np.maximum(r, 1e-4)


def _corner_seed(rng):
    """Deterministic seed emphasizing corners and edges: 4 corner circles,
    edge circles along each side, and interior fill from the hex layout."""
    pts = [
        [0.10, 0.10], [0.90, 0.10], [0.10, 0.90], [0.90, 0.90],
        [0.30, 0.06], [0.50, 0.06], [0.70, 0.06],
        [0.30, 0.94], [0.50, 0.94], [0.70, 0.94],
        [0.06, 0.30], [0.06, 0.50], [0.06, 0.70],
        [0.94, 0.30], [0.94, 0.50], [0.94, 0.70],
        [0.28, 0.28], [0.72, 0.28], [0.28, 0.72], [0.72, 0.72],
        [0.50, 0.28], [0.28, 0.50], [0.72, 0.50], [0.50, 0.72],
        [0.50, 0.50], [0.50, 0.88],
    ]
    pts = np.array(pts[:_N], dtype=float)
    pts += rng.uniform(-0.01, 0.01, pts.shape)
    return np.clip(pts, 0.04, 0.96)


def construct_packing():
    """
    Joint SLSQP optimization of all 78 variables (26 x, 26 y, 26 r) maximizing
    the sum of radii subject to pairwise non-overlap and square containment.
    Multistart from: (a) hex seed with greedy-maximal radii, (b) perturbed hex
    seed, (c) jittered 5x5+1 grid, (d) corner/edge-biased seed, (e) hex seed
    shifted toward walls. A bounded refinement loop re-solves from perturbed
    incumbents to escape flat regions. Falls back to the hex seed with
    conservative pairwise-maximal radii if SLSQP fails to improve.
    """
    # Incumbent fallback: hex layout with greedy-maximal radii
    hex_c = _hex_seed()
    fb_r = np.full(_N, 1.0)
    for i in range(_N):
        fb_r[i] = min(hex_c[i, 0], hex_c[i, 1], 1 - hex_c[i, 0], 1 - hex_c[i, 1])
    for i in range(_N):
        for j in range(i + 1, _N):
            dist = np.linalg.norm(hex_c[i] - hex_c[j])
            if fb_r[i] + fb_r[j] > dist:
                s = dist / (fb_r[i] + fb_r[j])
                fb_r[i] *= s
                fb_r[j] *= s
    best_c, best_r, best_sum = hex_c.copy(), fb_r.copy(), float(np.sum(fb_r))

    # Primary seed: optimized mixed-radius shell layout (heterogeneous radii),
    # replacing the pure equal-radius hex seed as the top structural candidate.
    try:
        shell_c, shell_r = _shell_seed()
        if np.sum(shell_r) > best_sum:
            best_c, best_r, best_sum = shell_c.copy(), shell_r.copy(), float(np.sum(shell_r))
    except Exception:
        shell_c, shell_r = hex_c.copy(), fb_r.copy()

    rng = np.random.default_rng(42)
    seeds = [
        (shell_c.copy(), np.full(_N, min(0.085, float(np.min(shell_r)) + 0.005))),
        (shell_c.copy(), np.clip(shell_r * 0.95, 1e-4, 0.5)),
        (hex_c.copy(), fb_r.copy()),  # greedy-maximal radii: feasible and already good
        (np.clip(hex_c + rng.uniform(-0.01, 0.01, hex_c.shape), 0.05, 0.95), np.full(_N, 0.09)),
        (_grid_seed(rng), np.full(_N, 0.08)),
        (_corner_seed(rng), np.full(_N, 0.075)),
        (np.clip(hex_c + rng.uniform(0.0, 0.012, hex_c.shape) * np.sign(hex_c - 0.5), 0.04, 0.96), np.full(_N, 0.088)),
    ]

    cons = _make_constraints()
    bnds = _bounds()

    def _run(c0, r0):
        """One bounded SLSQP solve; returns (centers, radii) or None."""
        z0 = _unpack_to_z(c0, r0)
        try:
            res = minimize(
                _objective, z0, method="SLSQP", bounds=bnds,
                constraints=cons,
                options={"maxiter": 400, "ftol": 1e-12},
            )
            cx, cy, rr = _pack(res.x)
            cand_c = np.column_stack([cx, cy])
            if _feasible(cand_c, rr):
                return cand_c, rr
        except Exception:
            pass
        return None

    for c0, r0 in seeds:
        out = _run(c0, r0)
        if out is not None and np.sum(out[1]) > best_sum:
            best_c, best_r, best_sum = out[0].copy(), out[1].copy(), float(np.sum(out[1]))

    # Circle-relocation refinement: evict a small-radius circle and reinsert
    # it at the largest empty site (Voronoi vertices of the remaining centers,
    # clipped into the square, plus a 9x9 grid fallback), recompute
    # greedy-maximal radii, and polish with one bounded SLSQP solve. Each round
    # tries the 3 smallest circles x top-3 clearance sites; the incumbent is
    # kept unless a relocation strictly improves the sum.
    def _greedy_radii(C):
        r = np.minimum(np.min(np.column_stack([C, 1.0 - C]), axis=1), 0.5)
        for i in range(_N):
            for j in range(i + 1, _N):
                d = np.linalg.norm(C[i] - C[j])
                if r[i] + r[j] > d and d > 1e-12:
                    s = d / (r[i] + r[j])
                    r[i] *= s
                    r[j] *= s
        return np.maximum(r, 1e-4)

    for _ in range(8):
        order = np.argsort(best_r)[:3]
        others_all = [np.delete(best_c, k, axis=0) for k in order]
        cands = []
        try:
            from scipy.spatial import Voronoi
            vr = Voronoi(best_c)
            for v in vr.vertices:
                if 0.0 <= v[0] <= 1.0 and 0.0 <= v[1] <= 1.0:
                    cands.append(v)
        except Exception:
            pass
        g = np.linspace(0.02, 0.98, 9)
        cands.extend([x, y] for y in g for x in g)
        if not cands:
            break
        C_c = np.array(cands, dtype=float)

        improved = False
        for k, others in zip(order, others_all):
            d = np.linalg.norm(C_c[:, None, :] - others[None, :, :], axis=2)
            clearance = np.min(d, axis=1)
            top = np.argsort(clearance)[-3:]
            for t in top:
                site = C_c[t]
                cand_c = np.vstack([others, np.atleast_2d(site)])
                cand_r = _greedy_radii(cand_c.copy())
                out = _run(cand_c, cand_r)
                if out is not None and np.sum(out[1]) > best_sum + 1e-9:
                    best_c, best_r = out[0].copy(), out[1].copy()
                    best_sum = float(np.sum(best_r))
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    return best_c, best_r, float(np.sum(best_r))


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
