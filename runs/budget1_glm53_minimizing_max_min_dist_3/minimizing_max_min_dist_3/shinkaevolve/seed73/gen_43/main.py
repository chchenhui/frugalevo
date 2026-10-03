# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

N, D = 14, 3
PAIRS_I, PAIRS_J = np.triu_indices(N, k=1)


def _pair_dists(P):
    return np.linalg.norm(P[PAIRS_I] - P[PAIRS_J], axis=1)


def _ratio(pts):
    ds = _pair_dists(pts)
    dmax = ds.max()
    if dmax <= 0:
        return -1.0
    return ds.min() / dmax


def _repulsion_stage(pts, iters=500, step=0.03):
    """Diameter-normalized repulsion to spread close pairs apart."""
    pts = pts.copy()
    for _ in range(iters):
        diff = pts[PAIRS_I] - pts[PAIRS_J]
        dist = np.linalg.norm(diff, axis=1)
        dmax = dist.max()
        if dmax <= 0:
            break
        pts /= dmax
        diff = pts[PAIRS_I] - pts[PAIRS_J]
        dist = np.sqrt(np.maximum((diff * diff).sum(-1), 1e-18))
        w = 1.0 / dist ** 12
        forces = (w[:, None] * diff) / dist[:, None]
        grad = np.zeros_like(pts)
        np.add.at(grad, PAIRS_I, forces)
        np.add.at(grad, PAIRS_J, -forces)
        norm = np.linalg.norm(grad, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        pts += step * grad / norm
        step *= 0.997
    return pts


def _slsqp_stage(pts, maxiter=500):
    """Polish: maximize dmin subject to dmax <= 1."""
    def objective(flat):
        return -_pair_dists(flat.reshape(N, D)).min()

    def constraint(flat):
        return 1.0 - _pair_dists(flat.reshape(N, D)).max()

    ds = _pair_dists(pts)
    if ds.max() > 0:
        pts = pts / ds.max()
    # Escalating tolerance cascade with tiny perturbation restarts to
    # escape stalled SLSQP active sets. Each pass is risk-free thanks to
    # the ratio-based acceptance check below.
    rng = np.random.default_rng(123)
    current = pts
    current_ratio = _ratio(pts)
    for maxit, ftol in ((maxiter, 1e-12), (400, 1e-14), (400, 1e-16)):
        res = minimize(
            objective,
            current.ravel(),
            method='SLSQP',
            constraints={'type': 'ineq', 'fun': constraint},
            options={'maxiter': maxit, 'ftol': ftol},
        )
        out = res.x.reshape(N, D)
        if np.all(np.isfinite(out)):
            r = _ratio(out)
            if r > current_ratio:
                current, current_ratio = out, r
            elif r >= current_ratio - 1e-9:
                # stalled: perturb slightly to escape the active set
                current = out + 1e-6 * rng.standard_normal(out.shape)
        else:
            current = current + 1e-6 * rng.standard_normal(current.shape)
    if current_ratio <= _ratio(pts):
        return pts
    return current


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def _icosahedron():
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ], dtype=float)
    return _unit(ico)


def _fibonacci_sphere(offset=0.0):
    k = np.arange(N) + 0.5
    ph = np.arccos(1.0 - 2.0 * k / N)
    th = np.pi * (1.0 + 5.0 ** 0.5) * k + offset
    return np.stack([np.cos(th) * np.sin(ph),
                     np.sin(th) * np.sin(ph),
                     np.cos(ph)], axis=1)


def _seeds(rng):
    seeds = []
    # Icosahedron + 2 perturbed poles
    ico = _icosahedron()
    assert ico.shape == (12, 3)
    for _ in range(3):
        extra = np.array([[0, 0, 1], [0, 0, -1]]) + 0.05 * rng.standard_normal((2, 3))
        s = np.vstack([ico, _unit(extra)])
        assert s.shape == (N, D)
        seeds.append(s)
    # Fibonacci variants
    for off in (0.0, 0.7, 1.9):
        s = _fibonacci_sphere(off)
        assert s.shape == (N, D)
        seeds.append(s)
    # Random on sphere and in ball
    for _ in range(5):
        seeds.append(_unit(rng.standard_normal((N, D))))
        seeds.append(rng.uniform(-1.0, 1.0, (N, D)))
    return seeds


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of
    minimum to maximum pairwise distance.

    Returns
        points: np.ndarray of shape (14, 3)
    """
    rng = np.random.default_rng(42)
    best_pts, best_ratio = None, -1.0

    # Known optimum dmin/dmax for N=14 in 3D is ~0.4899; stop early once
    # we are essentially at it to save evaluation time.
    TARGET_RATIO = 0.49
    for init in _seeds(rng):
        pre = _repulsion_stage(init)
        for cand in (pre, init):
            polished = _slsqp_stage(cand)
            assert polished.shape == (N, D)
            r = _ratio(polished)
            if r > best_ratio:
                best_ratio, best_pts = r, polished
        if best_ratio >= TARGET_RATIO:
            break

    if best_pts is None:
        np.random.seed(42)
        best_pts = np.random.randn(N, D)

    # Antipodal-symmetry refinement pass
    best_pts = _symmetrize_polish(best_pts)

    return np.asarray(best_pts, dtype=float)


def _symmetrize_polish(pts):
    """Enforce antipodal symmetry and polish with a tight SLSQP pass."""
    pts = pts - pts.mean(0)
    # Match each point with its best antipodal partner
    cost = np.linalg.norm(pts[:, None, :] + pts[None, :, :], axis=2)
    np.fill_diagonal(cost, np.inf)
    partner = np.argmin(cost, axis=1)
    # Only symmetrize if pairing is a valid involution (mutual matches)
    mutual = partner[partner] == np.arange(N)
    if not np.all(mutual):
        return pts
    sym = (pts - pts[partner]) / 2.0  # x_i and -x_j averaged -> (x_i - x_j)/2
    if _ratio(sym) <= 0:
        return pts
    out = _slsqp_tight(sym)
    if _ratio(out) > _ratio(pts):
        return out
    return pts


def _slsqp_tight(pts, maxiter=200):
    """Short, tight SLSQP polish from a symmetrized configuration."""
    ds = _pair_dists(pts)
    if ds.max() > 0:
        pts = pts / ds.max()

    def objective(flat):
        return -_pair_dists(flat.reshape(N, D)).min()

    def constraint(flat):
        return 1.0 - _pair_dists(flat.reshape(N, D)).max()

    res = minimize(
        objective,
        pts.ravel(),
        method='SLSQP',
        constraints={'type': 'ineq', 'fun': constraint},
        options={'maxiter': maxiter, 'ftol': 1e-16},
    )
    out = res.x.reshape(N, D)
    if not np.all(np.isfinite(out)):
        return pts
    return out


# EVOLVE-BLOCK-END