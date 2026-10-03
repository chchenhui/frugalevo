# EVOLVE-BLOCK-START
import numpy as np

try:
    from scipy.optimize import linprog
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


_TRIANGLES_13 = np.array(
    [(i, j, k) for i in range(13) for j in range(i + 1, 13)
     for k in range(j + 1, 13)],
    dtype=np.intp,
)

_HULL = np.array(
    [[0.0, 0.0],
     [1.0, 0.0],
     [0.0, 1.0]],
    dtype=float,
)


def _determinants(points: np.ndarray) -> np.ndarray:
    tri = points[_TRIANGLES_13]
    return (
        (tri[:, 1, 0] - tri[:, 0, 0]) *
        (tri[:, 2, 1] - tri[:, 0, 1]) -
        (tri[:, 1, 1] - tri[:, 0, 1]) *
        (tri[:, 2, 0] - tri[:, 0, 0])
    )


def _value(points: np.ndarray) -> float:
    return float(np.min(np.abs(_determinants(points))))


def _triangle_seed(rng: np.random.Generator, mode: int) -> np.ndarray:
    """
    Generate varied deterministic seeds inside the reference triangle.
    The mixed lattice/random family gives substantially more useful
    orientation patterns than purely independent uniform samples.
    """
    if mode % 3 == 0:
        # Perturbed triangular lattice, selected in a deliberately staggered order.
        lattice = np.array([
            [0.13, 0.11], [0.36, 0.09], [0.63, 0.10], [0.83, 0.08],
            [0.09, 0.34], [0.30, 0.29], [0.53, 0.27], [0.71, 0.20],
            [0.15, 0.60], [0.39, 0.43],
        ])
        p = lattice + rng.normal(0.0, 0.040, lattice.shape)
    elif mode % 3 == 1:
        # Stratified barycentric sample.
        u = (np.arange(10) + rng.random(10)) / 10.0
        v = (np.mod(np.arange(10) * 7 + mode, 10) + rng.random(10)) / 10.0
        u, v = np.minimum(u, v), np.maximum(u, v)
        p = np.column_stack((u, 1.0 - v))
        p += rng.normal(0.0, 0.018, p.shape)
    else:
        # Uniform barycentric sample with a weak repulsive ordering.
        z = rng.exponential(1.0, size=(10, 3))
        z /= z.sum(axis=1, keepdims=True)
        p = z[:, 1:3]

    p = np.maximum(p, 0.002)
    sums = p.sum(axis=1)
    over = sums > 0.996
    p[over] *= (0.996 / sums[over])[:, None]
    return np.vstack((_HULL, p))


def _lp_coordinate_step(points: np.ndarray, optimize_x: bool) -> np.ndarray | None:
    """
    Solve max t subject to sign(D_ijk) * D_ijk >= t for every triple.
    Holding one coordinate fixed makes all these constraints linear.
    """
    det = _determinants(points)
    signs = np.where(det >= 0.0, 1.0, -1.0)

    # Variables are coordinate values for points 3..12 followed by t.
    a_ub = np.zeros((len(_TRIANGLES_13), 11), dtype=float)
    b_ub = np.zeros(len(_TRIANGLES_13), dtype=float)

    fixed = points[:, 1] if optimize_x else points[:, 0]
    coord = points[:, 0] if optimize_x else points[:, 1]

    for row, (i, j, k) in enumerate(_TRIANGLES_13):
        if optimize_x:
            # D = x_i(y_k-y_j) + x_j(y_i-y_k) + x_k(y_j-y_i)
            coeff = (fixed[k] - fixed[j], fixed[i] - fixed[k],
                     fixed[j] - fixed[i])
        else:
            # D = y_i(x_j-x_k) + y_j(x_k-x_i) + y_k(x_i-x_j)
            coeff = (fixed[j] - fixed[k], fixed[k] - fixed[i],
                     fixed[i] - fixed[j])

        constant = 0.0
        for idx, c in zip((i, j, k), coeff):
            if idx < 3:
                constant += c * coord[idx]
            else:
                a_ub[row, idx - 3] = -signs[row] * c

        # -sign * (linear + constant) + t <= 0
        a_ub[row, 10] = 1.0
        b_ub[row] = signs[row] * constant

    # Keeping points in x>=0, y>=0, x+y<=1 is just a coordinate bound
    # because the complementary coordinate is held fixed.
    complementary = points[3:, 1] if optimize_x else points[3:, 0]
    bounds = [(0.0, max(0.0, 1.0 - float(q))) for q in complementary]
    bounds.append((0.0, None))

    result = linprog(
        c=np.r_[np.zeros(10), -1.0],
        A_ub=a_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs",
        options={"presolve": True},
    )
    if not result.success or result.x is None or not np.all(np.isfinite(result.x)):
        return None

    candidate = points.copy()
    if optimize_x:
        candidate[3:, 0] = result.x[:10]
    else:
        candidate[3:, 1] = result.x[:10]
    return candidate


def heilbronn_convex13() -> np.ndarray:
    """
    Construct thirteen points in the unit-area reference triangle.

    Alternating LP is deterministic and uses exact global optimization for
    each coordinate block.  The three reference vertices fix affine freedom;
    all remaining points are explicitly constrained to their convex hull.
    """
    if not _HAS_SCIPY:
        # Valid deterministic fallback for environments without SciPy.
        return np.vstack((
            _HULL,
            np.array([
                [0.13, 0.11], [0.36, 0.09], [0.63, 0.10], [0.83, 0.08],
                [0.09, 0.34], [0.30, 0.29], [0.53, 0.27], [0.71, 0.20],
                [0.15, 0.60], [0.39, 0.43],
            ])
        ))

    rng = np.random.default_rng(13031957)
    best = None
    best_score = -1.0

    # Each seed represents a different orientation/sign pattern.  Once a
    # promising pattern is found, LP refinement is much faster than annealing.
    for restart in range(30):
        points = _triangle_seed(rng, restart)
        score = _value(points)

        for _ in range(18):
            improved = False
            for optimize_x in (False, True):
                candidate = _lp_coordinate_step(points, optimize_x)
                if candidate is None:
                    continue
                candidate_score = _value(candidate)
                if candidate_score >= score - 1e-11:
                    if candidate_score > score + 1e-10:
                        improved = True
                    points, score = candidate, candidate_score
            if not improved:
                break

        if score > best_score:
            best_score = score
            best = points.copy()

    return best


# EVOLVE-BLOCK-END