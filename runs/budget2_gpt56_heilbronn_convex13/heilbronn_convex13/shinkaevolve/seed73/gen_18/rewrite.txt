# EVOLVE-BLOCK-START
import numpy as np

try:
    from scipy.optimize import linprog
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


_N = 13
_TRIANGLES = np.array(
    [(i, j, k) for i in range(_N - 2) for j in range(i + 1, _N - 1)
     for k in range(j + 1, _N)],
    dtype=np.intp,
)

# The containing reference triangle has area 1/2.  Consequently an absolute
# determinant is already the triangle area normalized by hull area.
_VERTICES = np.array(
    [[0.0, 0.0],
     [1.0, 0.0],
     [0.0, 1.0]],
    dtype=float,
)


def _determinants(points: np.ndarray) -> np.ndarray:
    tri = points[_TRIANGLES]
    return (
        (tri[:, 1, 0] - tri[:, 0, 0]) *
        (tri[:, 2, 1] - tri[:, 0, 1]) -
        (tri[:, 1, 1] - tri[:, 0, 1]) *
        (tri[:, 2, 0] - tri[:, 0, 0])
    )


def _score(points: np.ndarray) -> tuple[float, int]:
    values = np.abs(_determinants(points))
    worst = int(np.argmin(values))
    return float(values[worst]), worst


def _project_simplex(point: np.ndarray) -> np.ndarray:
    """Project a point to x >= 0, y >= 0, x + y <= 1."""
    x = max(0.0, float(point[0]))
    y = max(0.0, float(point[1]))
    total = x + y
    if total > 1.0:
        x /= total
        y /= total
    return np.array((x, y), dtype=float)


def _ring_seed(rng: np.random.Generator, restart: int) -> np.ndarray:
    """
    A threefold seed in barycentric coordinates.  The outer reference
    vertices remove affine degrees of freedom, while three perturbed inner
    rings and a displaced centre provide varied orientation patterns.
    """
    points = np.empty((_N, 2), dtype=float)
    points[:3] = _VERTICES

    centre = np.array((1.0 / 3.0, 1.0 / 3.0))
    radii = np.array((0.26, 0.49, 0.72))
    radii += rng.normal(0.0, 0.028, 3)
    phase = rng.uniform(-0.13, 0.13) + restart * 0.071

    # Affine images of equally spaced directions based on the three vertices.
    directions = _VERTICES - centre
    index = 3
    for ring, radius in enumerate(radii):
        shift = phase + (ring + 1) * rng.uniform(-0.085, 0.085)
        for j in range(3):
            d0 = directions[(j + ring) % 3]
            d1 = directions[(j + ring + 1) % 3]
            direction = (1.0 - shift) * d0 + shift * d1
            candidate = centre + radius * direction
            candidate += rng.normal(0.0, 0.010 + 0.003 * ring, 2)
            points[index] = _project_simplex(candidate)
            index += 1

    points[12] = _project_simplex(
        centre + rng.normal(0.0, 0.055, 2)
    )
    return points


def _lp_coordinate_step(points: np.ndarray, optimize_x: bool) -> np.ndarray | None:
    """
    With signs and one coordinate block fixed, maximin determinant refinement
    is a linear program.  The first three points are immutable hull vertices.
    """
    if not _HAS_SCIPY:
        return None

    det = _determinants(points)
    signs = np.where(det >= 0.0, 1.0, -1.0)
    count = len(_TRIANGLES)

    # Ten free coordinates followed by the common determinant lower bound.
    a_ub = np.zeros((count, 11), dtype=float)
    b_ub = np.zeros(count, dtype=float)

    fixed = points[:, 1] if optimize_x else points[:, 0]
    coordinate = points[:, 0] if optimize_x else points[:, 1]

    for row, (i, j, k) in enumerate(_TRIANGLES):
        if optimize_x:
            coeffs = (
                fixed[k] - fixed[j],
                fixed[i] - fixed[k],
                fixed[j] - fixed[i],
            )
        else:
            coeffs = (
                fixed[j] - fixed[k],
                fixed[k] - fixed[i],
                fixed[i] - fixed[j],
            )

        constant = 0.0
        for point_index, coefficient in zip((i, j, k), coeffs):
            if point_index < 3:
                constant += coefficient * coordinate[point_index]
            else:
                a_ub[row, point_index - 3] = -signs[row] * coefficient

        a_ub[row, 10] = 1.0
        b_ub[row] = signs[row] * constant

    other = points[3:, 1] if optimize_x else points[3:, 0]
    bounds = [(0.0, max(0.0, 1.0 - float(v))) for v in other]
    bounds.append((0.0, None))

    result = linprog(
        c=np.r_[np.zeros(10), -1.0],
        A_ub=a_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs",
        options={"presolve": True},
    )
    if (not result.success or result.x is None or
            not np.all(np.isfinite(result.x))):
        return None

    candidate = points.copy()
    if optimize_x:
        candidate[3:, 0] = result.x[:10]
    else:
        candidate[3:, 1] = result.x[:10]
    return candidate


def _polish(points: np.ndarray, score: float) -> tuple[np.ndarray, float]:
    """Alternating exact coordinate-block maximin optimization."""
    for _ in range(14):
        changed = False
        for optimize_x in (False, True):
            candidate = _lp_coordinate_step(points, optimize_x)
            if candidate is None:
                continue
            candidate_score, _ = _score(candidate)
            if candidate_score >= score - 1.0e-12:
                if candidate_score > score + 1.0e-10:
                    changed = True
                points = candidate
                score = candidate_score
        if not changed:
            break
    return points, score


def heilbronn_convex13() -> np.ndarray:
    """
    Return a deterministic high-quality configuration of thirteen points.

    All points lie in the reference triangle with vertices (0,0), (1,0),
    and (0,1).  This is a convex containing region, and the returned quality
    is invariant under conversion to any desired unit-area affine image.
    """
    rng = np.random.default_rng(91723)
    best_points = None
    best_score = -np.inf

    # Annealing supplies diverse determinant sign patterns; LP then extracts
    # substantially more value from each pattern than coordinate perturbations.
    for restart in range(12):
        points = _ring_seed(rng, restart)
        score, worst = _score(points)

        for iteration in range(5200):
            fraction = iteration / 5199.0
            step = 0.075 * (1.0 - fraction) ** 1.65 + 0.0012
            temperature = 0.0018 * (1.0 - fraction) ** 2.1 + 0.000004

            candidate = points.copy()
            if rng.random() < 0.86:
                moved = int(_TRIANGLES[worst, rng.integers(0, 3)])
                # Hull points are fixed; choose another point occasionally.
                if moved < 3:
                    moved = int(rng.integers(3, _N))
            else:
                moved = int(rng.integers(3, _N))

            candidate[moved] = _project_simplex(
                candidate[moved] + rng.normal(0.0, step, 2)
            )
            candidate_score, candidate_worst = _score(candidate)
            delta = candidate_score - score

            if delta >= 0.0 or rng.random() < np.exp(delta / temperature):
                points, score, worst = candidate, candidate_score, candidate_worst

        points, score = _polish(points, score)
        if score > best_score:
            best_score = score
            best_points = points.copy()

    if best_points is None or not np.all(np.isfinite(best_points)):
        raise RuntimeError("deterministic Heilbronn search failed")

    return best_points


# EVOLVE-BLOCK-END