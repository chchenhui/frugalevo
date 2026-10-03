# EVOLVE-BLOCK-START
import numpy as np


_SQRT3 = float(np.sqrt(3.0))
_HEIGHT = 0.5 * _SQRT3
_N = 11
_FREE = 8

# Affine coordinates (u, v):
# Cartesian coordinate = (u + v/2, sqrt(3)*v/2).
_CORNERS = np.array(
    [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
    dtype=float,
)

_TRIPLES = np.array(
    [
        (i, j, k)
        for i in range(_N)
        for j in range(i + 1, _N)
        for k in range(j + 1, _N)
    ],
    dtype=np.intp,
)


def _to_cartesian(uv: np.ndarray) -> np.ndarray:
    """Convert affine simplex coordinates to Cartesian coordinates."""
    out = np.empty_like(uv, dtype=float)
    out[:, 0] = uv[:, 0] + 0.5 * uv[:, 1]
    out[:, 1] = _HEIGHT * uv[:, 1]
    return out


def _project_simplex(uv: np.ndarray) -> np.ndarray:
    """Project arbitrary points into u >= 0, v >= 0, u + v <= 1."""
    z = np.asarray(uv, dtype=float).copy()
    z = np.maximum(z, 0.0)
    total = z[..., 0] + z[..., 1]
    outside = total > 1.0
    if np.any(outside):
        z[outside] /= total[outside, None]
    return z


def _signed_determinants(free: np.ndarray) -> np.ndarray:
    """All 165 signed normalized triangle areas in affine coordinates."""
    points = np.vstack((_CORNERS, free))
    q = points[_TRIPLES]
    return (
        (q[:, 1, 0] - q[:, 0, 0]) * (q[:, 2, 1] - q[:, 0, 1])
        - (q[:, 1, 1] - q[:, 0, 1]) * (q[:, 2, 0] - q[:, 0, 0])
    )


def _minimum_area(free: np.ndarray) -> float:
    """Minimum triangle area normalized by the enclosing triangle area."""
    return float(np.min(np.abs(_signed_determinants(free))))


def _feasible(free: np.ndarray) -> bool:
    return bool(
        np.all(np.isfinite(free))
        and np.all(free >= -1.0e-10)
        and np.all(np.sum(free, axis=1) <= 1.0 + 1.0e-10)
    )


def _tail_value(free: np.ndarray) -> float:
    """
    Global-search objective.

    The minimum determinant dominates the score.  A small lower-tail bonus
    differentiates otherwise equal plateaus and encourages simultaneous repair
    of several nearly active constraints.
    """
    violation = np.maximum(-free, 0.0).sum()
    violation += np.maximum(free.sum(axis=1) - 1.0, 0.0).sum()

    if violation > 0.0:
        return -2.0 - 10.0 * violation

    areas = np.abs(_signed_determinants(free))
    smallest = np.partition(areas, 11)[:12]
    return float(smallest[0] + 0.0025 * np.mean(smallest[1:]))


def _structured_seeds() -> list:
    """Several symmetry-related staggered arrangements for global seeding."""
    base = np.array(
        [
            [0.145, 0.070],
            [0.445, 0.055],
            [0.755, 0.070],
            [0.075, 0.335],
            [0.355, 0.275],
            [0.650, 0.235],
            [0.145, 0.625],
            [0.420, 0.470],
        ],
        dtype=float,
    )

    reflected = base.copy()
    reflected[:, 0] = 1.0 - base[:, 0] - base[:, 1]

    swapped = base[:, ::-1].copy()
    swapped = _project_simplex(swapped)

    return [base, reflected, swapped]


def _polish_epigraph(layout: np.ndarray) -> np.ndarray:
    """
    Solve a fixed-orientation epigraph problem.

    Once a promising arrangement has been found globally, signs of all
    determinants define a smooth chamber.  SLSQP then maximizes t subject to
    sign(det_i) * det_i >= t for every triple.
    """
    try:
        from scipy.optimize import minimize

        signed0 = _signed_determinants(layout)
        orientation = np.sign(signed0)
        orientation[orientation == 0.0] = 1.0
        initial = _minimum_area(layout)

        def constraint_values(x: np.ndarray) -> np.ndarray:
            free = x[:-1].reshape(_FREE, 2)
            level = x[-1]
            signed = _signed_determinants(free)
            return np.concatenate(
                (
                    orientation * signed - level,
                    1.0 - free.sum(axis=1),
                )
            )

        result = minimize(
            fun=lambda x: -x[-1],
            x0=np.r_[layout.ravel(), initial],
            method="SLSQP",
            bounds=[(0.0, 1.0)] * (2 * _FREE) + [(0.0, 0.2)],
            constraints={"type": "ineq", "fun": constraint_values},
            options={
                "maxiter": 1800,
                "ftol": 2.0e-13,
                "disp": False,
            },
        )

        if result.success and np.all(np.isfinite(result.x)):
            candidate = _project_simplex(result.x[:-1].reshape(_FREE, 2))
            if _feasible(candidate) and _minimum_area(candidate) >= _minimum_area(layout):
                return candidate
    except Exception:
        pass

    return layout


def _global_search() -> np.ndarray:
    """
    Deterministic global evolutionary search followed by chamber polishing.

    Differential evolution is intentionally used as a population-wide global
    optimizer rather than as a local mutation process.  It can cross the
    determinant-zero walls separating different combinatorial arrangements.
    """
    try:
        from scipy.optimize import differential_evolution

        rng_seed = 11031987
        seeds = _structured_seeds()
        candidates = []

        # Independent seeded global searches improve coverage of the important
        # reflected and rotated chambers without nondeterminism.
        for seed_number, seed_layout in enumerate(seeds):
            def objective(x: np.ndarray) -> float:
                return -_tail_value(x.reshape(_FREE, 2))

            result = differential_evolution(
                objective,
                bounds=[(0.0, 1.0)] * (2 * _FREE),
                strategy="currenttobest1bin",
                maxiter=700,
                popsize=18,
                tol=2.0e-8,
                atol=1.0e-10,
                mutation=(0.45, 1.15),
                recombination=0.78,
                seed=rng_seed + seed_number,
                polish=False,
                updating="immediate",
                workers=1,
                x0=seed_layout.ravel(),
            )

            if np.all(np.isfinite(result.x)):
                trial = _project_simplex(result.x.reshape(_FREE, 2))
                candidates.append(trial)

        candidates.extend(seeds)

        best = None
        best_value = -np.inf

        # Each candidate enters its own smooth orientation chamber.  Retaining
        # multiple chambers avoids relying on whichever DE individual happened
        # to have the best soft lower-tail score.
        for candidate in candidates:
            polished = _polish_epigraph(candidate)
            value = _minimum_area(polished)
            if value > best_value:
                best_value = value
                best = polished

        if best is None or not _feasible(best):
            raise RuntimeError("global optimization did not yield a layout")

        return best
    except Exception:
        # This is only used if SciPy or a numerical optimizer is unavailable.
        return np.array(
            [
                [0.145, 0.070],
                [0.445, 0.055],
                [0.755, 0.070],
                [0.075, 0.335],
                [0.355, 0.275],
                [0.650, 0.235],
                [0.145, 0.625],
                [0.420, 0.470],
            ],
            dtype=float,
        )


def _build_layout() -> np.ndarray:
    """Construct, validate, and cache the Cartesian eleven-point arrangement."""
    try:
        free = _global_search()
        affine = np.vstack((_CORNERS, free))
        result = _to_cartesian(affine)

        if result.shape != (11, 2) or not np.all(np.isfinite(result)):
            raise RuntimeError("invalid output")
        return result
    except Exception:
        affine = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.145, 0.070],
                [0.445, 0.055],
                [0.755, 0.070],
                [0.075, 0.335],
                [0.355, 0.275],
                [0.650, 0.235],
                [0.145, 0.625],
                [0.420, 0.470],
            ],
            dtype=float,
        )
        return _to_cartesian(affine)


_CACHED_LAYOUT = _build_layout()


def heilbronn_triangle11() -> np.ndarray:
    """
    Return eleven deterministic points in the unit equilateral triangle.

    Returns
    -------
    numpy.ndarray
        Cartesian coordinates with shape ``(11, 2)``.
    """
    return _CACHED_LAYOUT.copy()


# EVOLVE-BLOCK-END